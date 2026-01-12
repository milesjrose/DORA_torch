# DORA_bridge/load_network_from_json.py
# Script to load a Network object from JSON state produced by NewNetworkStateGenerator.

import json
from pathlib import Path
from logging import getLogger
logger = getLogger(__name__)
from typing import Dict, Any, Union, Optional

import torch

# Import from nodes/network
from nodes.network import Network, Params, Tokens, Semantics, Token_Tensor
from nodes.network.tokens import Links, Mapping
from nodes.network.tokens.connections.connections import Connections_Tensor
from nodes.enums import TF, SF, Type, Set, Mode, B, MappingFields, OntStatus, null, tensor_type


class NetworkLoader:
    """
    Load a Network object from JSON state produced by NewNetworkStateGenerator.
    
    Example usage:
        >>> loader = NetworkLoader()
        >>> network = loader.load('test_data/state.json')
        >>> # Or with chaining
        >>> network = NetworkLoader().load('test_data/state.json')
    """
    
    def __init__(self):
        """Initialize the NetworkLoader."""
        self._state: Optional[Dict[str, Any]] = None
        self._set_map = {
            'driver': Set.DRIVER,
            'recipient': Set.RECIPIENT,
            'memory': Set.MEMORY,
            'newSet': Set.NEW_SET,
            'new_set': Set.NEW_SET,
        }
        self._type_map = {
            'P': Type.P,
            'RB': Type.RB,
            'PO': Type.PO,
        }
        self._mode_map = {
            'CHILD': Mode.CHILD,
            'NEUTRAL': Mode.NEUTRAL,
            'PARENT': Mode.PARENT,
        }
        self._ont_map = {
            'state': OntStatus.STATE,
            'value': OntStatus.VALUE,
            'sdm': OntStatus.SDM,
            'ho': OntStatus.HO,
        }
    
    def load(self, file_path: Union[str, Path]) -> Network:
        """
        Load a Network from a JSON state file.
        
        Args:
            file_path: Path to the JSON state file
            
        Returns:
            Network: The reconstructed Network object
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the JSON structure is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"State file not found: {file_path}")
        
        logger.info(f"Loading network from {file_path}")

        with open(file_path, 'r') as f:
            self._state = json.load(f)
        
        return self.load_from_state()
        
    
    def load_from_state(self, state: Dict[str, Any] = None) -> Network:
        """
        Load a Network from a state dictionary.
        
        Args:
            state: State dictionary
        
        Returns:
            Network: The reconstructed Network object
        """
        if state is not None:
            self._state = state
        else:
            if self._state is None:
                raise ValueError("No state/file provided.")

        # Build all components
        connections = self._build_connections()
        token_tensor = self._build_token_tensor(connections)
        links = self._build_links()
        mapping = self._build_mapping()
        semantics = self._build_semantics()
        params = self._build_params()
        
        # Create Tokens container
        tokens = Tokens(token_tensor, connections, links, mapping)
        
        # Create and return Network
        network = Network(tokens, semantics, mapping, links, params)
        
        # Restore activation states
        self._restore_activations(network)
        
        # Recache to ensure masks are up to date
        network.recache()

        logger.info("Network loaded successfully!")

        self._print_summary(network)
        
        return network
    
    def _get_all_tokens(self) -> list:
        """Get all tokens from the state in a flat list with their type."""
        all_tokens = []
        for p in self._state['tokens']['Ps']:
            all_tokens.append(('P', p))
        for rb in self._state['tokens']['RBs']:
            all_tokens.append(('RB', rb))
        for po in self._state['tokens']['POs']:
            all_tokens.append(('PO', po))
        # Sort by original index to preserve order
        all_tokens.sort(key=lambda x: x[1]['index'])
        return all_tokens
    
    def _build_token_tensor(self, connections: Connections_Tensor) -> Token_Tensor:
        """Build the Token_Tensor from state."""
        all_tokens = self._get_all_tokens()
        num_tokens = len(all_tokens)
        
        if num_tokens == 0:
            # Create minimal empty tensor
            tokens_data = torch.zeros(1, len(TF), dtype=tensor_type)
            tokens_data[0, TF.DELETED] = B.TRUE
            names = {}
            return Token_Tensor(tokens_data, connections, names)
        
        # Create token tensor
        tokens_data = torch.full((num_tokens, len(TF)), null, dtype=tensor_type)
        names = {}
        
        for i, (token_type, token_data) in enumerate(all_tokens):
            original_idx = token_data['index']
            
            # Basic properties
            tokens_data[i, TF.ID] = original_idx + 1  # 1-indexed IDs
            tokens_data[i, TF.TYPE] = self._type_map[token_type]
            tokens_data[i, TF.SET] = self._set_map.get(token_data['set'], Set.MEMORY)
            tokens_data[i, TF.ANALOG] = token_data['analog'] if token_data['analog'] is not None else null
            
            # Activation states
            tokens_data[i, TF.ACT] = token_data.get('act', 0.0)
            tokens_data[i, TF.NET_INPUT] = token_data.get('net_input', 0.0)
            tokens_data[i, TF.TD_INPUT] = token_data.get('td_input', 0.0)
            tokens_data[i, TF.BU_INPUT] = token_data.get('bu_input', 0.0)
            tokens_data[i, TF.LATERAL_INPUT] = token_data.get('lateral_input', 0.0)
            tokens_data[i, TF.MAP_INPUT] = token_data.get('map_input', 0.0)
            tokens_data[i, TF.MAX_MAP] = token_data.get('max_map', 0.0)
            
            # Bool flags
            tokens_data[i, TF.INFERRED] = B.TRUE if token_data.get('inferred', False) else B.FALSE
            tokens_data[i, TF.DELETED] = B.FALSE
            
            # Type-specific properties
            if token_type == 'PO':
                pred_or_obj = token_data.get('predOrObj', 0)
                tokens_data[i, TF.PRED] = B.TRUE if pred_or_obj == 1 else B.FALSE
            
            # Store name
            names[i] = token_data['name']
        
        return Token_Tensor(tokens_data, connections, names)
    
    def _build_connections(self) -> Connections_Tensor:
        """Build the Connections_Tensor from state."""
        all_tokens = self._get_all_tokens()
        num_tokens = max(len(all_tokens), 1)
        
        connections_data = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
        
        if 'connections' not in self._state:
            return Connections_Tensor(connections_data)
        
        # Build name to index mapping
        name_to_idx = {}
        for i, (_, token_data) in enumerate(all_tokens):
            name_to_idx[token_data['name']] = i
        
        # Process P -> RB connections
        for conn in self._state['connections'].get('P_to_RB', []):
            parent_name = conn['parent']
            child_name = conn['child']
            if parent_name in name_to_idx and child_name in name_to_idx:
                parent_idx = name_to_idx[parent_name]
                child_idx = name_to_idx[child_name]
                connections_data[parent_idx, child_idx] = True
        
        # Process RB -> PO connections
        for conn in self._state['connections'].get('RB_to_PO', []):
            parent_name = conn['parent']
            child_name = conn['child']
            if parent_name in name_to_idx and child_name in name_to_idx:
                parent_idx = name_to_idx[parent_name]
                child_idx = name_to_idx[child_name]
                connections_data[parent_idx, child_idx] = True
        
        # Process RB -> child P connections (higher-order)
        for conn in self._state['connections'].get('RB_to_childP', []):
            parent_name = conn['parent']
            child_name = conn['child']
            if parent_name in name_to_idx and child_name in name_to_idx:
                parent_idx = name_to_idx[parent_name]
                child_idx = name_to_idx[child_name]
                connections_data[parent_idx, child_idx] = True
        
        return Connections_Tensor(connections_data)
    
    def _build_links(self) -> Links:
        """Build the Links tensor from state."""
        all_tokens = self._get_all_tokens()
        num_tokens = max(len(all_tokens), 1)
        num_sems = max(len(self._state.get('semantics', [])), 1)
        
        links_data = torch.zeros(num_tokens, num_sems, dtype=tensor_type)
        
        if 'links' not in self._state:
            return Links(links_data)
        
        # Build name to index mappings
        token_name_to_idx = {}
        for i, (_, token_data) in enumerate(all_tokens):
            token_name_to_idx[token_data['name']] = i
        
        sem_name_to_idx = {}
        for i, sem_data in enumerate(self._state.get('semantics', [])):
            sem_name_to_idx[sem_data['name']] = i
        
        # Process links list
        for link in self._state['links'].get('links_list', []):
            po_name = link['po_name']
            sem_name = link['sem_name']
            weight = link['weight']
            
            if po_name in token_name_to_idx and sem_name in sem_name_to_idx:
                po_idx = token_name_to_idx[po_name]
                sem_idx = sem_name_to_idx[sem_name]
                links_data[po_idx, sem_idx] = weight
        
        return Links(links_data)
    
    def _build_mapping(self) -> Mapping:
        """Build the Mapping tensor from state."""
        # Count driver and recipient tokens
        all_tokens = self._get_all_tokens()
        
        driver_count = sum(1 for _, t in all_tokens if t['set'] == 'driver')
        recipient_count = sum(1 for _, t in all_tokens if t['set'] == 'recipient')
        
        driver_count = max(driver_count, 1)
        recipient_count = max(recipient_count, 1)
        
        # Create mapping tensor: [recipient, driver, fields]
        mapping_data = torch.zeros(recipient_count, driver_count, len(MappingFields), dtype=tensor_type)
        
        if 'mappings' not in self._state:
            return Mapping(mapping_data)
        
        # Build name to local index mappings
        driver_tokens = [(i, t) for i, (_, t) in enumerate(all_tokens) if t['set'] == 'driver']
        recipient_tokens = [(i, t) for i, (_, t) in enumerate(all_tokens) if t['set'] == 'recipient']
        
        driver_name_to_local = {t['name']: local_idx for local_idx, (_, t) in enumerate(driver_tokens)}
        recipient_name_to_local = {t['name']: local_idx for local_idx, (_, t) in enumerate(recipient_tokens)}
        
        # Process mappings
        for mapping in self._state['mappings'].get('all_mappings', []):
            driver_name = mapping['driver_name']
            recipient_name = mapping['recipient_name']
            weight = mapping['weight']
            
            if driver_name in driver_name_to_local and recipient_name in recipient_name_to_local:
                d_local = driver_name_to_local[driver_name]
                r_local = recipient_name_to_local[recipient_name]
                mapping_data[r_local, d_local, MappingFields.WEIGHT] = weight
        
        return Mapping(mapping_data)
    
    def _build_semantics(self) -> Semantics:
        """Build the Semantics object from state."""
        sem_list = self._state.get('semantics', [])
        num_sems = len(sem_list)
        
        if num_sems == 0:
            # Create empty semantics with one placeholder
            nodes = torch.zeros(1, len(SF), dtype=tensor_type)
            nodes[0, SF.DELETED] = B.TRUE
            connections = torch.zeros(1, 1, dtype=tensor_type)
            ids = {}
            names = {}
            return Semantics(nodes, connections, ids, names)
        
        # Create semantic nodes tensor
        nodes = torch.zeros(num_sems, len(SF), dtype=tensor_type)
        ids = {}
        names = {}
        dimensions = {}
        
        for i, sem_data in enumerate(sem_list):
            sem_id = i + 1  # 1-indexed IDs
            
            nodes[i, SF.ID] = sem_id
            nodes[i, SF.TYPE] = Type.SEMANTIC
            nodes[i, SF.ACT] = sem_data.get('act', 0.0)
            nodes[i, SF.DELETED] = B.FALSE
            
            # Handle amount
            amount = sem_data.get('amount')
            if amount is not None:
                nodes[i, SF.AMOUNT] = amount
            else:
                nodes[i, SF.AMOUNT] = null
            
            # Handle ont_status
            ont_status = sem_data.get('ont_status')
            if ont_status is not None and ont_status in self._ont_map:
                nodes[i, SF.ONT] = self._ont_map[ont_status]
            else:
                nodes[i, SF.ONT] = null
            
            # Handle dimension
            dimension = sem_data.get('dimension')
            if dimension is not None and dimension != 'nil':
                # Create or get dimension key
                dim_key = None
                for k, v in dimensions.items():
                    if v == dimension:
                        dim_key = k
                        break
                if dim_key is None:
                    dim_key = len(dimensions) + 1
                    dimensions[dim_key] = dimension
                nodes[i, SF.DIM] = dim_key
            else:
                nodes[i, SF.DIM] = null
            
            ids[sem_id] = i
            names[sem_id] = sem_data['name']
        
        # Create semantic connections (none for now)
        connections = torch.zeros(num_sems, num_sems, dtype=tensor_type)
        
        semantics = Semantics(nodes, connections, ids, names)
        semantics.dimensions = dimensions
        
        return semantics
    
    def _build_params(self) -> Params:
        """Build the Params object from state metadata."""
        metadata = self._state.get('metadata', {})
        parameters = metadata.get('parameters', {})
        
        # Map parameter names to expected format
        param_mapping = {
            'asDORA': 'asDORA',
            'gamma': 'gamma',
            'delta': 'delta',
            'eta': 'eta',
            'HebbBias': 'HebbBias',
            'bias_retrieval_analogs': 'bias_retrieval_analogs',
            'use_relative_act': 'use_relative_act',
            'run_order': 'run_order',
            'run_cyles': 'run_cycles',  # Note: typo in source
            'run_cycles': 'run_cycles',
            'firingOrderRule': 'firingOrderRule',
            'strategic_mapping': 'strategic_mapping',
            'ignore_object_semantics': 'ignore_object_semantics',
            'ignore_memory_semantics': 'ignore_memory_semantics',
            'mag_decimal_precision': 'mag_decimal_precision',
            'exemplar_memory': 'exemplar_memory',
            'recent_analog_bias': 'recent_analog_bias',
            'lateral_input_level': 'lateral_input_level',
            'screen_width': 'screen_width',
            'screen_height': 'screen_height',
            'doGUI': 'doGUI',
            'GUI_update_rate': 'GUI_update_rate',
            'starting_iteration': 'starting_iteration',
            'ho_sem_act_flow': 'ho_sem_act_flow',
            'tokenize': 'tokenize',
            'remove_uncompressed': 'remove_uncompressed',
            'remove_compressed': 'remove_compressed',
        }
        
        mapped_params = {}
        for src_key, dest_key in param_mapping.items():
            if src_key in parameters:
                mapped_params[dest_key] = parameters[src_key]
        
        return Params(mapped_params)
    
    def _restore_activations(self, network: Network):
        """Restore any activation states that weren't set during tensor creation."""
        # Most activations are already restored in _build_token_tensor
        # This method can be extended if additional restoration is needed
        pass
    
    def _print_summary(self, network: Network):
        """Print a summary of the loaded network."""
        token_tensor = network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        num_ps = ((token_tensor.tensor[:, TF.TYPE] == Type.P) & non_deleted).sum().item()
        num_rbs = ((token_tensor.tensor[:, TF.TYPE] == Type.RB) & non_deleted).sum().item()
        num_pos = ((token_tensor.tensor[:, TF.TYPE] == Type.PO) & non_deleted).sum().item()
        num_sems = len(network.semantics.IDs)
        
        logger.info(f"  Ps: {num_ps}, RBs: {num_rbs}, POs: {num_pos}, Semantics: {num_sems}")


def load_network_from_json(file_path: Union[str, Path]) -> Network:
    """
    Convenience function to load a Network from a JSON state file.
    
    Args:
        file_path: Path to the JSON state file
        
    Returns:
        Network: The reconstructed Network object
        
    Example:
        >>> network = load_network_from_json('test_data/state.json')
    """
    return NetworkLoader().load(file_path)

def load_from_state(state: Dict[str, Any]) -> Network:
    """
    Load a Network from a state dictionary.
    
    Args:
        state: State dictionary
    
    Returns:
        Network: The reconstructed Network object
    """
    return NetworkLoader().load_from_state(state)


# ==================[ Main / Example Usage ]==================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_network_from_json.py <json_state_file>")
        print("\nExample:")
        print("  python load_network_from_json.py test_data/compnew.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    print(f"\nLoading network from: {json_file}")
    print("-" * 50)
    
    try:
        network = load_network_from_json(json_file)
        
        print("\n" + "=" * 50)
        print("Network loaded successfully!")
        print("=" * 50)
        
        # Print additional details
        print("\nDriver set:")
        driver_mask = network.token_tensor.tensor[:, TF.SET] == Set.DRIVER
        driver_mask &= network.token_tensor.tensor[:, TF.DELETED] == B.FALSE
        driver_indices = torch.where(driver_mask)[0]
        for idx in driver_indices[:10]:  # Show first 10
            name = network.token_tensor.names.get(idx.item(), f"token_{idx.item()}")
            token_type = Type(int(network.token_tensor.tensor[idx, TF.TYPE].item())).name
            print(f"  [{idx.item()}] {name} ({token_type})")
        if len(driver_indices) > 10:
            print(f"  ... and {len(driver_indices) - 10} more")
        
        print("\nRecipient set:")
        recipient_mask = network.token_tensor.tensor[:, TF.SET] == Set.RECIPIENT
        recipient_mask &= network.token_tensor.tensor[:, TF.DELETED] == B.FALSE
        recipient_indices = torch.where(recipient_mask)[0]
        for idx in recipient_indices[:10]:  # Show first 10
            name = network.token_tensor.names.get(idx.item(), f"token_{idx.item()}")
            token_type = Type(int(network.token_tensor.tensor[idx, TF.TYPE].item())).name
            print(f"  [{idx.item()}] {name} ({token_type})")
        if len(recipient_indices) > 10:
            print(f"  ... and {len(recipient_indices) - 10} more")
        
        print("\nSemantics:")
        for sem_id, sem_idx in list(network.semantics.IDs.items())[:10]:
            name = network.semantics.names.get(sem_id, f"sem_{sem_id}")
            print(f"  [{sem_idx}] {name}")
        if len(network.semantics.IDs) > 10:
            print(f"  ... and {len(network.semantics.IDs) - 10} more")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading network: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

