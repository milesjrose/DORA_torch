# DORA_bridge/new_network_state_generator.py
# Class for generating test data from the new nodes/network implementation.
# Produces the same JSON structure as TestDataGenerator for comparison.

import json
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from logging import getLogger
logger = getLogger(__name__)

import torch

# Import from nodes/network
from nodes.network import Network, Params
from nodes.builder import NetworkBuilder
from nodes.enums import TF, SF, Type, Set, Mode, B, MappingFields, OntStatus, null


class NewNetworkStateGenerator:
    """
    Generate and inspect test data from the new nodes/network implementation.
    
    This class provides the same interface and output structure as TestDataGenerator,
    enabling direct comparison between the old currVers implementation and the new
    tensorized implementation.
    
    Example usage:
        >>> gen = NewNetworkStateGenerator()
        >>> gen.load_sim('sims/testsim15.py')
        >>> state = gen.get_state()
        >>> gen.save_state('test_data/new_state.json', format='json')
        
        # Compare with old implementation:
        >>> old_state = load_state_json('test_data/old_state.json')
        >>> new_state = load_state_json('test_data/new_state.json')
    """
    
    def __init__(self, parameters: Optional[Dict] = None):
        """
        Initialize the NewNetworkStateGenerator.
        
        Args:
            parameters: Optional dict of DORA parameters. If None, uses defaults.
        """
        self.network: Optional[Network] = None
        self.parameters = parameters or self._default_parameters()
        self._sim_path = None
    
    def _default_parameters(self) -> Dict:
        """Return default DORA parameters."""
        return {
            "asDORA": True,
            "gamma": 0.3,
            "delta": 0.1,
            "eta": 0.9,
            "HebbBias": 0.5,
            "bias_retrieval_analogs": True,
            "use_relative_act": True,
            "run_order": ["cdr", "selectTokens", "r", "wp", "m", "p", "s", "f", "c"],
            "run_cyles": 5000,
            "write_network_state": False,
            "write_on_iteration": 100,
            "write_unit_states": False,
            "firingOrderRule": "random",
            "strategic_mapping": False,
            "ignore_object_semantics": False,
            "ignore_memory_semantics": True,
            "mag_decimal_precision": 0,
            "exemplar_memory": False,
            "recent_analog_bias": True,
            "driver_bias_on": True,
            "driver_bias_start_size": 2,
            "turn_driver_bias_off": False,
            "iters_of_driver_bias": 1000,
            "turn_driver_bias_off_size": 4,
            "lateral_input_level": 1,
            "screen_width": 1200,
            "screen_height": 700,
            "doGUI": False,
            "GUI_update_rate": 1,
            "starting_iteration": 0,
            "tokenize": False,
            "ho_sem_act_flow": 0,
            "remove_uncompressed": False,
            "remove_compressed": False,
        }
    
    # ==================[ Loading Functions ]==================
    
    def load_sim(self, sim_path: Union[str, Path]) -> 'NewNetworkStateGenerator':
        """
        Load a simulation file into the new network.
        
        Args:
            sim_path: Path to the simulation file (.py format)
            
        Returns:
            self (for method chaining)
            
        Raises:
            FileNotFoundError: If sim file doesn't exist
            ValueError: If sim file format is invalid
        """
        sim_path = Path(sim_path)
        if not sim_path.exists():
            raise FileNotFoundError(f"Simulation file not found: {sim_path}")
        
        self._sim_path = sim_path
        
        # Use NetworkBuilder to load and build the network
        builder = NetworkBuilder(file_path=str(sim_path))
        self.network = builder.build_network()
        
        # Recache to ensure masks are up to date
        self.network.recache()
        
        # Print summary
        token_tensor = self.network.token_tensor
        num_ps = (token_tensor.tensor[:, TF.TYPE] == Type.P).sum().item()
        num_rbs = (token_tensor.tensor[:, TF.TYPE] == Type.RB).sum().item()
        num_pos = (token_tensor.tensor[:, TF.TYPE] == Type.PO).sum().item()
        num_sems = len(self.network.semantics.IDs)
        
        logger.info(f"Loaded simulation from {sim_path}")
        print(f"  Ps: {num_ps}, RBs: {num_rbs}, POs: {num_pos}, Semantics: {num_sems}")
        
        return self
    
    def load_props(self, symProps: List[Dict]) -> 'NewNetworkStateGenerator':
        """
        Load a network from symProps directly (without a file).
        
        Args:
            symProps: List of proposition dictionaries
            
        Returns:
            self (for method chaining)
        """
        self._sim_path = None
        
        # Use NetworkBuilder to build the network from props
        builder = NetworkBuilder(symProps=symProps)
        self.network = builder.build_network()
        
        # Recache to ensure masks are up to date
        self.network.recache()
        
        # Print summary
        token_tensor = self.network.token_tensor
        num_ps = (token_tensor.tensor[:, TF.TYPE] == Type.P).sum().item()
        num_rbs = (token_tensor.tensor[:, TF.TYPE] == Type.RB).sum().item()
        num_pos = (token_tensor.tensor[:, TF.TYPE] == Type.PO).sum().item()
        num_sems = len(self.network.semantics.IDs)
        
        print(f"Loaded network from props")
        logger.info(f"  Ps: {num_ps}, RBs: {num_rbs}, POs: {num_pos}, Semantics: {num_sems}")
        
        return self
    
    # ==================[ State Extraction ]==================
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the complete detailed state of the new network.
        
        Returns:
            Dict containing:
                - tokens: All P, RB, PO tokens with properties
                - semantics: All semantic units with properties
                - links: PO-semantic link matrix with weights
                - mappings: All mapping connections with weights
                - analogs: Analog structure
                - driver: Current driver set contents
                - recipient: Current recipient set contents
                - metadata: Sim file path, parameters, etc.
        
        Raises:
            ValueError: If no network is loaded
        """
        if self.network is None:
            raise ValueError("No network loaded. Call load_sim() first.")
        
        return {
            'tokens': self._extract_tokens(),
            'semantics': self._extract_semantics(),
            'links': self._extract_links(),
            'mappings': self._extract_mappings(),
            'analogs': self._extract_analogs(),
            'connections': self._extract_connections(),
            'driver': self._extract_set_contents(Set.DRIVER),
            'recipient': self._extract_set_contents(Set.RECIPIENT),
            'metadata': self._extract_metadata(),
        }
    
    def _extract_tokens(self) -> Dict[str, List[Dict]]:
        """Extract all token data."""
        tokens = {'Ps': [], 'RBs': [], 'POs': []}
        
        token_tensor = self.network.token_tensor
        connections = self.network.tokens.connections
        
        # Get all non-deleted tokens
        non_deleted_mask = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        all_indices = torch.where(non_deleted_mask)[0]
        
        for idx in all_indices:
            idx_int = idx.item()
            tensor_row = token_tensor.tensor[idx_int]
            token_type = Type(int(tensor_row[TF.TYPE].item()))
            
            # Common properties for all token types
            base_props = {
                'index': idx_int,
                'name': token_tensor.names.get(idx_int, f"token_{idx_int}"),
                'set': self._set_to_string(Set(int(tensor_row[TF.SET].item()))),
                'analog': int(tensor_row[TF.ANALOG].item()) if tensor_row[TF.ANALOG].item() != null else None,
                'act': float(tensor_row[TF.ACT].item()),
                'net_input': float(tensor_row[TF.NET_INPUT].item()),
                'td_input': float(tensor_row[TF.TD_INPUT].item()),
                'bu_input': float(tensor_row[TF.BU_INPUT].item()),
                'lateral_input': float(tensor_row[TF.LATERAL_INPUT].item()),
                'map_input': float(tensor_row[TF.MAP_INPUT].item()),
                'max_map': float(tensor_row[TF.MAX_MAP].item()),
                'max_map_unit_name': self._get_max_map_unit_name(idx_int),
                'inferred': bool(tensor_row[TF.INFERRED].item() == B.TRUE),
            }
            
            if token_type == Type.P:
                # Get child RBs (children are RB type)
                child_rb_names = self._get_child_names(idx_int, Type.RB)
                # Get parent RBs (look for RBs that have this P as child)
                parent_rb_names = self._get_parent_rb_names_for_p(idx_int)
                
                tokens['Ps'].append({
                    **base_props,
                    'child_RB_names': child_rb_names,
                    'parent_RB_names': parent_rb_names,
                })
                
            elif token_type == Type.RB:
                # Get parent Ps
                parent_p_names = self._get_parent_names(idx_int, Type.P)
                # Get pred and obj POs
                pred_name, obj_name = self._get_rb_pred_obj_names(idx_int)
                # Get child P (for higher-order)
                child_p_name = self._get_rb_child_p_name(idx_int)
                
                tokens['RBs'].append({
                    **base_props,
                    'parent_P_names': parent_p_names,
                    'pred_name': pred_name,
                    'obj_name': obj_name,
                    'child_P_name': child_p_name,
                })
                
            elif token_type == Type.PO:
                # Get parent RBs
                parent_rb_names = self._get_parent_names(idx_int, Type.RB)
                # Get semantic names
                semantic_names = self._get_po_semantic_names(idx_int)
                # Get predOrObj
                pred_or_obj = 1 if tensor_row[TF.PRED].item() == B.TRUE else 0
                
                tokens['POs'].append({
                    **base_props,
                    'predOrObj': pred_or_obj,
                    'parent_RB_names': parent_rb_names,
                    'semantic_names': semantic_names,
                })
        
        return tokens
    
    def _extract_semantics(self) -> List[Dict]:
        """Extract all semantic unit data."""
        semantics = []
        sem_obj = self.network.semantics
        
        for sem_id, sem_idx in sem_obj.IDs.items():
            tensor_row = sem_obj.nodes[sem_idx]
            
            # Get dimension name
            dim_key = int(tensor_row[SF.DIM].item()) if tensor_row[SF.DIM].item() != null else None
            dimension = sem_obj.dimensions.get(dim_key, 'nil') if dim_key else 'nil'
            
            # Get ont_status
            ont_val = int(tensor_row[SF.ONT].item()) if tensor_row[SF.ONT].item() != null else None
            ont_status = OntStatus(ont_val).name.lower() if ont_val is not None else None
            
            # Get amount
            amount = float(tensor_row[SF.AMOUNT].item()) if tensor_row[SF.AMOUNT].item() != null else None
            
            semantics.append({
                'index': sem_idx,
                'name': sem_obj.names.get(sem_id, f"sem_{sem_id}"),
                'act': float(tensor_row[SF.ACT].item()),
                'dimension': dimension,
                'amount': amount,
                'ont_status': ont_status,
            })
        
        return semantics
    
    def _extract_links(self) -> Dict:
        """Extract PO-semantic links as a matrix and detailed list."""
        token_tensor = self.network.token_tensor
        links = self.network.links
        sem_obj = self.network.semantics
        
        # Get PO indices
        po_mask = (token_tensor.tensor[:, TF.TYPE] == Type.PO) & (token_tensor.tensor[:, TF.DELETED] == B.FALSE)
        po_indices = torch.where(po_mask)[0]
        
        num_pos = len(po_indices)
        num_sems = len(sem_obj.IDs)
        
        # Create the matrix (using PO order as found)
        matrix = [[0.0] * num_sems for _ in range(num_pos)]
        
        # Build PO names list in order
        po_names = [token_tensor.names.get(idx.item(), f"po_{idx.item()}") for idx in po_indices]
        
        # Build semantic names list (in ID order)
        semantic_names = []
        sem_idx_map = {}  # sem_idx -> position in our list
        for i, (sem_id, sem_idx) in enumerate(sorted(sem_obj.IDs.items(), key=lambda x: x[1])):
            semantic_names.append(sem_obj.names.get(sem_id, f"sem_{sem_id}"))
            sem_idx_map[sem_idx] = i
        
        # Build links list and populate matrix
        links_list = []
        
        for po_list_idx, po_global_idx in enumerate(po_indices):
            po_idx_int = po_global_idx.item()
            po_name = po_names[po_list_idx]
            
            # Get all semantic connections for this PO
            for sem_idx, list_pos in sem_idx_map.items():
                weight = links.adj_matrix[po_idx_int, sem_idx].item()
                if weight != 0:
                    matrix[po_list_idx][list_pos] = weight
                    links_list.append({
                        'po_index': po_idx_int,
                        'po_name': po_name,
                        'sem_index': sem_idx,
                        'sem_name': semantic_names[list_pos],
                        'weight': weight,
                    })
        
        return {
            'matrix': matrix,
            'po_names': po_names,
            'semantic_names': semantic_names,
            'links_list': links_list,
        }
    
    def _extract_mappings(self) -> Dict:
        """Extract all mapping connections as matrices and lists."""
        mappings = {
            'P_mappings': [],
            'RB_mappings': [],
            'PO_mappings': [],
            'all_mappings': [],
        }
        
        token_tensor = self.network.token_tensor
        mapping_tensor = self.network.mappings
        
        # Get driver and recipient indices
        driver_mask = (token_tensor.tensor[:, TF.SET] == Set.DRIVER) & (token_tensor.tensor[:, TF.DELETED] == B.FALSE)
        recipient_mask = (token_tensor.tensor[:, TF.SET] == Set.RECIPIENT) & (token_tensor.tensor[:, TF.DELETED] == B.FALSE)
        
        driver_indices = torch.where(driver_mask)[0]
        recipient_indices = torch.where(recipient_mask)[0]
        
        # Build index to local position mappings
        driver_to_local = {idx.item(): i for i, idx in enumerate(driver_indices)}
        recipient_to_local = {idx.item(): i for i, idx in enumerate(recipient_indices)}
        
        # Iterate through all non-zero mappings
        weight_matrix = mapping_tensor[MappingFields.WEIGHT]
        
        for r_local, r_global in enumerate(recipient_indices):
            for d_local, d_global in enumerate(driver_indices):
                weight = weight_matrix[r_local, d_local].item()
                if weight > 0:
                    r_idx = r_global.item()
                    d_idx = d_global.item()
                    
                    # Determine token type
                    token_type_val = int(token_tensor.tensor[d_idx, TF.TYPE].item())
                    token_type = Type(token_type_val)
                    
                    entry = {
                        'type': token_type.name,
                        'driver_name': token_tensor.names.get(d_idx, f"token_{d_idx}"),
                        'recipient_name': token_tensor.names.get(r_idx, f"token_{r_idx}"),
                        'weight': weight,
                    }
                    
                    mappings['all_mappings'].append(entry)
                    
                    if token_type == Type.P:
                        mappings['P_mappings'].append(entry)
                    elif token_type == Type.RB:
                        mappings['RB_mappings'].append(entry)
                    elif token_type == Type.PO:
                        mappings['PO_mappings'].append(entry)
        
        return mappings
    
    def _extract_analogs(self) -> List[Dict]:
        """Extract analog structure."""
        analogs = []
        token_tensor = self.network.token_tensor
        
        # Get unique analog numbers (excluding null)
        analog_col = token_tensor.tensor[:, TF.ANALOG]
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        valid_analogs = analog_col[non_deleted]
        valid_analogs = valid_analogs[valid_analogs != null]
        unique_analogs = torch.unique(valid_analogs).tolist()
        
        for analog_num in unique_analogs:
            analog_num_int = int(analog_num)
            
            # Get all tokens in this analog
            analog_mask = (token_tensor.tensor[:, TF.ANALOG] == analog_num) & non_deleted
            analog_indices = torch.where(analog_mask)[0]
            
            p_names = []
            rb_names = []
            po_names = []
            
            for idx in analog_indices:
                idx_int = idx.item()
                token_type = Type(int(token_tensor.tensor[idx_int, TF.TYPE].item()))
                name = token_tensor.names.get(idx_int, f"token_{idx_int}")
                
                if token_type == Type.P:
                    p_names.append(name)
                elif token_type == Type.RB:
                    rb_names.append(name)
                elif token_type == Type.PO:
                    po_names.append(name)
            
            analogs.append({
                'index': analog_num_int,
                'P_names': p_names,
                'RB_names': rb_names,
                'PO_names': po_names,
            })
        
        return analogs
    
    def _extract_connections(self) -> Dict:
        """Extract token hierarchy connections (P→RB→PO)."""
        connections = {
            'P_to_RB': [],
            'RB_to_PO': [],
            'RB_to_childP': [],
        }
        
        token_tensor = self.network.token_tensor
        connections_tensor = self.network.tokens.connections
        
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        all_indices = torch.where(non_deleted)[0]
        
        for idx in all_indices:
            idx_int = idx.item()
            token_type = Type(int(token_tensor.tensor[idx_int, TF.TYPE].item()))
            parent_name = token_tensor.names.get(idx_int, f"token_{idx_int}")
            
            # Get children
            child_mask = connections_tensor.connections[idx_int, :] == True
            child_indices = torch.where(child_mask)[0]
            
            for child_idx in child_indices:
                child_idx_int = child_idx.item()
                if token_tensor.tensor[child_idx_int, TF.DELETED] == B.TRUE:
                    continue
                    
                child_type = Type(int(token_tensor.tensor[child_idx_int, TF.TYPE].item()))
                child_name = token_tensor.names.get(child_idx_int, f"token_{child_idx_int}")
                
                if token_type == Type.P and child_type == Type.RB:
                    connections['P_to_RB'].append({
                        'parent': parent_name,
                        'child': child_name,
                    })
                elif token_type == Type.RB and child_type == Type.PO:
                    # Determine if pred or obj
                    is_pred = token_tensor.tensor[child_idx_int, TF.PRED].item() == B.TRUE
                    role = 'pred' if is_pred else 'obj'
                    connections['RB_to_PO'].append({
                        'parent': parent_name,
                        'child': child_name,
                        'role': role,
                    })
                elif token_type == Type.RB and child_type == Type.P:
                    connections['RB_to_childP'].append({
                        'parent': parent_name,
                        'child': child_name,
                    })
        
        return connections
    
    def _extract_set_contents(self, set_type: Set) -> Dict:
        """Extract contents of a specific set (driver/recipient)."""
        token_tensor = self.network.token_tensor
        
        set_mask = (token_tensor.tensor[:, TF.SET] == set_type) & (token_tensor.tensor[:, TF.DELETED] == B.FALSE)
        set_indices = torch.where(set_mask)[0]
        
        p_names = []
        rb_names = []
        po_names = []
        
        for idx in set_indices:
            idx_int = idx.item()
            token_type = Type(int(token_tensor.tensor[idx_int, TF.TYPE].item()))
            name = token_tensor.names.get(idx_int, f"token_{idx_int}")
            
            if token_type == Type.P:
                p_names.append(name)
            elif token_type == Type.RB:
                rb_names.append(name)
            elif token_type == Type.PO:
                po_names.append(name)
        
        return {
            'P_names': p_names,
            'RB_names': rb_names,
            'PO_names': po_names,
            'counts': {
                'Ps': len(p_names),
                'RBs': len(rb_names),
                'POs': len(po_names),
            },
        }
    
    def _extract_metadata(self) -> Dict:
        """Extract metadata about the network."""
        token_tensor = self.network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        num_ps = ((token_tensor.tensor[:, TF.TYPE] == Type.P) & non_deleted).sum().item()
        num_rbs = ((token_tensor.tensor[:, TF.TYPE] == Type.RB) & non_deleted).sum().item()
        num_pos = ((token_tensor.tensor[:, TF.TYPE] == Type.PO) & non_deleted).sum().item()
        num_sems = len(self.network.semantics.IDs)
        
        return {
            'sim_path': str(self._sim_path) if self._sim_path else None,
            'parameters': self.parameters,
            'token_counts': {
                'Ps': num_ps,
                'RBs': num_rbs,
                'POs': num_pos,
                'semantics': num_sems,
            },
        }
    
    # ==================[ Helper Functions ]==================
    
    def _set_to_string(self, set_type: Set) -> str:
        """Convert Set enum to string matching old format."""
        mapping = {
            Set.DRIVER: 'driver',
            Set.RECIPIENT: 'recipient',
            Set.MEMORY: 'memory',
            Set.NEW_SET: 'newSet',
        }
        return mapping.get(set_type, str(set_type))
    
    def _get_max_map_unit_name(self, idx: int) -> Optional[str]:
        """Get the name of the max mapping unit for a token."""
        max_map_unit = int(self.network.token_tensor.tensor[idx, TF.MAX_MAP_UNIT].item())
        if max_map_unit == 0 or max_map_unit == null:
            return None
        return self.network.token_tensor.names.get(max_map_unit, None)
    
    def _get_child_names(self, idx: int, child_type: Type) -> List[str]:
        """Get names of children of a specific type."""
        connections = self.network.tokens.connections
        token_tensor = self.network.token_tensor
        
        child_mask = connections.connections[idx, :] == True
        child_indices = torch.where(child_mask)[0]
        
        names = []
        for child_idx in child_indices:
            child_idx_int = child_idx.item()
            if token_tensor.tensor[child_idx_int, TF.DELETED] == B.TRUE:
                continue
            if Type(int(token_tensor.tensor[child_idx_int, TF.TYPE].item())) == child_type:
                names.append(token_tensor.names.get(child_idx_int, f"token_{child_idx_int}"))
        
        return names
    
    def _get_parent_names(self, idx: int, parent_type: Type) -> List[str]:
        """Get names of parents of a specific type."""
        connections = self.network.tokens.connections
        token_tensor = self.network.token_tensor
        
        parent_mask = connections.connections[:, idx] == True
        parent_indices = torch.where(parent_mask)[0]
        
        names = []
        for parent_idx in parent_indices:
            parent_idx_int = parent_idx.item()
            if token_tensor.tensor[parent_idx_int, TF.DELETED] == B.TRUE:
                continue
            if Type(int(token_tensor.tensor[parent_idx_int, TF.TYPE].item())) == parent_type:
                names.append(token_tensor.names.get(parent_idx_int, f"token_{parent_idx_int}"))
        
        return names
    
    def _get_parent_rb_names_for_p(self, p_idx: int) -> List[str]:
        """Get RBs that have this P as a child (for higher-order structures)."""
        connections = self.network.tokens.connections
        token_tensor = self.network.token_tensor
        
        # Find RBs that have this P as a child
        parent_mask = connections.connections[:, p_idx] == True
        parent_indices = torch.where(parent_mask)[0]
        
        names = []
        for parent_idx in parent_indices:
            parent_idx_int = parent_idx.item()
            if token_tensor.tensor[parent_idx_int, TF.DELETED] == B.TRUE:
                continue
            if Type(int(token_tensor.tensor[parent_idx_int, TF.TYPE].item())) == Type.RB:
                names.append(token_tensor.names.get(parent_idx_int, f"token_{parent_idx_int}"))
        
        return names
    
    def _get_rb_pred_obj_names(self, rb_idx: int) -> tuple:
        """Get predicate and object names for an RB."""
        connections = self.network.tokens.connections
        token_tensor = self.network.token_tensor
        
        child_mask = connections.connections[rb_idx, :] == True
        child_indices = torch.where(child_mask)[0]
        
        pred_name = None
        obj_name = None
        
        for child_idx in child_indices:
            child_idx_int = child_idx.item()
            if token_tensor.tensor[child_idx_int, TF.DELETED] == B.TRUE:
                continue
            if Type(int(token_tensor.tensor[child_idx_int, TF.TYPE].item())) == Type.PO:
                is_pred = token_tensor.tensor[child_idx_int, TF.PRED].item() == B.TRUE
                name = token_tensor.names.get(child_idx_int, f"token_{child_idx_int}")
                if is_pred:
                    pred_name = name
                else:
                    obj_name = name
        
        return pred_name, obj_name
    
    def _get_rb_child_p_name(self, rb_idx: int) -> Optional[str]:
        """Get child P name for an RB (higher-order structure)."""
        connections = self.network.tokens.connections
        token_tensor = self.network.token_tensor
        
        child_mask = connections.connections[rb_idx, :] == True
        child_indices = torch.where(child_mask)[0]
        
        for child_idx in child_indices:
            child_idx_int = child_idx.item()
            if token_tensor.tensor[child_idx_int, TF.DELETED] == B.TRUE:
                continue
            if Type(int(token_tensor.tensor[child_idx_int, TF.TYPE].item())) == Type.P:
                return token_tensor.names.get(child_idx_int, f"token_{child_idx_int}")
        
        return None
    
    def _get_po_semantic_names(self, po_idx: int) -> List[str]:
        """Get semantic names connected to a PO."""
        links = self.network.links
        sem_obj = self.network.semantics
        
        names = []
        for sem_id, sem_idx in sem_obj.IDs.items():
            weight = links.adj_matrix[po_idx, sem_idx].item()
            if weight != 0:
                names.append(sem_obj.names.get(sem_id, f"sem_{sem_id}"))
        
        return names
    
    # ==================[ Saving Functions ]==================
    
    def save_state(self, file_path: Union[str, Path], format: str = 'pickle') -> Path:
        """
        Save the current network state to a file.
        
        Args:
            file_path: Path to save the state file
            format: 'pickle' (binary, exact) or 'json' (human-readable)
            
        Returns:
            Path to the saved file
            
        Raises:
            ValueError: If no network is loaded or invalid format
        """
        if self.network is None:
            raise ValueError("No network loaded. Call load_sim() first.")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self.get_state()
        
        if format == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
        elif format == 'json':
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'pickle' or 'json'.")
        
        print(f"State saved to {file_path} ({format} format)")
        return file_path
    
    def save_state_pickle(self, file_path: Union[str, Path]) -> Path:
        """Convenience method to save as pickle."""
        return self.save_state(file_path, format='pickle')
    
    def save_state_json(self, file_path: Union[str, Path]) -> Path:
        """Convenience method to save as JSON."""
        return self.save_state(file_path, format='json')
    
    # ==================[ Inspection Utilities ]==================
    
    def print_summary(self):
        """Print a summary of the current network state."""
        if self.network is None:
            print("No network loaded.")
            return
        
        token_tensor = self.network.token_tensor
        non_deleted = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        
        num_ps = ((token_tensor.tensor[:, TF.TYPE] == Type.P) & non_deleted).sum().item()
        num_rbs = ((token_tensor.tensor[:, TF.TYPE] == Type.RB) & non_deleted).sum().item()
        num_pos = ((token_tensor.tensor[:, TF.TYPE] == Type.PO) & non_deleted).sum().item()
        num_sems = len(self.network.semantics.IDs)
        
        print("\n" + "="*60)
        print("NETWORK SUMMARY (New Implementation)")
        print("="*60)
        
        if self._sim_path:
            print(f"Source: {self._sim_path}")
        
        print(f"\nToken Counts:")
        print(f"  Ps:        {num_ps}")
        print(f"  RBs:       {num_rbs}")
        print(f"  POs:       {num_pos}")
        print(f"  Semantics: {num_sems}")
        
        # Count analogs
        analog_col = token_tensor.tensor[:, TF.ANALOG]
        valid_analogs = analog_col[non_deleted]
        valid_analogs = valid_analogs[valid_analogs != null]
        num_analogs = len(torch.unique(valid_analogs))
        print(f"  Analogs:   {num_analogs}")
        
        # Driver set
        driver_contents = self._extract_set_contents(Set.DRIVER)
        print(f"\nDriver Set:")
        print(f"  Ps:  {driver_contents['counts']['Ps']} - {driver_contents['P_names']}")
        print(f"  RBs: {driver_contents['counts']['RBs']} - {driver_contents['RB_names']}")
        print(f"  POs: {driver_contents['counts']['POs']} - {driver_contents['PO_names']}")
        
        # Recipient set
        recipient_contents = self._extract_set_contents(Set.RECIPIENT)
        print(f"\nRecipient Set:")
        print(f"  Ps:  {recipient_contents['counts']['Ps']} - {recipient_contents['P_names']}")
        print(f"  RBs: {recipient_contents['counts']['RBs']} - {recipient_contents['RB_names']}")
        print(f"  POs: {recipient_contents['counts']['POs']} - {recipient_contents['PO_names']}")
        
        # Count mappings
        mapping_tensor = self.network.mappings
        total_mappings = (mapping_tensor[MappingFields.WEIGHT] > 0).sum().item()
        print(f"\nMapping Connections: {total_mappings}")
        
        # Count semantic links
        links = self.network.links
        total_links = (links.adj_matrix > 0).sum().item()
        print(f"Semantic Links: {total_links}")
        
        print("="*60 + "\n")


# ==================[ Static Loading Functions ]==================

def load_state(file_path: Union[str, Path]) -> Dict:
    """
    Load a previously saved state from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The saved state dictionary
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_state_json(file_path: Union[str, Path]) -> Dict:
    """
    Load a previously saved state from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        The saved state dictionary
    """
    with open(file_path, 'r') as f:
        return json.load(f)


