# DORA_bridge/test_data_generator.py
# Class for generating and inspecting test data from currVers

import sys
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from logging import getLogger
logger = getLogger(__name__)


class TestDataGenerator:
    """
    Generate and inspect test data from the currVers DORA implementation.
    
    This class provides a bridge between the original DORA implementation
    and the new tensorized implementation, enabling:
    - Loading simulation files into currVers
    - Extracting detailed network state
    - Saving state snapshots for comparison testing
    
    Example usage:
        >>> gen = TestDataGenerator()
        >>> gen.load_sim('sims/testsim15.py')
        >>> state = gen.get_state()
        >>> gen.save_state('test_data/initial_state.pkl')
        
        # Run some operations
        >>> gen.network.do_retrieval()
        >>> gen.network.do_map()
        >>> gen.save_state('test_data/after_mapping.pkl')
    """
    
    def __init__(self, parameters: Optional[Dict] = None):
        """
        Initialize the TestDataGenerator.
        
        Args:
            parameters: Optional dict of DORA parameters. If None, uses defaults.
        """
        self._setup_currvers_path()
        self._import_currvers()
        
        self.memory = None
        self.network = None
        self.parameters = parameters or self._default_parameters()
        self._sim_path = None
    
    def _setup_currvers_path(self):
        """Add currVers to Python path if not already there."""
        currvers_dir = Path(__file__).parent / 'currVers'
        if str(currvers_dir) not in sys.path:
            sys.path.insert(0, str(currvers_dir))
    
    def _import_currvers(self):
        """Import currVers modules."""
        import buildNetwork
        import basicRunDORA
        import dataTypes
        
        self._buildNetwork = buildNetwork
        self._basicRunDORA = basicRunDORA
        self._dataTypes = dataTypes
    
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
            "doGUI": False,  # Disable GUI for testing
            "GUI_update_rate": 1,
            "starting_iteration": 0,
            "tokenize": False,
            "ho_sem_act_flow": 0,
            "remove_uncompressed": False,
            "remove_compressed": False,
        }
    
    # ==================[ Loading Functions ]==================
    
    def load_sim(self, sim_path: Union[str, Path]) -> 'TestDataGenerator':
        """
        Load a simulation file into the currVers network.
        
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
        
        # Read and parse the sim file
        with open(sim_path, 'r') as f:
            content = f.read()
        
        # Parse simType from first line
        first_line = content.split('\n')[0]
        simType = ""
        di = {"simType": simType}
        exec(first_line, di)
        simType = di["simType"]
        
        # Parse symProps based on simType
        if simType == "sym_file":
            symProps = []
            di = {"symProps": symProps}
            # Get everything after the first line
            rest = '\n'.join(content.split('\n')[1:])
            exec(rest, di)
            symProps = di["symProps"]
        elif simType == "json_sym":
            import json
            lines = content.split('\n')
            symProps = json.loads(lines[1])
        elif simType == "sim_file":
            symProps = []
            di = {"symProps": symProps}
            rest = '\n'.join(content.split('\n')[1:])
            exec(rest, di)
            symProps = di["symProps"]
        else:
            raise ValueError(f"Unknown simType: {simType}")
        
        # Build the network
        self.memory = self._buildNetwork.initializeMemorySet()
        mysym = self._buildNetwork.interpretSymfile(symProps)
        self.memory = self._buildNetwork.buildTheNetwork(mysym[0], self.memory)
        
        # Create runDORA object
        self.network = self._basicRunDORA.runDORA(self.memory, self.parameters)
        
        # Set up driver and recipient
        self.network.initialize_run(self.network.memory)

        logger.info(f"Loaded simulation from {sim_path}")
        logger.info(f"  Ps: {len(self.memory.Ps)}, RBs: {len(self.memory.RBs)}, "
              f"POs: {len(self.memory.POs)}, Semantics: {len(self.memory.semantics)}")
        
        return self
    
    def load_props(self, symProps: List[Dict]) -> 'TestDataGenerator':
        """
        Load a network from symProps directly (without a file).
        
        Args:
            symProps: List of proposition dictionaries
            
        Returns:
            self (for method chaining)
        """
        self._sim_path = None
        
        # Build the network
        self.memory = self._buildNetwork.initializeMemorySet()
        mysym = self._buildNetwork.interpretSymfile(symProps)
        self.memory = self._buildNetwork.buildTheNetwork(mysym[0], self.memory)
        
        # Create runDORA object
        self.network = self._basicRunDORA.runDORA(self.memory, self.parameters)
        
        # Set up driver and recipient
        self.network.memory = self._basicRunDORA.clearDriverSet(self.network.memory)
        self.network.memory = self._basicRunDORA.clearRecipientSet(self.network.memory)
        self.network.memory = self._basicRunDORA.findDriverRecipient(self.network.memory)
        
        logger.info(f"Loaded network from props")
        logger.info(f"  Ps: {len(self.memory.Ps)}, RBs: {len(self.memory.RBs)}, "
              f"POs: {len(self.memory.POs)}, Semantics: {len(self.memory.semantics)}")
        
        return self
    
    # ==================[ State Extraction ]==================
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the complete detailed state of the currVers network.
        
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
        if self.memory is None:
            raise ValueError("No network loaded. Call load_sim() first.")
        
        return {
            'tokens': self._extract_tokens(),
            'semantics': self._extract_semantics(),
            'links': self._extract_links(),
            'mappings': self._extract_mappings(),
            'analogs': self._extract_analogs(),
            'connections': self._extract_connections(),
            'driver': self._extract_set_contents('driver'),
            'recipient': self._extract_set_contents('recipient'),
            'metadata': {
                'sim_path': str(self._sim_path) if self._sim_path else None,
                'parameters': self.parameters,
                'token_counts': {
                    'Ps': len(self.memory.Ps),
                    'RBs': len(self.memory.RBs),
                    'POs': len(self.memory.POs),
                    'semantics': len(self.memory.semantics),
                },
            },
        }
    
    def _extract_tokens(self) -> Dict[str, List[Dict]]:
        """Extract all token data."""
        tokens = {'Ps': [], 'RBs': [], 'POs': []}
        
        # Helper to get analog index from myanalog object
        def get_analog_index(myanalog_obj):
            """Get the analog index number from the myanalog object."""
            if myanalog_obj is None:
                return None
            try:
                return self.memory.analogs.index(myanalog_obj)
            except ValueError:
                return None
        
        # Extract P tokens
        for i, myP in enumerate(self.memory.Ps):
            tokens['Ps'].append({
                'index': i,
                'name': myP.name,
                'set': myP.set,
                'analog': get_analog_index(myP.myanalog),
                'act': myP.act,
                'net_input': getattr(myP, 'net_input', 0.0),
                'td_input': getattr(myP, 'td_input', 0.0),
                'bu_input': getattr(myP, 'bu_input', 0.0),
                'lateral_input': getattr(myP, 'lateral_input', 0.0),
                'map_input': getattr(myP, 'map_input', 0.0),
                'max_map': myP.max_map,
                'max_map_unit_name': myP.max_map_unit.name if myP.max_map_unit else None,
                'inferred': getattr(myP, 'inferred', False),
                'child_RB_names': [rb.name for rb in myP.myRBs],
                'parent_RB_names': [rb.name for rb in myP.myParentRBs] if hasattr(myP, 'myParentRBs') else [],
            })
        
        # Extract RB tokens
        for i, myRB in enumerate(self.memory.RBs):
            tokens['RBs'].append({
                'index': i,
                'name': myRB.name,
                'set': myRB.set,
                'analog': get_analog_index(myRB.myanalog),
                'act': myRB.act,
                'net_input': getattr(myRB, 'net_input', 0.0),
                'td_input': getattr(myRB, 'td_input', 0.0),
                'bu_input': getattr(myRB, 'bu_input', 0.0),
                'lateral_input': getattr(myRB, 'lateral_input', 0.0),
                'map_input': getattr(myRB, 'map_input', 0.0),
                'max_map': myRB.max_map,
                'max_map_unit_name': myRB.max_map_unit.name if myRB.max_map_unit else None,
                'inferred': getattr(myRB, 'inferred', False),
                'parent_P_names': [p.name for p in myRB.myParentPs],
                'pred_name': myRB.myPred[0].name if myRB.myPred else None,
                'obj_name': myRB.myObj[0].name if myRB.myObj else None,
                'child_P_name': myRB.myChildP[0].name if myRB.myChildP else None,
            })
        
        # Extract PO tokens
        for i, myPO in enumerate(self.memory.POs):
            tokens['POs'].append({
                'index': i,
                'name': myPO.name,
                'set': myPO.set,
                'analog': get_analog_index(myPO.myanalog),
                'act': myPO.act,
                'net_input': getattr(myPO, 'net_input', 0.0),
                'td_input': getattr(myPO, 'td_input', 0.0),
                'bu_input': getattr(myPO, 'bu_input', 0.0),
                'lateral_input': getattr(myPO, 'lateral_input', 0.0),
                'map_input': getattr(myPO, 'map_input', 0.0),
                'predOrObj': myPO.predOrObj,  # 1 = pred, 0 = obj
                'max_map': myPO.max_map,
                'max_map_unit_name': myPO.max_map_unit.name if myPO.max_map_unit else None,
                'inferred': getattr(myPO, 'inferred', False),
                'parent_RB_names': [rb.name for rb in myPO.myRBs],
                'semantic_names': [link.mySemantic.name for link in myPO.mySemantics],
            })
        
        return tokens
    
    def _extract_semantics(self) -> List[Dict]:
        """Extract all semantic unit data."""
        semantics = []
        for i, sem in enumerate(self.memory.semantics):
            semantics.append({
                'index': i,
                'name': sem.name,
                'act': sem.act,
                'dimension': getattr(sem, 'dimension', 'nil'),
                'amount': getattr(sem, 'amount', None),
                'ont_status': getattr(sem, 'ont_status', None),
            })
        return semantics
    
    def _extract_links(self) -> Dict:
        """
        Extract PO-semantic links as a matrix and detailed list.
        """
        num_pos = len(self.memory.POs)
        num_sems = len(self.memory.semantics)
        
        # Initialize weight matrix
        matrix = [[0.0] * num_sems for _ in range(num_pos)]
        
        # Detailed link list
        links_list = []
        
        # Fill from PO links
        for po_idx, myPO in enumerate(self.memory.POs):
            for link in myPO.mySemantics:
                try:
                    sem_idx = self.memory.semantics.index(link.mySemantic)
                    matrix[po_idx][sem_idx] = link.weight
                    links_list.append({
                        'po_index': po_idx,
                        'po_name': myPO.name,
                        'sem_index': sem_idx,
                        'sem_name': link.mySemantic.name,
                        'weight': link.weight,
                    })
                except ValueError:
                    pass
        
        return {
            'matrix': matrix,
            'po_names': [po.name for po in self.memory.POs],
            'semantic_names': [sem.name for sem in self.memory.semantics],
            'links_list': links_list,
        }
    
    def _extract_mappings(self) -> Dict:
        """
        Extract all mapping connections as matrices and lists.
        """
        mappings = {
            'P_mappings': [],
            'RB_mappings': [],
            'PO_mappings': [],
            'all_mappings': [],
        }
        
        # Extract from all tokens that have mapping connections
        for myP in self.memory.Ps:
            for mc in myP.mappingConnections:
                entry = {
                    'type': 'P',
                    'driver_name': mc.driverToken.name,
                    'recipient_name': mc.recipientToken.name,
                    'weight': mc.weight,
                }
                mappings['P_mappings'].append(entry)
                mappings['all_mappings'].append(entry)
        
        for myRB in self.memory.RBs:
            for mc in myRB.mappingConnections:
                entry = {
                    'type': 'RB',
                    'driver_name': mc.driverToken.name,
                    'recipient_name': mc.recipientToken.name,
                    'weight': mc.weight,
                }
                mappings['RB_mappings'].append(entry)
                mappings['all_mappings'].append(entry)
        
        for myPO in self.memory.POs:
            for mc in myPO.mappingConnections:
                entry = {
                    'type': 'PO',
                    'driver_name': mc.driverToken.name,
                    'recipient_name': mc.recipientToken.name,
                    'weight': mc.weight,
                }
                mappings['PO_mappings'].append(entry)
                mappings['all_mappings'].append(entry)
        
        return mappings
    
    def _extract_connections(self) -> Dict:
        """Extract token hierarchy connections (P→RB→PO)."""
        connections = {
            'P_to_RB': [],
            'RB_to_PO': [],
            'RB_to_childP': [],
        }
        
        for myP in self.memory.Ps:
            for myRB in myP.myRBs:
                connections['P_to_RB'].append({
                    'parent': myP.name,
                    'child': myRB.name,
                })
        
        for myRB in self.memory.RBs:
            if myRB.myPred:
                connections['RB_to_PO'].append({
                    'parent': myRB.name,
                    'child': myRB.myPred[0].name,
                    'role': 'pred',
                })
            if myRB.myObj:
                connections['RB_to_PO'].append({
                    'parent': myRB.name,
                    'child': myRB.myObj[0].name,
                    'role': 'obj',
                })
            if myRB.myChildP:
                connections['RB_to_childP'].append({
                    'parent': myRB.name,
                    'child': myRB.myChildP[0].name,
                })
        
        return connections
    
    def _extract_analogs(self) -> List[Dict]:
        """Extract analog structure."""
        analogs = []
        for i, analog in enumerate(self.memory.analogs):
            analogs.append({
                'index': i,
                'P_names': [p.name for p in analog.myPs],
                'RB_names': [rb.name for rb in analog.myRBs],
                'PO_names': [po.name for po in analog.myPOs],
            })
        return analogs
    
    def _extract_set_contents(self, set_name: str) -> Dict:
        """Extract contents of a specific set (driver/recipient)."""
        if set_name == 'driver':
            set_obj = self.memory.driver
        elif set_name == 'recipient':
            set_obj = self.memory.recipient
        else:
            raise ValueError(f"Unknown set: {set_name}")
        
        return {
            'P_names': [p.name for p in set_obj.Ps],
            'RB_names': [rb.name for rb in set_obj.RBs],
            'PO_names': [po.name for po in set_obj.POs],
            'counts': {
                'Ps': len(set_obj.Ps),
                'RBs': len(set_obj.RBs),
                'POs': len(set_obj.POs),
            },
        }
    
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
        if self.memory is None:
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
        
        logger.info(f"State saved to {file_path} ({format} format)")
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
        if self.memory is None:
            print("No network loaded.")
            return
        
        print("\n" + "="*60)
        print("NETWORK SUMMARY")
        print("="*60)
        
        if self._sim_path:
            print(f"Source: {self._sim_path}")
        
        print(f"\nToken Counts:")
        print(f"  Ps:        {len(self.memory.Ps)}")
        print(f"  RBs:       {len(self.memory.RBs)}")
        print(f"  POs:       {len(self.memory.POs)}")
        print(f"  Semantics: {len(self.memory.semantics)}")
        print(f"  Analogs:   {len(self.memory.analogs)}")
        
        print(f"\nDriver Set:")
        print(f"  Ps:  {len(self.memory.driver.Ps)} - {[p.name for p in self.memory.driver.Ps]}")
        print(f"  RBs: {len(self.memory.driver.RBs)} - {[rb.name for rb in self.memory.driver.RBs]}")
        print(f"  POs: {len(self.memory.driver.POs)} - {[po.name for po in self.memory.driver.POs]}")
        
        print(f"\nRecipient Set:")
        print(f"  Ps:  {len(self.memory.recipient.Ps)} - {[p.name for p in self.memory.recipient.Ps]}")
        print(f"  RBs: {len(self.memory.recipient.RBs)} - {[rb.name for rb in self.memory.recipient.RBs]}")
        print(f"  POs: {len(self.memory.recipient.POs)} - {[po.name for po in self.memory.recipient.POs]}")
        
        # Count mappings
        total_mappings = sum(
            len(t.mappingConnections) 
            for t in self.memory.Ps + self.memory.RBs + self.memory.POs
        )
        print(f"\nMapping Connections: {total_mappings}")
        
        # Count semantic links
        total_links = sum(len(po.mySemantics) for po in self.memory.POs)
        print(f"Semantic Links: {total_links}")
        
        print("="*60 + "\n")
    
    def print_tokens(self, token_type: Optional[str] = None):
        """Print detailed token information."""
        if self.memory is None:
            print("No network loaded.")
            return
        
        # Helper to get analog index
        def get_analog_index(myanalog_obj):
            if myanalog_obj is None:
                return None
            try:
                return self.memory.analogs.index(myanalog_obj)
            except ValueError:
                return None
        
        if token_type is None or token_type.upper() == 'P':
            print("\n--- P Tokens ---")
            for i, p in enumerate(self.memory.Ps):
                analog_idx = get_analog_index(p.myanalog)
                print(f"  [{i}] {p.name} (set={p.set}, analog={analog_idx}, act={p.act:.3f})")
        
        if token_type is None or token_type.upper() == 'RB':
            print("\n--- RB Tokens ---")
            for i, rb in enumerate(self.memory.RBs):
                pred = rb.myPred[0].name if rb.myPred else "None"
                obj = rb.myObj[0].name if rb.myObj else "None"
                analog_idx = get_analog_index(rb.myanalog)
                print(f"  [{i}] {rb.name} (set={rb.set}, analog={analog_idx}, pred={pred}, obj={obj})")
        
        if token_type is None or token_type.upper() == 'PO':
            print("\n--- PO Tokens ---")
            for i, po in enumerate(self.memory.POs):
                role = "pred" if po.predOrObj == 1 else "obj"
                sems = [link.mySemantic.name for link in po.mySemantics]
                analog_idx = get_analog_index(po.myanalog)
                print(f"  [{i}] {po.name} ({role}, set={po.set}, analog={analog_idx}, sems={sems})")
    
    def print_semantics(self):
        """Print detailed semantic information."""
        if self.memory is None:
            print("No network loaded.")
            return
        
        print("\n--- Semantics ---")
        
        if len(self.memory.semantics) == 0:
            print("  No semantics found.")
            return
        
        # Build a map of which POs link to each semantic
        semantic_to_pos = {}
        for po_idx, myPO in enumerate(self.memory.POs):
            for link in myPO.mySemantics:
                sem_name = link.mySemantic.name
                if sem_name not in semantic_to_pos:
                    semantic_to_pos[sem_name] = []
                semantic_to_pos[sem_name].append({
                    'po_name': myPO.name,
                    'weight': link.weight,
                })
        
        for i, sem in enumerate(self.memory.semantics):
            dimension = getattr(sem, 'dimension', 'nil')
            amount = getattr(sem, 'amount', None)
            ont_status = getattr(sem, 'ont_status', None)
            
            # Build dimension/amount string
            dim_str = f", dimension={dimension}" if dimension != 'nil' else ""
            amount_str = f", amount={amount}" if amount is not None else ""
            ont_str = f", ont_status={ont_status}" if ont_status is not None else ""
            
            # Get linked POs
            linked_pos = semantic_to_pos.get(sem.name, [])
            po_info = ""
            if linked_pos:
                po_names = [f"{po['po_name']}({po['weight']:.2f})" for po in linked_pos[:3]]
                po_info = f", linked_to={po_names}"
                if len(linked_pos) > 3:
                    po_info += f", ... ({len(linked_pos)} total)"
            
            print(f"  [{i}] {sem.name} (act={sem.act:.3f}{dim_str}{amount_str}{ont_str}{po_info})")
    
    def print_links(self, show_matrix: bool = False, max_links: int = 20):
        """
        Print PO-semantic link information.
        
        Args:
            show_matrix: If True, print the full weight matrix (can be large)
            max_links: Maximum number of individual links to show (default 20)
        """
        if self.memory is None:
            print("No network loaded.")
            return
        
        print("\n--- PO-Semantic Links ---")
        
        # Get link data
        links_data = self._extract_links()
        links_list = links_data['links_list']
        matrix = links_data['matrix']
        po_names = links_data['po_names']
        semantic_names = links_data['semantic_names']
        
        if len(links_list) == 0:
            print("  No links found.")
            return
        
        # Print summary
        print(f"  Total links: {len(links_list)}")
        print(f"  POs: {len(po_names)}, Semantics: {len(semantic_names)}")
        
        # Print individual links (limited)
        print(f"\n  Individual Links (showing up to {max_links}):")
        for i, link in enumerate(links_list[:max_links]):
            print(f"    [{i}] {link['po_name']} <-> {link['sem_name']} "
                  f"(weight={link['weight']:.4f})")
        
        if len(links_list) > max_links:
            print(f"    ... and {len(links_list) - max_links} more links")
        
        # Print matrix if requested
        if show_matrix:
            print(f"\n  Weight Matrix ({len(po_names)} POs x {len(semantic_names)} Semantics):")
            print("    PO\\Sem", end="")
            for sem_name in semantic_names[:10]:  # Show first 10 semantic names
                print(f"  {sem_name[:6]:>6}", end="")
            if len(semantic_names) > 10:
                print("  ...")
            else:
                print()
            
            for po_idx, po_name in enumerate(po_names[:10]):  # Show first 10 POs
                print(f"    {po_name[:8]:>8}", end="")
                for sem_idx in range(min(10, len(semantic_names))):
                    weight = matrix[po_idx][sem_idx]
                    print(f"  {weight:6.3f}", end="")
                if len(semantic_names) > 10:
                    print("  ...")
                else:
                    print()
            
            if len(po_names) > 10:
                print("    ...")
        
        # Print statistics
        if links_list:
            weights = [link['weight'] for link in links_list]
            avg_weight = sum(weights) / len(weights)
            max_weight = max(weights)
            min_weight = min(weights)
            print(f"\n  Statistics:")
            print(f"    Average weight: {avg_weight:.4f}")
            print(f"    Max weight: {max_weight:.4f}")
            print(f"    Min weight: {min_weight:.4f}")
    
    def print_mappings(self):
        """Print current mapping connections."""
        if self.memory is None:
            print("No network loaded.")
            return
        
        print("\n--- Mapping Connections ---")
        
        has_mappings = False
        for token_type, tokens in [('P', self.memory.Ps), 
                                    ('RB', self.memory.RBs), 
                                    ('PO', self.memory.POs)]:
            for token in tokens:
                for mc in token.mappingConnections:
                    has_mappings = True
                    print(f"  {token_type}: {mc.driverToken.name} <-> "
                          f"{mc.recipientToken.name} (weight={mc.weight:.4f})")
        
        if not has_mappings:
            print("  No mapping connections established.")


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

