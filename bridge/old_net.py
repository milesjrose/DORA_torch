from .state_printer import StatePrinter
from .state import State
from nodes.enums import *

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from logging import getLogger
logger = getLogger("old_state_extractor")

class OldNet:
    """
    A class for extracting the state of the old network.
    """
    def __init__(self, parameters: Optional[Dict] = None, sim_path: Optional[str] = None):
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
        self.state = State()
        self.printer = StatePrinter(state=self.state)
        self.printer.header_text = "OLD"
    
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
    
    def load_sim(self, sim_path: Union[str, Path]):
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
        
        self.get_state()
    
    def load_props(self, symProps: List[Dict]):
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
        
        self.get_state()

    def get_state(self) -> State:
        """ Get the state of the network. """
        state = self.state
        state.clear()
        state.tokens = self._extract_tokens()
        state.tk_count = len(state.tokens)
        ids = [tk['ID'] for tk in state.tokens.values()]
        state.idxs = {id: i for i, id in enumerate(sorted(ids))}
        state.recipient_idxs = {id: i for i, id in enumerate(sorted(state.recipient))}
        state.driver_idxs = {id: i for i, id in enumerate(sorted(state.driver))}
        state.semantics = self._extract_semantics()
        sem_ids = [tk['ID'] for tk in state.semantics.values()]
        state.sem_idxs = {id: i for i, id in enumerate(sorted(sem_ids))}
        state.sem_count = len(state.semantics)
        state.links, state.links_list = self._extract_links()
        state.mappings = self._extract_mappings()
        state.connections = self._extract_connections()
        state.metadata = {
            'sim_path': str(self._sim_path) if self._sim_path else None,
            'parameters': self.parameters,
            'token_counts': {
                'Ps': len(self.memory.Ps),
                'RBs': len(self.memory.RBs),
                'POs': len(self.memory.POs),
                'semantics': len(self.memory.semantics),
            },
        }
        return self.state
    
    def _extract_tokens(self) -> dict[int, dict]:
        """ Extract all token data. """
        tokens = {}
        for myP in self.memory.Ps:
            tokens[myP.ID] = self._extract_token_data(myP)
        for myRB in self.memory.RBs:
            tokens[myRB.ID] = self._extract_token_data(myRB)
        for myPO in self.memory.POs:
            tokens[myPO.ID] = self._extract_token_data(myPO)
        logger.debug(f"Extracted {len(tokens)} tokens.")
        return tokens
    
    def _encode_data(self, data, enum_class) -> any:
        """ Encode data into a standard format. """
        match enum_class:
            case 'type':
                return {
                    'P': Type.P,
                    'RB': Type.RB,
                    'PO': Type.PO,
                }[data]
            case 'set':
                return {
                    'driver': Set.DRIVER,
                    'recipient': Set.RECIPIENT,
                    'memory': Set.MEMORY,
                    'newSet': Set.NEW_SET,
                }[data]
            case 'mode':
                return {
                    -1: Mode.CHILD,
                    0: Mode.NEUTRAL,
                    1: Mode.PARENT,
                }[data]
            case 'ont_status':
                return {
                    'state': OntStatus.STATE,
                    'value': OntStatus.VALUE,
                    'sdm': OntStatus.SDM,
                    'ho': OntStatus.HO,
                }[data]
            case 'sdm':
                return {
                    'more': SDM.MORE,
                    'less': SDM.LESS,
                    'same': SDM.SAME,
                    'diff': SDM.DIFF,
                }[data]

    def _extract_token_data(self, token) -> dict:
        """ Extract data from a token. """
        logger.info(f"Analog:{token.myanalog.ID}")
        data = {
            'name': token.name,
            'ID': token.ID,
            'set': self._encode_data(token.set, 'set'),
            'type': self._encode_data(token.my_type, 'type'),
            'myanalog': token.myanalog.ID,
            'act': token.act,
            'max_act': token.max_act,
            'my_index': token.my_index,
            'inhibitor_input': token.inhibitor_input,
            'inhibitor_act': token.inhibitor_act,
            'mappingHypotheses': token.mappingHypotheses,
            'mappingConnections': token.mappingConnections,
            'max_map_unit': token.max_map_unit,
            'max_map': token.max_map,
            'td_input': token.td_input,
            'bu_input': token.bu_input,
            'lateral_input': token.lateral_input,
            'map_input': token.map_input,
            'net_input': token.net_input,
            'GUI_unit': token.GUI_unit,
            'my_made_unit': token.my_made_unit.ID if token.my_made_unit else None,
            'my_made_units': [made.ID for made in token.my_made_units],
            'my_maker_unit': token.my_maker_unit.ID if token.my_maker_unit else None,
            'inferred': token.inferred,
            'retrieved': token.retrieved,
            'copy_for_DR': token.copy_for_DR,
            'copied_DR_index': token.copied_DR_index,
            'sim_made': token.sim_made,
            'inhibitorThreshold': token.inhibitorThreshold
        }
        match token.my_type:
            case 'P':
                data['myRBs'] = [rb.ID for rb in token.myRBs]
                data['myParentRBs'] = [rb.ID for rb in token.myParentRBs]
                data['myGroups'] = [group.ID for group in token.myGroups]
                data['mode'] = self._encode_data(token.mode, 'mode')
                self.state.token_ids[Type.P].append(token.ID)
            case 'RB':
                data['myParentPs'] = [p.ID for p in token.myParentPs]
                data['myPred'] = [pred.ID for pred in token.myPred]
                data['myObj'] = [obj.ID for obj in token.myObj]
                data['myChildP'] = [childP.ID for childP in token.myChildP]
                data['timesFired'] = token.timesFired
                self.state.token_ids[Type.RB].append(token.ID)
            case 'PO':
                data['predOrObj'] = token.predOrObj
                data['myRBs'] = [rb.ID for rb in token.myRBs]
                data['same_RB_POs'] = [po.ID for po in token.same_RB_POs]
                data['mySemantics'] = [link.mySemantic.ID for link in token.mySemantics]
                data['semNormalization'] = token.semNormalization
                data['max_sem_weight'] = token.max_sem_weight
                self.state.token_ids[Type.PO].append(token.ID)
            case _:
                raise ValueError(f"Unknown token type: {token.my_type}")
        if data['set'] == Set.DRIVER:
            self.state.driver.append(token.ID)
        if data['set'] == Set.RECIPIENT:
            self.state.recipient.append(token.ID)
        return data

    def _extract_semantics(self) -> dict[int, dict]:
        """ Extract all semantic data. """
        semantics = {}
        for sem in self.memory.semantics:
            semantics[sem.ID] = {
                'name': sem.name,
                'ID': sem.ID,
                'my_type': 'semantic',
                'dimension': sem.dimension,
                'amount': sem.amount,
                'ont_status': self._encode_data(sem.ont_status, 'ont_status'),
                'myinput': sem.myinput,
                'max_sem_input': sem.max_sem_input,
                'act': sem.act,
                'myPOs': [link.myPO.ID for link in sem.myPOs],
                'myGroups': [link.group.ID for link in sem.myGroups],
                'semConnect': [link.mySemantic.ID for link in sem.semConnect],
                'semConnectWeights': sem.semConnectWeights,
            }
        logger.debug(f"Extracted {len(semantics)} semantics.")
        return semantics
    
    def _extract_links(self) -> list[list[float]]:
        """ Extract all link data. """
        links = [[0.0] * self.state.sem_count for _ in range(self.state.tk_count)]
        count = 0
        links_list = {}
        for po in self.memory.POs:
            for link in po.mySemantics:
                weight = link.weight
                if weight > 0:
                    sem_id = link.mySemantic.ID
                    po_id = link.myPO.ID
                    links[self.state.idxs[po_id]][self.state.sem_idxs[sem_id]] = link.weight
                    if po_id not in links_list:
                        links_list[po_id] = []
                    if sem_id not in links_list[po_id]:
                        links_list[po_id].append(sem_id)
                    else:
                        raise ValueError(f"Duplicate link found for {po.name} -> {link.mySemantic.name}")
                    count += 1
        logger.debug(f"Extracted {count} links.")
        return links, links_list
    
    def _extract_mappings(self) -> list[list[dict[MappingFields, float]]]:
        """ Extract all mapping data. """
        r_count = len(self.state.recipient)
        d_count = len(self.state.driver)
        mappings = [[{MappingFields.WEIGHT: 0.0, MappingFields.HYPOTHESIS: 0.0, MappingFields.MAX_HYP: 0.0} for _ in range(r_count)] for _ in range(d_count)]
        for tk_list in [self.memory.Ps, self.memory.RBs, self.memory.POs]:
            for token in tk_list:
                # hypotheses
                for hyp in token.mappingHypotheses:
                    info = {
                        MappingFields.WEIGHT: hyp.myMappingConnection.weight,
                        MappingFields.HYPOTHESIS: hyp.hypothesis,
                        MappingFields.MAX_HYP: hyp.max_hyp,
                    }
                    rec_idx = self.state.recipient_idxs[hyp.myMappingConnection.recipientToken.ID]
                    dri_idx = self.state.driver_idxs[hyp.myMappingConnection.driverToken.ID]
                    mappings[rec_idx][dri_idx] = info
                # connections
                for mc in token.mappingConnections:
                    rec_idx = self.state.recipient_idxs[hyp.myMappingConnection.recipientToken.ID]
                    dri_idx = self.state.driver_idxs[hyp.myMappingConnection.driverToken.ID]
                    weight = mappings[rec_idx][dri_idx][MappingFields.WEIGHT]
                    if weight != 0.0 and weight != mc.weight:
                        raise ValueError(f"Mapping weight mismatch for {token.name} -> {mc.driverToken.name}")
                    else:
                        mappings[rec_idx][dri_idx][MappingFields.WEIGHT] = mc.weight
        return mappings

    def _extract_connections(self) -> list[list[bool]]:
        """ Extract all connection data. """
        connections = [[0.0] * self.state.tk_count for _ in range(self.state.tk_count)]
        for myP in self.memory.Ps:
            for myRB in myP.myRBs:
                connections[self.state.idxs[myP.ID]][self.state.idxs[myRB.ID]] = True
        for myRB in self.memory.RBs:
            for myChildP in myRB.myChildP:
                connections[self.state.idxs[myRB.ID]][self.state.idxs[myChildP.ID]] = True
        for myPO in self.memory.POs:
            for myRB in myPO.myRBs:
                connections[self.state.idxs[myPO.ID]][self.state.idxs[myRB.ID]] = True
        return connections

    def save_state(self, file_path: str = None):
        """
        Convert the state to a JSON-serializable dictionary.
        
        Args:
            filepath: Optional path to save the JSON file. If None, only returns the dict.
            
        Returns:
            A JSON-serializable dictionary representation of the state.
        """
        return self.state.to_json(file_path)
    
    def load_state(self, source: str | dict):
        """
        Load a State instance from JSON data.
        
        Args:
            source: Either a filepath (str) to a JSON file, or a dictionary of state data.
            
        Returns:
            A new State instance populated with the loaded data.
        """
        return self.state.from_json(source)