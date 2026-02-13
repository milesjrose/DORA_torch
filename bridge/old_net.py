from .state_printer import StatePrinter
from .state import State
from nodes.enums import *

import sys
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

from logging import getLogger
logger = getLogger("OLD_NET")

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
        self.num_cons = {
            Type.P: 0,
            Type.RB: 0,
            Type.PO: 0,
        }
        self.state = State()
        self.printer = StatePrinter(state=self.state, logger=logger)
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
        from dataTypes import reset_iterators
        self._reset_iterators = reset_iterators
        
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
        self._reset_iterators()
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
        
        self.get_state()

    def get_new_state(self) -> State:
        """ Returns a new generated state object, without affecting current state object. """
        state = self.state
        new_state = State()
        self.state = new_state
        self.get_state(log=False)
        self.state = state
        return new_state

    def get_state(self, log: bool = True) -> State:
        """ Get the state of the network. """
        state = self.state
        state.clear()
        state.tokens, tk_count = self._extract_tokens()
        state.tk_count = tk_count
        ids = sorted([tk['ID'] for tk in state.tokens.values()])
        for i, id in enumerate(ids):
            state.idxs[id] = i
            state.ids[i] = id
        state.recipient_idxs = {id: i for i, id in enumerate(sorted(state.recipient))}
        state.driver_idxs = {id: i for i, id in enumerate(sorted(state.driver))}
        state.semantics, sem_count = self._extract_semantics()
        sem_ids = [tk['ID'] for tk in state.semantics.values()]
        state.sem_idxs = {id: i for i, id in enumerate(sorted(sem_ids))}
        state.sem_ids = {i: id for id, i in state.sem_idxs.items()}
        state.sem_count = sem_count
        state.links, state.links_list, link_count = self._extract_links()
        state.mappings, m_count, h_count = self._extract_mappings()
        state.connections, con_count = self._extract_connections()
        state.sem_connections, sem_con_count = self._extract_semantic_connections()
        met = state.metadata
        met['sim_path'] = str(self._sim_path) if self._sim_path else None
        met['parameters'] = self.parameters
        met['token_counts'] = {
            Type.P: len(self.memory.Ps),
            Type.RB: len(self.memory.RBs),
            Type.PO: len(self.memory.POs),
            Type.SEMANTIC: len(self.memory.semantics),
        }
        state.firing_order = [node.ID for node in self.network.firingOrder] if self.network.firingOrder is not None else []
        if log:
            logger.debug(f"Extracted {tk_count} tk, {sem_count} sem, {link_count} link, {m_count} maps, {con_count} cons, {sem_con_count} sem_cons")
        return state
    
    def _extract_tokens(self) -> tuple[dict[int, dict], int]:
        """ Extract all token data. 

        Returns: 
        tuple[dict[int, dict], int]: A tuple containing:
            tokens (dict[int, dict]): A dictionary of tokens by ID.
            tk_count (int): The number of tokens.
        """
        tokens = {}
        for myP in self.memory.Ps:
            tokens[myP.ID] = self._extract_token_data(myP)
        for myRB in self.memory.RBs:
            tokens[myRB.ID] = self._extract_token_data(myRB)
        for myPO in self.memory.POs:
            tokens[myPO.ID] = self._extract_token_data(myPO)
        return tokens, len(tokens)
    
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
        """ Extract data from a token. 

        Returns:
            dict: A dictionary containing the token data.
        """
        non_zero_hyps = []
        for hyp in token.mappingHypotheses:
            if hyp.hypothesis != 0.0 and hyp.myMappingConnection.weight != 0.0:
                non_zero_hyps.append(hyp)
        non_zero_mcs = []
        for mc in token.mappingConnections:
            if mc.weight != 0.0:
                non_zero_mcs.append(mc)
        hyps = [f"{hyp.driverToken.ID}.{hyp.recipientToken.ID}" for hyp in non_zero_hyps]
        unique_mcs = []
        for mc in non_zero_mcs:
            tk_str = f"{mc.driverToken.ID}.{mc.recipientToken.ID}"
            if tk_str not in hyps:
                unique_mcs.append(mc)

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
            'mappingHypotheses': [(hyp.driverToken.ID, hyp.recipientToken.ID, hyp.hypothesis, hyp.max_hyp, hyp.myMappingConnection.weight) for hyp in non_zero_hyps],
            'mappingConnections': [(mc.driverToken.ID, mc.recipientToken.ID, mc.weight) for mc in unique_mcs],
            'max_map_unit': token.max_map_unit.ID if token.max_map_unit else None,
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
                self.state.metadata['con_counts'][Type.P][Type.RB]['child'] += len(token.myRBs)
                self.state.metadata['con_counts'][Type.P][Type.RB]['parent'] += len(token.myParentRBs)
                self.state.metadata['con_counts'][Type.P][Type.GROUP] += len(token.myGroups)
                self.state.token_ids[Type.P].append(token.ID)
            case 'RB':
                data['myParentPs'] = [p.ID for p in token.myParentPs]
                data['myPred'] = [pred.ID for pred in token.myPred]
                data['myObj'] = [obj.ID for obj in token.myObj]
                data['myChildP'] = [childP.ID for childP in token.myChildP]
                data['timesFired'] = token.timesFired
                self.state.metadata['con_counts'][Type.RB][Type.P]['parent'] += len(token.myParentPs)
                self.state.metadata['con_counts'][Type.RB][Type.PO]['pred'] += len(token.myPred)
                self.state.metadata['con_counts'][Type.RB][Type.PO]['obj'] += len(token.myObj)
                self.state.metadata['con_counts'][Type.RB][Type.P]['child'] += len(token.myChildP)
                self.state.token_ids[Type.RB].append(token.ID)
            case 'PO':
                data['predOrObj'] = True if token.predOrObj == 1 else False
                data['myRBs'] = [rb.ID for rb in token.myRBs]
                data['same_RB_POs'] = [po.ID for po in token.same_RB_POs]
                data['mySemantics'] = [link.mySemantic.ID for link in token.mySemantics]
                data['semNormalization'] = token.semNormalization
                data['max_sem_weight'] = token.max_sem_weight
                self.state.metadata['con_counts'][Type.PO][Type.RB] += len(token.myRBs)
                self.state.token_ids[Type.PO].append(token.ID)
            case _:
                raise ValueError(f"Unknown token type: {token.my_type}")
        if data['set'] == Set.DRIVER:
            self.state.driver.append(token.ID)
        if data['set'] == Set.RECIPIENT:
            self.state.recipient.append(token.ID)
        return data

    def _extract_semantics(self) -> tuple[dict[int, dict], int]:
        """ Extract all semantic data. 

        Returns:
        tuple[dict[int, dict], int]: A tuple containing:
            semantics (dict[int, dict]): A dictionary of semantics by ID.
            sem_count (int): The number of semantics.
        """
        semantics = {}
        for sem in self.memory.semantics:
            sem_connect_weights: List = sem.semConnectWeights
            while True:
                try:
                    sem_connect_weights.remove(0.0)
                except ValueError:
                    break
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
                'semConnectWeights': sem_connect_weights,
            }
        return semantics, len(semantics)
    
    def _extract_semantic_connections(self) -> tuple[list[list[float]], int]:
        """ Extract all semantic connection data. 

        Returns:
        tuple[list[list[float]], int]: A tuple containing:
            connections (list[list[float]]): The semantic connections matrix.
            count (int): The number of semantic connections.
        """
        count = 0
        connections = [[0.0] * self.state.sem_count for _ in range(self.state.sem_count)]
        for sem in self.memory.semantics:
            for link in sem.semConnect:
                connections[self.state.sem_idxs[sem.ID]][self.state.sem_idxs[link.mySemantic.ID]] = link.weight
                if link.weight != 0:
                    count += 1
        return connections, count
    
    def _extract_links(self) -> tuple[list[list[float]], int]:
        """ Extract all link data. 
        Returns:
            tuple[list[list[float]], int]: A tuple containing:
                - links (list[list[float]]): The links matrix.
                - links_list (dict[int, list[int]]): A dictionary mapping token IDs to the IDs of the semantics they are linked to.
                - count (int): The number of links.
        """
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
        return links, links_list, count
    
    def _extract_mappings(self) -> tuple[list[list[dict[MappingFields, float]]], int, int]:
        """ Extract all mapping data. 
        Returns:
            tuple[list[list[dict[MappingFields, float]]], int, int]: A tuple containing:
                - mappings (list[list[dict[MappingFields, float]]]): The mappings matrix.
                - m_count (int): The number of mappings.
                - h_count (int): The number of hypotheses.
        """
        m_count = 0
        h_count = 0
        r_count = len(self.state.recipient)
        d_count = len(self.state.driver)
        mappings = [[{MappingFields.WEIGHT: 0.0, MappingFields.HYPOTHESIS: 0.0, MappingFields.MAX_HYP: 0.0} for _ in range(d_count)] for _ in range(r_count)]
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
                    if info[MappingFields.WEIGHT] != 0.0:
                        m_count += 1
                    if info[MappingFields.HYPOTHESIS] != 0.0:
                        h_count += 1
                # connections
                for mc in token.mappingConnections:
                    rec_idx = self.state.recipient_idxs[hyp.myMappingConnection.recipientToken.ID]
                    dri_idx = self.state.driver_idxs[hyp.myMappingConnection.driverToken.ID]
                    weight = mappings[rec_idx][dri_idx][MappingFields.WEIGHT]
                    if weight != 0.0 and weight != mc.weight:
                        logger.error(f"Mapping weight mismatch for {token.name}({token.ID}) -> {mc.driverToken.name}({mc.driverToken.ID}) ({weight} != {mc.weight})")
                    else:
                        mappings[rec_idx][dri_idx][MappingFields.WEIGHT] = mc.weight
        return mappings, m_count, h_count

    def _extract_connections(self) -> tuple[list[list[bool]], int]:
        """ Extract all connection data. 

        Returns:
        tuple[list[list[bool]], int]: A tuple containing:
            connections (list[list[bool]]): The connections matrix.
            count (int): The number of connections.
        """
        # init matrix
        connections = [[False] * self.state.tk_count for _ in range(self.state.tk_count)]
        # extract connections
        for myP in self.memory.Ps:
            for myRB in myP.myRBs:
                connections[self.state.idxs[myP.ID]][self.state.idxs[myRB.ID]] = True
        for myRB in self.memory.RBs:
            for myChildP in myRB.myChildP:
                connections[self.state.idxs[myRB.ID]][self.state.idxs[myChildP.ID]] = True
        for myPO in self.memory.POs:
            for myRB in myPO.myRBs:
                connections[self.state.idxs[myRB.ID]][self.state.idxs[myPO.ID]] = True
        # count the number of connections
        num_cons_final = 0
        for row in connections:
            for val in row:
                if val:
                    num_cons_final += 1
        return connections, num_cons_final

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