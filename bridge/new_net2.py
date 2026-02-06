from _pytest.mark.expression import IDENT_PREFIX
from .state_printer import StatePrinter
from .state import State
import torch
from nodes import NetworkBuilder
from nodes.enums import *
from nodes.network import Network, Params, Semantics
from nodes.network.tokens import Tokens, Token_Tensor,Links, Mapping,Connections_Tensor
from logging import getLogger
from pathlib import Path
logger = getLogger(__name__)

class NewNet:
    """
    A class for holding the new network object.
    """

    def __init__(self):
        self.network = None
        self.state = State()
        self.sim_path = None
        self.printer = StatePrinter(state=self.state)
        self.printer.header_text = "NEW"
    
    def set_network(self, network: Network):
        """ Set the network object. 
        Args:
            network: Network object.
        """
        self.network = network

    def set_state(self, state: State):
        """ Set the state object. 
        Args:
            state: State object.
        """
        self.state = state
        self.printer.set_state(self.state)
    
    def load_sim(self, sim_path):
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

    def get_state(self) -> State:
        """ Generate the state from the network. """
        state = self.state
        state.clear()
        state.tokens = self._extract_tokens()
        state.tk_count = len(state.tokens)
        state.semantics = self._extract_semantics()
        state.links, state.links_list = self._extract_links()
        state.mappings = self._extract_mappings()
        state.connections = self._extract_connections()
        state.metadata = {
            'sim_path': str(self.sim_path) if self.sim_path else None,
            'parameters': self.network.params.get_params_dict(),
            'token_counts': {
                'P': len(torch.where(self.network.token_tensor.tensor[:, TF.TYPE] == Type.P)[0]),
                'RB': len(torch.where(self.network.token_tensor.tensor[:, TF.TYPE] == Type.RB)[0]),
                'PO': len(torch.where(self.network.token_tensor.tensor[:, TF.TYPE] == Type.PO)[0]),
                'semantics': len(state.sem_ids),
            },
        }
        return state
    

# =============================== EXTRACTORS ===============================
    def _extract_tokens(self) -> dict[int, dict]:
        """ Extract the tokens from the network. """
        tokens = {}
        token_tensor = self.network.token_tensor
        non_deleted_mask = token_tensor.tensor[:, TF.DELETED] == B.FALSE
        all_indices = torch.where(non_deleted_mask)[0].tolist()
        for idx in all_indices:
            id = int(token_tensor.tensor[idx, TF.ID].item())
            tokens[id] = self._extract_token_data(idx)
        logger.debug(f"Extracted {len(tokens)} tokens.")
        return tokens
    
    def _extract_token_data(self, idx: int) -> dict:
        """ Extract the data from a token. """
        net = self.network
        nodes = net.node_ops
        id = nodes.tk_valc(idx, TF.ID)
        self.state.idxs[id] = idx
        self.state.ids[idx] = id
        # common data
        data = { 
            'name': net.get_name(idx),
            'ID': id,
            'set': nodes.tk_valc(idx, TF.SET),
            'type': nodes.tk_valc(idx, TF.TYPE),
            'myanalog': nodes.tk_valc(idx, TF.ANALOG),
            'act': nodes.tk_valc(idx, TF.ACT),
            'max_act': nodes.tk_valc(idx, TF.MAX_ACT),
            'my_index': idx,
            'inhibitor_input': nodes.tk_valc(idx, TF.INHIBITOR_INPUT),
            'inhibitor_act': nodes.tk_valc(idx, TF.INHIBITOR_ACT),
            'mappingHypotheses': None,
            'mappingConnections': None,
            'max_map_unit': nodes.tk_valc(idx, TF.MAX_MAP_UNIT),
            'max_map': nodes.tk_valc(idx, TF.MAX_MAP),
            'td_input': nodes.tk_valc(idx, TF.TD_INPUT),
            'bu_input': nodes.tk_valc(idx, TF.BU_INPUT),
            'lateral_input': nodes.tk_valc(idx, TF.LATERAL_INPUT),
            'map_input': nodes.tk_valc(idx, TF.MAP_INPUT),
            'net_input': nodes.tk_valc(idx, TF.NET_INPUT),
            'GUI_unit': None,
            'my_made_unit': nodes.tk_valc(idx, TF.MADE_UNIT),
            'my_made_units': nodes.tk_valc(idx, TF.MADE_UNIT), # I can't hold more that one at, should add another data structure if this is needed.
            'my_maker_unit': nodes.tk_valc(idx, TF.MAKER_UNIT),
            'inferred': nodes.tk_valc(idx, TF.INFERRED),
            'retrieved': nodes.tk_valc(idx, TF.RETRIEVED),
            'copy_for_DR': nodes.tk_valc(idx, TF.COPY_FOR_DR),
            'copied_DR_index': nodes.tk_valc(idx, TF.COPIED_DR_INDEX),
            'sim_made': nodes.tk_valc(idx, TF.SIM_MADE),
            'inhibitorThreshold': nodes.tk_valc(idx, TF.INHIBITOR_THRESHOLD)
        }
        # Extract type specific data
        match data['type']:
            case Type.P:
                data['myRBs'] = []
                data['myParentRBs'] = []
                data['myGroups'] = []
                data['mode'] = nodes.tk_valc(idx, TF.MODE)
                self.state.token_ids[Type.P].append(id)
            case Type.RB:
                data['myParentPs'] = []
                data['myPred'] = []
                data['myObj'] = []
                data['myChildP'] = []
                data['timesFired'] = nodes.tk_valc(idx, TF.TIMES_FIRED)
                self.state.token_ids[Type.RB].append(id)
            case Type.PO:
                data['predOrObj'] = nodes.tk_valc(idx, TF.PRED)
                data['myRBs'] = []
                data['same_RB_POs'] = []
                data['mySemantics'] = []
                data['semNormalization'] = nodes.tk_valc(idx, TF.SEM_COUNT)
                data['max_sem_weight'] = nodes.tk_valc(idx, TF.MAX_SEM_WEIGHT)
                self.state.token_ids[Type.PO].append(id)
            case _:
                raise ValueError(f"Unknown token type: {data['type']}")
        if data['set'] == Set.DRIVER:
            self.state.driver.append(data['ID'])
        if data['set'] == Set.RECIPIENT:
            self.state.recipient.append(data['ID'])
        return data
    
    def _extract_semantics(self) -> dict[int, dict]:
        """ Extract the semantics from the network. """
        sems = {}
        st = self.network.semantics
        non_deleted_mask = st.nodes[:, SF.DELETED] == B.FALSE
        all_indices = torch.where(non_deleted_mask)[0].tolist()
        for idx in all_indices:
            id = int(st.nodes[idx, SF.ID].item())
            self.state.sem_idxs[id] = idx
            self.state.sem_ids[idx] = id
            sems[id] = {
                'name': st.get_name(idx),
                'ID': id,
                'my_type': 'semantic',
                'dimension': st.get_dim_name(st.get_dim(idx)),
                'amount': st.getc(idx, SF.AMOUNT),
                'ont_status': st.getc(idx, SF.ONT),
                'myinput': st.getc(idx, SF.INPUT),
                'max_sem_input': st.getc(idx, SF.MAX_INPUT),
                'act': st.getc(idx, SF.ACT),
                'myPOs': [],
                'myGroups': [],
                'semConnect': [],
                'semConnectWeights': [],
            }
        logger.debug(f"Extracted {len(sems)} semantics.")
        return sems
    
    def _extract_links(self) -> list[list[float]]:
        """ Extract the links from the network. """
        state = self.state
        net = self.network
        tk_tens = net.token_tensor.tensor
        all_tks = torch.where(tk_tens[:, TF.DELETED] == B.FALSE)[0]
        all_sems = torch.where(net.semantics.nodes[:, SF.DELETED] == B.FALSE)[0]
        links = net.links.adj_matrix[all_tks, :][:, all_sems].tolist()
        # Get lists for each token
        links_list = {}
        linked_tokens = torch.where(net.links.adj_matrix[all_tks, :].any(dim=1))[0].tolist()
        for tk_idx in linked_tokens:
            tk_id = state.ids[tk_idx]
            linked_sems = torch.where(net.links.adj_matrix[tk_idx, :] > 0.0)[0].tolist()
            for sem_idx in linked_sems:
                sem_id = state.sem_ids[sem_idx]
                state.tokens[tk_id]['mySemantics'].append(sem_id)
                state.semantics[sem_id]['myPOs'].append(tk_id)
                if tk_id not in links_list:
                    links_list[tk_id] = []
                if sem_id not in links_list[tk_id]:
                    links_list[tk_id].append(sem_id)
                else:
                    raise ValueError(f"Duplicate link found for {tk_id} -> {sem_id}")
        logger.debug(f"Extracted {len(links_list)} links.")
        return links, links_list
        
    def _extract_mappings(self) -> list[list[dict[MappingFields, float]]]:
        """ Extract the mappings from the network. """
        map_tens = self.network.mappings.adj_matrix
        r_count = len(self.state.recipient)
        d_count = len(self.state.driver)
        m_c = 0
        h_c = 0
        mappings = [[{MappingFields.WEIGHT: 0.0, MappingFields.HYPOTHESIS: 0.0, MappingFields.MAX_HYP: 0.0} for _ in range(d_count)] for _ in range(r_count)]
        for idx in range(r_count):
            for d_idx in range(d_count):
                weight = map_tens[idx, d_idx, MappingFields.WEIGHT].item()
                hypothesis = map_tens[idx, d_idx, MappingFields.HYPOTHESIS].item()
                max_hyp = map_tens[idx, d_idx, MappingFields.MAX_HYP].item()
                mappings[idx][d_idx] = {
                    MappingFields.WEIGHT: weight,
                    MappingFields.HYPOTHESIS: hypothesis,
                    MappingFields.MAX_HYP: max_hyp,
                }
                if weight != 0.0:
                    m_c += 1
                if hypothesis != 0.0:
                    h_c += 1
        logger.debug(f"Extracted {m_c} mappings and {h_c} hypotheses.")
        return mappings
    
    def _extract_connections(self) -> list[list[bool]]:
        """ Extract the connections from the network. """
        all_tks = torch.where(self.network.token_tensor.tensor[:, TF.DELETED] == B.FALSE)[0]
        tokens = self.network.tokens
        con_tens = self.network.tokens.connections.tensor[all_tks, :][:, all_tks]
        connections = con_tens.tolist()
        con_tens = self.network.tokens.connections.tensor
        po_mask = tokens.arb_mask({TF.TYPE: Type.PO})
        rb_mask = tokens.arb_mask({TF.TYPE: Type.RB})
        po_idx_list = torch.where(po_mask)[0].tolist()
        po_shared_rb = self._shared(po_mask, po_mask, rb_mask, con_tens, con_tens.t())
        for id in self.state.idxs.keys():
            idx = self.state.idxs[id]
            tk = self.state.tokens[id]
            match tk['type']:
                case Type.P:
                    # myRBs
                    rbs = tokens.arb_mask({TF.TYPE: Type.RB})
                    p_childs = con_tens[idx, :] == True
                    my_rbs = torch.where(rbs & p_childs)[0].tolist()
                    for rb_idx in my_rbs:
                        rb_id = self.state.ids[rb_idx]
                        tk['myRBs'].append(rb_id)
                    # myParentRBs
                    p_parents = con_tens[:, idx] == True
                    my_parent_rbs = torch.where(p_parents & rbs)[0].tolist()
                    for p_rb_idx in my_parent_rbs:
                        p_rb_id = self.state.ids[p_rb_idx]
                        tk['myParentRBs'].append(p_rb_id)
                    # myGroups
                    pass
                case Type.RB:
                    # myParentPs
                    ps = tokens.arb_mask({TF.TYPE: Type.P})
                    rb_parents = con_tens[idx, :] == True
                    my_parent_ps = torch.where(rb_parents & ps)[0].tolist()
                    for p_idx in my_parent_ps:
                        p_id = self.state.ids[p_idx]
                        tk['myParentPs'].append(p_id)
                    # myPred
                    preds = tokens.arb_mask({TF.TYPE: Type.PO, TF.PRED: B.TRUE})
                    rb_preds = con_tens[idx, :] == True
                    my_preds = torch.where(rb_preds & preds)[0].tolist()
                    for pred_idx in my_preds:
                        pred_id = self.state.ids[pred_idx]
                        tk['myPred'].append(pred_id)
                    # myObj
                    objs = tokens.arb_mask({TF.TYPE: Type.PO, TF.PRED: B.FALSE})
                    rb_objs = con_tens[idx, :] == True
                    my_objs = torch.where(rb_objs & objs)[0].tolist()
                    for obj_idx in my_objs:
                        obj_id = self.state.ids[obj_idx]
                        tk['myObj'].append(obj_id)
                    # myChildP
                    child_ps = tokens.arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.CHILD})
                    rb_child_ps = con_tens[:, idx] == True
                    my_child_ps = torch.where(rb_child_ps & child_ps)[0].tolist()
                    for child_p_idx in my_child_ps:
                        child_p_id = self.state.ids[child_p_idx]
                        tk['myChildP'].append(child_p_id)
                    pass
                case Type.PO:
                    # myRBs
                    rbs = tokens.arb_mask({TF.TYPE: Type.RB})
                    po_parents = con_tens[:, idx] == True
                    my_rbs = torch.where(po_parents & rbs)[0].tolist()
                    for rb_idx in my_rbs:
                        rb_id = self.state.ids[rb_idx]
                        tk['myRBs'].append(rb_id)
                    # same_RB_POs
                    po_idx = po_idx_list.index(idx)
                    same_rb_pos = torch.where(po_shared_rb[po_idx, :] > 0)[0].tolist()
                    for same_rb_pos_idx in same_rb_pos:
                        same_rb_pos_id = self.state.ids[same_rb_pos_idx]
                        tk['same_RB_POs'].append(same_rb_pos_id)
                    pass
                case _:
                    raise ValueError(f"Unknown token type: {tk['type']}")
        logger.debug(f"Extracted {(con_tens > 0).sum()} connections.")
        return connections
    
    def _shared(self, child1_mask, child2_mask, parent_mask, con_tensor, parent_cons):
        """ Returns a child1xchild2 tensor of 1 if child1 and child2 are not both connected to the same parent, 0 o.w """
        c1 = child1_mask
        c2 = child2_mask
        p = parent_mask
        shared = torch.matmul(                                  # c1xc2 tensor, shared[i][j] > 1 if c1[i] and c2[j] share a parent, 0 o.w
                parent_cons[c1][:, p].float(),
                con_tensor[p][:, c2].float()                              
            ) 
        shared = torch.gt(shared, 0).int()                      # now shared[i][j] = 1 if c1[i] and c2[j] share a parent, 0 o.w
        return shared

# =============================== SAVE/LOAD ===============================
    def save_state(self, file_path: str):
        """ Save the state to a file. """
        self.state.to_json(file_path)
    
    def load_state(self, file_path: str):
        """ Load the state from a file. """
        self.state.from_json(file_path)

# =============================== BUILDER ===============================
    def load_from_state(self):
        """ Load network from state. """
        if self.state is None:
            raise ValueError("State is not set.")
        
        # Build all components
        connections = self._build_connections()
        token_tensor = self._build_token_tensor()
        links = self._build_links()
        mapping = self._build_mapping()
        semantics = self._build_semantics()
        params = self._build_params()
        
        # Create Tokens container
        tokens = Tokens(token_tensor, connections, links, mapping)
        
        # Create and return Network
        network = Network(tokens, semantics, params)

    def _build_connections(self) -> Connections_Tensor:
        """ Build the connections from the state. """
        return Connections_Tensor(torch.tensor(self.state.connections))
    
    def _build_token_tensor(self) -> Token_Tensor:
        """ Build the token tensor from the state. """
        IDs = list(self.state.tokens.keys())
        num_tokens = len(IDs)
        
        tokens_data = torch.zeros(num_tokens, len(TF), null, dtype=torch.float32)
        for id in IDs:
            idx = self.state.idxs[id]
            tk = self.state.tokens[id]
            tokens_data[idx, TF.ID] = id
            tokens_data[idx, TF.TYPE] = tk['type']
            tokens_data[idx, TF.SET] = tk['set']
            tokens_data[idx, TF.ANALOG] = tk['myanalog']

            tokens_data[idx, TF.ACT] = tk['act']
            tokens_data[idx, TF.NET_INPUT] = tk['net_input']
            tokens_data[idx, TF.TD_INPUT] = tk['td_input']
            tokens_data[idx, TF.BU_INPUT] = tk['bu_input']
            tokens_data[idx, TF.LATERAL_INPUT] = tk['lateral_input']
            tokens_data[idx, TF.MAP_INPUT] = tk['map_input']
            tokens_data[idx, TF.MAX_MAP] = tk['max_map']

            tokens_data[idx, TF.MAX_MAP_UNIT] = tk['max_map_unit']
            tokens_data[idx, TF.MADE_UNIT] = tk['my_made_unit']
            tokens_data[idx, TF.MAKER_UNIT] = tk['my_maker_unit']

            tokens_data[idx, TF.INHIBITOR_THRESHOLD] = tk['inhibitorThreshold']
            tokens_data[idx, TF.GROUP_LAYER] = tk['group_layer']
            tokens_data[idx, TF.MODE] = tk['mode']
            tokens_data[idx, TF.TIMES_FIRED] = tk['times_fired']
            tokens_data[idx, TF.SEM_COUNT] = tk['sem_count']
            tokens_data[idx, TF.INHIBITOR_INPUT] = tk['inhibitor_input']
            tokens_data[idx, TF.INHIBITOR_ACT] = tk['inhibitor_act']
            tokens_data[idx, TF.INFERRED] = tk['inferred']
            tokens_data[idx, TF.RETRIEVED] = tk['retrieved']
            tokens_data[idx, TF.COPY_FOR_DR] = tk['copy_for_DR']
            tokens_data[idx, TF.COPIED_DR_INDEX] = tk['copied_DR_index']
            tokens_data[idx, TF.SIM_MADE] = tk['sim_made']





