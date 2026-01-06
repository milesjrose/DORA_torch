from ..enums import *
from ..network import Network, Params, Tokens, Semantics, Mapping, Links, Token_Tensor, default_params
from ..network.tokens.connections.connections import Connections_Tensor
from ..network.tokens.connections.links import LD
from ..network.single_nodes import Token, Semantic
from time import monotonic
import torch


class NetworkBuilder(object):
    """
    A class for building the network object from sym files.

    Attributes:
        symProps (list): A list of symProps.
        file_path (str): The path to the sym file.
        params (Params): The parameters for the network.
        do_timing (bool): Whether to record timing of the build process.
    """
    def __init__(self, symProps: list[dict] = None, file_path: str = None, params: Params = None, do_timing: bool = False):
        """
        Initialise the NetworkBuilder with symProps and file_path.

        Args:
            symProps (list): A list of symProps.
            file_path (str): The path to the sym file.
            params (Params, optional): The parameters for the network.
            do_timing (bool, optional): Whether to record timing of the build process.
        """
        self.do_timing = do_timing
        self.last_time = {0: monotonic()}
        self.symProps = symProps
        self.file_path = file_path
        self.params = params if params is not None else default_params()
        self.set_map = {
            "driver": Set.DRIVER,
            "recipient": Set.RECIPIENT,
            "memory": Set.MEMORY,
            "new_set": Set.NEW_SET
        }
        
        # Build state tracking
        self._next_token_id = 1
        self._next_sem_id = 1
        self._token_list: list[Token] = []
        self._token_names: list[str] = []
        self._semantic_dict: dict[str, int] = {}  # name -> index in sem list
        self._semantic_list: list[Semantic] = []
        self._semantic_names: dict[int, str] = {}  # id -> name
        self._semantic_ids: dict[int, int] = {}  # id -> index in tensor
        
        # Connection tracking
        self._parent_child_pairs: list[tuple[int, int]] = []  # (parent_idx, child_idx)
        self._token_sem_links: list[tuple[int, int, float]] = []  # (token_idx, sem_idx, weight)

    def build_network(self) -> Network:
        """
        Build the network object.

        Returns:
            network (Network): The network object.
        
        Raises:
            ValueError: If no symProps or file_path set.
        """
        if self.file_path is not None:
            read_time = []
            self.timer(read_time, start=True)
            self.get_symProps_from_file()
            self.timer(read_time)
        
        if self.symProps is not None:
            # 1. Parse symProps and build tokens/semantics
            self._parse_symProps()
            
            # 2. Create tensors from the collected data
            token_tensor = self._build_token_tensor()
            connections = self._build_connections_tensor()
            links = self._build_links_tensor()
            mapping = self._build_mapping_tensor()
            semantics = self._build_semantics_object()
            
            # 3. Create the Tokens container
            tokens = Tokens(token_tensor, connections, links, mapping)
            
            # 4. Create and return Network
            network = Network(tokens, semantics, mapping, links, self.params)
            return network
        else:
            raise ValueError("No symProps or file_path provided")
    
    def _parse_symProps(self):
        """Parse symProps list and build tokens and semantics."""
        analog_counter = {}  # set -> current analog number
        
        for prop in self.symProps:
            prop_name = prop['name']
            prop_set = self.set_map[prop['set']]
            prop_analog = prop.get('analog', 0)
            
            # Track analog numbers per set
            if prop_set not in analog_counter:
                analog_counter[prop_set] = 0
            
            # Create P token for the proposition
            p_idx = self._add_token(
                name=prop_name,
                token_type=Type.P,
                token_set=prop_set,
                analog=prop_analog,
                mode=Mode.NEUTRAL
            )
            
            # Process each RB (role binding) in the proposition
            for rb in prop['RBs']:
                rb_idx = self._process_rb(rb, prop_set, prop_analog, p_idx)
    
    def _process_rb(self, rb: dict, token_set: Set, analog: int, parent_p_idx: int) -> int:
        """Process a role binding and create RB and PO tokens."""
        pred_name = rb['pred_name']
        pred_sems = rb['pred_sem']
        obj_name = rb['object_name']
        obj_sems = rb['object_sem']
        is_higher_order = rb.get('higher_order', False)
        
        # Create RB token
        rb_name = f"{pred_name}_{obj_name}"
        rb_idx = self._add_token(
            name=rb_name,
            token_type=Type.RB,
            token_set=token_set,
            analog=analog
        )
        
        # Connect P -> RB
        self._parent_child_pairs.append((parent_p_idx, rb_idx))
        
        # Create predicate PO token
        pred_idx = self._add_token(
            name=pred_name,
            token_type=Type.PO,
            token_set=token_set,
            analog=analog,
            is_pred=True
        )
        
        # Connect RB -> Pred PO
        self._parent_child_pairs.append((rb_idx, pred_idx))
        
        # Add semantics for predicate
        for sem_name in pred_sems:
            sem_idx = self._get_or_create_semantic(sem_name)
            self._token_sem_links.append((pred_idx, sem_idx, 1.0))
        
        # Create object PO token
        obj_idx = self._add_token(
            name=obj_name,
            token_type=Type.PO,
            token_set=token_set,
            analog=analog,
            is_pred=False
        )
        
        # Connect RB -> Object PO
        self._parent_child_pairs.append((rb_idx, obj_idx))
        
        # Add semantics for object
        for sem_name in obj_sems:
            sem_idx = self._get_or_create_semantic(sem_name)
            self._token_sem_links.append((obj_idx, sem_idx, 1.0))
        
        return rb_idx
    
    def _add_token(self, name: str, token_type: Type, token_set: Set, 
                   analog: int = 0, mode: Mode = None, is_pred: bool = None) -> int:
        """Add a token and return its index."""
        features = {
            TF.ID: self._next_token_id,
            TF.TYPE: token_type,
            TF.SET: token_set,
            TF.ANALOG: analog,
        }
        
        if mode is not None:
            features[TF.MODE] = mode
        
        if is_pred is not None:
            features[TF.PRED] = B.TRUE if is_pred else B.FALSE
        
        token = Token(type=token_type, set=token_set, features=features, name=name)
        idx = len(self._token_list)
        self._token_list.append(token)
        self._token_names.append(name)
        self._next_token_id += 1
        
        return idx
    
    def _get_or_create_semantic(self, name: str) -> int:
        """Get existing semantic index or create a new one."""
        if name in self._semantic_dict:
            return self._semantic_dict[name]
        
        sem = Semantic(name=name, features={SF.TYPE: Type.SEMANTIC})
        idx = len(self._semantic_list)
        self._semantic_list.append(sem)
        self._semantic_dict[name] = idx
        self._semantic_ids[self._next_sem_id] = idx
        self._semantic_names[self._next_sem_id] = name
        self._next_sem_id += 1
        
        return idx
    
    def _build_token_tensor(self) -> Token_Tensor:
        """Build the Token_Tensor from collected tokens."""
        num_tokens = len(self._token_list)
        
        # Create token tensor
        tokens_data = torch.zeros(num_tokens, len(TF), dtype=tensor_type)
        for i, token in enumerate(self._token_list):
            tokens_data[i, :] = token.tensor
        
        # Create names dict
        names = {i: name for i, name in enumerate(self._token_names)}
        
        # Create connections (will be populated separately)
        connections = Connections_Tensor(torch.zeros(num_tokens, num_tokens, dtype=torch.bool))
        
        return Token_Tensor(tokens_data, connections, names)
    
    def _build_connections_tensor(self) -> Connections_Tensor:
        """Build the Connections_Tensor from parent-child pairs."""
        num_tokens = len(self._token_list)
        connections = torch.zeros(num_tokens, num_tokens, dtype=torch.bool)
        
        for parent_idx, child_idx in self._parent_child_pairs:
            connections[parent_idx, child_idx] = True
        
        return Connections_Tensor(connections)
    
    def _build_links_tensor(self) -> Links:
        """Build the Links tensor connecting tokens to semantics."""
        num_tokens = len(self._token_list)
        num_sems = len(self._semantic_list)
        
        # Ensure minimum size
        num_sems = max(num_sems, 1)
        
        links_data = torch.zeros(num_tokens, num_sems, dtype=tensor_type)
        
        for token_idx, sem_idx, weight in self._token_sem_links:
            links_data[token_idx, sem_idx] = weight
        
        return Links(links_data)
    
    def _build_mapping_tensor(self) -> Mapping:
        """Build the Mapping tensor for driver-recipient mappings."""
        # Count tokens per set
        driver_count = sum(1 for t in self._token_list if t.tensor[TF.SET] == Set.DRIVER)
        recipient_count = sum(1 for t in self._token_list if t.tensor[TF.SET] == Set.RECIPIENT)
        
        # Ensure minimum size
        driver_count = max(driver_count, 1)
        recipient_count = max(recipient_count, 1)
        
        # Create mapping tensor: [recipient, driver, fields]
        mapping_data = torch.zeros(recipient_count, driver_count, len(MappingFields), dtype=tensor_type)
        
        return Mapping(mapping_data)
    
    def _build_semantics_object(self) -> Semantics:
        """Build the Semantics object."""
        num_sems = len(self._semantic_list)
        
        # Ensure minimum size
        if num_sems == 0:
            # Create empty semantics with one placeholder
            nodes = torch.zeros(1, len(SF), dtype=tensor_type)
            nodes[0, SF.DELETED] = B.TRUE
            connections = torch.zeros(1, 1, dtype=tensor_type)
            ids = {}
            names = {}
        else:
            # Create semantic nodes tensor
            nodes = torch.zeros(num_sems, len(SF), dtype=tensor_type)
            for i, sem in enumerate(self._semantic_list):
                nodes[i, :] = sem.tensor
                nodes[i, SF.ID] = i + 1  # 1-indexed IDs
            
            # Create semantic connections (no internal semantic connections for now)
            connections = torch.zeros(num_sems, num_sems, dtype=tensor_type)
            
            # Build ID and name mappings
            ids = {i + 1: i for i in range(num_sems)}  # id -> index
            names = {i + 1: self._semantic_list[i].name for i in range(num_sems)}  # id -> name
        
        return Semantics(nodes, connections, ids, names)

    def get_symProps_from_file(self):
        """Read the symProps from the file into a list of dicts."""
        import json
        file = open(self.file_path, "r")    
        if file:
            simType = ""
            di = {"simType": simType}
            file.seek(0)
            exec(file.readline(), di)
            if di["simType"] == "sym_file":
                symstring = ""
                for line in file:
                    symstring += line
                symProps = []
                di = {"symProps": symProps}
                exec(symstring, di)
                self.symProps = di["symProps"]
            elif di["simType"] == "json_sym":
                symProps = json.loads(file.readline())
                self.symProps = symProps
            else:
                print(
                    "\nThe sym file you have loaded is formatted incorrectly. "
                    "\nPlease check your sym file and try again."
                )
            file.close()

    def timer(self, times, timer_index=0, start=False):
        """Get time since last timer call."""
        if self.do_timing:  
            if not start:
                times.append(monotonic() - self.last_time[timer_index])
            self.last_time[timer_index] = monotonic()
