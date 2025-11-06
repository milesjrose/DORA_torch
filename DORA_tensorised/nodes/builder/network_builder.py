# nodes/builder/network_builder.py
# Builds the network object.

import torch
from time import monotonic

from ..enums import *
from ..network import Network, Params, Links, Mappings
from ..network.sets import *

from .build_set import Build_set
from .build_sems import Build_sems
from .build_children import Build_children
from .build_connections import build_con_tensors, build_links_tensors
from ..network.network_params import default_params


class NetworkBuilder(object):                              # Builds tensors for each set, memory, and semantic objects. Finally build the nodes object.
    """
    A class for building the network object.

    Attributes:
        symProps (list): A list of symProps.
        file_path (str): The path to the sym file.
        token_sets (dict): A dictionary of token sets, mapping set to token set object.
        mappings (dict): A dictionary of mapping object, mappings sets to mapping object.
        set_map (dict): A dictionary of set mappings, mapping set name to set. Used for reading set from symProps file.
    """
    def __init__(self, symProps: list[dict] = None, file_path: str = None, params: Params = None, do_timing: bool = False):
        """
        Initialise the nodeBuilder with symProps and file_path.

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
        self.token_sets = {}
        self.mappings = {}
        self.params = params if params is not None else default_params()
        self.set_map = {
            "driver": Set.DRIVER,
            "recipient": Set.RECIPIENT,
            "memory": Set.MEMORY,
            "new_set": Set.NEW_SET
        }

        # Built objects
        self.built_sets = {}
        self.built_mappings = None
        self.built_links = None

    def build_network(self):          # Build the network object
        """
        Build the network object.

        Returns:
            network (Network): The network object.
        
        Raises:
            ValueError: If no symProps or file_path set.
        """
        
        if self.file_path is not None:
            read_time = []
            self.timer(read_time, start = True)
            self.get_symProps_from_file()
            self.timer(read_time)

        if self.symProps is not None:
            set_times = self.build_set_tensors()
            node_times = self.build_set_objects()
            con_times = self.build_inter_set_connections()
            self.network = Network(
                dict_sets=self.built_sets, 
                semantics=self.built_semantics, 
                mappings=self.built_mappings, 
                links=self.built_links, 
                params=self.params)
            self.network.set_params(self.params)
            return self.network
        
        else:
            raise ValueError("No symProps or file_path provided")

    def build_set_tensors(self):    # Build sem_set, token_sets
        """
        Build the sem_set and token_sets.
        """
        props = {}
        times = []
        self.timer(times, start = True)

        # 1). Build the sems
        build_sems = Build_sems(self.symProps)
        self.sems = build_sems.build_sems()
        self.timer(times)

        # 2). Initiliase empty lists for each set
        for set in Set:                                 
            props[set] = []
        self.timer(times)

        # 3). Add the prop to the correct set
        for prop in self.symProps:
            props[self.set_map[prop["set"]]].append(prop)    
        self.timer(times)
        
        # 4). Build the basic token sets
        for set in Set:                                
            build_set = Build_set(props[set], set)
            self.token_sets[set] = build_set.build_set()
        self.timer(times)

        # 5). Build the children lists of IDs
        for set in Set:
            build_children = Build_children(set, self.token_sets[set], self.sems, props[set])
            build_children.get_children()
        self.timer(times)

        # 7). Tensorise the sets 
        for set in Set:
            self.token_sets[set].tensorise()
        self.sems.tensorise()

        self.timer(times)
        return times

    def build_set_objects(self):   # Build the set objects
        """
        Build the set objects.
        """
        # Function to build a given set
        def build_set(token_set, Set_Class):
            if not isinstance(token_set.connections, torch.Tensor):
                raise TypeError(f"connections must be torch.Tensor, not {type(token_set.connections)}.")
            
            # Convert id_dict from dict[int, Inter_Token] to dict[int, int] (ID -> tensor index)
            # Convert name_dict from dict[str, Inter_Token] to dict[int, str] (ID -> name)
            IDs = {}
            names = {}
            for token_id, token_obj in token_set.id_dict.items():
                IDs[token_id] = token_obj.ID  # token_obj.ID is the tensor index
                names[token_id] = token_obj.name
            
            return Set_Class(
                token_set.token_tensor, 
                token_set.connections, 
                IDs, 
                names
                )
        set_classes = {
            Set.DRIVER: Driver,
            Set.RECIPIENT: Recipient,
            Set.MEMORY: Memory,
            Set.NEW_SET: New_Set
        }
        # Set timer
        times = []
        self.timer(times, start = True)
        # Build the connections tensors
        build_con_tensors(self.token_sets, self.sems)
        self.timer(times)

        # Build Semantics
        self.built_semantics: Semantics = Semantics(self.sems.node_tensor, self.sems.connections_tensor, self.sems.id_dict, self.sems.name_dict)
        self.timer(times)

        # Build the sets
        for set in Set:
            self.built_sets[set] = build_set(self.token_sets[set], set_classes[set])
            self.timer(times)

        return times
    
    def build_inter_set_connections(self):
        """Build the inter set connections."""
        times = []
        self.timer(times, start = True)
        # =============== Build tensors ================
        build_links_tensors(self.token_sets, self.sems)
        self.timer(times)

        # ============ Build the links object ============
        links = {}
        for set in Set:
            links[set] = self.token_sets[set].links
        self.built_links = Links(links, self.built_semantics)
        self.timer(times)

        # ============ Build the mapping dictionary ============
        for set in [Set.RECIPIENT]:
            # Get sizes of tensors
            map_size = torch.zeros(self.built_sets[set].nodes.shape[0], self.built_sets[Set.DRIVER].nodes.shape[0], dtype=tensor_type)

            # Make tensor for each mapping field
            mapping_tensors = {}
            for field in MappingFields:
                mapping_tensors[field] = torch.zeros_like(map_size, dtype=tensor_type)

            # Create mappings object
            self.built_mappings = Mappings(self.built_sets[Set.DRIVER], mapping_tensors)
        self.timer(times)
        
        return times

    def get_symProps_from_file(self):
        """
        Read the symProps from the file into a list of dicts.
        """
        import json
        file = open(self.file_path, "r")    
        if file:
            simType = ""
            di = {"simType": simType}  # porting from Python2 to Python3
            file.seek(0)  # to get to the beginning of the file.
            exec(file.readline(), di)  # porting from Python2 to Python3
            if di["simType"] == "sym_file":  # porting from Python2 to Python3
                symstring = ""
                for line in file:
                    symstring += line
                do_run = True
                symProps = []  # porting from Python2 to Python3
                di = {"symProps": symProps}  # porting from Python2 to Python3
                exec(symstring, di)  # porting from Python2 to Python3
                self.symProps = di["symProps"]  # porting from Python2 to Python3
            # now load the parameter file, if there is one.
            # if self.parameterFile:
            #     parameter_string = ''
            #     for line in self.parameterFile:
            #         parameter_string += line
            #     try:
            #         exec (parameter_string)
            #     except:
            #         print ('\nYour loaded paramter file is wonky. \nI am going to run anyway, but with preset parameters.')
            #         keep_running = input('Would you like to continue? (Y) or any key to exit>')
            #         if keep_running.upper() != 'Y':
            #             do_run = False
            elif di["simType"] == "json_sym":  # porting from Python2 to Python3
                # you've loaded a json generated sym file, which means that it's in json format, and thus must be loaded via the json.load() routine.
                # load the second line of the sym file via json.load().
                symProps = json.loads(file.readline())
                self.symProps = symProps
            else:
                print(
                    "\nThe sym file you have loaded is formatted incorrectly. \nPlease check your sym file and try again."
                )
                input("Enter any key to return to the MainMenu>")

    def timer(self, times, timer_index = 0, start = False, ):
        """Get time since last timer call."""
        if self.do_timing:  
            if not start:
                times.append(monotonic() - self.last_time[timer_index])
            self.last_time[timer_index] = monotonic()