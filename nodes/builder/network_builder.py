# nodes/builder/node_builder.py
# Builds the network object.

import torch

from nodes.enums import *
from nodes.network import Network
from nodes.sets import *
from nodes.sets.connections import Links, Mappings

from nodes.builder import Build_set, Build_sems, Build_children, Build_connections

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
    def __init__(self, symProps: list[dict] = None, file_path: str = None):
        """
        Initialise the nodeBuilder with symProps and file_path.

        Args:
            symProps (list): A list of symProps.
            file_path (str): The path to the sym file.
        """
        self.symProps = symProps
        self.file_path = file_path
        self.token_sets = {}
        self.mappings = {}
        self.set_map = {
            "driver": Set.DRIVER,
            "recipient": Set.RECIPIENT,
            "memory": Set.MEMORY,
            "new_set": Set.NEW_SET
        }

    def build_nodes(self, DORA_mode= True):          # Build the nodes object
        """
        Build the nodes object.

        Args:
            DORA_mode (bool): Whether to use DORA mode.
        
        Returns:
            network (Network): The network object.
        
        Raises:
            ValueError: If no symProps or file_path set.
        """
        if self.file_path is not None:
            self.get_symProps_from_file()
        if self.symProps is not None:
            self.build_set_tensors()
            self.build_node_tensors()
            self.network = Network(self.driver_tensor, self.recipient_tensor, self.memory_tensor, self.new_set_tensor, self.semantics_tensor, self.mappings, DORA_mode)
            return self.network
        else:
            raise ValueError("No symProps or file_path provided")

    def build_set_tensors(self):    # Build sem_set, token_sets
        """
        Build the sem_set and token_sets.
        """
        props = {}

        # 1). Build the sems
        build_sems = Build_sems(self.symProps)
        self.sems = build_sems.build_sems()

        # 2). Initiliase empty lists for each set
        for set in Set:                                 
            props[set] = []

        # 3). Add the prop to the correct set
        for prop in self.symProps:
            props[self.set_map[prop["set"]]].append(prop)    
        
        # 4). Build the basic token sets
        for set in Set:                                
            build_set = Build_set(props[set], set)
            self.token_sets[set] = build_set.build_set()
        
        # 5). Build the children lists of IDs
        for set in Set:
            build_children = Build_children(set, self.token_sets[set], self.sems, props[set])
            build_children.get_children()
        
        # 6). Build the connections and links matrices                              
        build_connections = Build_connections(self.token_sets, self.sems)
        build_connections.build_connections_links()
        
        # 7). Tensorise the sets
        for set in Set:
            self.token_sets[set].tensorise()
        self.sems.tensorise()

    def build_link_object(self):    # Build the mem objects
        """
        Build the mem objects. (links, mappings)
        """
        # Create links object
        driver_links = self.token_sets[Set.DRIVER].links_tensor
        recipient_links = self.token_sets[Set.RECIPIENT].links_tensor
        memory_links = self.token_sets[Set.MEMORY].links_tensor
        new_set_links = self.token_sets[Set.NEW_SET].links_tensor
        self.links = Links(driver_links, recipient_links, memory_links, new_set_links, None)
        return self.links
    
    def build_map_object(self,set):
        # Create mapping tensors
        map_cons = torch.zeros(self.token_sets[set].num_tokens, self.token_sets[Set.DRIVER].num_tokens)
        map_weights = torch.zeros_like(map_cons)
        map_hyp = torch.zeros_like(map_cons)
        map_max_hyp = torch.zeros_like(map_cons)
                # Create mappings object
        mappings = Mappings(self.driver_tensor, map_cons, map_weights, map_hyp, map_max_hyp)
        self.mappings[set] = mappings
        return mappings
        

    def build_node_tensors(self):   # Build the node tensor objects
        """
        Build the per set tensor objects. (driver, recipient, memory, new_set, semantics)
        """
        self.build_link_object()
        self.semantics_tensor: Semantics = Semantics(self.sems.node_tensor, self.sems.connections_tensor, self.links, self.sems.id_dict)
        self.links.semantics = self.semantics_tensor

        driver_set = self.token_sets[Set.DRIVER]
        self.driver_tensor = Driver(driver_set.token_tensor, driver_set.connections_tensor, self.links, driver_set.id_dict)
        
        recipient_set = self.token_sets[Set.RECIPIENT]
        recipient_maps = self.build_map_object(Set.RECIPIENT)
        self.recipient_tensor = Recipient(recipient_set.token_tensor, recipient_set.connections_tensor, self.links, recipient_maps, recipient_set.id_dict)

        memory_set = self.token_sets[Set.MEMORY]
        mem_maps = self.build_map_object(Set.MEMORY)
        self.memory_tensor = Memory(memory_set.token_tensor, memory_set.connections_tensor, self.links, mem_maps, memory_set.id_dict)

        new_set = self.token_sets[Set.NEW_SET]
        self.new_set_tensor = New_Set(new_set.token_tensor, new_set.connections_tensor, self.links, new_set.id_dict)
    
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
