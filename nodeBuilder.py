from nodeEnums import *
import numpy as np
import torch
from nodes import Nodes
from nodeTensors import *
from nodeMemObjects import Links

# ===========[ INTERMIDIATE DATA STRUCTURES ]===========
class Node(object):
    def __init__(self, name):
        self.name = name
        self.features = None
        self.ID = None
    
    def set(self, feature, value: float):
        self.features[feature] = value
    
    def set_ID(self, ID):
        self.ID = ID
        self.features[TF.ID] = ID

class Semantic(Node):
    def __init__(self, name):
        super().__init__(name)
        self.features = [None] * len(SF)
        self.initialise_defaults()
    
    def initialise_defaults(self):      # TODO: Check defaults
        self.features[SF.TYPE] = Type.SEMANTIC
        self.features[SF.ONT_STATUS] = 0
        self.features[SF.AMOUNT] = 0
        self.features[SF.INPUT] = 0
        self.features[SF.MAX_INPUT] = 0
        self.features[SF.ACT] = 0
    
    def floatate_features(self):
        for feature in SF:
            if self.features[feature] is not None:
                self.features[feature] = float(self.features[feature])
            else:
                self.features[feature] = 0.0

class Token(Node):
    def __init__(self, name, set: Set, analog: int):
        super().__init__(name)
        self.features = [None] * len(TF)
        self.initialise_defaults()
        self.set(TF.SET, set)
        self.set(TF.ANALOG, analog)
        self.children = []
    
    def initialise_defaults(self):   # TODO: Check defaults
        self.features[TF.ID] = None
        self.features[TF.TYPE] = None
        self.features[TF.SET] = None
        self.features[TF.ANALOG] = None
        self.features[TF.MAX_MAP_UNIT] = 0
        self.features[TF.MADE_UNIT] = 0
        self.features[TF.MAKER_UNIT] = 0
        self.features[TF.INHIBITOR_THRESHOLD] = 0
        self.features[TF.GROUP_LAYER] = 0
        self.features[TF.MODE] = 0
        self.features[TF.TIMES_FIRED] = 0
        self.features[TF.SEM_COUNT] = 0
        self.features[TF.ACT] = 0
        self.features[TF.MAX_ACT] = 0
        self.features[TF.INHIBITOR_INPUT] = 0
        self.features[TF.INHIBITOR_ACT] = 0
        self.features[TF.MAX_MAP] = 0
        self.features[TF.NET_INPUT] = 0
        self.features[TF.MAX_SEM_WEIGHT] = 0
        self.features[TF.INFERRED] = False
        self.features[TF.RETRIEVED] = False
        self.features[TF.COPY_FOR_DR] = False
        self.features[TF.COPIED_DR_INDEX] = 0
        self.features[TF.SIM_MADE] = False
        self.features[TF.DELETED] = False
        self.features[TF.PRED] = False
    
    def floatate_features(self):
        for feature in TF:
            if self.features[feature] is not None:
                self.features[feature] = float(self.features[feature])
            else:
                self.features[feature] = 0.0

class Prop(Token):
    def __init__(self, name, set, analog):
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.P)

class RB(Token):
    def __init__(self, name, set, analog):
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.RB)

class PO(Token):
    def __init__(self, name, set, analog, is_pred: bool):
        super().__init__(name, set, analog)
        self.set(TF.TYPE, Type.PO)
        self.set(TF.PRED, is_pred)

class Token_set(object):
    def __init__(self, set: Set, tokens: dict[Type, list[Token]], name_dict: dict[str, Token], id_dict: dict[int, Token]):
        self.set = set
        self.tokens = tokens
        self.name_dict = name_dict
        self.id_dict = id_dict
        self.num_tokens = sum([len(self.tokens[type]) for type in Type])
        self.connections = np.zeros((self.num_tokens, self.num_tokens))
        self.links = np.zeros((self.num_tokens, self.num_tokens))
    
    def get_token(self, name):
        return self.name_dict[name]

    def get_token_by_id(self, ID):
        return self.id_dict[ID] 
    
    def get_token_tensor(self):
        token_tensor = torch.zeros((self.num_tokens, len(TF)))
        for type in Type:
            for token in self.tokens[type]:
                token.floatate_features()
                token_tensor[token.ID] = torch.tensor(token.features)
        return token_tensor
    
    def tensorise(self):
        self.token_tensor = self.get_token_tensor()
        self.connections_tensor = torch.tensor(self.connections)
        self.links_tensor = torch.tensor(self.links)

class Sem_set(object):
    def __init__(self, sems: list[Semantic], name_dict: dict[str, Semantic], id_dict: dict[int, Semantic]):
        self.sems = sems
        self.name_dict = name_dict
        self.id_dict = id_dict
        self.num_sems = len(sems)
        self.connections = np.zeros((self.num_sems, self.num_sems))
    
    def get_sem(self, name):
        return self.name_dict[name]
    
    def get_sem_by_id(self, ID):
        return self.id_dict[ID]

    def tensorise(self):
        self.node_tensor = torch.zeros((self.num_sems, len(SF)))
        self.connections_tensor = torch.tensor(self.connections)
        for sem in self.sems:
            sem.floatate_features()
            self.node_tensor[sem.ID] = torch.tensor(sem.features)
# ------------------------------------------------------


# ===================[ BUILD CLASSES ]==================
class Build_set(object):                                # Builds the nodes for a given set
    def __init__(self, symProps: list[dict], set: Set):
        self.symProps = symProps
        self.tokens = {}
        self.set = set
        self.name_dict = {}
        self.id_dict = {}
        self.names = []
        self.token_set = None
    
    def build_set(self):    #  Returns a new token_set
        self.get_nodes()       # Get the nodes from the symProps
        self.id_tokens()       # Create basic tokens
        self.token_set = Token_set(self.set, self.tokens, self.name_dict, self.id_dict)   # Return new token_set
        return self.token_set


    def get_nodes(self):         # Gets lists of unique tokens by type
        for type in Type:                    # Initialise empty lists for each type
            self.tokens[type] = []
        
        non_exist = "non_exist"
        for prop in self.symProps:
            prop_name = prop['name']
            analog = prop['analog']
            if prop_name != non_exist:
                self.create_token(prop_name, Prop, analog)
            for rb in prop['RBs']:
                pred_name = rb['pred_name']
                obj_name = rb['object_name']
                self.create_token(pred_name + "_" + obj_name, RB, analog)
                if pred_name != non_exist:
                    self.create_token(pred_name, PO, analog, True)
                if obj_name != non_exist:
                    self.create_token(obj_name, PO, analog, False)


    def id_tokens(self):   # Give each token an ID
        self.names = list(set(self.names))
        i = 0
        for name in self.names:
            node_obj =  self.name_dict[name]
            node_obj.set_ID(i)
            self.id_dict[i] = name
            i += 1

    def create_token(self, name, token_class, analog, is_pred = None): # Create a token and add it to the name/dict
        if name not in self.names:
            if is_pred is not None:
                obj = token_class(name, self.set, analog, is_pred)
            else:
                obj = token_class(name, self.set, analog)
            self.tokens[obj.features[TF.TYPE]].append(obj)
            self.name_dict[name] = obj
            self.names.append(name)


class Build_sems(object):                               # Builds the semantic objects
    def __init__(self, symProps: list[dict]):
        self.sems = []
        self.nodes = []
        self.name_dict = {}
        self.id_dict = {}
        self.num_sems = 0
        self.symProps = symProps
    
    def build_sems(self):
        self.get_sems(self.symProps)
        self.tokenise()
        self.sem_set = Sem_set(self.nodes, self.name_dict, self.id_dict)
        return self.sem_set

    def tokenise(self):                                 # Turn each unique semantic into a semantic object:
        self.num_sems = 0
        for sem in self.sems:
            new_sem = Semantic(sem)
            new_sem.set_ID(self.num_sems)
            self.nodes.append(new_sem)
            self.id_dict[self.num_sems] = sem
            self.name_dict[sem] = new_sem
            self.num_sems += 1
    
    def get_sems(self, symProps: list[dict]):           # Depth first search to get list of all nodes in the symProps:
        for prop in symProps:
            for rb in prop['RBs']:
                if rb['pred_sem'] is not []:
                    for sem in rb['pred_sem']:
                        self.sems.append(sem)
                if rb['object_sem'] is not []:
                    for sem in rb['object_sem']:
                        self.sems.append(sem)

        self.sems = list(set(self.sems))
        self.num_sems = len(self.sems)


class Build_children(object):                           # Builds the children of each token
    def __init__(self, set: Set, tokens: Token_set, sems: Sem_set, symProps: list[dict]):
        self.set = set
        self.token_set = tokens
        self.sems = sems
        self.symProps = symProps

    def get_children(self):                                 # Add child nodes to each token object
        for prop in self.symProps:
            self.get_prop_children(prop)
        for type in Type:                                   # Remove any duplicates
            for token in self.token_set.tokens[type]:
                token.children = list(set(token.children))
    
    def get_prop_children(self, prop: dict):                # Add children to the prop object, recursively call lower get child functions
        prop_obj = self.get_object(prop['name'])            # Get the prop object
        for rb in prop['RBs']:
            child = self.get_rb_children(rb)                # Find rbs children, and return rb ID
            if prop_obj is not None:
                prop_obj.children.append(child)             # Append child ID to prop object

    def get_rb_children(self, rb: dict):                    # Add children to the rb object, recursively call lower get child functions
        pred_name = rb['pred_name']
        obj_name = rb['object_name']
        rb_obj = self.get_object(pred_name + "_" + obj_name)
        child_pred = self.get_po_children(pred_name, rb['pred_sem'])      # Get children of pred, return pred ID
        child_obj = self.get_po_children(obj_name, rb['object_sem'])   # Get children of obj, return obj ID
        rb_obj.children.append(child_pred)
        rb_obj.children.append(child_obj)
        return rb_obj.ID

    def get_po_children(self, name, sems: list):       # Add children to the po object, return po ID
        po_obj = self.get_object(name)
        for sem in sems:
            sem_obj = self.sems.get_sem(sem)
            po_obj.children.append(sem_obj.ID)
        return po_obj.ID
    
    def get_object(self, name):                             # Returns token object, if it exists.  O.w returns None
        non_exist = "non_exist"
        non_exist_rb = "non_exist_non_exist"
        obj = None
        if name != non_exist and name != non_exist_rb:
            obj = self.token_set.get_token(name)
        return obj  


class Build_connections(object):                        # Builds links and connections for each set
    def __init__(self, token_sets: dict[Set, Token_set], sems: Sem_set):
        self.token_sets = token_sets
        self.sems = sems

    def build_connections_links(self):
        for set in Set:
            token_set = self.token_sets[set]
            token_set.connections = self.build_set_connections(token_set)
            token_set.links = self.build_set_links(token_set)

    def build_set_connections(self, token_set: Token_set):  # Returns matrix of all connections for a given set
        num_tks = token_set.num_tokens
        connections = np.zeros((num_tks, num_tks))          # len tokens x len tokens matrix for connections.
        for type in Type:
            if type != Type.PO:
                for node in token_set.tokens[type]:
                    for child in node.children:
                        connections[node.ID][child] = 1
        return connections
    
    def build_set_links(self, token_set: Token_set):        # Returns matrix of all po -> sem links for a given set
        num_tks = token_set.num_tokens
        num_sems = self.sems.num_sems
        links = np.zeros((num_tks, num_sems))               # Len tokens x len sems matrix for links.
        for po in token_set.tokens[Type.PO]:
            for child in po.children:
                links[po.ID][child] = 1
        return links


class nodeBuilder(object):                            # Builds tensors for each set, memory, and semantic objects. Finally build the nodes object.
    def __init__(self, symProps: list[dict]):
        self.symProps = symProps
        self.token_sets = {}
        self.set_map = {
            "driver": Set.DRIVER,
            "recipient": Set.RECIPIENT,
            "memory": Set.MEMORY,
            "new_set": Set.NEW_SET
        }

    def build_nodes(self, DORA_mode= True):          # Build the nodes object
        self.build_set_tensors()
        self.build_mem_objects()
        self.build_node_tensors()
        self.nodes = Nodes(self.driver_tensor, self.recipient_tensor, self.memory_tensor, self.new_set_tensor, self.semantics_tensor, self.mappings, DORA_mode)
        return self.nodes

    def build_set_tensors(self):    # Build sem_set, token_sets
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

    def build_mem_objects(self):    # Build the mem objects
        self.links = Links(self.token_sets[Set.DRIVER].links, self.token_sets[Set.RECIPIENT].links, self.token_sets[Set.MEMORY].links)
        self.mappings = Mappings(torch.zeros(self.token_sets[Set.RECIPIENT].num_tokens, self.token_sets[Set.DRIVER].num_tokens, 3))

    def build_node_tensors(self):   # Build the node tensor objects
        driver_set = self.token_sets[Set.DRIVER]
        self.driver_tensor = DriverTensor(driver_set.token_tensor, driver_set.connections_tensor, self.links, driver_set.id_dict)
        
        recipient_set = self.token_sets[Set.RECIPIENT]
        self.recipient_tensor = RecipientTensor(recipient_set.token_tensor, recipient_set.connections_tensor, self.links, recipient_set.id_dict)

        memory_set = self.token_sets[Set.MEMORY]
        self.memory_tensor = TokenTensor(memory_set.token_tensor, memory_set.connections_tensor, self.links, memory_set.id_dict)

        new_set = self.token_sets[Set.NEW_SET]
        self.new_set_tensor = TokenTensor(new_set.token_tensor, new_set.connections_tensor, self.links, new_set.id_dict)

        self.semantics_tensor = SemanticTensor(self.sems.node_tensor, self.sems.connections_tensor, self.links, self.sems.id_dict)
# ------------------------------------------------------

# ===================[ MAIN FUNCTION ]==================
def main(symProps: list[dict]):
    from time import time
    time_start = time()
    builder = nodeBuilder(symProps)
    nodes = builder.build_nodes()
    time_end = time()
    print(f"Time taken: {time_end - time_start} seconds")
    return nodes

if __name__ == "__main__":
    # example symProps
    symProps = [{'name': 'lovesMaryTom', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Mary', 'object_sem': ['mary1', 'mary2', 'mary3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Tom', 'object_sem': ['tom1', 'tom2', 'tome3'], 'P': 'non_exist'}], 'set': 'driver', 'analog': 0}, 
    {'name': 'lovesTomJane', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Tom', 'object_sem': ['tom1', 'tom2', 'tome3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Jane', 'object_sem': ['jane1', 'jane2', 'mary2'], 'P': 'non_exist'}], 'set': 'driver', 'analog': 0},
    {'name': 'jealousMaryJane', 'RBs': [{'pred_name': 'jealous_act', 'pred_sem': ['jel1', 'jel2', 'jel3'], 'higher_order': False, 'object_name': 'Mary', 'object_sem': ['mary1', 'mary2', 'mary3'], 'P': 'non_exist'}, {'pred_name': 'jealous_pat', 'pred_sem': ['jel4', 'jel5', 'jel6'], 'higher_order': False, 'object_name': 'Jane', 'object_sem': ['jane1', 'jane2', 'mary2'], 'P': 'non_exist'}], 'set': 'driver', 'analog': 0},
    {'name': 'lovesJohnKathy', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'John', 'object_sem': ['john1', 'john2', 'john3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Kathy', 'object_sem': ['kathy1', 'kathy2', 'kathy3'], 'P': 'non_exist'}], 'set': 'recipient', 'analog': 0}, 
    {'name': 'lovesKathySam', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Kathy', 'object_sem': ['kathy1', 'kathy2', 'kathy3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Sam', 'object_sem': ['sam1', 'sam2', 'sam3'], 'P': 'non_exist'}], 'set': 'recipient', 'analog': 0}]

    main(symProps)
# ------------------------------------------------------