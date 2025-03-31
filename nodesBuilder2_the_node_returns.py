from nodeEnums import *
import numpy as np
import torch

# ===========[ INTERMIDIATE DATA STRUCTURES ]===========

class Node(object):
    def __init__(self, name):
        self.name = name
        self.features = None

    def floatate_features(self):
        for feature in self.features:
            if feature is not None:
                feature = float(feature)
            else:
                feature = 0.0
    
    def set(self, feature: TF, value: float):
        self.features[feature] = value


class Semantic(Node):
    def __init__(self, name, ID):
        super().__init__(name)
        self.features = [None] * len(SF)
        self.initialise_defaults(ID)
    
    def initialise_defaults(self, ID):
        self.features[SF.ID] = ID
        self.features[SF.TYPE] = Type.SEMANTIC
        self.features[SF.ONT_STATUS] = None
        self.features[SF.AMOUNT] = 0
        self.features[SF.INPUT] = 0
        self.features[SF.MAX_INPUT] = 0
        self.features[SF.ACT] = 0


class Token(Node):
    def __init__(self, name, ID):
        super().__init__(name)
        self.features = [None] * len(TF)
        self.initialise_defaults(ID)
        self.children = []
    
    def initialise_defaults(self, ID):
        self.features[TF.ID] = ID
        self.features[TF.TYPE] = Type.TOKEN
        self.features[TF.SET] = None
        self.features[TF.ANALOG] = 0
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


class Prop(Token):
    def __init__(self, name, ID):
        super().__init__(name, ID)
        self.features[TF.TYPE] = Type.P


class RB(Token):
    def __init__(self, name, ID):
        super().__init__(name, ID)
        self.features[TF.TYPE] = Type.RB


class PO(Token):
    def __init__(self, name, ID, is_pred: bool):
        super().__init__(name, ID)
        self.features[TF.TYPE] = Type.PO
        self.features[TF.PRED] = B.TRUE if is_pred else B.FALSE

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

class Sem_set(object):
    def __init__(self, sems: list[Semantic], name_dict: dict[str, Semantic], id_dict: dict[int, Semantic]):
        self.sems = sems
        self.name_dict = name_dict
        self.id_dict = id_dict

    def get_sem(self, name):
        return self.name_dict[name]
    
    def get_sem_by_id(self, ID):
        return self.id_dict[ID]

# ------------------------------------------------------


# ===================[ BUILD CLASSES ]==================
class Build_set(object):                                # Builds the nodes for a given set
    def __init__(self, symProps: list[dict], set: Set):
        self.symProps = symProps
        self.props = []
        self.RBs = []
        self.preds = []
        self.objs = []
        self.tokens = {}
        self.set = set
        self.name_dict = {}
        self.id_dict = {}
        self.type_map = {
            Type.P: Prop,
            Type.RB: RB,
            Type.PO: PO
        }
    
    def build_set(self):    #  Returns a new token_set
        self.get_nodes(self.symProps)                                           # Get the nodes from the symProps
        self.create_tokens()                                                     # Create basic tokens
        return Token_set(self.set, self.props, self.RBs, self.preds, self.objs, self.name_dict, self.id_dict) #  Return new token_set

    
    def get_nodes(self, symProps: list[dict]):              # Gets lists of unique tokens by type
        for type in Type:                                   # Initialise empty lists for each type
            self.tokens[type] = []
        non_exist = "non_exist"
        for prop in self.symProps:
            prop_name = prop['name']
            if prop_name != non_exist:
                self.tokens[Type.P].append(prop_name)
            for rb in prop['RBs']:
                pred_name = rb['pred_name']
                if pred_name != non_exist:
                    self.tokens[Type.PO].append((pred_name, True))
                obj_name = rb['object_name']
                if obj_name != non_exist:
                    self.tokens[Type.PO].append((obj_name, False))
                self.tokens[Type.RB].append(pred_name + "_" + obj_name)
        for type in Type:
            self.tokens[type] = list(set(self.tokens[type])) # Remove duplicates

    def create_tokens(self):                                 # Create basic tokens, based on default values
        i = 0
        for type in Type:
            for token in self.tokens[type]:
                self.create_token(token, i, type)
                i += 1

    def create_token(self, name, ID, type):             # Create a token and add it to the name and id dictionaries
        if isinstance(name, tuple):                     # If name is a tuple, it is a PO object, as contains name and is_pred
            obj = PO(name[0], ID, name[1])
        else:                                           # Otherwise, it is a basic token
            obj_class = self.type_map[type]
            obj = obj_class(name, ID)
        self.name_dict[name] = obj
        self.id_dict[ID] = obj


class Build_sems(object):                               # Builds the semantic objects
    def __init__(self, symProps: list[dict]):
        self.sems = []
        self.nodes = []
        self.node_dict = {}
        self.id_dict = {}
        self.num_sems = 0
    
    def build_sems(self):
        self.get_sems(self.symProps)
        self.tokenise()
        return Sem_set(self.nodes, self.name_dict, self.id_dict)

    def tokenise(self):                                 # Turn each unique semantic into a semantic object:
        for sem in self.sems:
            self.id_dict[sem] = self.num_sems
            new_sem = Semantic(sem, self.num_sems)
            self.nodes.append(new_sem)
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
    
    def build(self):
        self.get_sems(self.symProps)
        self.tokenise()


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
        rb_obj = self.get_object(rb['object_name'] + "_" + rb['pred_name'])
        child_pred = self.get_po_children(rb['pred_name'], rb['pred_sem'])      # Get children of pred, return pred ID
        child_obj = self.get_po_children(rb['object_name'], rb['object_sem'])   # Get children of obj, return obj ID
        rb_obj.children.append(child_pred)
        rb_obj.children.append(child_obj)
        return rb_obj.ID

    def get_po_children(self, obj: dict, sems: list):       # Add children to the po object, return po ID
        po_obj = obj['name']
        for sem in sems:
            sem_obj = self.sems.get_sem(sem)
            po_obj.children.append(sem_obj.ID)
        return po_obj.ID
    
    def get_object(self, name):                             # Returns token object, if it exists.  O.w returns None
        non_exist = "non_exist"
        non_exist_rb = "non_exist_non_exist"
        obj = None
        if name != non_exist and name != non_exist_rb:
            obj = self.token_set.get_node(name)
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
            for node in token_set.tokens[type]:
                for child in node.children:
                    connections[node.ID][child.ID] = 1
        return connections
    
    def build_set_links(self, token_set: Token_set):        # Returns matrix of all po -> sem links for a given set
        num_tks = token_set.num_tokens
        num_sems = self.sems.num_sems
        links = np.zeros((num_tks, num_sems))               # Len tokens x len sems matrix for links.
        for po in token_set.tokens[Type.PO]:
            for child in po.children:
                links[po.ID][child.ID] = 1
        return links


class Build_tensors(object):
    def __init__(self, symProps: list[dict]):
        self.symProps = symProps
        self.token_sets = {}     # Map set -> Token_set
        self.sems = None         # Sem_set
        self.set_map = {
            "driver": Set.DRIVER,
            "recipient": Set.RECIPIENT,
            "memory": Set.MEMORY,
            "new_set": Set.NEW_SET
        }

    def build_node_sets(self):   # Build sem_set, token_sets
        props = {}

        # 1). Build the sems
        build_sems = Build_sems(self.symProps)
        sems = build_sems.build()

        # 2). Initiliase empty lists for each set
        for set in Set:                                 
            props[set] = []

        # 3). Add the prop to the correct set
        for prop in self.symProps:
            props[self.set_map[prop["set"]]].append(prop)    
        
        # 4). Build the basic token sets
        for set in Set:                                
            build_set = Build_set(self.props[set], set)
            self.token_sets[set] = build_set.build_set()
        
        # 5). Build the children lists of IDs
        for set in Set:
            build_children = Build_children(set, self.token_sets[set], sems, self.props[set])
            build_children.get_children()
        
        # 6). Build the connections and links matrices
        for set in Set:                                 
            build_connections = Build_connections(self.token_sets[set], self.sems)
            build_connections.build_connections_links()
        
# ------------------------------------------------------

# ===================[ MAIN FUNCTION ]==================
def main(symProps: list[dict]):
    props = {}
    token_sets = {}
    sems = None
    set_map = {
        "driver": Set.DRIVER,
        "recipient": Set.RECIPIENT,
        "memory": Set.MEMORY,
        "new_set": Set.NEW_SET
    }

    # 1). Build the sems
    build_sems = Build_sems(symProps)
    sems = build_sems.build()

    # 2). Initiliase empty lists for each set
    for set in Set:                                 
        props[set] = []

    # 3). Add the prop to the correct set
    for prop in symProps:
        props[set_map[prop["set"]]].append(prop)    
    
    # 4). Build the token_sets
    for set in Set:                                
        build_set = Build_set(props[set], set)
        token_sets[set] = build_set.build_set()
    
    # 5). Build the children lists of IDs
    for set in Set:
        build_children = Build_children(set, token_sets[set], sems, props[set])
        build_children.get_children()

    # 6). Build the connections and links matrices
    connections = Build_connections(token_sets, sems)
    connections.build_connections_links()


if __name__ == "__main__":
    symProps = [{'name': 'lovesMaryTom', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Mary', 'object_sem': ['mary1', 'mary2', 'mary3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Tom', 'object_sem': ['tom1', 'tom2', 'tome3'], 'P': 'non_exist'}], 'set': 'driver', 'analog': 0}, 
    {'name': 'lovesTomJane', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Tom', 'object_sem': ['tom1', 'tom2', 'tome3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Jane', 'object_sem': ['jane1', 'jane2', 'mary2'], 'P': 'non_exist'}], 'set': 'driver', 'analog': 0},
    {'name': 'jealousMaryJane', 'RBs': [{'pred_name': 'jealous_act', 'pred_sem': ['jel1', 'jel2', 'jel3'], 'higher_order': False, 'object_name': 'Mary', 'object_sem': ['mary1', 'mary2', 'mary3'], 'P': 'non_exist'}, {'pred_name': 'jealous_pat', 'pred_sem': ['jel4', 'jel5', 'jel6'], 'higher_order': False, 'object_name': 'Jane', 'object_sem': ['jane1', 'jane2', 'mary2'], 'P': 'non_exist'}], 'set': 'driver', 'analog': 0},
    {'name': 'lovesJohnKathy', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'John', 'object_sem': ['john1', 'john2', 'john3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Kathy', 'object_sem': ['kathy1', 'kathy2', 'kathy3'], 'P': 'non_exist'}], 'set': 'recipient', 'analog': 0}, 
    {'name': 'lovesKathySam', 'RBs': [{'pred_name': 'lover', 'pred_sem': ['lover1', 'lover2', 'lover3'], 'higher_order': False, 'object_name': 'Kathy', 'object_sem': ['kathy1', 'kathy2', 'kathy3'], 'P': 'non_exist'}, {'pred_name': 'beloved', 'pred_sem': ['beloved1', 'beloved2', 'beloved3'], 'higher_order': False, 'object_name': 'Sam', 'object_sem': ['sam1', 'sam2', 'sam3'], 'P': 'non_exist'}], 'set': 'recipient', 'analog': 0}]

    main(symProps)