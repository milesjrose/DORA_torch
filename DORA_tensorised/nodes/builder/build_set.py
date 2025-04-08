# nodes/builder/build_set.py
# Builds the nodes for a given set.

from .intermediate_types import *

class Build_set(object):                                # Builds the nodes for a given set
    """
    A class for building the nodes for a given set.

    Attributes:
        symProps (list): A list of symProps relating to the set.
        tokens (dict): A dictionary of tokens, mapping type to list of tokens.
        set (Set): The set to be built.
        name_dict (dict): A dictionary of tokens, mapping name to token in tokens.
        id_dict (dict): A dictionary of tokens, mapping ID to token in tokens.
    """
    def __init__(self, symProps: list[dict], set: Set):
        """
        Initialise the build set with symProps, set, and empty dictionaries for tokens, name_dict, and id_dict.

        Args:
            symProps (list): A list of symProps relating to the set.
            set (Set): The set to be built.
        """
        self.symProps = symProps
        self.tokens = {}
        self.set = set
        self.name_dict = {}
        self.id_dict = {}
        self.names = []
        self.token_set = None
    
    def build_set(self):    #  Returns a new token_set
        """
        Returns a new token_set object
        """
        self.get_nodes()       # Get the nodes from the symProps
        self.id_tokens()       # Create basic tokens
        self.token_set = Token_set(self.set, self.tokens, self.name_dict, self.id_dict)   # Return new token_set
        return self.token_set


    def get_nodes(self):         # Gets lists of unique tokens by type
        """
        Gets lists of unique tokens by type.
        """
        for type in Type:                    # Initialise empty lists for each type
            self.tokens[type] = []
        
        non_exist = "non_exist"
        for prop in self.symProps:
            prop_name = prop['name']
            analog = prop['analog']
            if prop_name != non_exist:
                self.create_token(prop_name, Inter_Prop, analog)
            for rb in prop['RBs']:
                pred_name = rb['pred_name']
                obj_name = rb['object_name']
                self.create_token(pred_name + "_" + obj_name, Inter_RB, analog)
                if pred_name != non_exist:
                    self.create_token(pred_name, Inter_PO, analog, True)
                if obj_name != non_exist:
                    self.create_token(obj_name, Inter_PO, analog, False)


    def id_tokens(self):   # Give each token an ID
        """
        Assign each token an ID, unique for the set.
        """
        self.names = list(set(self.names))
        i = 0
        for name in self.names:
            node_obj =  self.name_dict[name]
            node_obj.set_ID(i)
            self.id_dict[i] = name
            i += 1

    def create_token(self, name, token_class, analog, is_pred = None): # Create a token and add it to the name/dict
        """
        Create a token object and add it to the name/dict.

        Args:
            name (str): The name of the token.
            token_class (Token): The class of the token.
            analog (int): The analog of the token.
            is_pred (bool): Whether the token is a predicate.
        """
        if name not in self.names:
            if is_pred is not None:
                obj = token_class(name, self.set, analog, is_pred)
            else:
                obj = token_class(name, self.set, analog)
            self.tokens[obj.features[TF.TYPE]].append(obj)
            self.name_dict[name] = obj
            self.names.append(name)
