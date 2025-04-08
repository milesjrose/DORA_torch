from .intermediate_types import *

class Build_children(object):                           # Builds the children of each token
    """
    A class for building the list of children for each token.

    Attributes:
        set (Set): The set of the tokens.
        tokens (Token_set): The token set.
        sems (Sem_set): The semantic set.
        symProps (list): A list of symProps relating to the set.
    """
    def __init__(self, set: Set, tokens: Token_set, sems: Sem_set, symProps: list[dict]):
        """
        Initialise the build children with set, tokens, sems, and symProps.

        Args:
            set (Set): The set of the tokens.
            tokens (Token_set): The token set object.
            sems (Sem_set): The semantic set object.
            symProps (list): A list of symProps relating to the set.
        """
        self.set = set
        self.token_set = tokens
        self.sems = sems
        self.symProps = symProps

    def get_children(self):                                 # Add child nodes to each token object
        """
        Recursively add child nodes IDs to each token objects children list.
        """
        for prop in self.symProps:
            self.get_prop_children(prop)
        for type in Type:                                   # Remove any duplicates
            for token in self.token_set.tokens[type]:
                token.children = list(set(token.children))
    
    def get_prop_children(self, prop: dict):                # Add children to the prop object, recursively call lower get child functions
        """
        Step one in recursively adding child nodes IDs to each token objects children list.
        """
        prop_obj = self.get_object(prop['name'])            # Get the prop object
        for rb in prop['RBs']:
            child = self.get_rb_children(rb)                # Find rbs children, and return rb ID
            if prop_obj is not None:
                prop_obj.children.append(child)             # Append child ID to prop object

    def get_rb_children(self, rb: dict):                    # Add children to the rb object, recursively call lower get child functions
        """
        Step two in recursively adding child nodes IDs to each token objects children list.
        """
        pred_name = rb['pred_name']
        obj_name = rb['object_name']
        rb_obj = self.get_object(pred_name + "_" + obj_name)
        child_pred = self.get_po_children(pred_name, rb['pred_sem'])      # Get children of pred, return pred ID
        child_obj = self.get_po_children(obj_name, rb['object_sem'])   # Get children of obj, return obj ID
        rb_obj.children.append(child_pred)
        rb_obj.children.append(child_obj)
        return rb_obj.ID

    def get_po_children(self, name, sems: list):       # Add children to the po object, return po ID
        """
        Step three in recursively adding child nodes IDs to each token objects children list.
        """
        po_obj = self.get_object(name)
        for sem in sems:
            sem_obj = self.sems.get_sem(sem)
            po_obj.children.append(sem_obj.ID)
        return po_obj.ID
    
    def get_object(self, name):                             # Returns token object, if it exists.  O.w returns None
        """
        Return token object if it exists. Else return None

        Returns:
            token (Token): The token object.
            None: If the token does not exist.
        """
        non_exist = "non_exist"
        non_exist_rb = "non_exist_non_exist"
        obj = None
        if name != non_exist and name != non_exist_rb:
            obj = self.token_set.get_token(name)
        return obj  
