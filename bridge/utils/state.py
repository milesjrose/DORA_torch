from nodes.enums import *
import json

class State:
    """
    A class for holding the state of the network, in a format that can be used to easily compare between
    old and new implementations.
    """
    def __init__(self):
        self.tokens = {}
        f""" Dictionary of tokens by ID. (tokens[ID] = token_data)"""
        self.semantics = {}
        f""" Dictionary of semantics by ID. (semantics[ID] = semantic_data)"""
        self.links = [[]]
        f""" Matrix of links between POs and semantics. (links[PO_ID][SEM_ID] = weight)"""
        self.mappings = [[{}]]
        f""" Matrix of mappings between driver and recipient tokens. (mappings[DRIVER_ID][RECIPIENT_ID][MappingField] = value)"""
        self.connections = [[]]
        f""" Matrix of connections between tokens. (connections[TOKEN_ID][TOKEN_ID] = weight)"""
        self.metadata = {
            'sim_path': None,
            'parameters': None,
            'token_counts': {
                'P': None,
                'RB': None,
                'PO': None,
                'semantics': None,
            }
        }
        self.token_ids = {
            Type.P: [],
            Type.RB: [],
            Type.PO: [],
        }
        """ Dict of token IDs by type. (token_ids[Type] = [ID1, ID2, ...])"""
        self.idxs = {}
        """ Dict of token indexes by ID. (idxs[ID] = [index1, index2, ...])"""
        self.tk_count = 0
        self.sem_count = 0
        self.recipient_idxs = {}
        """ Dict of recipient token indexes by ID. (recipient_idxs[ID] = index)"""
        self.driver_idxs = {}
        """ Dict of driver token indexes by ID. (driver_idxs[ID] = index)"""
        self.recipient = []
        """ List of recipient token ids """
        self.driver = []
        """ List of driver token ids """
        self.sem_idxs = {}
        """ Dict of semantic token indexes by ID. (sem_idxs[ID] = index)"""
    
    def to_json(self, filepath: str = None) -> dict:
        """
        Convert the state to a JSON-serializable dictionary.
        
        Args:
            filepath: Optional path to save the JSON file. If None, only returns the dict.
            
        Returns:
            A JSON-serializable dictionary representation of the state.
        """
        # Convert token_ids keys from Type enum to string names
        token_ids_serializable = {
            type_key.name: ids for type_key, ids in self.token_ids.items()
        }
        
        data = {
            'tokens': self.tokens,
            'semantics': self.semantics,
            'links': self.links,
            'mappings': self.mappings,
            'connections': self.connections,
            'metadata': self.metadata,
            'token_ids': token_ids_serializable,
            'idxs': self.idxs,
            'tk_count': self.tk_count,
            'sem_count': self.sem_count,
            'recipient_idxs': self.recipient_idxs,
            'driver_idxs': self.driver_idxs,
            'recipient': self.recipient,
            'driver': self.driver,
            'sem_idxs': self.sem_idxs,
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        return data
    
    @classmethod
    def from_json(cls, source) -> 'State':
        """
        Create a State instance from JSON data.
        
        Args:
            source: Either a filepath (str) to a JSON file, or a dictionary of state data.
            
        Returns:
            A new State instance populated with the loaded data.
        """
        if isinstance(source, str):
            with open(source, 'r') as f:
                data = json.load(f)
        else:
            data = source
        
        state = cls()
        
        state.tokens = data.get('tokens', {})
        state.semantics = data.get('semantics', {})
        state.links = data.get('links', [[]])
        state.mappings = data.get('mappings', [[{}]])
        state.connections = data.get('connections', [[]])
        state.metadata = data.get('metadata', state.metadata)
        
        # Convert token_ids keys from string names back to Type enum
        token_ids_raw = data.get('token_ids', {})
        state.token_ids = {
            Type[type_name]: ids for type_name, ids in token_ids_raw.items()
        }
        
        state.idxs = data.get('idxs', {})
        state.tk_count = data.get('tk_count', 0)
        state.sem_count = data.get('sem_count', 0)
        state.recipient_idxs = data.get('recipient_idxs', {})
        state.driver_idxs = data.get('driver_idxs', {})
        state.recipient = data.get('recipient', [])
        state.driver = data.get('driver', [])
        state.sem_idxs = data.get('sem_idxs', {})
        
        return state
    
    
    
