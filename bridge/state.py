from nodes.enums import *
import json
from logging import getLogger
logger = getLogger("STATE")

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
        self.links_list = {}
        """ Dictionary of token IDs to list of semantic IDs. (links_list[TOKEN_ID] = [SEM_ID1, SEM_ID2, ...])"""
        self.mappings = [[{}]]
        f""" Matrix of mappings between driver and recipient tokens. (mappings[DRIVER_ID][RECIPIENT_ID][MappingField] = value)"""
        self.connections = [[]]
        f""" Matrix of connections between tokens. (connections[TOKEN_ID][TOKEN_ID] = weight)"""
        self.sem_connections = [[]]
        f""" Matrix of connections between semantics. (sem_connections[SEM_ID][SEM_ID] = weight)"""
        self.metadata = {
            'sim_path': None,
            'parameters': None,
            'token_counts': {
                Type.P: None,
                Type.RB: None,
                Type.PO: None,
                Type.SEMANTIC: None,
            },
            'con_counts': {
                Type.P: {
                    Type.RB: {'child': 0,'parent': 0,},
                    Type.GROUP: 0,
                },
                Type.RB: {
                    Type.P: {'parent': 0,'child': 0,},
                    Type.PO: {'pred': 0,'obj': 0,},
                },
                Type.PO: {
                    Type.RB: 0,
                },
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
        self.ids = {}
        """ Dict of token IDs by index. (ids[index] = [ID1, ID2, ...])"""
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
        self.sem_ids = {}
        """ Dict of semantic IDs by index. (sem_ids[index1] = ID1)"""
        self.firing_order = []
        """ List of firing order. (firing_order = [token_id1, token_id2, ...])"""
    
    def copy_from(self, state: 'State'):
        """ Copy the state object info into current state.
        Args:
            state: State object.
        """
        self.tokens = state.tokens.copy()
        self.semantics = state.semantics.copy()
        self.links = state.links.copy()
        self.links_list = state.links_list.copy()
        self.mappings = state.mappings.copy()
        self.connections = state.connections.copy()
        self.sem_connections = state.sem_connections.copy()
        self.metadata = state.metadata.copy()
        self.token_ids = state.token_ids.copy()
        self.idxs = state.idxs.copy()
        self.ids = state.ids.copy()
        self.tk_count = state.tk_count
        self.sem_count = state.sem_count
        self.recipient_idxs = state.recipient_idxs.copy()
        self.driver_idxs = state.driver_idxs.copy()
        self.recipient = state.recipient.copy()
        self.driver = state.driver.copy()
        self.sem_idxs = state.sem_idxs.copy()
        self.sem_ids = state.sem_ids.copy()
        self.firing_order = state.firing_order.copy()

    def clear(self):
        """ Clear the state. """
        self.tokens = {}
        self.semantics = {}
        self.links = [[]]
        self.mappings = [[{}]]
        self.connections = [[]]
        self.metadata = {
            'sim_path': None,
            'parameters': None,
            'token_counts': {
                Type.P: None,
                Type.RB: None,
                Type.PO: None,
                Type.SEMANTIC: None,
            },
            'con_counts': {
                Type.P: {
                    Type.RB: {'child': 0,'parent': 0,},
                    Type.GROUP: 0,
                },
                Type.RB: {
                    Type.P: {'parent': 0,'child': 0,},
                    Type.PO: {'pred': 0,'obj': 0,},
                },
                Type.PO: {
                    Type.RB: 0,
                },
            }
        }
        self.token_ids = {
            Type.P: [],
            Type.RB: [],
            Type.PO: [],
        }
        self.idxs = {}
        self.ids = {}
        self.sem_ids = {}
        self.tk_count = 0
        self.sem_count = 0
        self.recipient_idxs = {}
        self.driver_idxs = {}
        self.recipient = []
        self.driver = []
        self.sem_idxs = {}
        self.firing_order = []

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
            'firing_order': self.firing_order,
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
        state.firing_order = data.get('firing_order', [])
        return state
    
    def debug_print(self):
        """ Print the state in a debug format. """
        output = " ---------------STATE---------------- \n"
        output += f"Tokens: ids={list(self.tokens.keys())}\n"
        output += f"Semantics: ids={list(self.semantics.keys())}\n"
        output += f"Links: size={len(self.links)}x{len(self.links[0])}\n"
        output += f"Mappings: size={len(self.mappings)}x{len(self.mappings[0])}x{len(self.mappings[0][0])}\n"
        output += f"Connections: size={len(self.connections)}x{len(self.connections[0])}\n"
        output += f"Sem Connections: size={len(self.sem_connections)}x{len(self.sem_connections[0])}\n"
        output += f"ID_dict_counts: tokens={len(self.ids.keys())} semantics={len(self.sem_ids.keys())}\n"
        output += f"index_dict_counts: tokens={len(self.idxs.keys())} semantics={len(self.sem_idxs.keys())}\n"
        logger.debug(output)
    
    def debug_con_counts(self):
        """ Print the connection counts. """
        output = "CON_COUNTS: "
        output += f"(P: RB_child={self.metadata['con_counts'][Type.P][Type.RB]['child']},"
        output += f" RB_parent={self.metadata['con_counts'][Type.P][Type.RB]['parent']} "
        output += f" group={self.metadata['con_counts'][Type.P][Type.GROUP]})"
        output += f" | (RB: P_parent={self.metadata['con_counts'][Type.RB][Type.P]['parent']},"
        output += f" P_child={self.metadata['con_counts'][Type.RB][Type.P]['child']},"
        output += f" pred={self.metadata['con_counts'][Type.RB][Type.PO]['pred']},"
        output += f" obj={self.metadata['con_counts'][Type.RB][Type.PO]['obj']})"
        output += f" | (PO: RB={self.metadata['con_counts'][Type.PO][Type.RB]})"
        logger.debug(output)
