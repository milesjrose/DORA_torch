
import os
from .state import State
from nodes.utils import OutputType, TablePrinter
from typing import List, Tuple
from logging import getLogger
from nodes.enums import *
from typing import Dict

class StatePrinter:
    """
    A class for printing state dictionary data in table format.
    
    Follows the style of nodes/utils/printer/printer.py but operates on
    the state dictionary format used by DORA_bridge.
    
    Attributes:
        log_file (str): Optional file path to log output to.
        print_to_console (bool): Whether to print output to console.
    """
    
    def __init__(self, log_file: str = None, state: State = None, output_type: OutputType = OutputType.SINGLE_LOG_CONSOLE, logger = None):
        """
        Initialize the StatePrinter.
        
        Args:
            log_file (str): Optional file to log output to.
            print_to_console (bool): Whether to print to console. Default True.
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self.log_file = log_file
        self._state = state
        self.output_type = OutputType.SINGLE_LOG_CONSOLE
        self.header_text = None
        self.logger = logger if logger is not None else getLogger("STATE_PRINTER")
        self._temp_logger = None
    
    def set_state(self, state: State):
        """ Set the state of the StatePrinter. 
        Args:
            state: State object.
        """
        self._state = state
    
    def get_state(self) -> State:
        """ Get the state of the StatePrinter. """
        return self._state
    
    def _output(self, message: str):
        """
        Output a message to console and/or log file.
        
        Args:
            message (str): The message to output.
        """
        logger = self._temp_logger if self._temp_logger is not None else self.logger
        if self.output_type == OutputType.PRINT_CONSOLE:
            print(message)
        elif self.output_type == OutputType.LOG_CONSOLE:
            logger.info(message)
        elif self.output_type == OutputType.SINGLE_LOG_CONSOLE:
            logger.info(message)
        elif self.output_type == OutputType.BUILD_STRING:
            logger.info(message)
        elif self.output_type == OutputType.BUILD_STR_LIST:
            logger.info(message)
        elif self.output_type == OutputType.LOG_FILE:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + "\n")
        else:
            raise ValueError(f"Invalid output type: {self.output_type}")
    
    def _print_table(self, columns: List[str], rows: List[List[str]], header_text: str):
        """
        Print a table using tablePrinter.
        
        Args:
            columns (list[str]): Column headers.
            rows (list[list[str]]): Row data.
            header_text (str): Header text for the table.
        """
        if not rows:
            self._output(f"{header_text}: (empty)")
            return
        
        table = TablePrinter(
            columns=columns,
            rows=rows,
            headers=[header_text],
            log_file=self.log_file,
            print_to_console=self.print_to_console,
            output_type=self.output_type
        )
        table.print_table(header=True, column_names=True, split=False)
    
    def tokens(self, token_types: List[str] = None, 
                     features: List[str] = None, 
                     filter_set: str = None,
                     names: list[str] = None,
                     IDs: list[int] = None,
                     logger = None):
        """ Print the tokens in the state.

        Args:
            token_types (list[str], optional): List of token types to print. Default [P, RB, PO].
            features (list[str], optional): List of features to print. Default None.
            filter_set (str, optional): Filter the tokens by set. Default None.
            names (list[str], optional): List of names to filter the tokens by. Default None.
            IDs (list[int], optional): List of IDs to filter the tokens by. Default None.
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self._temp_logger = logger
        tokens = self._state.tokens
        ID_list = list(tokens.keys())
        if token_types is None:
            token_types = [Type.P, Type.RB, Type.PO]
        output = []
        for id in sorted(ID_list):
            token = tokens[id]
            if token['type'] not in token_types:
                self._output(f"{id} type[{token['type']}] not in token_types[{token_types}]")
                continue
            if filter_set is not None and token['set'] != filter_set:
                self._output(f"{id} set[{token['set']}] not in filter_set[{filter_set}]")
                continue
            if names is not None and token['name'] not in names:
                self._output(f"{id} name[{token['name']}] not in names[{names}]")
                continue
            if IDs is not None and id not in IDs:
                self._output(f"{id} not in IDs[{IDs}]")
                continue
            output.append(token)
        self._output(f"Tokens: {len(output)} tokens")
        self._print_tokens(output)
        self._temp_logger = None
    
    def semantics(self, show_zero_act: bool = True, names: list[str] = None, logger = None):
        """ Print the semantics in the state.

        Args:
            show_zero_act (bool, optional): Whether to show semantics with zero activation. Default True.
            names (list[str], optional): List of names to filter the semantics by. Default None.
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self._temp_logger = logger
        semantics = self._state.semantics
        ID_list = list(semantics.keys())
        output = []
        for id in ID_list:
            sem = semantics[id]
            if show_zero_act is False and sem['act'] == 0.0:
                continue
            if names is not None and sem['name'] not in names:
                continue
            output.append(sem)
        self._print_semantics(output)
        self._temp_logger = None

    def links(self, show_weights: bool = True, 
              min_weight: float = 0.0,
              names: list[str] = None,
              ids: list[int] = None,
              as_matrix: bool = False,
              logger = None):
        """ Print the links in the state.

        Args:
            show_weights (bool, optional): Whether to show weights. Default True.
            min_weight (float, optional): Minimum weight to display. Default 0.0.
            names (list[str], optional): List of names to filter the links by. Default None.
            ids (list[int], optional): List of IDs to filter the links by. Default None.
            as_matrix (bool, optional): Whether to print the links as a matrix. Default False.
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self._temp_logger = logger
        if as_matrix:
            links = self._state.links
            i = 0
            for row in links:
                row.insert(0, i)
                i += 1
            columns = ['t/s'].extend([f's{i}' for i in range(len(links[0]))])
            header_text = f"Links Matrix({len(links)} tokens x {len(links[0])} semantics)" if self.header_text is None else f"{self.header_text} ({len(links)} tokens x {len(links[0])} semantics)"
            self._print_table(columns, links, 'Links')
        else:
            links_list = self._state.links_list
            link_data = []
            if ids is None:
                ids = list(self._state.idxs.keys())
            tk_idxs = self._state.idxs
            sem_idxs = self._state.sem_idxs
            rows = []
            for tk_id in ids:
                row = []
                links = links_list[tk_id]
                for sem_id in links:
                    weight = link_data[tk_idxs[tk_id]][sem_idxs[sem_id]]
                    if weight > 0.0:
                        row.append(f's{sem_id}[{sem_idxs[sem_id]}] ({weight})')
                if row != []:
                    sem_text = ""
                    for sem in row:
                        sem_text += sem + ", "
                    rows.append[f"{tk_id}[{tk_idxs[tk_id]}]", '→', sem_text]
            if len(rows) == 0:
                self._output(f"Links: (none above min_weight={min_weight})")
                return
            header_text = f"({len(rows)} links)" if self.header_text is None else f"{self.header_text} ({len(rows)} links)"
            columns = ['Token[idx]', '', 'Semantics[idx] (weight)']
            self._print_table(columns, rows, header_text)
        self._temp_logger = None
        
    def mappings(self, show_weights: bool = True, 
                 min_weight: float = 0.0,
                 names: list[str] = None,
                 ids: list[int] = None,
                 as_matrix: bool = False,
                 logger = None):
        """ Print the mappings in the state.

        Args:
            show_weights (bool, optional): Whether to show weights. Default True.
            min_weight (float, optional): Minimum weight to display. Default 0.0.
            names (list[str], optional): List of names to filter the mappings by. Default None.
            ids (list[int], optional): List of IDs to filter the mappings by. Default None.
            as_matrix (bool, optional): Whether to print the mappings as a matrix. Default False.
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self._temp_logger = logger
        raise NotImplementedError("Not done yet.")
        self._temp_logger = None

    def connections(self, show_weights: bool = True, 
                 min_weight: float = 0.0,
                 names: list[str] = None,
                 ids: list[int] = None,
                 as_matrix: bool = False,
                 get_children: bool = True,
                 get_parents: bool = False,
                 logger = None):
        """ Print the connections in the state.

        Args:
            show_weights (bool, optional): Whether to show weights. Default True.
            min_weight (float, optional): Minimum weight to display. Default 0.0.
            names (list[str], optional): List of names to filter the connections by. Default None.
            ids (list[int], optional): List of IDs to filter the connections by. Default None.
            as_matrix (bool, optional): Whether to print the connections as a matrix. Default False.
            get_children (bool, optional): Whether to get the children of the connections. Default True.
            get_parents (bool, optional): Whether to get the parents of the connections. Default False.
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self._temp_logger = logger
        if as_matrix:
            raise NotImplementedError("Not done yet.")
        else:
            # get connections list by going through each token and getting its connections
            tokens = self._state.tokens
            rows = []
            all_ids = list(self._state.idxs.keys())
            for id in all_ids:
                token = tokens[id]
                children = []
                parents = []
                if get_children:
                    children = self._get_children(token)
                if get_parents:
                    parents = self._get_parents(token)
                if len(children) == 0 and len(parents) == 0:
                    continue
                if children and parents:
                    rows.append([f"{id}[{self._state.idxs[id]}]", '→', children, '', parents])
                elif children:
                    rows.append([f"{id}[{self._state.idxs[id]}]", '→', children])
                elif parents:
                    rows.append([f"{id}[{self._state.idxs[id]}]", '→', parents])
            if len(rows) == 0:
                self._output(f"Connections: (None found for given ids)")
                return
            else:
                header_text = f"({len(rows)} connections)" if self.header_text is None else f"{self.header_text} ({len(rows)} connections)"
                columns = ['Token[idx]', '', 'Children[idx]',] if get_children else ['Token[idx]', '', 'Parents[idx]']
                if get_children and get_parents:
                    columns = ['Token[idx]', '', 'Children[idx]', '', 'Parents[idx]']
                self._print_table(columns, rows, header_text)
        self._temp_logger = None
        
    def summary(self, logger = None):
        """ 
        Print a short summary of the state. 
        (Counts of tokens, semantics, connections, mappings, links, and sets.)

        Args:
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self._temp_logger = logger
        s = self._state
        
        num_cons = 0
        for row in s.connections:
            for conn in row:
                num_cons += int(conn)
        num_mappings = 0
        for row in s.mappings:
            for mapping in row:
                num_mappings += int(mapping[MappingFields.WEIGHT] > 0.0)
        num_links = 0
        for row in s.links:
            for link in row:
                if link > 0.0:
                    num_links += 1
        os = "Summary of the state: \n"
        os += f"Nodes: \n    > {len(s.tokens)} tokens,\n" 
        os += f"    --> {len(s.token_ids[Type.P])} P,\n    --> {len(s.token_ids[Type.RB])} RB,\n    --> {len(s.token_ids[Type.PO])} PO\n"
        os += f"    > {len(s.semantics)} semantics\n"
        os += f"Set_counts: \n    > {len(s.driver)} driver, \n    > {len(s.recipient)} recipient\n"
        os += f"Cons: \n    > {num_cons} Connections, \n    > {num_mappings} Mappings, \n    > {num_links} Links\n"
        self._output(os)
        self._temp_logger = None

    def token_data(self, id: int, logger = None):
        """
        Print the data for a token.
        Args:
            id: The ID of the token.
            logger (Logger, optional): Logger to use. Defaults to local logger.
        """
        self._temp_logger = logger
        token = self._state.tokens[id]
        idx = self._state.idxs[id]
        data = [['idx', f"{idx}"]]
        for feature in list(token.keys()):
            value = token[feature]
            if isinstance(value, Type):
                value = value.name
            elif isinstance(value, Set):
                value = value.name
            elif isinstance(value, Mode):
                value = value.name
            elif isinstance(value, OntStatus):
                value = value.name
            elif isinstance(value, SDM):
                value = value.name
            data.append([feature, value])
        cols = ['Feat', "Val"]
        rows = data
        self._print_table(cols, rows, f"Token {id}")
        self._temp_logger = None
        
    def _get_children(self, token: Dict, ids: List[int] = None) -> List[int]:
        """ Get the children of a token. """
        c = []
        tk_type = token['type']
        match tk_type:
            case Type.P:
                c.extend(token['myRBs'])
            case Type.RB:
                c.extend(token['myChildP'], token['myPred'], token['myObj'])
            case _ :
                return []
        if ids is not None:
            for child in c:
                if child not in ids:
                    c.remove(child)
        return c

    def _get_parents(self, token: Dict, ids: List[int] = None) -> List[int]:
        """ Get the parents of a token. """
        p = []
        tk_type = token['type']
        match tk_type:
            case Type.P:
                p.extend(token['myParentRBs'])
                p.extend(token['myGroups'])
            case Type.RB:
                p.extend(token['myParentPs'])
            case Type.PO:
                p.extend(token['myRBs'])
            case _ :
                return []
        if ids is not None:
            for parent in p:
                if parent not in ids:
                    p.remove(parent)
        return p

    # ================================[ Helper Functions ]===============================
    def _print_tokens(self, tokens: List[Dict], features: List[Tuple[str, str]] = None):
        """ Print the tokens in the state.
        Args:
            tokens: List of tokens to print.
        """
        header_text = 'Tokens'
        if features is None:
            features = [('idx', 'ID'), 
                        ('Name', 'name'), 
                        ('ID', 'ID'), 
                        ('Set', 'set'), 
                        ('Type', 'type'), 
                        ('Analog', 'myanalog'), 
                        ('Act', 'act'), 
                        ('Mode', 'mode'), 
                        ('Net Input', 'net_input'), 
                        ('TD Input', 'td_input'), 
                        ('BU Input', 'bu_input'), 
                        ('Lateral Input', 'lateral_input'), 
                        ('Map Input', 'map_input')
                        ]
        columns = [feature[0] for feature in features]
        rows = []
        for token in tokens:
            row = []
            id = token['ID']
            index = self._state.idxs[id]
            for feature in features:
                value = token.get(feature[1], None)
                row.append(self._format(value))
            row[0] = f"{index}"
            rows.append(row)
        if self.header_text is not None:
            header_text = f"{self.header_text} ({len(tokens)} tokens)"
        self._print_table(columns, rows, header_text)
    
    def _print_semantics(self, semantics: List[Dict]):
        """ Print the semantics in the state.
        Args:
            semantics: List of semantics to print.
        """
        header_text = 'Semantics'
        if features is None:
            features = [('Name', 'name'), 
                        ('Index', 'index'), 
                        ('Act', 'act'), 
                        ('Input', 'input'), 
                        ('Max Input', 'max_input'), 
                        ('Dimension', 'dimension'), 
                        ('Amount', 'amount'), 
                        ('Ont Status', 'ont_status'), 
                        ]
        columns = [feature[0] for feature in features]
        rows = []
        for sem in semantics:
            row = []
            for feature in features:
                value = sem.get(feature[1], None)
                row.append(self._format(value))
            rows.append(row)
        if self.header_text is not None:
            header_text = f"{self.header_text} ({len(semantics)} semantics)"
        self._print_table(columns, rows, header_text)
        
    def _format(self, value) -> str:
        """ Format a value.
        Args:
            value: The value to format.
        """
        if value is None:
            return '-'
        if isinstance(value, Type):
            return value.name
        if isinstance(value, Set):
            return value.name
        if isinstance(value, Mode):
            return value.name
        if isinstance(value, OntStatus):
            return value.name
        if isinstance(value, SDM):
            return value.name
        if isinstance(value, float):
            if value == 0.0:
                return "0.0"
            elif value == int(value):
                return str(int(value)) + ".0"
            else:
                return f"{value:.4f}".rstrip('0').rstrip('.')
        return str(value)

    def _print_table(self, columns: List[str], rows: List[List[str]], header_text: str):
        """
        Print a table using tablePrinter.
        
        Args:
            columns (list[str]): Column headers.
            rows (list[list[str]]): Row data.
            header_text (str): Header text for the table.
        """
        if not rows:
            self._output(f"{header_text}: (empty)")
            return
        
        table = TablePrinter(
            columns=columns,
            rows=rows,
            headers=[header_text],
            log_file=self.log_file,
            print_to_console=True,
            output_type=self.output_type
        )
        table.print_table(header=True, column_names=True, split=False)
        