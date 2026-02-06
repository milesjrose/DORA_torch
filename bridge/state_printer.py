
import os
from .state import State
from nodes.utils import OutputType, TablePrinter
from typing import List, Tuple
from logging import getLogger
from nodes.enums import *
from typing import Dict
logger = getLogger("STATE_PRINTER")

class StatePrinter:
    """
    A class for printing state dictionary data in table format.
    
    Follows the style of nodes/utils/printer/printer.py but operates on
    the state dictionary format used by DORA_bridge.
    
    Attributes:
        log_file (str): Optional file path to log output to.
        print_to_console (bool): Whether to print output to console.
    """
    
    def __init__(self, log_file: str = None, state: State = None, output_type: OutputType = OutputType.SINGLE_LOG_CONSOLE):
        """
        Initialize the StatePrinter.
        
        Args:
            log_file (str): Optional file to log output to.
            print_to_console (bool): Whether to print to console. Default True.
        """
        self.log_file = log_file
        self._state = state
        self.output_type = OutputType.SINGLE_LOG_CONSOLE
        self.header_text = None
    
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
                     IDs: list[int] = None):
        """ Print the tokens in the state.
        Args:
            token_types: List of token types to print.
            features: List of features to print.
            filter_set: Filter the tokens by set.
            names: List of names to filter the tokens by.
            IDs: List of IDs to filter the tokens by.
        """
        tokens = self._state.tokens
        ID_list = list(tokens.keys())
        if token_types is None:
            token_types = [Type.P, Type.RB, Type.PO]
        output = []
        for id in sorted(ID_list):
            token = tokens[id]
            if token['type'] not in token_types:
                logger.info(f"{id} type[{token['type']}] not in token_types[{token_types}]")
                continue
            if filter_set is not None and token['set'] != filter_set:
                logger.info(f"{id} set[{token['set']}] not in filter_set[{filter_set}]")
                continue
            if names is not None and token['name'] not in names:
                logger.info(f"{id} name[{token['name']}] not in names[{names}]")
                continue
            if IDs is not None and id not in IDs:
                logger.info(f"{id} not in IDs[{IDs}]")
                continue
            output.append(token)
        logger.info(f"Tokens: {len(output)} tokens")
        self._print_tokens(output)
    
    def semantics(self, show_zero_act: bool = True, names: list[str] = None):
        """ Print the semantics in the state.

        Args:
            show_zero_act: Whether to show semantics with zero activation.
            names: List of names to filter the semantics by.
        """
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
    
    def links(self, show_weights: bool = True, 
              min_weight: float = 0.0,
              names: list[str] = None,
              ids: list[int] = None,
              as_matrix: bool = False):
        """ Print the links in the state.
        Args:
            show_weights: Whether to show weights.
            min_weight: Minimum weight to display.
        """
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
    
    def mappings(self, show_weights: bool = True, 
                 min_weight: float = 0.0,
                 names: list[str] = None,
                 ids: list[int] = None,
                 as_matrix: bool = False):
        """ Print the mappings in the state.
        Args:
            show_weights: Whether to show weights.
            min_weight: Minimum weight to display.
        """
        raise NotImplementedError("Not done yet.")

    def connections(self, show_weights: bool = True, 
                 min_weight: float = 0.0,
                 names: list[str] = None,
                 ids: list[int] = None,
                 as_matrix: bool = False,
                 get_children: bool = True,
                 get_parents: bool = False,
                 ):
        """ Print the connections in the state.
        Args:
            connection_types: List of connection types to print.
            min_weight: Minimum weight to display.
        """
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

    def summary(self):
        """ Print the summary of the state. """
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
        logger.info(os)

    def token_data(self, id: int):
        """
        Print the data for a token.
        Args:
            id: The ID of the token.
        """
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
        