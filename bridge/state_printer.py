
import os
from .state import State
from nodes.utils import OutputType, TablePrinter
from typing import List, Tuple
from logging import getLogger, INFO, DEBUG
debug_logger = getLogger("SP_DEBUG")
debug_logger.setLevel(INFO)
from nodes.enums import *
from typing import Dict

empty_char = "-"

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
        self.header_text = ""
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
        logger = self._temp_logger if self._temp_logger is not None else self.logger
        if not rows:
            self._output(f"{header_text}: (empty)")
            return
        
        table = TablePrinter(
            columns=columns,
            rows=rows,
            headers=[header_text],
            log_file=self.log_file,
            print_to_console=self.print_to_console,
            output_type=self.output_type,
            logger=logger
        )
        table.print_table(header=True, column_names=True, split=False)
    
    def tokens(self, token_types: List[str] = None, 
                     features: List[str] = None, 
                     filter_set: str = None,
                     names: list[str] = None,
                     ids: list[int] = None,
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
                debug_logger.debug(f"{id} type[{token['type']}] not in token_types[{token_types}]")
                continue
            if filter_set is not None and token['set'] != filter_set:
                debug_logger.debug(f"{id} set[{token['set']}] not in filter_set[{filter_set}]")
                continue
            if names is not None and token['name'] not in names:
                debug_logger.debug(f"{id} name[{token['name']}] not in names[{names}]")
                continue
            if ids is not None and id not in ids:
                debug_logger.debug(f"{id} not in IDs[{ids}]")
                continue
            output.append(token)
        self._print_tokens(output, ids=ids)
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
        if as_matrix:
            mappings = self._state.mappings
            r_ids = sorted(self._state.recipient)
            d_ids = sorted(self._state.driver)
            if len(r_ids) == 0 or len(d_ids) == 0:
                self._output(f"Mappings: (no driver/recipient tokens)")
                self._temp_logger = None
                return
            # Build column headers from driver IDs
            cols = ["r \\ d"] + [f"{self._get_tag(d_id, names is not None)}" for d_id in d_ids]
            rows = []
            for r_local_idx, r_id in enumerate(r_ids):
                if ids is not None and r_id not in ids:
                    # Check if any driver in ids maps to this recipient
                    has_match = False
                    for d_local_idx, d_id in enumerate(d_ids):
                        if d_id in ids:
                            entry = mappings[r_local_idx][d_local_idx]
                            if entry[MappingFields.WEIGHT] > min_weight or entry[MappingFields.HYPOTHESIS] != 0.0:
                                has_match = True
                                break
                    if not has_match:
                        continue
                r_tag = self._get_tag(r_id, names is not None)
                r_label = f"*{r_tag}*" if ids is not None and r_id in ids else f"{r_tag}"
                row = [r_label]
                for d_local_idx, d_id in enumerate(d_ids):
                    entry = mappings[r_local_idx][d_local_idx]
                    weight = entry[MappingFields.WEIGHT]
                    hyp = entry[MappingFields.HYPOTHESIS]
                    max_hyp = entry[MappingFields.MAX_HYP]
                    if weight > min_weight or hyp != 0.0:
                        parts = []
                        if show_weights:
                            if weight > 0.0:
                                parts.append(f"w={self._format(weight)}")
                            if hyp != 0.0:
                                parts.append(f"h={self._format(hyp)}")
                            if max_hyp != 0.0:
                                parts.append(f"mh={self._format(max_hyp)}")
                        row.append(" ".join(parts) if parts else "●")
                    else:
                        row.append("·")
                rows.append(row)
            if len(rows) == 0:
                self._output(f"Mappings: (none above min_weight={min_weight})")
            else:
                header_text = self._get_header_text("Mappings", len(rows), ids)
                self._print_table(cols, rows, header_text)
        else:
            # List mode: show each non-zero mapping as a row
            mappings = self._state.mappings
            r_ids = sorted(self._state.recipient)
            d_ids = sorted(self._state.driver)
            if len(r_ids) == 0 or len(d_ids) == 0:
                self._output(f"Mappings: (no driver/recipient tokens)")
                self._temp_logger = None
                return
            rows = []
            for r_local_idx, r_id in enumerate(r_ids):
                for d_local_idx, d_id in enumerate(d_ids):
                    entry = mappings[r_local_idx][d_local_idx]
                    weight = entry[MappingFields.WEIGHT]
                    hyp = entry[MappingFields.HYPOTHESIS]
                    max_hyp = entry[MappingFields.MAX_HYP]
                    if weight <= min_weight and hyp == 0.0:
                        continue
                    # Apply id filter
                    if ids is not None and r_id not in ids and d_id not in ids:
                        continue
                    # Apply name filter
                    if names is not None:
                        r_name = self._state.tokens[r_id]['name'] if r_id in self._state.tokens else None
                        d_name = self._state.tokens[d_id]['name'] if d_id in self._state.tokens else None
                        if r_name not in names and d_name not in names:
                            continue
                    d_tag = self._get_tag(d_id, names is not None)
                    r_tag = self._get_tag(r_id, names is not None)
                    d_label = f"*{d_tag}*" if ids is not None and d_id in ids else d_tag
                    r_label = f"*{r_tag}*" if ids is not None and r_id in ids else r_tag
                    if show_weights:
                        row = [d_label, '↔', r_label, self._format(weight), self._format(hyp), self._format(max_hyp)]
                    else:
                        row = [d_label, '↔', r_label]
                    rows.append(row)
            if len(rows) == 0:
                self._output(f"Mappings: (none above min_weight={min_weight})")
            else:
                header_text = self._get_header_text("Mappings", len(rows), ids)
                if show_weights:
                    columns = ['Driver', '', 'Recipient', 'Weight', 'Hypothesis', 'Max Hyp']
                else:
                    columns = ['Driver', '', 'Recipient']
                self._print_table(columns, rows, header_text)
        self._temp_logger = None

    def connections(self, show_weights: bool = True, 
                 min_weight: float = 0.0,
                 names: list[str] = None,
                 ids: list[int] = None,
                 as_matrix: bool = False,
                 get_children: bool = True,
                 get_parents: bool = False,
                 logger = None,
                 use_names: bool = False):
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
            use_names (bool, optional): Whether to use names instead of IDs. Default False.
        """
        self._temp_logger = logger
        get_both = get_children and get_parents
        if as_matrix:
            cons = self._state.connections
            id_dict = self._state.ids
            idxs = self._state.idxs
            no_nodes = len(cons)
            rows = []
            filter_idxs = [idxs[id] for id in ids]
            if ids is not None:
                col_list = []
                row_list = []
                # Go through each row, check if parent or child connection to ids.
                for row_idx, row in enumerate(cons):
                    if row_idx in filter_idxs: # If parent is in ids
                        row_list.append(row_idx)  # Add row
                        for col_idx in range(len(row)):
                            if row[col_idx]:  # If i is child of parent
                                col_list.append(col_idx)  # Add col
                    else: # Parent not in ids
                        for col_idx in filter_idxs:
                            if row[col_idx]:
                                # If connect to node in ids, add row and col.
                                row_list.append(row_idx)
                                col_list.append(col_idx)
                # Remove duplicates and sort.
                col_list = sorted(list(set(col_list)))
                row_list = sorted(list(set(row_list)))
                # build matrix of rows and cols we want.
                for row_idx, row in enumerate(cons):
                    if row_idx in row_list:
                        tag = self._get_tag(id_dict[row_idx], use_names)
                        new_row = [f"*{tag}*"] if id_dict[row_idx] in ids else [f"{tag}"]
                        for col_idx in col_list:
                            formatted_val = "●" if row[col_idx] else "·"
                            new_row.append(formatted_val)
                        rows.append(new_row)
                cols = ["p->c"] 
                for col_idx in col_list:
                    tag = self._get_tag(id_dict[col_idx], use_names)
                    tag_str = f"*{tag}*" if id_dict[row_idx] in ids else f"{tag}"
                    cols.append(tag_str)
            else:
                cols = ["p->c"] + [self._get_tag(id_dict[i], use_names) for i in range(no_nodes)]
                for i, row in enumerate(cons):
                    new_row = [f"{id_dict[i]}"]
                    for val in row:
                        formatted_val = "●" if val else "·"
                        new_row.append(formatted_val)
                    rows.append(new_row)
            if len(rows) != 0:
                header_text = self._get_header_text("Connections", len(rows), ids)
                self._print_table(cols, rows, header_text)
            else:
                self._output(f"Connections: (None found for given ids)")
        else:
            # get connections list by going through each token and getting its connections
            tokens = self._state.tokens
            rows = []
            all_ids = list(self._state.idxs.keys())
            for id in all_ids:
                token = tokens[id]
                children = []
                parents = []
                # Get parents and children, filter by ids if provided.
                if get_children:
                    children = self._filter_con_str(use_names, self._get_children(token), id, get_both, ids)
                if get_parents:
                    parents = self._filter_con_str(use_names, self._get_parents(token), id, get_both, ids)
                # Add row to list
                id_str = self._get_tag(id, use_names)
                if get_children and get_parents and (children != empty_char or parents != empty_char):
                    rows.append([f"{id_str}", '→', children, '', parents])
                else:
                    id_str = f"*{id_str}*" if id in ids else id_str
                    if get_children and children != empty_char:
                        rows.append([id_str, '→', children])
                    elif get_parents and parents != empty_char:
                        rows.append([id_str, '→', parents])
            if len(rows) == 0:
                self._output(f"Connections: (None found for given ids)")
                return
            else:
                header_text = self._get_header_text("Connections", len(rows), ids)
                columns = ['Token', '', 'Children',] if get_children else ['Token', '', 'Parents']
                if get_children and get_parents:
                    columns = ['Token', '', 'Children', '', 'Parents']
                self._print_table(columns, rows, header_text)
        self._temp_logger = None

    def _filter_con_str(self, use_names: bool,con_list: List[int], id: int =None, get_both: bool = False, ids: List[int]=None) -> List[str]:
        """ Filter a list of connections by ids. """
        if ids is not None:
            in_ids = (id in ids)
            if get_both and not in_ids:
                return empty_char
            filtered = []
            for child in con_list:
                if child in ids:
                    filtered.append(f"*{self._get_tag(child, use_names)}*")
                elif in_ids:
                    filtered.append(f"{self._get_tag(child, use_names)}")
        else:
            filtered = [self._get_tag(id, use_names) for id in con_list]
        if len(filtered) == 0:
            return empty_char
        else:
            output = ", ".join(filtered)
            return output
    
    def _get_tag(self, id: int, use_names: bool) -> str:
        """ Get the tag for a token or semantic. 
        Args:
            id: The ID of the token or semantic.
            use_names: Whether to use names instead of IDs.
        Returns:
            str: The tag for the token or semantic.
        """
        if use_names:
            return f"{self._state.tokens[id]['name']}"
        else:
            return f"{id}"
        
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
                c.extend(token['myChildP'])
                c.extend(token['myPred'])
                c.extend(token['myObj'])
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
    def _print_tokens(self, tokens: List[Dict], features: List[Tuple[str, str]] = None, ids: list[int] = None):
        """ Print the tokens in the state.
        Args:
            tokens: List of tokens to print.
        """
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
                        ('Map Input', 'map_input'),
                        ('Inhib Input', 'inhibitor_input'),
                        ('mappingConnections', 'mappingConnections'),
                        ('mappingHypotheses', 'mappingHypotheses'),
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
        header_text = self._get_header_text("Tokens", len(tokens), ids)
        self._print_table(columns, rows, header_text)

    def _get_header_text(self, label: str,num_items: int, ids: list[int] = None) -> str:
        """ Get the header text. """
        header_text = f"{self.header_text} {label} ({num_items})"
        if ids is not None:
            header_text += f" | {ids}"
        return header_text
    
    def _print_semantics(self, semantics: List[Dict], features: List[Tuple[str, str]] = None):
        """ Print the semantics in the state.
        Args:
            semantics: List of semantics to print.
        """
        header_text = 'Semantics'
        if features is None:
            features = [('Idx', 'index'), 
                        ('Name', 'name'), 
                        ('ID', 'ID'),
                        ('Act', 'act'), 
                        ('Input', 'myinput'), 
                        ('Max Input', 'max_sem_input'), 
                        ('Dimension', 'dimension'), 
                        ('Amount', 'amount'), 
                        ('Ont Status', 'ont_status'), 
                        ]
        columns = [feature[0] for feature in features]
        rows = []
        for sem in semantics:
            row = []
            for feature in features:
                if feature[1] == 'index':
                    value = self._state.sem_idxs[sem['ID']]
                else:
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
        