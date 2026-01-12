# DORA_bridge/print_state.py
# Provides functions for printing state dictionary data in table format.

import os
from typing import Dict, List, Optional
from .print_table import tablePrinter


class StatePrinter:
    """
    A class for printing state dictionary data in table format.
    
    Follows the style of nodes/utils/printer/printer.py but operates on
    the state dictionary format used by DORA_bridge.
    
    Attributes:
        log_file (str): Optional file path to log output to.
        print_to_console (bool): Whether to print output to console.
    """
    
    def __init__(self, log_file: str = None, print_to_console: bool = True):
        """
        Initialize the StatePrinter.
        
        Args:
            log_file (str): Optional file to log output to.
            print_to_console (bool): Whether to print to console. Default True.
        """
        self.log_file = log_file
        self.print_to_console = print_to_console
    
    def _output(self, message: str):
        """
        Output a message to console and/or log file.
        
        Args:
            message (str): The message to output.
        """
        if self.print_to_console:
            print(message)
        if self.log_file is not None:
            mode = 'a' if os.path.exists(self.log_file) else 'w'
            with open(self.log_file, mode, encoding='utf-8') as f:
                f.write(message + "\n")
    
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
        
        table = tablePrinter(
            columns=columns,
            rows=rows,
            headers=[header_text],
            log_file=self.log_file,
            print_to_console=self.print_to_console
        )
        table.print_table(header=True, column_names=True, split=False)
    
    def print_tokens(self, state: Dict, token_types: List[str] = None, 
                     show_all_fields: bool = False, filter_set: str = None) -> None:
        """
        Print the tokens in the state dictionary.
        
        Args:
            state (Dict): The state dictionary containing token data.
            token_types (list[str]): Which token types to print ('Ps', 'RBs', 'POs').
                                     If None, prints all types.
            show_all_fields (bool): If True, shows all available fields.
                                    If False, shows compact view. Default False.
            filter_set (str): Filter tokens by set ('driver', 'recipient'). 
                              If None, shows all.
        """
        if token_types is None:
            token_types = ['Ps', 'RBs', 'POs']
        
        tokens_data = state.get('tokens', {})
        
        for token_type in token_types:
            tokens = tokens_data.get(token_type, [])
            
            if not tokens:
                self._output(f"\n{token_type}: (none)")
                continue
            
            # Filter by set if specified
            if filter_set:
                tokens = [t for t in tokens if t.get('set') == filter_set]
                if not tokens:
                    self._output(f"\n{token_type} ({filter_set}): (none)")
                    continue
            
            self._print_token_type(tokens, token_type, show_all_fields, filter_set)
    
    def _print_token_type(self, tokens: List[Dict], token_type: str, 
                          show_all_fields: bool, filter_set: Optional[str]):
        """
        Print a specific token type.
        """
        if show_all_fields:
            # Full field view
            if token_type == 'Ps':
                columns = ['Name', 'Set', 'Analog', 'Act', 'Inferred', 'Child RBs']
                rows = []
                for t in tokens:
                    child_rbs = ', '.join(t.get('child_RB_names', [])[:3])
                    if len(t.get('child_RB_names', [])) > 3:
                        child_rbs += '...'
                    rows.append([
                        t.get('name', ''),
                        t.get('set', ''),
                        str(t.get('analog', '')),
                        self._format_float(t.get('act', 0.0)),
                        str(t.get('inferred', False)),
                        child_rbs
                    ])
            elif token_type == 'RBs':
                columns = ['Name', 'Set', 'Analog', 'Act', 'Pred', 'Obj', 'Parent Ps']
                rows = []
                for t in tokens:
                    parent_ps = ', '.join(t.get('parent_P_names', [])[:2])
                    if len(t.get('parent_P_names', [])) > 2:
                        parent_ps += '...'
                    rows.append([
                        t.get('name', ''),
                        t.get('set', ''),
                        str(t.get('analog', '')),
                        self._format_float(t.get('act', 0.0)),
                        t.get('pred_name', '') or '-',
                        t.get('obj_name', '') or '-',
                        parent_ps
                    ])
            else:  # POs
                columns = ['Name', 'Set', 'Analog', 'Act', 'Type', 'Parent RBs', 'Semantics']
                rows = []
                for t in tokens:
                    parent_rbs = ', '.join(t.get('parent_RB_names', [])[:2])
                    if len(t.get('parent_RB_names', [])) > 2:
                        parent_rbs += '...'
                    sems = ', '.join(t.get('semantic_names', [])[:3])
                    if len(t.get('semantic_names', [])) > 3:
                        sems += '...'
                    po_type = 'pred' if t.get('predOrObj') == 1 else 'obj'
                    rows.append([
                        t.get('name', ''),
                        t.get('set', ''),
                        str(t.get('analog', '')),
                        self._format_float(t.get('act', 0.0)),
                        po_type,
                        parent_rbs,
                        sems
                    ])
        else:
            # Compact view
            columns = ['Name', 'Set', 'Analog', 'Act', 'Net Input', 'TD Input', 'BU Input', 'Lateral Input', 'Map Input']
            rows = []
            for t in tokens:
                rows.append([
                    t.get('name', ''),
                    t.get('set', ''),
                    str(t.get('analog', '')),
                    self._format_float(t.get('act', 0.0)),
                    self._format_float(t.get('net_input', 0.0)),
                    self._format_float(t.get('td_input', 0.0)),
                    self._format_float(t.get('bu_input', 0.0)),
                    self._format_float(t.get('lateral_input', 0.0)),
                    self._format_float(t.get('map_input', 0.0)),
                ])
        
        set_str = f" ({filter_set})" if filter_set else ""
        header_text = f"{token_type}{set_str} ({len(tokens)} tokens)"
        self._print_table(columns, rows, header_text)
    
    def print_semantics(self, state: Dict, show_zero_act: bool = True) -> None:
        """
        Print the semantics in the state dictionary.
        
        Args:
            state (Dict): The state dictionary containing semantic data.
            show_zero_act (bool): Whether to show semantics with zero activation.
                                  Default True.
        """
        semantics = state.get('semantics', [])
        
        if not semantics:
            self._output("Semantics: (none)")
            return
        
        if not show_zero_act:
            semantics = [s for s in semantics if s.get('act', 0.0) != 0.0]
            if not semantics:
                self._output("Semantics: (all zero activation)")
                return
        
        columns = ['Name', 'Index', 'Act', 'Ont Status', 'Dimension', 'Amount']
        rows = []
        
        for sem in semantics:
            rows.append([
                sem.get('name', ''),
                str(sem.get('index', '')),
                self._format_float(sem.get('act', 0.0)),
                sem.get('ont_status', '') or '-',
                sem.get('dimension') or '-',
                str(sem.get('amount', '')) if sem.get('amount') is not None else '-'
            ])
        
        header_text = f"Semantics ({len(semantics)} units)"
        self._print_table(columns, rows, header_text)
    
    def print_links(self, state: Dict, show_weights: bool = True, 
                    min_weight: float = 0.0, as_matrix: bool = False) -> None:
        """
        Print the links between POs and semantics in the state dictionary.
        
        Args:
            state (Dict): The state dictionary containing link data.
            show_weights (bool): If True, show weight values. If False, show "●" for linked.
            min_weight (float): Minimum weight to display. Links below this are hidden.
            as_matrix (bool): If True, print as matrix. If False, print as list.
        """
        links_data = state.get('links', {})
        links_list = links_data.get('links_list', [])
        
        if not links_list:
            self._output("Links: (none)")
            return
        
        # Filter by minimum weight
        links_list = [l for l in links_list if l.get('weight', 0.0) >= min_weight]
        
        if not links_list:
            self._output(f"Links: (none above min_weight={min_weight})")
            return
        
        if as_matrix:
            self._print_links_matrix(state, links_list, show_weights)
        else:
            self._print_links_list(links_list, show_weights)
    
    def _print_links_list(self, links_list: List[Dict], show_weights: bool):
        """Print links as a list."""
        # Group links by PO
        po_links = {}
        for link in links_list:
            po_name = link.get('po_name', '')
            po_idx = link.get('po_index', 0)
            key = (po_idx, po_name)
            if key not in po_links:
                po_links[key] = []
            
            sem_name = link.get('sem_name', '')
            weight = link.get('weight', 1.0)
            if show_weights:
                if weight == 1.0:
                    po_links[key].append(f"{sem_name}(1.0)")
                else:
                    weight_str = f"{weight:.3f}".rstrip('0').rstrip('.')
                    po_links[key].append(f"{sem_name}({weight_str})")
            else:
                po_links[key].append(sem_name)
        
        columns = ['PO', 'Idx', '', 'Semantics']
        rows = []
        
        for (po_idx, po_name), sems in sorted(po_links.items(), key=lambda x: x[0][0]):
            sems_str = ', '.join(sems)
            rows.append([po_name, str(po_idx), '→', sems_str])
        
        header_text = f"Links List ({len(links_list)} links, {len(po_links)} POs)"
        self._print_table(columns, rows, header_text)
    
    def _print_links_matrix(self, state: Dict, links_list: List[Dict], show_weights: bool):
        """Print links as a matrix."""
        links_data = state.get('links', {})
        po_names = links_data.get('po_names', [])
        sem_names = links_data.get('semantic_names', [])
        matrix = links_data.get('matrix', [])
        
        if not matrix or not po_names or not sem_names:
            self._output("Links Matrix: (insufficient data)")
            return
        
        # Find semantics that actually have links
        active_sems = set()
        for link in links_list:
            active_sems.add(link.get('sem_name'))
        
        active_sem_indices = [i for i, name in enumerate(sem_names) if name in active_sems]
        
        if len(active_sem_indices) > 15:
            self._output(f"Links Matrix: Too many semantics ({len(active_sem_indices)}), showing list view instead.")
            self._print_links_list(links_list, show_weights)
            return
        
        # Build column headers
        col_headers = ['PO'] + [sem_names[i] for i in active_sem_indices]
        
        rows = []
        for po_idx, po_name in enumerate(po_names):
            row = [po_name]
            for sem_idx in active_sem_indices:
                weight = matrix[po_idx][sem_idx] if po_idx < len(matrix) and sem_idx < len(matrix[po_idx]) else 0.0
                if weight > 0:
                    if show_weights:
                        if weight == 1.0:
                            row.append("1.0")
                        else:
                            row.append(f"{weight:.2f}".rstrip('0').rstrip('.'))
                    else:
                        row.append("●")
                else:
                    row.append("·")
            rows.append(row)
        
        header_text = f"Links Matrix ({len(po_names)} POs × {len(active_sem_indices)} semantics)"
        self._print_table(col_headers, rows, header_text)
    
    def print_mappings(self, state: Dict, mapping_types: List[str] = None,
                       show_weights: bool = True, min_weight: float = 0.0) -> None:
        """
        Print the mappings between driver and recipient tokens in the state dictionary.
        
        Args:
            state (Dict): The state dictionary containing mapping data.
            mapping_types (list[str]): Which mapping types to print ('P', 'RB', 'PO').
                                       If None, prints all types with mappings.
            show_weights (bool): If True, show weight values. If False, show "●".
            min_weight (float): Minimum weight to display.
        """
        mappings_data = state.get('mappings', {})
        
        if mapping_types is None:
            mapping_types = ['P', 'RB', 'PO']
        
        total_mappings = 0
        
        for mapping_type in mapping_types:
            type_key = f'{mapping_type}_mappings'
            mappings = mappings_data.get(type_key, [])
            
            # Filter by minimum weight
            mappings = [m for m in mappings if m.get('weight', 0.0) >= min_weight]
            
            if not mappings:
                continue
            
            total_mappings += len(mappings)
            
            columns = ['Driver', '', 'Recipient', 'Weight']
            rows = []
            
            for m in mappings:
                driver = m.get('driver_name', '')
                recipient = m.get('recipient_name', '')
                weight = m.get('weight', 0.0)
                
                if show_weights:
                    if weight == 1.0:
                        weight_str = "1.0"
                    else:
                        weight_str = f"{weight:.4f}".rstrip('0').rstrip('.')
                else:
                    weight_str = "●" if weight > 0 else "·"
                
                rows.append([driver, '↔', recipient, weight_str])
            
            header_text = f"{mapping_type} Mappings ({len(mappings)})"
            self._print_table(columns, rows, header_text)
        
        if total_mappings == 0:
            self._output(f"Mappings: (none above min_weight={min_weight})")
    
    def print_connections(self, state: Dict, connection_types: List[str] = None) -> None:
        """
        Print the structural connections between tokens in the state dictionary.
        
        Args:
            state (Dict): The state dictionary containing connection data.
            connection_types (list[str]): Which connection types to print 
                                          ('P_to_RB', 'RB_to_PO', 'RB_to_childP').
                                          If None, prints all types.
        """
        connections_data = state.get('connections', {})
        
        if connection_types is None:
            connection_types = ['P_to_RB', 'RB_to_PO', 'RB_to_childP']
        
        for conn_type in connection_types:
            connections = connections_data.get(conn_type, [])
            
            if not connections:
                continue
            
            if conn_type == 'RB_to_PO':
                # Include role information
                columns = ['Parent', '', 'Child', 'Role']
                rows = []
                for c in connections:
                    rows.append([
                        c.get('parent', ''),
                        '→',
                        c.get('child', ''),
                        c.get('role', '')
                    ])
            else:
                columns = ['Parent', '', 'Child']
                rows = []
                for c in connections:
                    rows.append([
                        c.get('parent', ''),
                        '→',
                        c.get('child', '')
                    ])
            
            # Format header
            conn_display = conn_type.replace('_to_', ' → ').replace('_', ' ')
            header_text = f"{conn_display} ({len(connections)} connections)"
            self._print_table(columns, rows, header_text)
    
    def print_analogs(self, state: Dict) -> None:
        """
        Print the analog groups in the state dictionary.
        
        Args:
            state (Dict): The state dictionary containing analog data.
        """
        analogs = state.get('analogs', [])
        
        if not analogs:
            self._output("Analogs: (none)")
            return
        
        for analog in analogs:
            idx = analog.get('index', 0)
            p_names = analog.get('P_names', [])
            rb_names = analog.get('RB_names', [])
            po_names = analog.get('PO_names', [])
            
            header_text = f"Analog {idx}"
            columns = ['Type', 'Count', 'Names']
            rows = [
                ['Ps', str(len(p_names)), ', '.join(p_names) if p_names else '-'],
                ['RBs', str(len(rb_names)), ', '.join(rb_names[:5]) + ('...' if len(rb_names) > 5 else '') if rb_names else '-'],
                ['POs', str(len(po_names)), ', '.join(po_names[:5]) + ('...' if len(po_names) > 5 else '') if po_names else '-']
            ]
            
            self._print_table(columns, rows, header_text)
    
    def print_summary(self, state: Dict) -> None:
        """
        Print a summary of the state dictionary.
        
        Args:
            state (Dict): The state dictionary.
        """
        metadata = state.get('metadata', {})
        counts = metadata.get('token_counts', {})
        driver = state.get('driver', {}).get('counts', {})
        recipient = state.get('recipient', {}).get('counts', {})
        
        columns = ['Category', 'Total', 'Driver', 'Recipient']
        rows = [
            ['Ps', str(counts.get('Ps', 0)), str(driver.get('Ps', 0)), str(recipient.get('Ps', 0))],
            ['RBs', str(counts.get('RBs', 0)), str(driver.get('RBs', 0)), str(recipient.get('RBs', 0))],
            ['POs', str(counts.get('POs', 0)), str(driver.get('POs', 0)), str(recipient.get('POs', 0))],
            ['Semantics', str(counts.get('semantics', 0)), '-', '-'],
        ]
        
        # Count connections and mappings
        connections = state.get('connections', {})
        conn_count = sum(len(connections.get(k, [])) for k in ['P_to_RB', 'RB_to_PO', 'RB_to_childP'])
        
        mappings = state.get('mappings', {})
        map_count = sum(len(mappings.get(f'{t}_mappings', [])) for t in ['P', 'RB', 'PO'])
        
        links_count = len(state.get('links', {}).get('links_list', []))
        
        rows.append(['Connections', str(conn_count), '-', '-'])
        rows.append(['Links', str(links_count), '-', '-'])
        rows.append(['Mappings', str(map_count), '-', '-'])
        
        header_text = "Network State Summary"
        self._print_table(columns, rows, header_text)
    
    def print_state(self, state: Dict, verbose: bool = True) -> None:
        """
        Print a complete overview of the state dictionary.
        
        Args:
            state (Dict): The state dictionary to print.
            verbose (bool): If True, print detailed information. Default True.
        """
        self.print_summary(state)
        self._output("")
        
        if verbose:
            self.print_tokens(state, show_all_fields=True)
            self._output("")
            
            self.print_semantics(state)
            self._output("")
            
            self.print_connections(state)
            self._output("")
            
            self.print_links(state)
            self._output("")
            
            self.print_mappings(state)
    
    def _format_float(self, value: float) -> str:
        """Format a float value nicely."""
        if value == 0.0:
            return "0.0"
        elif value == int(value):
            return str(int(value)) + ".0"
        else:
            return f"{value:.4f}".rstrip('0').rstrip('.')


# ======================[ MODULE-LEVEL FUNCTIONS ]======================
# Convenience functions that create a StatePrinter instance

def print_tokens(state: Dict, token_types: List[str] = None,
                 show_all_fields: bool = False, filter_set: str = None) -> None:
    """
    Print the tokens in the state dictionary.
    
    Args:
        state (Dict): The state dictionary containing token data.
        token_types (list[str]): Which token types to print ('Ps', 'RBs', 'POs').
        show_all_fields (bool): If True, shows all available fields.
        filter_set (str): Filter tokens by set ('driver', 'recipient').
    """
    printer = StatePrinter()
    printer.print_tokens(state, token_types, show_all_fields, filter_set)


def print_semantics(state: Dict, show_zero_act: bool = True) -> None:
    """
    Print the semantics in the state dictionary.
    
    Args:
        state (Dict): The state dictionary containing semantic data.
        show_zero_act (bool): Whether to show semantics with zero activation.
    """
    printer = StatePrinter()
    printer.print_semantics(state, show_zero_act)


def print_links(state: Dict, show_weights: bool = True,
                min_weight: float = 0.0, as_matrix: bool = False) -> None:
    """
    Print the links between POs and semantics in the state dictionary.
    
    Args:
        state (Dict): The state dictionary containing link data.
        show_weights (bool): If True, show weight values.
        min_weight (float): Minimum weight to display.
        as_matrix (bool): If True, print as matrix. If False, print as list.
    """
    printer = StatePrinter()
    printer.print_links(state, show_weights, min_weight, as_matrix)


def print_mappings(state: Dict, mapping_types: List[str] = None,
                   show_weights: bool = True, min_weight: float = 0.0) -> None:
    """
    Print the mappings between driver and recipient tokens in the state dictionary.
    
    Args:
        state (Dict): The state dictionary containing mapping data.
        mapping_types (list[str]): Which mapping types to print ('P', 'RB', 'PO').
        show_weights (bool): If True, show weight values.
        min_weight (float): Minimum weight to display.
    """
    printer = StatePrinter()
    printer.print_mappings(state, mapping_types, show_weights, min_weight)


def print_connections(state: Dict, connection_types: List[str] = None) -> None:
    """
    Print the structural connections between tokens in the state dictionary.
    
    Args:
        state (Dict): The state dictionary containing connection data.
        connection_types (list[str]): Which connection types to print.
    """
    printer = StatePrinter()
    printer.print_connections(state, connection_types)
