# nodes/utils/new_printer/printer.py
# Provides a Printer class for printing token tensors.

import os
import torch
from .print_table import tablePrinter
from ...enums import TF, TF_type, SF, Type, Set, Mode, OntStatus, B, null, MappingFields


class Printer:
    """
    A class for printing token tensor data in table format.
    
    Attributes:
        use_labels (bool): If True, convert float values to enum labels where applicable.
                          If False, print raw float values.
        log_file (str): Optional file path to log output to.
        print_to_console (bool): Whether to print output to console.
    """
    
    def __init__(self, use_labels: bool = True, log_file: str = None, print_to_console: bool = True):
        """
        Initialize the Printer.
        
        Args:
            use_labels (bool): Toggle between labels (True) and raw floats (False). Default True.
            log_file (str): Optional file to log output to.
            print_to_console (bool): Whether to print to console. Default True.
        """
        self.use_labels = use_labels
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
    
    def print_token_tensor(self, token_tensor, cols_per_table: int = 8, 
                           show_deleted: bool = False, indices: torch.Tensor = None):
        """
        Print a Token_Tensor object in table format.
        
        Args:
            token_tensor: The Token_Tensor object to print.
            cols_per_table (int): Number of feature columns per table. 
                                  If None or 0, shows all columns in one table.
            show_deleted (bool): Whether to include deleted tokens. Default False.
            indices (torch.Tensor): Optional specific indices to print. 
                                    If None, prints all (non-deleted) tokens.
        """
        tensor = token_tensor.tensor
        names = token_tensor.names
        
        if tensor.size(0) == 0:
            self._output("Empty tensor")
            return
        
        # Determine which indices to display
        if indices is not None:
            display_indices = indices
        elif show_deleted:
            display_indices = torch.arange(tensor.size(0))
        else:
            non_deleted_mask = tensor[:, TF.DELETED] == B.FALSE
            display_indices = torch.where(non_deleted_mask)[0]
        
        if len(display_indices) == 0:
            self._output("No tokens to display")
            return
        
        # Get feature names as column headers
        tf_headers = [tf.name for tf in TF]
        
        # Check if we have names to use
        has_names = any(idx.item() in names for idx in display_indices)
        id_header = "Name" if has_names else "Idx"
        
        # Build row data
        rows = []
        for idx in display_indices:
            row = []
            # Add identifier (name or index)
            if has_names:
                name = names.get(idx.item(), "")
                row.append(name if name else str(idx.item()))
            else:
                row.append(str(idx.item()))
            
            # Add feature values
            for tf in TF:
                value = tensor[idx, tf].item()
                formatted = self._format_value(tf, value)
                row.append(formatted)
            
            rows.append(row)
        
        # All columns: [id_header] + tf_headers
        all_columns = [id_header] + tf_headers
        
        # Print tables (split by cols_per_table if needed)
        if cols_per_table is None or cols_per_table <= 0:
            # Print all in one table
            self._print_table(all_columns, rows, f"Token Tensor ({len(display_indices)} tokens)")
        else:
            # Split into multiple tables
            num_features = len(tf_headers)
            table_num = 1
            
            for start_idx in range(0, num_features, cols_per_table):
                end_idx = min(start_idx + cols_per_table, num_features)
                
                # Columns for this chunk: id column + feature columns slice
                chunk_columns = [id_header] + tf_headers[start_idx:end_idx]
                
                # Rows for this chunk: id value + feature values slice
                chunk_rows = []
                for row in rows:
                    # row[0] is the id, row[1:] are the feature values
                    chunk_row = [row[0]] + row[1 + start_idx:1 + end_idx]
                    chunk_rows.append(chunk_row)
                
                # Determine header text
                feature_range = f"Features {start_idx}-{end_idx - 1}"
                header_text = f"Token Tensor - {feature_range}"
                
                self._print_table(chunk_columns, chunk_rows, header_text)
                
                # Add spacing between tables
                if end_idx < num_features:
                    self._output("")
                
                table_num += 1
    
    def print_connections(self, token_tensor, show_deleted: bool = False, 
                          indices: torch.Tensor = None, use_names: bool = True,
                          connected_char: str = "●", empty_char: str = "·"):
        """
        Print the connections tensor as a matrix table.
        Rows are parent tokens, columns are child tokens.
        connections[parent, child] = True means parent connects to child.
        
        Args:
            token_tensor: The Token_Tensor object containing connections.
            show_deleted (bool): Whether to include deleted tokens. Default False.
            indices (torch.Tensor): Optional specific indices to print.
                                    If None, prints all (non-deleted) tokens.
            use_names (bool): Whether to use token names instead of indices. Default True.
            connected_char (str): Character to show for connections. Default "●".
            empty_char (str): Character to show for no connection. Default "·".
        """
        tensor = token_tensor.tensor
        names = token_tensor.names
        connections = token_tensor.connections
        
        # Handle Connections_Tensor wrapper or raw tensor
        if hasattr(connections, 'connections'):
            conn_tensor = connections.connections
        else:
            conn_tensor = connections
        
        if tensor.size(0) == 0:
            self._output("Empty tensor - no connections to display")
            return
        
        # Determine which indices to display
        if indices is not None:
            display_indices = indices
        elif show_deleted:
            display_indices = torch.arange(tensor.size(0))
        else:
            non_deleted_mask = tensor[:, TF.DELETED] == B.FALSE
            display_indices = torch.where(non_deleted_mask)[0]
        
        if len(display_indices) == 0:
            self._output("No tokens to display")
            return
        
        # Get labels for tokens (name or index)
        def get_label(idx: int) -> str:
            if use_names and idx in names and names[idx]:
                return names[idx]
            return str(idx)
        
        # Build column headers (child tokens)
        col_headers = ["Parent\\Child"] + [get_label(idx.item()) for idx in display_indices]
        
        # Build row data
        rows = []
        for parent_idx in display_indices:
            row = [get_label(parent_idx.item())]  # Row label (parent token)
            
            for child_idx in display_indices:
                # Check if connection exists
                is_connected = conn_tensor[parent_idx, child_idx].item()
                if is_connected:
                    row.append(connected_char)
                else:
                    row.append(empty_char)
            
            rows.append(row)
        
        # Count total connections
        # Get the submatrix for display_indices
        sub_connections = conn_tensor[display_indices][:, display_indices]
        total_connections = sub_connections.sum().item()
        
        header_text = f"Connections Matrix ({len(display_indices)} tokens, {int(total_connections)} connections)"
        self._print_table(col_headers, rows, header_text)
    
    def print_connections_list(self, token_tensor, show_deleted: bool = False,
                               indices: torch.Tensor = None, use_names: bool = True):
        """
        Print connections as a list of parent -> child relationships.
        More readable for sparse connection matrices.
        
        Args:
            token_tensor: The Token_Tensor object containing connections.
            show_deleted (bool): Whether to include deleted tokens. Default False.
            indices (torch.Tensor): Optional specific indices to print.
                                    If None, prints all (non-deleted) tokens.
            use_names (bool): Whether to use token names instead of indices. Default True.
        """
        tensor = token_tensor.tensor
        names = token_tensor.names
        connections = token_tensor.connections
        
        # Handle Connections_Tensor wrapper or raw tensor
        if hasattr(connections, 'connections'):
            conn_tensor = connections.connections
        else:
            conn_tensor = connections
        
        if tensor.size(0) == 0:
            self._output("Empty tensor - no connections to display")
            return
        
        # Determine which indices to display
        if indices is not None:
            display_indices = indices
        elif show_deleted:
            display_indices = torch.arange(tensor.size(0))
        else:
            non_deleted_mask = tensor[:, TF.DELETED] == B.FALSE
            display_indices = torch.where(non_deleted_mask)[0]
        
        if len(display_indices) == 0:
            self._output("No tokens to display")
            return
        
        # Get labels for tokens (name or index)
        def get_label(idx: int) -> str:
            if use_names and idx in names and names[idx]:
                return names[idx]
            return str(idx)
        
        # Convert display_indices to set for quick lookup
        display_set = set(display_indices.tolist())
        
        # Build list of connections
        rows = []
        for parent_idx in display_indices:
            parent_label = get_label(parent_idx.item())
            children = []
            
            for child_idx in display_indices:
                if conn_tensor[parent_idx, child_idx].item():
                    children.append(get_label(child_idx.item()))
            
            if children:
                children_str = ", ".join(children)
                rows.append([parent_label, "→", children_str])
            else:
                rows.append([parent_label, "→", "(none)"])
        
        # Count total connections
        sub_connections = conn_tensor[display_indices][:, display_indices]
        total_connections = sub_connections.sum().item()
        
        header_text = f"Connections List ({len(display_indices)} tokens, {int(total_connections)} connections)"
        columns = ["Parent", "", "Children"]
        self._print_table(columns, rows, header_text)
    
    def print_links(self, links, token_names: dict[int, str] = None, 
                    semantic_names: dict[int, str] = None, 
                    token_indices: torch.Tensor = None,
                    semantic_indices: torch.Tensor = None,
                    min_weight: float = 0.0,
                    show_weights: bool = True):
        """
        Print the links tensor as a matrix showing token-to-semantic connections.
        Links tensor shape: [tokens, semantics] with float weights.
        
        Args:
            links: Links object or raw tensor of shape [tokens, semantics].
            token_names (dict[int, str]): Optional dict mapping token index to name.
            semantic_names (dict[int, str]): Optional dict mapping semantic index to name.
            token_indices (torch.Tensor): Optional specific token indices to show.
                                          If None, shows all tokens.
            semantic_indices (torch.Tensor): Optional specific semantic indices to show.
                                             If None, shows all semantics with at least one link.
            min_weight (float): Minimum weight to display. Links below this are shown as empty.
            show_weights (bool): If True, show weight values. If False, show "●" for linked.
        """
        # Handle Links wrapper or raw tensor
        if hasattr(links, 'adj_matrix'):
            links_tensor = links.adj_matrix
        else:
            links_tensor = links
        
        if links_tensor.size(0) == 0 or links_tensor.size(1) == 0:
            self._output("Empty links tensor")
            return
        
        num_tokens = links_tensor.size(0)
        num_semantics = links_tensor.size(1)
        
        # Determine which token indices to display
        if token_indices is not None:
            display_token_indices = token_indices
        else:
            display_token_indices = torch.arange(num_tokens)
        
        # Determine which semantic indices to display
        if semantic_indices is not None:
            display_sem_indices = semantic_indices
        else:
            # Show semantics that have at least one link above min_weight
            sem_has_links = (links_tensor >= min_weight).any(dim=0)
            display_sem_indices = torch.where(sem_has_links)[0]
        
        if len(display_token_indices) == 0:
            self._output("No tokens to display")
            return
        
        if len(display_sem_indices) == 0:
            self._output("No linked semantics to display")
            return
        
        # Helper to get token label
        def get_token_label(idx: int) -> str:
            if token_names and idx in token_names and token_names[idx]:
                return token_names[idx]
            return f"T{idx}"
        
        # Helper to get semantic label
        def get_sem_label(idx: int) -> str:
            if semantic_names and idx in semantic_names and semantic_names[idx]:
                return semantic_names[idx]
            return f"S{idx}"
        
        # Build column headers (semantic labels)
        col_headers = ["Token"] + [get_sem_label(idx.item()) for idx in display_sem_indices]
        
        # Build row data
        rows = []
        total_links = 0
        for tk_idx in display_token_indices:
            row = [get_token_label(tk_idx.item())]
            
            for sem_idx in display_sem_indices:
                weight = links_tensor[tk_idx, sem_idx].item()
                
                if weight >= min_weight and weight > 0:
                    total_links += 1
                    if show_weights:
                        # Format weight nicely
                        if weight == 1.0:
                            row.append("1.0")
                        else:
                            row.append(f"{weight:.3f}".rstrip('0').rstrip('.'))
                    else:
                        row.append("●")
                else:
                    row.append("·")
            
            rows.append(row)
        
        header_text = f"Links Matrix ({len(display_token_indices)} tokens × {len(display_sem_indices)} semantics, {total_links} links)"
        self._print_table(col_headers, rows, header_text)
    
    def print_links_list(self, links, token_names: dict[int, str] = None, 
                         semantic_names: dict[int, str] = None,
                         token_indices: torch.Tensor = None,
                         min_weight: float = 0.0,
                         show_weights: bool = True):
        """
        Print the links tensor as a list showing each token's linked semantics.
        More readable for sparse matrices.
        
        Args:
            links: Links object or raw tensor of shape [tokens, semantics].
            token_names (dict[int, str]): Optional dict mapping token index to name.
            semantic_names (dict[int, str]): Optional dict mapping semantic index to name.
            token_indices (torch.Tensor): Optional specific token indices to show.
                                          If None, shows all tokens with at least one link.
            min_weight (float): Minimum weight to display. Default 0.0.
            show_weights (bool): If True, show weight values with semantics. Default True.
        """
        # Handle Links wrapper or raw tensor
        if hasattr(links, 'adj_matrix'):
            links_tensor = links.adj_matrix
        else:
            links_tensor = links
        
        if links_tensor.size(0) == 0 or links_tensor.size(1) == 0:
            self._output("Empty links tensor")
            return
        
        num_tokens = links_tensor.size(0)
        num_semantics = links_tensor.size(1)
        
        # Determine which token indices to display
        if token_indices is not None:
            display_token_indices = token_indices
        else:
            # Show tokens that have at least one link above min_weight
            token_has_links = (links_tensor >= min_weight).any(dim=1)
            has_links_mask = token_has_links & (links_tensor.sum(dim=1) > 0)
            display_token_indices = torch.where(has_links_mask)[0]
        
        if len(display_token_indices) == 0:
            self._output("No tokens with links to display")
            return
        
        # Helper to get token label
        def get_token_label(idx: int) -> str:
            if token_names and idx in token_names and token_names[idx]:
                return token_names[idx]
            return f"T{idx}"
        
        # Helper to get semantic label
        def get_sem_label(idx: int) -> str:
            if semantic_names and idx in semantic_names and semantic_names[idx]:
                return semantic_names[idx]
            return f"S{idx}"
        
        # Build list of links
        rows = []
        total_links = 0
        for tk_idx in display_token_indices:
            token_label = get_token_label(tk_idx.item())
            linked_sems = []
            
            for sem_idx in range(num_semantics):
                weight = links_tensor[tk_idx, sem_idx].item()
                if weight >= min_weight and weight > 0:
                    total_links += 1
                    sem_label = get_sem_label(sem_idx)
                    if show_weights:
                        if weight == 1.0:
                            linked_sems.append(f"{sem_label}(1.0)")
                        else:
                            weight_str = f"{weight:.3f}".rstrip('0').rstrip('.')
                            linked_sems.append(f"{sem_label}({weight_str})")
                    else:
                        linked_sems.append(sem_label)
            
            if linked_sems:
                sems_str = ", ".join(linked_sems)
                rows.append([token_label, "→", sems_str])
            else:
                rows.append([token_label, "→", "(none)"])
        
        header_text = f"Links List ({len(display_token_indices)} tokens, {total_links} links)"
        columns = ["Token", "", "Semantics"]
        self._print_table(columns, rows, header_text)
    
    def print_mappings(self, mapping, 
                       driver_names: dict[int, str] = None,
                       recipient_names: dict[int, str] = None,
                       driver_indices: torch.Tensor = None,
                       recipient_indices: torch.Tensor = None,
                       field: MappingFields = MappingFields.WEIGHT,
                       min_value: float = 0.0,
                       show_values: bool = True):
        """
        Print the mapping tensor as a matrix showing recipient-to-driver mappings.
        Mapping tensor shape: [recipient_nodes, driver_nodes, fields]
        
        Args:
            mapping: Mapping object or raw tensor of shape [recipients, drivers, fields].
            driver_names (dict[int, str]): Optional dict mapping driver index to name.
            recipient_names (dict[int, str]): Optional dict mapping recipient index to name.
            driver_indices (torch.Tensor): Optional specific driver indices to show.
                                           If None, shows all drivers with at least one mapping.
            recipient_indices (torch.Tensor): Optional specific recipient indices to show.
                                              If None, shows all recipients with at least one mapping.
            field (MappingFields): Which field to display. Default WEIGHT.
            min_value (float): Minimum value to display. Values below this shown as empty.
            show_values (bool): If True, show values. If False, show "●" for non-zero.
        """
        # Handle Mapping wrapper or raw tensor
        if hasattr(mapping, 'adj_matrix'):
            mapping_tensor = mapping.adj_matrix
        else:
            mapping_tensor = mapping
        
        if mapping_tensor.size(0) == 0 or mapping_tensor.size(1) == 0:
            self._output("Empty mapping tensor")
            return
        
        num_recipients = mapping_tensor.size(0)
        num_drivers = mapping_tensor.size(1)
        
        # Get the specific field's 2D slice
        field_tensor = mapping_tensor[:, :, field]
        
        # Determine which recipient indices to display
        if recipient_indices is not None:
            display_rec_indices = recipient_indices
        else:
            # Show recipients that have at least one mapping above min_value
            rec_has_mappings = (field_tensor >= min_value).any(dim=1) & (field_tensor > 0).any(dim=1)
            display_rec_indices = torch.where(rec_has_mappings)[0]
        
        # Determine which driver indices to display
        if driver_indices is not None:
            display_dri_indices = driver_indices
        else:
            # Show drivers that have at least one mapping above min_value
            dri_has_mappings = (field_tensor >= min_value).any(dim=0) & (field_tensor > 0).any(dim=0)
            display_dri_indices = torch.where(dri_has_mappings)[0]
        
        if len(display_rec_indices) == 0:
            self._output("No recipients with mappings to display")
            return
        
        if len(display_dri_indices) == 0:
            self._output("No drivers with mappings to display")
            return
        
        # Helper to get recipient label
        def get_rec_label(idx: int) -> str:
            if recipient_names and idx in recipient_names and recipient_names[idx]:
                return recipient_names[idx]
            return f"R{idx}"
        
        # Helper to get driver label
        def get_dri_label(idx: int) -> str:
            if driver_names and idx in driver_names and driver_names[idx]:
                return driver_names[idx]
            return f"D{idx}"
        
        # Build column headers (driver labels)
        col_headers = ["Rec\\Dri"] + [get_dri_label(idx.item()) for idx in display_dri_indices]
        
        # Build row data
        rows = []
        total_mappings = 0
        for rec_idx in display_rec_indices:
            row = [get_rec_label(rec_idx.item())]
            
            for dri_idx in display_dri_indices:
                value = field_tensor[rec_idx, dri_idx].item()
                
                if value >= min_value and value > 0:
                    total_mappings += 1
                    if show_values:
                        # Format value nicely
                        if value == 1.0:
                            row.append("1.0")
                        else:
                            row.append(f"{value:.3f}".rstrip('0').rstrip('.'))
                    else:
                        row.append("●")
                else:
                    row.append("·")
            
            rows.append(row)
        
        field_name = field.name
        header_text = f"Mappings [{field_name}] ({len(display_rec_indices)} rec × {len(display_dri_indices)} dri, {total_mappings} mappings)"
        self._print_table(col_headers, rows, header_text)
    
    def print_mappings_list(self, mapping,
                            driver_names: dict[int, str] = None,
                            recipient_names: dict[int, str] = None,
                            recipient_indices: torch.Tensor = None,
                            field: MappingFields = MappingFields.WEIGHT,
                            min_value: float = 0.0,
                            show_values: bool = True):
        """
        Print the mapping tensor as a list showing each recipient's mapped drivers.
        More readable for sparse matrices.
        
        Args:
            mapping: Mapping object or raw tensor of shape [recipients, drivers, fields].
            driver_names (dict[int, str]): Optional dict mapping driver index to name.
            recipient_names (dict[int, str]): Optional dict mapping recipient index to name.
            recipient_indices (torch.Tensor): Optional specific recipient indices to show.
                                              If None, shows all recipients with at least one mapping.
            field (MappingFields): Which field to display. Default WEIGHT.
            min_value (float): Minimum value to display. Default 0.0.
            show_values (bool): If True, show values with drivers. Default True.
        """
        # Handle Mapping wrapper or raw tensor
        if hasattr(mapping, 'adj_matrix'):
            mapping_tensor = mapping.adj_matrix
        else:
            mapping_tensor = mapping
        
        if mapping_tensor.size(0) == 0 or mapping_tensor.size(1) == 0:
            self._output("Empty mapping tensor")
            return
        
        num_recipients = mapping_tensor.size(0)
        num_drivers = mapping_tensor.size(1)
        
        # Get the specific field's 2D slice
        field_tensor = mapping_tensor[:, :, field]
        
        # Determine which recipient indices to display
        if recipient_indices is not None:
            display_rec_indices = recipient_indices
        else:
            # Show recipients that have at least one mapping above min_value
            rec_has_mappings = (field_tensor >= min_value).any(dim=1) & (field_tensor > 0).any(dim=1)
            display_rec_indices = torch.where(rec_has_mappings)[0]
        
        if len(display_rec_indices) == 0:
            self._output("No recipients with mappings to display")
            return
        
        # Helper to get recipient label
        def get_rec_label(idx: int) -> str:
            if recipient_names and idx in recipient_names and recipient_names[idx]:
                return recipient_names[idx]
            return f"R{idx}"
        
        # Helper to get driver label
        def get_dri_label(idx: int) -> str:
            if driver_names and idx in driver_names and driver_names[idx]:
                return driver_names[idx]
            return f"D{idx}"
        
        # Build list of mappings
        rows = []
        total_mappings = 0
        for rec_idx in display_rec_indices:
            rec_label = get_rec_label(rec_idx.item())
            mapped_drivers = []
            
            for dri_idx in range(num_drivers):
                value = field_tensor[rec_idx, dri_idx].item()
                if value >= min_value and value > 0:
                    total_mappings += 1
                    dri_label = get_dri_label(dri_idx)
                    if show_values:
                        if value == 1.0:
                            mapped_drivers.append(f"{dri_label}(1.0)")
                        else:
                            value_str = f"{value:.3f}".rstrip('0').rstrip('.')
                            mapped_drivers.append(f"{dri_label}({value_str})")
                    else:
                        mapped_drivers.append(dri_label)
            
            if mapped_drivers:
                drivers_str = ", ".join(mapped_drivers)
                rows.append([rec_label, "→", drivers_str])
            else:
                rows.append([rec_label, "→", "(none)"])
        
        field_name = field.name
        header_text = f"Mappings List [{field_name}] ({len(display_rec_indices)} recipients, {total_mappings} mappings)"
        columns = ["Recipient", "", "Drivers"]
        self._print_table(columns, rows, header_text)
    
    def _print_table(self, columns: list[str], rows: list[list[str]], header_text: str):
        """
        Print a table using tablePrinter.
        
        Args:
            columns (list[str]): Column headers.
            rows (list[list[str]]): Row data.
            header_text (str): Header text for the table.
        """
        table = tablePrinter(
            columns=columns,
            rows=rows,
            headers=[header_text],
            log_file=self.log_file,
            print_to_console=self.print_to_console
        )
        table.print_table(header=True, column_names=True, split=False)
    
    def _format_value(self, feature: TF, value: float) -> str:
        """
        Format a tensor value based on the feature type.
        
        Args:
            feature (TF): The feature enum indicating the value type.
            value (float): The raw tensor value.
        
        Returns:
            str: Formatted value - either raw float or converted label.
        """
        # Handle null values
        if value == null:
            return "null"
        
        # If not using labels, return raw float representation
        if not self.use_labels:
            return self._format_raw(value)
        
        # Get the type for this feature
        feature_type = TF_type(feature)
        
        # Convert enum types to labels
        if feature_type == Type:
            try:
                return Type(int(value)).name
            except (ValueError, KeyError):
                return str(int(value))
        
        elif feature_type == Set:
            try:
                return Set(int(value)).name
            except (ValueError, KeyError):
                return str(int(value))
        
        elif feature_type == Mode:
            try:
                return Mode(int(value)).name
            except (ValueError, KeyError):
                return str(int(value))
        
        # Convert bool types
        elif feature_type == bool:
            if value == B.TRUE:
                return "True"
            elif value == B.FALSE:
                return "False"
            else:
                return str(bool(value))
        
        # Format int types
        elif feature_type == int:
            return str(int(value))
        
        # Format float types
        elif feature_type == float:
            return f"{value:.4f}".rstrip('0').rstrip('.')
        
        # Fallback
        return str(value)
    
    def _format_raw(self, value: float) -> str:
        """
        Format a value as a raw float string.
        
        Args:
            value (float): The value to format.
        
        Returns:
            str: The formatted float string.
        """
        # Check if it's effectively an integer
        if value == int(value):
            return str(int(value))
        else:
            return f"{value:.4f}"

