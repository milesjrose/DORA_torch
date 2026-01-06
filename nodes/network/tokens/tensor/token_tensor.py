import torch
from ...single_nodes import Token, Ref_Token
from ....enums import *
from ..connections.connections import Connections_Tensor
from ..connections.mapping import Mapping
from typing import List
from .cache import Cache
from ....utils import tensor_ops as tOps
from logging import getLogger
from ..tensor_view import TensorView
logger = getLogger(__name__)

class Token_Tensor:
    """
    A class for holding all the tokens in the network.
    """
    def __init__(self, tokens: torch.Tensor, connections: Connections_Tensor, names: dict[int, str]):
        """
        Initialize the Tokens object.
        Args:
            tokens: torch.Tensor - The tensor of tokens.
            connections: Connections_Tensor - The connections object.
            names: dict[int, str] - The dictionary of names.
        """
        self.tensor: torch.Tensor = tokens
        self.connections: Connections_Tensor = connections
        self.names: dict[int, str] = names # idx -> name
        self.expansion_factor = 1.1
        self.cache = Cache(tokens)
        """holds the cache object"""
    
    def get_count(self) -> int:
        """
        Get the number of tokens in the tensor.
        Returns:
            int - The number of tokens in the tensor.
        """
        return self.tensor.size(dim=0)
    
    def get_set_count(self, set: Set) -> int:
        """
        Get the number of tokens in the tensor for the given set.
        Args:
            set: Set - The set to get the count for.
        Returns:
            int - The number of tokens in the tensor for the given set.
        """
        return self.cache.get_set_mask(set).sum()
    
    def add_tokens(self, tokens: torch.Tensor, names: List[str]) -> torch.Tensor:
        """
        add a tokens to the tensor.
        Args:
            tokens: torch.Tensor - The tensor of tokens to add.
            names: List[str] - The list of names to add.
        Returns:
            tuple[list[Set], torch.Tensor] - The sets that were changed and the indices that were replaced.
        """
        num_to_add = tokens.size(dim=0)
        logger.info(f"Adding {num_to_add} tokens to the tensor.")
        logger.debug(f"Tokens: {tokens}")
        logger.debug(f"Names: {names}")
        num_deleted = (self.tensor[:, TF.DELETED]==B.TRUE).sum()
        if num_to_add > num_deleted:
            # Expand to ensure we have enough deleted slots
            # We need at least (num_to_add - num_deleted) more slots
            min_additional_slots = num_to_add - num_deleted
            self.expand_tensor(min_additional_slots)
        # Add tokens to deleted indices (recalculate after potential expansion)
        deleted_mask = self.tensor[:, TF.DELETED] == B.TRUE
        deleted_idxs = torch.where(deleted_mask)[0]
        replace_idxs = deleted_idxs[:num_to_add]
        self.tensor[replace_idxs, :] = tokens
        # Add names to names dictionary
        for idx, name in zip(replace_idxs, names):
            self.names[idx.item()] = name
        # Get sets that were changed to update masks
        sets_changed = [Set(int(set)) for set in torch.unique(tokens[:, TF.SET])]
        self.cache.cache_sets(sets_changed)
        return replace_idxs
    
    def del_tokens(self, indices: torch.Tensor):
        """
        Delete the tokens at the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to delete.
        """
        logger.info(f"Deleting {len(indices)} tokens from the tensor.")
        logger.debug(f"Indices: {indices}")
        # Get SET values before setting to null (needed for cache update)
        sets_to_cache = []
        if len(indices) > 0:
            set_values = self.tensor[indices, TF.SET]
            unique_sets = torch.unique(set_values)
            # Filter out null values before converting to Set enum
            valid_sets = unique_sets[unique_sets != null]
            if len(valid_sets) > 0:
                sets_to_cache = [Set(int(set_val.item())) for set_val in valid_sets]
        
        # Set all values to null, then set DELETED flag to TRUE
        self.tensor[indices, :] = null
        self.tensor[indices, TF.DELETED] = B.TRUE
        self.cache.cache_analogs()
        if sets_to_cache:
            self.cache.cache_sets(sets_to_cache)
    
    def get_feature(self, indices: torch.Tensor, feature: TF) -> torch.Tensor:
        """
        Get the features for the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to get the features of.
            feature: TF - The feature to get.
        Returns:
            torch.Tensor - The features for the given indices.
        """
        return self.tensor[indices, feature]
    
    def set_feature(self, indices: torch.Tensor, feature: TF, value: float):
        """
        Set the features for the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to set the features of.
            feature: TF - The feature to set.
            value: float - The value to set the feature to.
        """
        self.tensor[indices, feature] = value
    
    def get_features(self, indices: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Get the features for the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to get the features of.
            features: torch.Tensor - The feature indices to get (TF enum values).
        Returns:
            torch.Tensor - The features for the given indices, shape (len(indices), len(features)).
        """
        return self.tensor[indices[:, None], features]
    
    def set_features(self, indices: torch.Tensor, features: torch.Tensor, values: torch.Tensor):
        """
        Set the features for the given indices.
        Args:
            indices: torch.Tensor - The indices of the tokens to set the features of.
            features: torch.Tensor - The feature indices to set (TF enum values).
            values: torch.Tensor - The values to set the features to, shape (len(indices), len(features)).
        """
        self.tensor[indices[:, None], features] = values
    
    def get_max(self, feature: TF, indicies: torch.Tensor) -> float:
        """
        Get the maximum value of the given feature for the given indices.
        Args:
            feature: TF - The feature to get the maximum value of.
            indices: torch.Tensor - The indices of the tokens to get the maximum value of.
        Returns:
            int, float - The index and value of the maximum value of the given feature for the given indices.
        """
        max_val, max_idx = self.tensor[indicies, feature].max(dim=0)
        return max_idx.item(), max_val.item()
    
    def get_view(self, indices: torch.Tensor) -> TensorView:
        """
        Get a view of the tensor for the given indices.
        Changes to the view will update the original tensor.
        
        Args:
            indices: torch.Tensor - The indices of the tokens to create a view of.
        Returns:
            TensorView - A view-like object that maps operations back to the original tensor.
        
        Example:
            # Get a view of tokens at indices [0, 5, 10]
            view = token_tensor.get_view(torch.tensor([0, 5, 10]))
            
            # Access with view-local indices (0, 1, 2 map to original indices 0, 5, 10)
            view[0, TF.ACT] = 1.5  # Updates token_tensor.tensor[0, TF.ACT]
            view[1, TF.SET] = Set.DRIVER  # Updates token_tensor.tensor[5, TF.SET]
            
            # Use masks/indices local to the view
            mask = torch.tensor([True, False, True])
            view[mask, TF.ACT] = 0.0  # Updates tokens at original indices 0 and 10
        """
        return TensorView(self.tensor, indices)
    
    def get_set_view(self, set: Set) -> TensorView:
        """
        Get a view of the tensor for the given set.
        Args:
            set: Set - The set to create a view of.
        Returns:
            TensorView - A view-like object that maps operations back to the original tensor.
        """
        return self.get_view(self.cache.get_set_indices(set))
    
    def expand_tensor(self, min_expansion: int = 5):
        """
        Expand the tensor by the expansion factor.
        Args:
            min_expansion: Minimum number of additional slots to add (not total size).
        """
        current_size = self.tensor.size(dim=0)
        # Calculate new size: at least current_size + min_expansion, or expanded by factor
        expanded_size = int(current_size * self.expansion_factor)
        new_size = max(current_size + min_expansion, expanded_size)
        # Create new tensor
        new_tokens = torch.full((new_size, len(TF)), null, dtype=tensor_type)
        new_tokens[:, TF.DELETED] = B.TRUE
        # Copy over old tokens
        new_tokens[:current_size, :] = self.tensor
        # Update tokens
        self.tensor = new_tokens
        # Update cache's tensor reference to point to the new tensor
        self.cache.tensor = self.tensor
        logger.info(f"Expanded tensor: {current_size} -> {new_size}")
    
    def move_tokens(self, indices: torch.Tensor, to_set: Set):
        """
        Move the tokens at the given indices to the given set.
        Args:
            indices: torch.Tensor - The indices of the tokens to move.
            to_set: Set - The set to move the tokens to.
        """
        logger.info(f"Moving tokens to set: {to_set}")
        logger.debug(f"Indices: {indices}")
        # Get old set values before moving (needed for cache update)
        sets_to_cache = [to_set]
        if len(indices) > 0:
            old_set_values = self.tensor[indices, TF.SET]
            unique_old_sets = torch.unique(old_set_values)
            # Filter out null values before converting to Set enum
            valid_old_sets = unique_old_sets[unique_old_sets != null]
            if len(valid_old_sets) > 0:
                old_sets = [Set(int(set_val.item())) for set_val in valid_old_sets]
                sets_to_cache.extend(old_sets)
        self.tensor[indices, TF.SET] = to_set
        self.cache.cache_sets(sets_to_cache)
    
    def copy_tokens(self, indices: torch.Tensor, to_set: Set) -> torch.Tensor:
        """
        Copy the tokens at the given indices to the given set.
        Args:
            indices: torch.Tensor - The indices of the tokens to copy.
            to_set: Set - The set to copy the tokens to.
        Returns:
            torch.Tensor - The indices of the tokens that were replaced.
        """
        logger.info(f"Copying {len(indices)} tokens to set: {to_set}")
        logger.debug(f"Indices: {indices}")
        tensor = self.tensor[indices, :].clone()
        # Update SET field to the target set
        tensor[:, TF.SET] = to_set
        replace_idxs = self.add_tokens(tensor, [self.names[idx.item()] for idx in indices])
        return replace_idxs
    
    def get_mapped_pos(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Get the indices of the POs that are mapped to.
        Args:
            idxs: torch.Tensor - The indices of the tokens to get the mapped POs of.
        Returns:
            torch.Tensor - The indices of the mapped POs.
        """
        pos = self.tensor[idxs, TF.TYPE] == Type.PO
        mapped_pos = self.tensor[idxs, TF.MAX_MAP] > 0.0
        mapped_pos = tOps.sub_union(pos, mapped_pos)
        return torch.where(mapped_pos)[0]
    
    def get_ref_string(self, idx: int) -> str:
        """
        Get the string to reference a token at the given index.
        Args:
            idx: int - The index of the token to get the string representation of.
        Returns:
            str - The string representation of the token.
        """
        return f"{self.tensor[idx, TF.SET]}[{idx}]"
    
    def get_name(self, idx: int) -> str:
        """
        Get the name of the token at the given index.
        Args:
            idx: int - The index of the token to get the name of.
        Returns:
            str - The name of the token.
        """
        return self.names[idx]
    
    def set_name(self, idx: int, name: str):
        """
        Set the name of the token at the given index.
        Args:
            idx: int - The index of the token to set the name of.
            name: str - The name to set the token to.
        """
        self.names[idx] = name
    
    def get_tokens_where(self, feature: TF, value: float, indices: torch.Tensor, local=False) -> torch.Tensor:
        """
        Get the indices of the tokens that have the given feature and value.
        
        Args:
            feature: TF - The feature to check.
            value: float - The value to match.
            indices: torch.Tensor - The indices to check within the tensor.
            local: bool - Whether to return indices into the input indices tensor or the global tensor.
        
        Returns:
            torch.Tensor - The indices of the tokens that match.
        """
        matching_tokens = (self.tensor[indices, feature] == value)
        if not torch.any(matching_tokens):
            return torch.tensor([], dtype=torch.long)  # Return empty tensor (No matching tokens)
        
        if local:
            return torch.where(matching_tokens)[0]
        else:
            matching_token_indices = indices[matching_tokens]
            return matching_token_indices
    
    def get_tokens_where_not(self, feature: TF, value: float, indices: torch.Tensor, local=False) -> torch.Tensor:
        """
        Get the indices of the tokens that do not have the given feature and value.
        Args:
            feature: TF - The feature to check.
            value: float - The value to match.
            indices: torch.Tensor - The indices to check within the tensor.
            local: bool - Whether to return indices into the input indices tensor or the global tensor.
        """
        matching_tokens = (self.tensor[indices, feature] != value)
        if not torch.any(matching_tokens):
            return torch.tensor([], dtype=torch.long)  # Return empty tensor (No matching tokens)
        
        if local:
            return torch.where(matching_tokens)[0]
        else:
            matching_token_indices = indices[matching_tokens]
            return matching_token_indices
    
    def get_string(self, cols_per_table: int = 16) -> str:
        """
        Get the string representation of the tensor.
        Returns a formatted table with all tokens and their features,
        with values converted based on their types (enums to labels, bools to True/False).
        
        Args:
            cols_per_table: Optional[int] - Number of feature columns per table.
                          If None or 0, shows all columns in a single table.
                          If specified, splits output into multiple tables.
                          The name/index column is always included in each table.
        """
        if self.tensor.size(0) == 0:
            return "Empty tensor"
        
        # Get only non-deleted tokens
        non_deleted_mask = self.tensor[:, TF.DELETED] == B.FALSE
        non_deleted_indices = torch.where(non_deleted_mask)[0]
        
        if len(non_deleted_indices) == 0:
            return "No active tokens (all deleted)"
        
        # Get column headers (TF feature names)
        tf_headers = [tf.name for tf in TF]
        
        # Check if we should use names or indices
        has_names = any(idx.item() in self.names for idx in non_deleted_indices)
        id_header = "Name" if has_names else "Idx"
        
        # Format all row data
        all_row_data = []
        for idx in non_deleted_indices:
            row_data = []
            for tf in TF:
                value = self.tensor[idx, tf].item()
                formatted_value = self._format_value(tf, value)
                row_data.append(formatted_value)
            
            # Add name or index at the beginning
            if has_names:
                name = self.names.get(idx.item(), "")
                id_value = name if name else str(idx.item())
            else:
                id_value = str(idx.item())
            
            all_row_data.append((id_value, row_data))
        
        # If cols_per_table is None or 0, show all columns
        if cols_per_table is None or cols_per_table <= 0:
            return self._build_table([id_header], tf_headers, all_row_data)
        
        # Split into multiple tables
        all_tables = []
        num_tf_cols = len(tf_headers)
        
        for start_idx in range(0, num_tf_cols, cols_per_table):
            end_idx = min(start_idx + cols_per_table, num_tf_cols)
            chunk_headers = tf_headers[start_idx:end_idx]
            
            # Create row data for this chunk (include id column)
            chunk_rows = []
            for id_value, row_data in all_row_data:
                chunk_row_data = [id_value] + row_data[start_idx:end_idx]
                chunk_rows.append(chunk_row_data)
            
            # Build table for this chunk
            table = self._build_table([id_header], chunk_headers, 
                                     [(row[0], row[1:]) for row in chunk_rows])
            all_tables.append(table)
            
            # Add separator between tables (except after last one)
            if end_idx < num_tf_cols:
                all_tables.append("")
        
        return "\n".join(all_tables)
    
    def _build_table(self, id_headers: list[str], feature_headers: list[str], 
                     row_data: list[tuple[str, list[str]]]) -> str:
        """
        Build a formatted table from headers and row data.
        
        Args:
            id_headers: List of identifier column headers (e.g., ["Name"] or ["Idx"])
            feature_headers: List of feature column headers
            row_data: List of tuples (id_value, feature_values_list)
        
        Returns:
            str - Formatted table string
        """
        # Combine headers
        headers = id_headers + feature_headers
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for id_value, feature_values in row_data:
            # Check id column width
            col_widths[0] = max(col_widths[0], len(str(id_value)))
            # Check feature column widths
            for i, value in enumerate(feature_values, start=1):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(value)))
        
        # Build the table
        lines = []
        
        # Header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Rows
        for id_value, feature_values in row_data:
            row = [id_value] + feature_values
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            lines.append(row_line)
        
        return "\n".join(lines)
    
    def _format_value(self, feature: TF, value: float) -> str:
        """
        Format a value based on its feature type.
        Args:
            feature: TF - The feature enum
            value: float - The raw tensor value
        Returns:
            str - The formatted value
        """
        # Handle null values
        if value == null:
            return "null"
        
        feature_type = TF_type(feature)
        
        # Handle enum types - convert to label
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
        
        # Handle bool types
        elif feature_type == bool:
            if value == B.TRUE:
                return "True"
            elif value == B.FALSE:
                return "False"
            else:
                return str(bool(value))
        
        # Handle int types
        elif feature_type == int:
            return str(int(value))
        
        # Handle float types
        elif feature_type == float:
            return f"{value:.4f}".rstrip('0').rstrip('.')
        
        # Fallback
        else:
            return str(value)   