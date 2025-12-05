from ...enums import *
from ...utils import tensor_ops as tOps
import torch
from logging import getLogger

logger = getLogger(__name__)

class FieldView:
    """
    A wrapper for field slices that propagates assignments back to the original tensor.
    Used when accessing a specific field from a TensorView (e.g., view[:, :, MappingFields.WEIGHT]).
    """
    def __init__(self, tensor: torch.Tensor, row_indices: torch.Tensor, field_key):
        """
        Initialize a field view.
        Args:
            tensor: The original tensor
            row_indices: The row indices (mapped from view space)
            field_key: The field key (can be a single field or tuple for multi-dimensional access)
        """
        self._tensor = tensor
        self._row_indices = row_indices
        self._field_key = field_key
        # Compute the actual slice for reading
        if isinstance(field_key, tuple):
            self._data = self._tensor[tuple([self._row_indices] + list(field_key))]
        else:
            self._data = self._tensor[self._row_indices, field_key]
        self._shape = self._data.shape
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._data.dtype
    
    @property
    def device(self):
        return self._data.device
    
    def __len__(self):
        """Get the length of the field view."""
        return len(self._data)
    
    def __getitem__(self, key):
        """Get items from the field view."""
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Set items in the field view, propagating back to the original tensor."""
        if isinstance(key, tuple):
            # Multi-dimensional indexing: weight_view[0, 1] = 0.95
            row_key, col_key = key[0], key[1:]
            if isinstance(row_key, int):
                # Single row: weight_view[0, 1] = 0.95
                mapped_row = self._row_indices[row_key].item()
                if isinstance(self._field_key, tuple):
                    # 3D+ tensor: tensor[row, col, field1, field2, ...]
                    self._tensor[(mapped_row,) + col_key + self._field_key[1:]] = value
                else:
                    # 3D tensor: tensor[row, col, field]
                    if len(col_key) == 0:
                        self._tensor[mapped_row, :, self._field_key] = value
                    else:
                        self._tensor[mapped_row, col_key[0], self._field_key] = value
            else:
                # Multiple rows
                if isinstance(row_key, slice):
                    mapped_rows = self._row_indices[row_key]
                elif isinstance(row_key, (list, torch.Tensor)):
                    if isinstance(row_key, list):
                        row_key = torch.tensor(row_key, dtype=torch.long)
                    mapped_rows = self._row_indices[row_key]
                else:
                    mapped_rows = self._row_indices[row_key]
                
                if isinstance(self._field_key, tuple):
                    self._tensor[tuple([mapped_rows] + list(col_key) + list(self._field_key[1:]))] = value
                else:
                    if len(col_key) == 0:
                        self._tensor[mapped_rows, :, self._field_key] = value
                    else:
                        self._tensor[mapped_rows, col_key[0], self._field_key] = value
        else:
            # Single dimension indexing
            if isinstance(key, int):
                mapped_row = self._row_indices[key].item()
                if isinstance(self._field_key, tuple):
                    self._tensor[(mapped_row,) + self._field_key[1:]] = value
                else:
                    self._tensor[mapped_row, :, self._field_key] = value
            elif isinstance(key, slice):
                mapped_rows = self._row_indices[key]
                if isinstance(self._field_key, tuple):
                    self._tensor[tuple([mapped_rows] + list(self._field_key[1:]))] = value
                else:
                    self._tensor[mapped_rows, :, self._field_key] = value
            else:
                if isinstance(key, list):
                    key = torch.tensor(key, dtype=torch.long)
                mapped_rows = self._row_indices[key]
                if isinstance(self._field_key, tuple):
                    self._tensor[tuple([mapped_rows] + list(self._field_key[1:]))] = value
                else:
                    self._tensor[mapped_rows, :, self._field_key] = value
        # Refresh the data slice to reflect changes
        if isinstance(self._field_key, tuple):
            self._data = self._tensor[tuple([self._row_indices] + list(self._field_key))]
        else:
            self._data = self._tensor[self._row_indices, self._field_key]
    
    def item(self):
        """Get a scalar value (for single-element views)."""
        return self._data.item()
    
    def clone(self):
        """Create a copy of the field view data."""
        return self._data.clone()

class TensorView:
    """
    A view-like wrapper for a subset of a tensor that maps operations back to the original tensor.
    This allows non-contiguous indices to be accessed as if they were a contiguous tensor,
    with all modifications propagating back to the original tensor.
    """
    def __init__(self, tensor: torch.Tensor, indices: torch.Tensor):
        """
        Initialize a tensor view.
        Args:
            tensor: torch.Tensor - The original tensor to create a view of.
            indices: torch.Tensor - The indices into the first dimension of the tensor.
        """
        self._tensor = tensor
        self._indices = indices
        self._shape = (len(indices),) + tensor.shape[1:]

    def to_local(self, idxs):
        """
        Convert global indicies to local (set view) indicies
        
        Args:
            idxs: torch.Tensor - Global indices (positions in the original tensor)
        
        Returns:
            torch.Tensor - Local indices (positions in the view where these global indices appear)
        
        Raises:
            ValueError: If any global index is not found in the view
        """
        # Convert to tensor if needed
        if not isinstance(idxs, torch.Tensor):
            idxs = torch.tensor(idxs, dtype=torch.long, device=self._indices.device)
        
        # Handle empty input
        if len(idxs) == 0:
            return torch.tensor([], dtype=torch.long, device=self._indices.device)
        
        # Use broadcasting to find matches: (len(self._indices), len(idxs))
        matches = (self._indices.unsqueeze(1) == idxs.unsqueeze(0))
        
        # Check which indices actually have matches
        has_match = matches.any(dim=0)
        
        # Check for invalid indices
        invalid_mask = ~has_match
        if invalid_mask.any():
            invalid_indices = idxs[invalid_mask].tolist()
            logger.error(f"Invalid global indices provided to to_local: {invalid_indices}. "
                        f"These indices are not present in the view. View contains indices: {self._indices.tolist()}")
            raise ValueError(f"Invalid global indices: {invalid_indices}. These indices are not present in the view.")
        
        # Find the first occurrence of each global index in self._indices
        # Convert boolean to float for argmax (True=1.0, False=0.0)
        # argmax returns the first True value (first 1.0)
        matches_float = matches.float()
        local_positions = matches_float.argmax(dim=0)
        
        return local_positions

    def to_global(self, idxs):
        """
        Convert local indicies to global indicies
        
        Args:
            idxs: torch.Tensor or int - Local indices (positions in the view)
        
        Returns:
            torch.Tensor - Global indices (positions in the original tensor)
        
        Raises:
            IndexError: If any local index is out of bounds for the view
        """
        # Convert to tensor if needed, ensuring it's 1-d
        if not isinstance(idxs, torch.Tensor):
            # If it's a single int, wrap it in a list to create 1-d tensor
            if isinstance(idxs, int):
                idxs = torch.tensor([idxs], dtype=torch.long, device=self._indices.device)
            else:
                idxs = torch.tensor(idxs, dtype=torch.long, device=self._indices.device)
        
        # Ensure it's 1-d (handle 0-d tensors)
        if idxs.dim() == 0:
            idxs = idxs.unsqueeze(0)
        
        # Check for out-of-bounds indices
        view_size = len(self._indices)
        if (idxs < 0).any() or (idxs >= view_size).any():
            invalid_mask = (idxs < 0) | (idxs >= view_size)
            invalid_indices = idxs[invalid_mask].tolist()
            logger.error(f"Invalid local indices provided to to_global: {invalid_indices}. "
                        f"View size is {view_size}, valid indices are 0 to {view_size - 1}")
            raise IndexError(f"Invalid local indices: {invalid_indices}. View size is {view_size}, "
                           f"valid indices are 0 to {view_size - 1}")
        
        # Simply index into self._indices to get the global indices
        result = self._indices[idxs]
        # Ensure result is always 1-d (handle case where single element might return 0-d)
        if result.dim() == 0:
            result = result.unsqueeze(0)
        return result
    
    @property
    def shape(self):
        """Get the shape of the view."""
        return self._shape
    
    @property
    def dtype(self):
        """Get the dtype of the view."""
        return self._tensor.dtype
    
    @property
    def device(self):
        """Get the device of the view."""
        return self._tensor.device
    
    def __len__(self):
        """Get the length of the view (first dimension)."""
        return len(self._indices)
    
    def __getitem__(self, key):
        """
        Get items from the view. Supports:
        - Integer indexing: view[0]
        - Slice indexing: view[0:5]
        - Tensor/List indexing: view[[0, 2, 3]]
        - Multi-dimensional indexing: view[0, TF.ACT] or view[[0, 1], [TF.ACT, TF.SET]]
        """
        if isinstance(key, tuple):
            # Multi-dimensional indexing
            row_key, col_key = key[0], key[1:]
            
            # Map row indices from view space to original tensor space
            if isinstance(row_key, int):
                mapped_row = self._indices[row_key].item()
            elif isinstance(row_key, slice):
                mapped_rows = self._indices[row_key]
            elif isinstance(row_key, (list, torch.Tensor)):
                if isinstance(row_key, list):
                    row_key = torch.tensor(row_key, dtype=torch.long)
                # Handle boolean masks - self._indices[row_key] works for both bool and int
                mapped_rows = self._indices[row_key]
            else:
                raise TypeError(f"Unsupported row index type: {type(row_key)}")
            
            # Access the original tensor
            if isinstance(row_key, int):
                if len(col_key) == 0:
                    return self._tensor[mapped_row]
                elif len(col_key) == 1:
                    return self._tensor[mapped_row, col_key[0]]
                else:
                    return self._tensor[(mapped_row,) + col_key]
            else:
                if len(col_key) == 0:
                    return self._tensor[mapped_rows]
                elif len(col_key) == 1:
                    # For 2D tensors, return the tensor slice directly
                    # For 3D+ tensors, use FieldView to handle field access
                    if len(self._tensor.shape) == 2:
                        # 2D tensor (e.g., token tensor) - return tensor slice directly
                        return self._tensor[mapped_rows, col_key[0]]
                    else:
                        # 3D+ tensor (e.g., mapping tensor) - use FieldView
                        return FieldView(self._tensor, mapped_rows, col_key[0])
                else:
                    # For 3D+ tensors with field access (e.g., view[:, :, MappingFields.WEIGHT])
                    # Always use FieldView to ensure assignments propagate back
                    return FieldView(self._tensor, mapped_rows, col_key)
        else:
            # Single dimension indexing - returns another view
            if isinstance(key, int):
                mapped_idx = self._indices[key].item()
                return self._tensor[mapped_idx]
            elif isinstance(key, slice):
                mapped_indices = self._indices[key]
                return TensorView(self._tensor, mapped_indices)
            elif isinstance(key, (list, torch.Tensor)):
                if isinstance(key, list):
                    key = torch.tensor(key, dtype=torch.long)
                # Handle boolean masks
                if key.dtype == torch.bool:
                    mapped_indices = self._indices[key]
                else:
                    mapped_indices = self._indices[key]
                return TensorView(self._tensor, mapped_indices)
            else:
                raise TypeError(f"Unsupported index type: {type(key)}")
    
    def __setitem__(self, key, value):
        """
        Set items in the view. Modifications propagate back to the original tensor.
        Supports the same indexing as __getitem__.
        """
        if isinstance(key, tuple):
            # Multi-dimensional indexing
            row_key, col_key = key[0], key[1:]
            
            # Map row indices from view space to original tensor space
            if isinstance(row_key, int):
                mapped_row = self._indices[row_key].item()
            elif isinstance(row_key, slice):
                mapped_rows = self._indices[row_key]
            elif isinstance(row_key, (list, torch.Tensor)):
                if isinstance(row_key, list):
                    row_key = torch.tensor(row_key, dtype=torch.long)
                # Handle boolean masks
                if row_key.dtype == torch.bool:
                    mapped_rows = self._indices[row_key]
                else:
                    mapped_rows = self._indices[row_key]
            else:
                raise TypeError(f"Unsupported row index type: {type(row_key)}")
            
            # Set values in the original tensor
            if isinstance(row_key, int):
                if len(col_key) == 0:
                    self._tensor[mapped_row] = value
                elif len(col_key) == 1:
                    self._tensor[mapped_row, col_key[0]] = value
                else:
                    self._tensor[(mapped_row,) + col_key] = value
            else:
                if len(col_key) == 0:
                    self._tensor[mapped_rows] = value
                elif len(col_key) == 1:
                    self._tensor[mapped_rows, col_key[0]] = value
                else:
                    self._tensor[tuple([mapped_rows] + list(col_key))] = value
        else:
            # Single dimension indexing
            if isinstance(key, int):
                mapped_idx = self._indices[key].item()
                self._tensor[mapped_idx] = value
            elif isinstance(key, slice):
                mapped_indices = self._indices[key]
                self._tensor[mapped_indices] = value
            elif isinstance(key, (list, torch.Tensor)):
                if isinstance(key, list):
                    key = torch.tensor(key, dtype=torch.long)
                # Handle boolean masks
                if key.dtype == torch.bool:
                    mapped_indices = self._indices[key]
                else:
                    mapped_indices = self._indices[key]
                self._tensor[mapped_indices] = value
            else:
                raise TypeError(f"Unsupported index type: {type(key)}")
    
    def clone(self):
        """Create a copy of the view's data (not a view)."""
        return self._tensor[self._indices].clone()
    
    def __repr__(self):
        return f"TensorView(shape={self.shape}, indices={self._indices.tolist()})"