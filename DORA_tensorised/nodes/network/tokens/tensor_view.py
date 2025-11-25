from ...enums import *
from ...utils import tensor_ops as tOps
import torch

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
                    return self._tensor[mapped_rows, col_key[0]]
                else:
                    return self._tensor[tuple([mapped_rows] + list(col_key))]
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