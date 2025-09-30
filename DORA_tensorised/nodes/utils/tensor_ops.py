# tensorOps.py
# Misc tensor operations
import torch

# return MxM matrix, with all ones except the diagonal from T[0, 0] to T[M, M]
def diag_zeros(M):
    """
    Return MxM matrix of all ones except the diagonal from T[0, 0] to T[M, M]
    Args:
        M (int): The size of the matrix
    Returns:
        torch.Tensor: A MxM matrix of all ones except the diagonal from T[0, 0] to T[M, M]
    """
    diag_zeroes = torch.ones((M, M), dtype=torch.float32)       # Create all ones matrix
    diag_zeroes -= torch.eye(M)                                 # Remove ones in diagonal, connecting value in row to each column, exceps same index as itself
    return diag_zeroes


def undirected(T: torch.Tensor):
    """
    Returns the undirected matrix made by OR of both directions of a given matrix T
    Args:
        T (torch.Tensor): The input matrix
    Returns:
        torch.Tensor: The undirected matrix made by OR of both directions of T
    """
    b_tensor = T.bool()
    return(torch.bitwise_or(b_tensor, b_tensor.t()))            # Or a matrix with its transpose, giving matrix that = 1  if [i]->[j] or [j]->[i]

def refine_mask(tensor, mask, index, value, in_place = False):
    """
    Returns a mask, that is the union of mask and the submask where tensor[mask, index] == value
    Args:
        tensor (torch.Tensor): The input tensor
        mask (torch.Tensor): The input mask
        index (int): The index of the value to check
        value (int): The value to check for
        in_place (bool): Whether to modify the input mask in place
    Returns:
        torch.Tensor: Mask(size of input mask) with union of input mask and submask where tensor[mask, index] == value
    """
    submask = (tensor[mask, index] == value)
    return sub_union(mask, submask, in_place)                   # return new mask

def sub_union(mask, submask, in_place = False):
    """
    Returns a mask, that is the union of the input mask and its submask
    Args:
        mask (torch.Tensor): The input mask
        submask (torch.Tensor): The submask
        in_place (bool): Whether to modify the mask in place
    Returns:
        torch.Tensor: Mask(size of input mask) with union of input mask and submask
    """
    if not in_place:
        new_mask = mask.clone().bool()
    else:
        new_mask = mask.bool()
    
    # Ensure submask is boolean
    submask = submask.bool()
    
    true_indices = torch.where(new_mask)[0]
    
    # Ensure submask is not larger than the number of true elements in new_mask
    if submask.shape[0] != true_indices.shape[0]:
        if submask.nelement() == 0 and true_indices.nelement() > 0:
            new_mask[true_indices] = False # no elements from submask are true
            return new_mask
        elif submask.nelement() == 0 and true_indices.nelement() == 0:
            return new_mask # nothing to do
        raise ValueError("sub_union: submask size does not match mask's true elements.")

    new_mask[true_indices] &= submask
    return new_mask

def max_broadcast(a, b):
    """
    Take two tensors, broadcast them to the same shape, and return the maximum of the two.
    Args:
        a (torch.Tensor): The first tensor
        b (torch.Tensor): The second tensor
    Returns:
        torch.Tensor: The maximum of the two tensors
    """
    a_expanded = a.unsqueeze(1)
    
    # Unsqueeze b to be a row vector (1 x n)
    b_expanded = b.unsqueeze(0)
    
    # Use torch.max to perform the broadcasted element-wise maximum
    # The result will be an m x n tensor
    return torch.max(a_expanded, b_expanded)

def efficient_local_max_excluding_self(tensor: torch.Tensor) -> torch.Tensor:
    """
    For each element (i, j) in a 2D tensor, finds the maximum of all other elements
    in row i and column j.

    Args:
        tensor (torch.Tensor): The 2D input tensor.

    Returns:
        torch.Tensor: A tensor where each element (i, j) is the local max
                      excluding the original element (i, j).
    """
    n_rows, n_cols = tensor.shape

    # 1. Find the top two values and indices for each row
    # top_row_vals will have shape (n_rows, 2)
    # top_row_indices will have shape (n_rows, 2)
    top_row_vals, _ = torch.topk(tensor, k=2, dim=1)

    # 2. Find the top two values and indices for each column
    # We transpose, find topk, then transpose back
    top_col_vals, _ = torch.topk(tensor.T, k=2, dim=1)
    top_col_vals = top_col_vals.T

    # 3. Get the #1 max value for each row and column and expand to the original shape
    # These tensors tell us what the absolute max of each row/column is.
    max_val_rows = top_row_vals[:, 0].unsqueeze(1).expand_as(tensor)
    max_val_cols = top_col_vals[0, :].unsqueeze(0).expand_as(tensor)

    # 4. Create a boolean mask where True indicates the element is the max of its row/column
    is_max_in_row = (tensor == max_val_rows)
    is_max_in_col = (tensor == max_val_cols)

    # 5. Select the #2 max where the element is the #1 max, otherwise select the #1 max
    # We use the mask to choose between the first and second largest values.
    row_max_excluding_self = torch.where(
        is_max_in_row,
        top_row_vals[:, 1].unsqueeze(1).expand_as(tensor), # Use 2nd max
        top_row_vals[:, 0].unsqueeze(1).expand_as(tensor)  # Use 1st max
    )

    col_max_excluding_self = torch.where(
        is_max_in_col,
        top_col_vals[1, :].unsqueeze(0).expand_as(tensor), # Use 2nd max
        top_col_vals[0, :].unsqueeze(0).expand_as(tensor)  # Use 1st max
    )

    # 6. Return the maximum of the two resulting tensors
    return torch.max(row_max_excluding_self, col_max_excluding_self)
