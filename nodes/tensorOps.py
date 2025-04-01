# tensorOps.py
# Misc tensor operations
import torch

# return MxM matrix, with all ones except the diagonal from T[0, 0] to T[M, M]
def diag_zeros(M):
    diag_zeroes = torch.ones((M, M), dtype=torch.float32)       # Create all ones matrix
    diag_zeroes -= torch.eye(M)                                 # Remove ones in diagonal, connecting value in row to each column, exceps same index as itself
    return diag_zeroes

# retuns undirected matrix made by OR of both directions
def undirected(T):
    return(torch.bitwise_or(T,  torch.transpose(T)))            # Or a matrix with its transpose, giving matrix that = 1  if [i]->[j] or [j]->[i]

# returns global mask for (mask == True) AND (T[mask,index] == value)
def refine_mask(tensor, mask, index, value, in_place = False):
    submask = (tensor[mask, index] == value)
    return sub_union(mask, submask, in_place)                   # return new mask

# Returns new_mask(size of input mask), of union of mask and its submask
def sub_union(mask, submask, in_place = False):
    if not in_place:
        mask = mask.copy()
    mask[mask] &= submask
    return mask
