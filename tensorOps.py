# tensorOps.py
# Misc tensor operations
import torch

# return MxM matrix, with all ones except the diagonal from T[0, 0] to T[M, M]
def diag_zeros(M):
    diag_zeroes = torch.ones((M, M), dtype=torch.float32)       # Create all ones matrix, size of sub-tensor of p nodes in parent mode
    diag_zeroes -= torch.eye(M)                                 # Remove ones in diagonal to create adj matrix connection connecting parent ps to all but themsel
    return diag_zeroes

# retuns undirected matrix made by OR of both directions
def undirected(T):
    return(torch.bitwise_or(T,  torch.transpose(T)))            # Or a matrix with its transpose, giving matrix that = 1  if [i]->[j] or [j]->[i]

# returns global mask for (mask == True) AND (T[mask,index] == value)
def refine_mask(tensor, mask, index, value, in_place = False):
    submask = (tensor[mask, index] == value)                    # sub-mask
    if in_place:
        mask[mask] &= submask                                   # set in place
        return None                                             # return None
    else:
        mask_copy = mask.copy()                                 # make copy
        mask_copy[mask_copy] &= submask  
        return mask_copy                                        # return new mask