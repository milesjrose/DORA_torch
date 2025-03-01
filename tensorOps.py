# tensorOps.py
# Misc tensor operations
import torch

def diag_zeros(M):
    diag_zeroes = torch.ones((M, M), dtype=torch.float32)   # Create all ones matrix, size of sub-tensor of p nodes in parent mode
    diag_zeroes -= torch.eye(M)                             # Remove ones in diagonal to create adj matrix connection connecting parent ps to all but themsel
    return diag_zeroes