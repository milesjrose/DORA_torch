import torch
import pandas as pd
from DORA_tensorised.nodes.utils import tensor_ops as tOps

def export_tensor_to_csv(tensor, filename):
    pd.DataFrame(tensor.numpy()).to_csv(filename, index=False, header=False)

# Read CSV files starting from cell A1
# Assuming the CSV files contain 11x16 data matrices
weight_data = pd.read_csv('scripts/weight_data.csv', header=None, index_col=None)
hyp_data = pd.read_csv('scripts/hyp_data.csv', header=None, index_col=None)

# Convert to tensors
weight = torch.tensor(weight_data.values, dtype=torch.float32)
hyp = torch.tensor(hyp_data.values, dtype=torch.float32)

# Ensure the tensors have the expected shape (11, 16)
if weight.shape != (11, 16):
    print(f"Warning: weight tensor shape is {weight.shape}, expected (11, 16)")
if hyp.shape != (11, 16):
    print(f"Warning: hyp tensor shape is {hyp.shape}, expected (11, 16)")

def get_max_hyp(hyp):
    max_recipient = hyp.max(dim=1).values
    max_driver = hyp.max(dim=0).values
    max_values = tOps.max_broadcast(max_recipient, max_driver)
    return max_values


from DORA_tensorised.nodes.utils import nodePrinter
printer = nodePrinter(print_to_file=False)
printer.print_weight_tensor(weight)
printer.print_weight_tensor(hyp, headers=["Hyp"])

# Divisive normalisation
print(f"============================== DIVISIVE NORMALISATION ==============================")
max_hyp = get_max_hyp(hyp)
export_tensor_to_csv(max_hyp, "scripts/max_hyp1.csv")
printer.print_weight_tensor(max_hyp, headers=["Max Hyp"])
norm_hyp = hyp / max_hyp
export_tensor_to_csv(norm_hyp, "scripts/norm_hyp.csv")
printer.print_weight_tensor(norm_hyp, headers=["Norm Hyp"])

# Subtractive normalisation
print(f"============================== SUBTRACTIVE NORMALISATION ==============================")
max_hyp = tOps.efficient_local_max_excluding_self(norm_hyp)
export_tensor_to_csv(max_hyp, "scripts/max_hyp2.csv")
printer.print_weight_tensor(max_hyp, headers=["Max Hyp"])
sub_hyp = norm_hyp - max_hyp
export_tensor_to_csv(sub_hyp, "scripts/sub_hyp.csv")
printer.print_weight_tensor(sub_hyp, headers=["Sub Hyp"])


# Update connections
eta = 0.9

print(f"============================== UPDATE CONNECTIONS ==============================")
print("eta: ", eta)
weight = torch.clamp(
    eta * (1.1 - weight) * sub_hyp, 
    0, 1)
export_tensor_to_csv(weight, "scripts/weight_updated.csv")
printer.print_weight_tensor(weight, headers=["Weight Updated"])


