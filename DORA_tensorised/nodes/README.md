# Nodes

The nodes package provides classes for the network used in DORA, implemented using tensors.

#### Structure
- Builder: Classes for building a network of nodes from a sim file.
- Enums: Enums used in encoding set/token/semantic information
- Network:
	- Network: Contain the sets of nodes, and provide access to network methods
	- Params: Hold parameters for the network
	- Sets: Classes for each set type (Driver, Recipient, etc.)
	- Connections: Classes to represent inter-set connections (Links, Mappings)
	- Single_nodes: 
		- Token/Semantic: Hold a 1D tensor, representing a single node
		- Ref_Token/Ref_Semantic: Hold Set/ID to reference a node in the network
- Tests: Collection of PyTest files
- Utils:
	- Printer: Classes for printing/logging tensors.
	- TensorOps: Contains useful tensor operations

## 1). Usage:

### 1.1). Generating a network:

A network can be generated from a sim file or list of props using "build_network".
A parameters file must be provided to run most network functions. This will be set to default values if not provided, but can be set later:

```
network = nodes.build_network(file="sims/testsim.py", params=parameters)
# Or
network = nodes.build_network(props=symProps)
network.set_params(parameters)
```

## 2). Network operations:

All the functions from the basic datatypes have been ported over. After building the network, these can be accesed through network methods:

```
# Update acts in the driver:
network.update_acts(Set.DRIVER)

# Update acts in semantics:
network.update_acts_sem()

# Update acts in driver, recipient, new_set and semantics
network.update_acts_am()
```


It's still possible to access set functions directly, but should be avoided if possible.

```
network.sets[Set.set].function()
or
network[Set.set].function()

# E.g, to update po tokens in the driver:
network.sets[Set.DRIVER].update_input_po()
```

## 3). Nodes:
### 3.1). Node Features
Node features are give by: 
	- tokens: nodes.enums.TF
	- semantics : nodes.enums.SF
Hovering over an enum class reference will show the hover-over documentation listing all features. 
For features that encode a value, for example p_mode, these can be referenced using a matching enum. E.g.,
```
token[TF.MODE] = Mode.PARENT
```

### 3.2). Single Nodes
Nodes can be added to a set using the single node classes. Adding a node returns a reference that can be used to set/get values or remove the token.
```
# Token
token = network.token(Type.PO, {TF.SET = Set.DRIVER})

# Add to set
ref_token = network.add_token(token)

# Set a token value
network.set(ref_token, TF.ACT, 0.9)

# Delete a token
network.del_token(ref_token)
```

## 4). Cuda:
Network performance can be improved by running with CUDA on a GPU. Once the model grows large enough to use up all dedicated vram, the gpu will start using system memory. This ruins performance, and so for models larger than around 10-15k nodes it will likely be better to run on the CPU instead.

To enable CUDA, include the following before running the network:
```
if torch.cuda.is_available():
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	# Or to use a specific device:
	# torch.cuda.set_device(0)  # Use GPU 0
	print(f"Using GPU: {torch.cuda.get_device_name(0)}")
```

## 5). Inter-set connections:
Adding/Removing links and mappings, as well as mapping updates, are not currently implemented. However, links included in a props file or links/mappings that are manually added will affect node updates.