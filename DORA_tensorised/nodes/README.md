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
- Utils:
	- Printer: Classes for printing/logging tensors.
	- TensorOps: Contains useful tensor operations

## 1). Usage:

### 1.1). Generating a network:

A network can be generated from a sim file or list of props using "build_network".
A parameters file must be provided to run most network functions. This can be included in generation, or added later. E.g:

```
network = nodes.build_network(file="sims/testsim.py", params=parameters)
# Or
network = nodes.build_network(props=symProps)
network.set_params(parameters)
```

## 2). Network operations:

### 2.1). Direct access to set functions
All functions from types in dataTypes that are covered in the DORA paper have been ported over, and can be directly accessed by:

```
network.set.function()

# E.g, to update po tokens in the driver:
network.sets[Set.DRIVER].update_input_po()
```

However, to keep the code modular, and allow for easy future modifications to the nodes backend, it is preferred to use the network methods:
###  2.2). Network methods

Set operations can be accessed through network methods, by providing a set, or using "_am"  or "_sem" versions of the function:

```
# Update acts in the driver:
network.update_acts(Set.DRIVER)

# Update acts in semantics:
network.update_acts_sem()

# Update acts in driver, recipient, new_set and semantics
network.update_acts_am()
```

## 2). Nodes:
Nodes are added by using the single node classes. Adding a node returns a reference to later access token information. This can be used to set/get values or remove the token.

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
