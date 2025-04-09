# nodes/builder/run_build.py
# Provides a simple function for building the network object.

from nodes.network.network_params import Params
from .network_builder import NetworkBuilder

def build_network(file=None, props=None, params=None):
    """
    Build the nodes object.

    Args:
        file (str): The path to the file containing the symProps.
        props (list): A list of symProps.
        params (dict) (Optional): The parameters for the network.

    Returns:
        network (Network): The network object.
    
    Raises:
        ValueError: If no file or symProps provided.
    """
    if params is not None:
        params = Params(params)
    if file is not None:
        builder = NetworkBuilder(file=file, params=params)
        network = builder.build_nodes()
        return network
    elif props is not None:
        builder = NetworkBuilder(symProps=props, params=params)
        network = builder.build_nodes()
        return network
    else:
        raise ValueError("No file or symProps provided")

