# nodes/builder/run_build.py
# Provides a simple function for building the network object.

from ..network.network_params import Params
from .network_builder import NetworkBuilder
from ..network.network_params import default_params

def build_network(file=None, props=None, params=None):
    """
    Build the nodes object.

    Args:
        file (str): The path to the file containing the symProps.
        props (list): A list of symProps.
        params (dict or Params) (Optional): The parameters for the network.

    Returns:
        network (Network): The network object.
    
    Raises:
        ValueError: If no file or symProps provided.
        ValueError: If invalid parameters provided.
    """
    if params is not None:
        if isinstance(params, dict):
            params = Params(params)
        elif isinstance(params, Params):
            pass
        else:
            raise ValueError("Invalid parameters provided")
    
    if file is not None:
        builder = NetworkBuilder(file_path=file, params=params)
        network = builder.build_network()
        return network
    elif props is not None:
        builder = NetworkBuilder(symProps=props, params=params)
        network = builder.build_network()
        return network
    else:
        raise ValueError("No file or symProps provided")