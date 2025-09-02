# Load a network from sym file.
from .builder import build_network
from .network import Network
from .enums import Set

def load_network_old(file_path: str):
    """Load a network from old sym file."""
    network = build_network(file_path=file_path)
    return network

def load_network_new(file_path: str):
    """Load a network from new sym file."""
    network = build_network(file_path=file_path)
    return network

def save_network(network: Network, file_path: str):
        """
        Write memory state to symfile. Should probably devise new tensor based file type (e.g. sym.tensors).
        """
        # define json structure for network state

        # network data:
        # - params (dictionary of all parameters from Params object)
        # - inhibitors (local_inhibitor, global_inhibitor floats)
        # - links (dictionary of tensors from Links object: dict[Set, torch.Tensor])
        # - mappings (dictionary of Mappings objects, each containing an adj_matrix tensor)

        # set data (for each set in network.sets and network.semantics):
        # - nodes (torch.Tensor)
        # - connections (torch.Tensor)
        # - IDs (dict[int, int])
        # - names (dict[int, str])

        # write to file -> json, zip into .sym file: allows for single file, and easy loading.
        # file for each set contains nodes, connections, IDs, names (set_nodes.json)
        # file for each set contains mappings (set_maps.json)
        # file for each set links (set_links.json)
        # file for network data [params, inhibitors] (network_data.json)
        # zip into .sym file

        set_jsons = get_set_jsons(network)
        network_data = network_data_json(network)
        zip_files(set_jsons, network_data, file_path)

def get_set_jsons(network: Network):
    """Get json objects for each set."""
    set_jsons = {}
    for set in Set:
        jsons = {}
        jsons["nodes"] = set_nodes_json(network, set)
        jsons["maps"] = set_maps_json(network, set)
        jsons["links"] = set_links_json(network, set)
        set_jsons[set] = jsons
    return set_jsons

def set_nodes_json(network: Network, set: Set):
    """Generate json object for set nodes."""
    pass

def set_maps_json(network: Network, set: Set):
    """Generate json object for set mappings."""
    pass

def set_links_json(network: Network, set: Set):
    """Generate json object for set links."""
    pass

def network_data_json(network: Network):
    """Generate json object for network data."""
    pass

def zip_files(set_jsons: dict[Set, dict], network_data: dict, file_path: str):
    """Zip json files into a .sym file."""
    pass

