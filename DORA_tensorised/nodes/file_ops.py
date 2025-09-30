# Load a network from sym file.
from .builder import build_network
from .network import Network
from .enums import Set
import torch
import json
import tempfile
import os
import shutil
from .network.network_params import Params
from .network.sets.semantics import Semantics
from .network.sets.driver import Driver
from .network.sets.recipient import Recipient
from .network.sets.memory import Memory
from .network.sets.new_set import New_Set
from .network.connections.mappings import Mappings
from .network.connections.links import Links
from .enums import MappingFields

def load_network_old(file_path: str):
    """Load a network from old sym file."""
    network = build_network(file=file_path)
    return network

def load_network_new(file_path: str):
    """Load a network from new sym file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Unzip the .sym file
        shutil.unpack_archive(file_path, temp_dir, 'zip')
        
        # Reconstruct the network from the unzipped files
        network = read_zipped_files(temp_dir)
        
    return network

def read_zipped_files(source_dir: str):
    """Read network data from unzipped files."""
    # Load network data
    network_data_path = os.path.join(source_dir, "network_data.json")
    with open(network_data_path, 'r') as f:
        network_data = json.load(f)
    
    params = Params(network_data['params'])
    
    # Load semantics
    semantics_dir = os.path.join(source_dir, "semantics")
    semantics_metadata_path = os.path.join(semantics_dir, "semantics_metadata.json")
    with open(semantics_metadata_path, 'r') as f:
        semantics_metadata = json.load(f)
    
    semantics_nodes_path = os.path.join(semantics_dir, semantics_metadata['nodes_file'])
    semantics_connections_path = os.path.join(semantics_dir, semantics_metadata['connections_file'])
    
    semantics_nodes = torch.load(semantics_nodes_path)
    semantics_connections = torch.load(semantics_connections_path)
    
    # Convert keys from string (from JSON) to int
    semantic_ids = {int(k): v for k, v in semantics_metadata['IDs'].items()}
    semantic_names = {int(k): v for k, v in semantics_metadata['names'].items()} if semantics_metadata['names'] is not None else {}

    semantics = Semantics(
        nodes=semantics_nodes,
        connections=semantics_connections,
        IDs=semantic_ids,
        names=semantic_names
    )
    
    # Load sets
    dict_sets = {}
    set_map = {
        Set.DRIVER: Driver,
        Set.RECIPIENT: Recipient,
        Set.MEMORY: Memory,
        Set.NEW_SET: New_Set,
    }
    for s in Set:
        set_dir = os.path.join(source_dir, s.name)
        nodes_metadata_path = os.path.join(set_dir, "nodes_metadata.json")
        with open(nodes_metadata_path, 'r') as f:
            nodes_metadata = json.load(f)
        
        nodes_path = os.path.join(set_dir, nodes_metadata['nodes_file'])
        connections_path = os.path.join(set_dir, nodes_metadata['connections_file'])
        
        set_nodes = torch.load(nodes_path)
        set_connections = torch.load(connections_path)
        
        # Convert keys from string (from JSON) to int
        set_ids = {int(k): v for k, v in nodes_metadata['IDs'].items()}
        set_names = {int(k): v for k, v in nodes_metadata['names'].items()} if nodes_metadata['names'] is not None else {}
        
        set_class = set_map[s]
        dict_sets[s] = set_class(
            nodes=set_nodes,
            connections=set_connections,
            IDs=set_ids,
            names=set_names
        )

    # Load mappings and links
    mappings = {}
    links_tensors = {}
    
    driver_set = dict_sets[Set.DRIVER]

    for s in Set:
        set_dir = os.path.join(source_dir, s.name)
        
        # Load mappings
        if not s == Set.NEW_SET and not s == Set.DRIVER:
            maps_metadata_path = os.path.join(set_dir, "maps_metadata.json")
            with open(maps_metadata_path, 'r') as f:
                maps_metadata = json.load(f)
            
            adj_matrix_path = os.path.join(set_dir, maps_metadata['adj_matrix_file'])
            adj_matrix = torch.load(adj_matrix_path)
            
            map_fields = {field: adj_matrix[:, :, field.value] for field in MappingFields}
            
            mappings[s] = Mappings(driver=driver_set, map_fields=map_fields) 
        
        # Load links
        links_metadata_path = os.path.join(set_dir, "links_metadata.json")
        with open(links_metadata_path, 'r') as f:
            links_metadata = json.load(f)
        
        links_path = os.path.join(set_dir, links_metadata['links_file'])
        links_tensors[s] = torch.load(links_path)

    # Create Links object
    links = Links(links=links_tensors, semantics=semantics)
    
    # Create Network object
    network = Network(dict_sets=dict_sets, semantics=semantics, mappings=mappings, links=links, params=params)
    
    network.inhibitor.local = network_data['inhibitor']['local']
    network.inhibitor.glbal = network_data['inhibitor']['global']

    return network

def save_network(network: Network, file_path: str):
        """
        Write memory state to symfile. Should probably devise new tensor based file type (e.g. sym.tensors).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # save network data
            network_data_json(network, temp_dir)

            # save semantics data
            semantics_dir = os.path.join(temp_dir, "semantics")
            os.makedirs(semantics_dir, exist_ok=True)
            semantics_json(network, semantics_dir)

            # save sets data
            for s in Set:
                set_dir = os.path.join(temp_dir, s.name)
                os.makedirs(set_dir, exist_ok=True)
                set_nodes_json(network, s, set_dir)
                if s != Set.NEW_SET and s != Set.DRIVER:
                    set_maps_json(network, s, set_dir)
                set_links_json(network, s, set_dir)

            # zip files
            zip_files(temp_dir, file_path)


def set_nodes_json(network: Network, s: Set, output_dir: str):
    """Generate json object for set nodes."""
    set_obj = network.sets[s]
    
    # Save tensors
    nodes_path = os.path.join(output_dir, "nodes.pt")
    connections_path = os.path.join(output_dir, "connections.pt")
    torch.save(set_obj.nodes, nodes_path)
    torch.save(set_obj.connections, connections_path)
    
    # Save metadata
    ids = {k: v.item() if torch.is_tensor(v) else v for k, v in set_obj.IDs.items()}
    names = {k: v.item() if torch.is_tensor(v) else v for k, v in set_obj.names.items()} if set_obj.names else None
    metadata = {
        "IDs": ids,
        "names": names,
        "nodes_file": "nodes.pt",
        "connections_file": "connections.pt"
    }
    
    metadata_path = os.path.join(output_dir, "nodes_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def semantics_json(network: Network, output_dir: str):
    """Generate json object for semantics."""
    semantics_obj = network.semantics
    
    # Save tensors
    nodes_path = os.path.join(output_dir, "nodes.pt")
    connections_path = os.path.join(output_dir, "connections.pt")
    torch.save(semantics_obj.nodes, nodes_path)
    torch.save(semantics_obj.connections, connections_path)
    
    # Save metadata
    ids = {k: v.item() if torch.is_tensor(v) else v for k, v in semantics_obj.IDs.items()}
    names = {k: v.item() if torch.is_tensor(v) else v for k, v in semantics_obj.names.items()} if semantics_obj.names else None

    metadata = {
        "IDs": ids,
        "names": names,
        "nodes_file": "nodes.pt",
        "connections_file": "connections.pt"
    }
    
    metadata_path = os.path.join(output_dir, "semantics_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def set_maps_json(network: Network, s: Set, output_dir: str):
    """Generate json object for set mappings."""
    mappings = network.mappings[s]
    adj_matrix_path = os.path.join(output_dir, "adj_matrix.pt")
    torch.save(mappings.adj_matrix, adj_matrix_path)
    
    metadata = {
        "adj_matrix_file": "adj_matrix.pt"
    }
    metadata_path = os.path.join(output_dir, "maps_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def set_links_json(network: Network, s: Set, output_dir: str):
    """Generate json object for set links."""
    links_tensor = network.links[s]
    links_path = os.path.join(output_dir, "links.pt")
    torch.save(links_tensor, links_path)
    
    metadata = {
        "links_file": "links.pt"
    }
    metadata_path = os.path.join(output_dir, "links_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)


def network_data_json(network: Network, output_dir: str):
    """Generate json object for network data."""
    local_inhibitor = network.inhibitor.local
    global_inhibitor = network.inhibitor.glbal

    network_data = {
        "params": network.params.get_params_dict(),
        "inhibitor": {
            "local": local_inhibitor.tolist() if torch.is_tensor(local_inhibitor) else local_inhibitor,
            "global": global_inhibitor.tolist() if torch.is_tensor(global_inhibitor) else global_inhibitor
        }
    }
    
    output_path = os.path.join(output_dir, "network_data.json")
    with open(output_path, 'w') as f:
        json.dump(network_data, f, indent=4)

def zip_files(source_dir: str, output_path: str):
    """Zip json files into a .sym file."""
    if not output_path.endswith('.sym'):
        output_path += '.sym'
    
    # Create .zip
    archive_name = output_path.replace('.sym', '')
    shutil.make_archive(archive_name, 'zip', source_dir)
    
    # Rename the .zip file to .sym
    os.rename(archive_name + '.zip', output_path)

