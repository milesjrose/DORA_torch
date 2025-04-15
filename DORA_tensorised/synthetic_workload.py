
from .nodes.network import Network
from time import monotonic

def test_update_inputs(iterations: int, network: Network, print_time = False):
    start_time = monotonic()
    for i in range(iterations):
        network.update_inputs_am()
    end_time = monotonic()
    if print_time:
        print(f"Time taken: {end_time - start_time} seconds")
    return end_time - start_time

