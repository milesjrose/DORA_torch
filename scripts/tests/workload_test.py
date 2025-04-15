from DORA_tensorised.synthetic_workload import test_update_inputs as tens_test_update_inputs
from DORA_tensorised.nodes import build_network as tens_build_network
from DORA_tensorised.nodes import Params
from DORA_tensorised.nodes import default_params
from DORA_tensorised.nodes.tests.sims.sim import symProps

from DORA.synthetic_workload import test_update_inputs as oop_test_update_inputs
import DORA.buildNetwork as oop_build
from .get_objects.generate_network import expand_network
from .get_objects.generate_props import generate_props
from time import monotonic
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Or to use a specific device:
    # torch.cuda.set_device(0)  # Use GPU 0
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def test_workload(num_analogs, iterations):
    start_time = monotonic()
    gen_props = generate_props(symProps, num_analogs)
    props_time = monotonic()

    # Build the network
    network = tens_build_network(props=symProps)
    network = expand_network(network, num_analogs)
    network_time = monotonic()
    # Test network time
    tens_time = tens_test_update_inputs(iterations, network)

    # Build the memory
    memory = oop_build.initializeMemorySet()
    currentSym = oop_build.interpretSymfile(gen_props)[0]
    memory = oop_build.buildTheNetwork(currentSym, memory)
    memory_time = monotonic()
    # Test the oop time
    oop_time = oop_test_update_inputs(iterations, memory, params=default_params())

    memory_time = monotonic() - network_time
    network_time = monotonic() - props_time
    props_time = monotonic() - start_time
    # Print the results
    return tens_time, oop_time, props_time, network_time, memory_time

def test_workloads_range(ananlogs, iterations):
    results = []
    for num_analogs in ananlogs:
        tens_time, oop_time, props_time, network_time, memory_time = test_workload(num_analogs, iterations)
        print(f"Analogs: {num_analogs}, Iterations: {iterations}", "Tensorised_time: {:.4f}, OOP_time: {:.4f}".format(tens_time, oop_time))
        results.append((num_analogs, tens_time, oop_time, props_time, network_time, memory_time))
    return results

def save_results(results, filename):
    with open(filename, 'w') as f:
        f.write("results = [")
        for result in results:
            f.write(f"[{result[0]}, {result[1]}, {result[2]}],\n")
        f.write("]")

def log(message):
    with open("test_results/test.log", 'a') as f:
        f.write(message + "\n")

def run_test(start, end, step, iterations):
    ananlogs = list(range(start, end, step))
    log("Running test with start: " + str(start) + ", end: " + str(end) + ", step: " + str(step) + ", iterations: " + str(iterations))
    results = test_workloads_range(start, ananlogs, iterations)
    save_results(results, "test_results/results.txt")
    return results

run_test(500, 1.1, 100, 100)






