from DORA_tensorised.synthetic_workload import test_update_inputs as tens_test_update_inputs
from DORA_tensorised.nodes import build_network as tens_build_network
from DORA_tensorised.nodes import Params
from DORA_tensorised.nodes import default_params
from DORA_tensorised.nodes.tests.sims.sim import symProps

from DORA.synthetic_workload import test_update_inputs as oop_test_update_inputs
import DORA.buildNetwork as oop_build
from .get_objects.generate_network import create_large_network
from .get_objects.generate_props import generate_props
from time import monotonic
import torch
import random
import time

cuda = False
gen = False
mem = False
net = False
build_time = False
defer_print = False

def build_network(num_analogs):
    """Build the network"""
    if gen:
        gen_props = generate_props(symProps, num_analogs)
        start_time = monotonic()
        network = tens_build_network(props=gen_props)
        build_time = monotonic() - start_time
    else:
        start_time = monotonic()
        network = tens_build_network(props=symProps)
        network = create_large_network(network, num_analogs)
        build_time = monotonic() - start_time
    return network, network.get_count(), build_time

def build_memory(num_analogs, ):
    """Build the memory"""
    gen_props = generate_props(symProps, num_analogs)
    start_time = monotonic()
    memory = oop_build.initializeMemorySet()
    currentSym = oop_build.interpretSymfile(gen_props)[0]
    memory = oop_build.buildTheNetwork(currentSym, memory)
    build_time = monotonic() - start_time
    return memory, memory.get_count(), build_time

def test_memory(num_analogs, iterations, print_time=False):
    # Build the memory
    memory, memory_count, memory_build_time = build_memory(num_analogs)
    # Test the oop time
    oop_time = oop_test_update_inputs(iterations, memory, params=default_params(), print_time=print_time)
    return oop_time, memory_count, memory_build_time

def test_network(num_analogs, iterations, print_time=False):
    # Build the network
    network, network_count, network_build_time = build_network(num_analogs)
    # Test the tensorised time
    tens_time = tens_test_update_inputs(iterations, network, print_time=print_time)
    return tens_time, network_count, network_build_time

def test_workloads_range(ananlogs, iterations):
    """Run workload test for a range of analogs"""
    mem_results = []
    net_results = []
    if defer_print:
        for num_analogs in ananlogs:
            if mem:
                test_memory(num_analogs, iterations, print_time=True)
            if net:
                test_network(num_analogs, iterations, print_time=True)
    else:
        for num_analogs in ananlogs:
            if mem: 
                oop_time, memory_count, memory_build_time = test_memory(num_analogs, iterations)
                mem_result = (num_analogs, oop_time, memory_build_time, memory_count)
                mem_results.append(mem_result)
            if net:
                tens_time, network_count, network_build_time = test_network(num_analogs, iterations)
                net_result = (num_analogs, tens_time, network_build_time, network_count)
                net_results.append(net_result)
            if net and mem:
                log((f"Analogs: {num_analogs}, mem_time: {mem_result[1]:.4f}, net_time: {net_result[1]:.4f}, mem_count: {mem_result[3]}, net_count: {net_result[3]}"), terminal=True)
            elif mem:
                log((f"Analogs: {num_analogs}, mem_time: {mem_result[1]:.4f}, mem_count: {mem_result[3]}"), terminal=True)
            elif net:
                log((f"Analogs: {num_analogs}, net_time: {net_result[1]:.4f}, net_count: {net_result[3]}"), terminal=True)
    
    return mem_results, net_results
    


def save_results(results, tag):
    """Save the results to a file"""
    with open("scripts/tests/test_results/results.log", 'a') as f:
        f.write(f"Results{tag} = [\n")
        for result in results:
            f.write(f"[{result[0]}, {result[1]}, {result[2]}, {result[3]}],\n")
        f.write("]\n")

def log(message, terminal=False, file=True):
    if terminal:
        print(message)
    if file:
        with open("scripts/tests/test_results/test.log", 'a') as f:
            f.write(message + "\n")

def run_test(start, end, step, iterations, args):
    global cuda, gen, mem, net, build_time, defer_print
    if "cuda" in args:
        cuda = True
    if "gen" in args:
        gen = True
    if "mem" in args:
        mem = True
    if "net" in args:
        net = True
    if "build_time" in args:
        build_time = True
    if "defer_print" in args:
        defer_print = True
    
    if cuda:
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # Or to use a specific device:
            # torch.cuda.set_device(0)  # Use GPU 0
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    ananlogs = list(range(start, end, step))
    tag = time.strftime("_%Y%m%d_%H_%M_%S")
    log((f"{tag}: Running test with start: {start}, end: {end}, step: {step}, iterations: {iterations}"), terminal=True)
    log((f"{tag}: Params: cuda={cuda}, gen={gen}, mem={mem}, net={net}, build_time={build_time}"), terminal=True)
    mem_results, net_results = test_workloads_range(ananlogs, iterations)
    if mem:
        save_results(mem_results, tag + "_mem")
    if net:
        if cuda:
            save_results(net_results, tag + "_net_cuda")
        else:
            save_results(net_results, tag + "_net")

    return mem_results, net_results






