from DORA_tensorised.nodes.network import Network
from time import monotonic
from DORA_tensorised.nodes.enums import Set
import torch

class Timer(object):
    def __init__(self):
        self.times = {}
        self.timer = monotonic()
        self.elapsed_timer = monotonic()
        self.use_timer = True
        for set in Set:
            self.times[set] = []
    def log_time(self, set):
        if self.use_timer:
            self.times[set].append(monotonic() - self.timer)
            self.timer = monotonic()
    def start_timer(self):
        self.timer = monotonic() 
    def get_times(self):
        return self.times
    def get_average_time(self):
        return {set: sum(self.times[set]) / len(self.times[set]) for set in Set}
    def get_iterations(self):
        iterations = len(self.times[Set.DRIVER])
        for set in Set:
            if len(self.times[set]) != iterations:
                raise ValueError(f"Number of iterations for {set} is not the same as for {Set.DRIVER}.")
        return iterations
    def elapsed_start(self):
        self.elapsed_timer = monotonic()
    def elapsed_end(self):
        return monotonic() - self.elapsed_timer

def do_iterations(iterations: int, network: Network, timer: Timer):
    for i in range(iterations):
        timer.start_timer()
        for set in Set:
            network.update_inputs(set)
            timer.log_time(set)

def test_update_inputs(iterations: int, network: Network, print_time = False):
    timer = Timer()
    do_iterations(iterations, network, timer)
    count = network.get_count()
    elapsed_time = timer.elapsed_end()
    average_time = timer.get_average_time()
    times = timer.get_times()
    if timer.get_iterations() != iterations:
        raise ValueError(f"Number of iterations for {Set.DRIVER} is not the same as for {Set.DRIVER}.")
    if print_time:
        print(f"Elapsed time: {elapsed_time}, average time: {average_time}, times: {times}, count: {count}")
    return elapsed_time

