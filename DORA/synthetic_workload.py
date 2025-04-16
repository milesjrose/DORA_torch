from time import monotonic
from DORA.basicRunDORA import update_driver_inputs, update_recipient_inputs, update_memory_inputs, update_newSet_inputs
from DORA_tensorised.nodes import Params
from DORA_tensorised.nodes.enums import Set
from DORA.basicRunDORA import make_AM, update_same_RB_POs

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

# index all items in memory.
def indexMemory(memory):
    for Group in memory.Groups:
        Group.get_index(memory)
    for myP in memory.Ps:
        myP.get_index(memory)
    for myRB in memory.RBs:
        myRB.get_index(memory)
    for myPO in memory.POs:
        myPO.get_index(memory)
    # returns.
    return memory

def initialize_run(memory):
    # index memory.
    memory = indexMemory(memory)
    # set up driver and recipient.
    memory.driver.Groups = []
    memory.driver.Ps = []
    memory.driver.RBs = []
    memory.driver.POs = []
    memory.recipient.Groups = []
    memory.recipient.Ps = []
    memory.recipient.RBs = []
    memory.recipient.POs = []
    memory = make_AM(memory)
    # initialize .same_RB_POs field for POs.
    memory = update_same_RB_POs(memory)
    # initialize GUI if necessary.
    for myPO in memory.POs:
        myPO.get_weight_length()
    return memory

def update_am(memory, params: Params, timer):
    timer.start_timer()
    update_driver_inputs(memory, params.as_DORA, params.lateral_input_level)
    timer.log_time(Set.DRIVER)
    update_recipient_inputs(memory, params.as_DORA, params.phase_set, params.lateral_input_level, params.ignore_object_semantics)
    timer.log_time(Set.RECIPIENT)
    update_memory_inputs(memory, params.as_DORA, params.lateral_input_level)
    timer.log_time(Set.MEMORY)
    update_newSet_inputs(memory)
    timer.log_time(Set.NEW_SET)

def do_iterations(iterations: int, memory, params: Params, timer: Timer):
    for i in range(iterations):
        update_am(memory, params, timer)






def test_update_inputs(iterations: int, memory, params: Params, print_time = False):
    timer = Timer()
    memory = initialize_run(memory)
    timer.elapsed_start()
    do_iterations(iterations, memory, params, timer)
    elapsed_time = timer.elapsed_end()
    count = memory.get_count()
    average_time = timer.get_average_time()
    times = timer.get_times()
    if timer.get_iterations() != iterations:
        raise ValueError(f"Number of iterations for {Set.DRIVER} is not the same as for {Set.DRIVER}.")
    if print_time:
        print(f"Elapsed time: {elapsed_time}, average time: {average_time}, times: {times}, count: {count}")
    return elapsed_time



