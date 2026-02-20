import pdb
from bridge import Bridge
from nodes.enums import *
from logging import getLogger, INFO, DEBUG
logger = getLogger("test")
old_net = getLogger("OLD_NET")
new_net = getLogger("NEW_NET")
old_net.setLevel(INFO)
new_net.setLevel(INFO)
po_log = getLogger("PO_LOG")
po_log.setLevel(INFO)
old_input_log = getLogger("OLD_INPUTS")
old_input_log.setLevel(INFO)
do_log = getLogger("DORA")
do_log.setLevel(DEBUG)
memory_log = getLogger("MEMORY")
memory_log.setLevel(INFO)

def test_firing_order():
    b = Bridge()
    b.load_both("sims/testsim15.py")
    b.old.network.create_firing_order()
    b.new.dora.create_firing_order()
    assert b.compare_states(), "Mismatch after do_1_to_3"
    b.new.network.params.firing_order_rule = "by_top_random"
    b.old.network.firingOrderRule = b.new.network.params.firing_order_rule
    b.old.network.create_firing_order()
    #b.new.dora.create_firing_order()
    #assert b.compare_states(), "Mismatch after do_1_to_3"


class TestDoMap:
    def test_do_map(self):
        b = Bridge()
        b.old.load_sim("sims/testsim15.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        so = b.old.network.get_state()
        sn = b.new.dora.get_state()
        assert b.compare_states_arg(so, sn), "Mismatch after do_1_to_3"
        b.new.dora.do_map()
        b.old.network.do_map()
        b.update_states()
        b.old.printer.mappings()
        b.new.printer.mappings()
        b.old.printer.tokens()
        b.new.printer.tokens()
        assert b.compare_states(), "Mismatch after do_map"

class TestDoRetrieval:
    def test_emtpy_memory(self):
        b = Bridge()
        b.old.load_sim("sims/testsim15.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        b.new.dora.do_retrieval()
        b.old.network.do_retrieval()
        print_tokens(b)
        assert b.compare_states(), "Mismatch after do_retrieval"

    def test_retrieval_analog_bias(self):
        """Test retrieval with memory tokens using analog bias (bias_retrieval_analogs=True)."""
        b = Bridge()
        b.old.load_sim("sims/testsim_retrieval.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        # Ensure analog bias retrieval is enabled (default)
        b.old.network.bias_retrieval_analogs = True
        b.old.network.use_relative_act = True
        b.new.network.params.bias_retrieval_analogs = True
        b.new.network.params.use_relative_act = True
        b.new.dora.do_retrieval()
        b.old.network.do_retrieval()
        print_tokens(b)
        assert b.compare_states(), "Mismatch after do_retrieval with analog bias"

    def test_retrieval_no_analog_bias(self):
        """Test retrieval with memory tokens using token-level retrieval (bias_retrieval_analogs=False)."""
        b = Bridge()
        b.old.load_sim("sims/testsim_retrieval.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        # Disable analog bias to test token-level retrieval
        b.old.network.bias_retrieval_analogs = False
        b.old.network.use_relative_act = False
        b.new.network.params.bias_retrieval_analogs = False
        b.new.network.params.use_relative_act = False
        b.new.dora.do_retrieval()
        b.old.network.do_retrieval()
        print_tokens(b)
        assert b.compare_states(), "Mismatch after do_retrieval without analog bias"

    def test_retrieval_analog_bias_no_relative_act(self):
        """Test retrieval with analog bias but without relative activation transform."""
        b = Bridge()
        b.old.load_sim("sims/testsim_retrieval.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        # Analog bias ON, relative act OFF
        b.old.network.bias_retrieval_analogs = True
        b.old.network.use_relative_act = False
        b.new.network.params.bias_retrieval_analogs = True
        b.new.network.params.use_relative_act = False
        b.new.dora.do_retrieval()
        b.old.network.do_retrieval()
        match, old, new = b.compare_logged_states()
        if not match:
            new_b = Bridge()
            new_b.old.state.copy_from(old)
            new_b.new.state.copy_from(new)
            np = new_b.new.printer
            op = new_b.old.printer
            from nodes.utils import OutputType
            np.output_type = OutputType.PRINT_CONSOLE
            op.output_type = OutputType.PRINT_CONSOLE
            pdb.set_trace()
        assert match, "Mismatch after do_retrieval analog bias no relative act"

def print_tokens(b: Bridge):
    b.update_states()
    b.old.printer.tokens()
    b.new.printer.tokens()