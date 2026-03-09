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
old_input_log.setLevel(DEBUG)
mem_p_log = getLogger("MEM_P")
mem_p_log.setLevel(INFO)
mem_po_log = getLogger("MEM_PO")
mem_po_log.setLevel(INFO)
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
        # as retreival is probabilistic, we ignore the sets field, and just check everything else.
        # Checking retrieval sets is done manually rn, but should be a better way to do it.
        match = b.compare_states(ignore=["set", "retrieved"]) 
        assert match, "Mismatch after do_retrieval analog bias no relative act"

class TestDoPredication:
    def test_no_map(self):
        """Predication without prior mapping: no mapping connections exist,
        so predication requirements cannot be met and both implementations
        should be no-ops with matching states."""
        b = Bridge()
        b.old.load_sim("sims/testsim_predication.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        b.new.dora.do_predication()
        b.old.network.do_predication()
        print_tokens(b)
        assert b.compare_states(), "Mismatch after do_predication (no prior mapping)"

    def test_no_map_ts15(self):
        """Predication on testsim15 without prior mapping: verifies both
        implementations agree when predication is a no-op on a more complex
        network topology."""
        b = Bridge()
        b.old.load_sim("sims/testsim15.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        b.new.dora.do_predication()
        b.old.network.do_predication()
        print_tokens(b)
        assert b.compare_states(), "Mismatch after do_predication (testsim15, no mapping)"

    def test_post_map(self):
        """Run mapping to establish mapping connections, then run predication.
        Both implementations should produce matching states, whether or not
        predication actually triggers new predicate creation."""
        b = Bridge()
        b.old.load_sim("sims/testsim_predication.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        # Establish mapping connections first
        b.new.dora.do_map()
        b.old.network.do_map()
        assert b.compare_states(), "Mismatch after do_map (pre-predication)"
        # Now run predication on both
        b.new.dora.do_predication()
        b.old.network.do_predication()
        print_tokens(b)
        assert b.compare_states(), "Mismatch after do_predication (post-mapping)"

    def test_post_map_ts15(self):
        """Predication after mapping on testsim15: verifies both implementations
        agree on a network with richer structure and shared predicate semantics."""
        b = Bridge()
        b.old.load_sim("sims/testsim15.py")
        b.old.network.do_1_to_3(mapping=False)
        b.load_new_from_old()
        b.old.network.set_old_net(b.old)
        b.new.dora.set_new_net(b.new)
        b.new.dora.do_map()
        b.old.network.do_map()
        assert b.compare_states(), "Mismatch after do_map (pre-predication, testsim15)"
        b.new.dora.do_predication()
        b.old.network.do_predication()
        print_tokens(b)
        assert b.compare_states(), "Mismatch after do_predication (post-mapping, testsim15)"

class TestDoRelForm:
    def test_do_rel_form(self):
        pass

class TestDoSchematisation:
    def test_do_schematisation(self):
        pass

class TestDoRelGen:
    def test_do_rel_gen(self):
        pass


def print_tokens(b: Bridge):
    b.update_states()
    b.old.printer.tokens()
    b.new.printer.tokens()