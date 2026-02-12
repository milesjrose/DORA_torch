from bridge import Bridge
from nodes.enums import *

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


def test_do_map():
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
    assert b.compare_logged_states(), "Mismatch after do_map"
