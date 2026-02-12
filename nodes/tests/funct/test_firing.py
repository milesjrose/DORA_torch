from bridge import Bridge
from nodes.enums import TF
def donttest_firing():
    b = Bridge()
    b.old.load_sim("sims/testsim15.py")
    b.old.network.do_1_to_3(mapping=False)
    b.load_new_from_old()
    b.new.dora.set_count_by_rbs()
    b.old.network.set_old_net(b.old)
    b.new.dora.set_new_net(b.new)
    assert b.compare_states(), "Mismatch after do_1_to_3"
    idx = b.new.network.firing_ops.firing_order[0]
    fire_id = b.new.network.node_ops.get_id(idx)
    inhibitor = b.new.network.inhibitor.get_global
    b.new.dora.fire_token(100, idx, inhibitor)
    fire_token = None
    for rb in b.old.memory.RBs:
        if rb.ID == fire_id:
            fire_token = rb
            break
    assert fire_token is not None, "Fire token not found"
    b.old.network.fire_token(fire_token, 0)
    assert b.compare_states(), "Mismatch after fire_token"


def test_time_step_activations():
    b = Bridge()
    b.old.load_sim("sims/testsim15.py")
    b.old.network.do_1_to_3(mapping=False)
    b.load_new_from_old()
    b.new.dora.set_count_by_rbs()
    b.old.network.set_old_net(b.old)
    b.new.dora.set_new_net(b.new)
    assert b.compare_states(), "Mismatch after do_1_to_3"
    idx = 14
    assert idx is not None, "Firing order not found"
    print(f"idx: {idx}")
    fire_id = b.new.network.node_ops.get_id(idx)
    fire_token = None
    for rb in b.old.memory.RBs:
        if rb.ID == fire_id:
            fire_token = rb
            break
    fire_token.act = 1.0
    b.new.network.node_ops.set_tk_value(idx, TF.ACT, 1.0)
    assert b.compare_states(), "Mismatch after set tk value"
    b.new.dora.time_step_activations()
    b.old.network.time_step_activations(0)
    b.old.memory = b.old.network.memory
    assert b.compare_states(), "Mismatch after time step activations"
    b.new.network.params.phase_set = 1
    b.new.dora.time_step_activations()
    b.old.network.time_step_activations(1)
    b.update_states()
    b.old.printer.tokens()
    b.new.printer.tokens()
    assert b.compare_logged_states(), "Mismatch after multi time steps"
    assert b.compare_states(), "Mismatch after time step activations"


    
