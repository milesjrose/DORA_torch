from bridge import Bridge
from nodes.enums import TF
from logging import getLogger, INFO
logger = getLogger("TEST")
from time import monotonic
nn_log = getLogger("NEW_NET")
on_log = getLogger("OLD_NET")
nn_log.setLevel(INFO)
on_log.setLevel(INFO)



def test_firing():
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
    fire_token = None
    for rb in b.old.memory.RBs:
        if rb.ID == fire_id:
            fire_token = rb
            break
    assert fire_token is not None, "Fire token not found"
    logger.info(f"Firing old token: {fire_token.ID}")
    b.old.network.fire_token(fire_token, 0)
    logger.info(f"Fired old token: {fire_token.ID}")
    logger.info(f"Firing new token: {fire_id}")
    b.new.dora.fire_token(100, idx, inhibitor)
    logger.info(f"Fired new token: {fire_id}")
    b.update_states()
    b.old.printer.tokens()
    b.new.printer.tokens()
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
    b.old.printer.semantics()
    b.new.printer.semantics()
    assert b.compare_logged_states(), "Mismatch after multi time steps"
    assert b.compare_states(), "Mismatch after time step activations"

def test_time_step_activations_loop():
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
    times = [[], []]
    for i in range(220):
        b.new.network.params.phase_set += 1
        t0 = monotonic()
        b.new.dora.time_step_activations()
        times[1].append(monotonic() - t0)
        t1 = monotonic()
        b.old.network.time_step_activations(b.new.network.params.phase_set)
        times[0].append(monotonic() - t1)
        assert b.compare_states(), "Mismatch after time step activations"
    factor = 1000
    for i in range(len(times[0])):
        times[0][i] *= factor
        times[1][i] *= factor
    print(f"Old avg time: {sum(times[0]) / len(times[0])} ms")
    print(f"New avg time: {sum(times[1]) / len(times[1])} ms")
    print(f"Old max time: {max(times[0])} ms")
    print(f"New max time: {max(times[1])} ms")
    print(f"Old min time: {min(times[0])} ms")
    print(f"New min time: {min(times[1])} ms")




    
