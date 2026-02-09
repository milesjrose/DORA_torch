from bridge.compare_states import CompareStates
from bridge.state import State
from bridge.old_net import OldNet
from bridge.new_net import NewNet
from logging import getLogger
logger = getLogger("TEST_COMP")
from nodes.enums import SF

def test_compare_two_old():
    net = OldNet()
    net.load_sim("sims/testsim15.py")
    state = net.get_state()
    compare_states = CompareStates(state, state)
    match, diffs = compare_states.compare()
    assert match, "States do not match"
    compare_states.print_diffs(diffs)


    match, diffs = compare_states.compare()
    assert match, "States do not match"
    compare_states.print_diffs(diffs)

def test_compare_states():
    old_net = OldNet()
    new_net = NewNet()
    # Get states from sims
    old_net.load_sim("sims/testsim15.py")
    old_state = old_net.get_state()
    new_net.set_state(old_state)
    new_net.build_network()
    new_state = new_net.get_state()

    new_net.printer.tokens(ids=[37, 38,41,35,36,38])
    
    old_net.printer.connections(ids=[37, 38,41,35,36,38],get_parents=True)
    new_net.printer.connections(ids=[37, 38,41,35,36,38],get_parents=True)


    compare_states = CompareStates(old_state, new_state)
    match, diffs = compare_states.compare(verbose=False)
    compare_states.print_diffs(diffs, table=True)
    assert match, "States do not match"

def test_compare_states_fail():
    old_net = OldNet()
    new_net = NewNet()
    # Get states from sims
    old_net.load_sim("sims/testsim15.py")
    old_state = old_net.get_state()
    new_net.set_state(old_state)
    new_net.build_network()
    new_net.network.semantics.nodes[0, SF.AMOUNT] = 100
    logger.debug(f"ID: {new_net.network.semantics.nodes[0, SF.ID]}")
    new_state = new_net.get_state()
    compare_states = CompareStates(old_state, new_state)
    match, diffs = compare_states.compare(verbose=False)
    assert not match, "States should not match"
