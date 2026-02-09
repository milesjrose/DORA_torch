from bridge.new_net import NewNet

def test_new_net():
    """
    Test the NewNet class.
    """
    new_net = NewNet()
    new_net.load_sim("sims/testsim15.py")
    state = new_net.get_state()
    new_net.save_state("test_data/new_state.json")
    new_net.load_state("test_data/new_state.json")
    new_net.set_state(state)
    new_net.build_network()
    