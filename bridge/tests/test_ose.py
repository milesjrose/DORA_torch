from bridge import OldNet

def test_old_net():
    """
    Test the OldStateExtractor class.
    """
    ose = OldNet()
    ose.load_sim("sims/testsim15.py")
    state = ose.get_state()
    ose.save_state("test_data/old_state.json")
    ose.load_state("test_data/old_state.json")
    ose.printer.summary()
    ose.printer.tokens()
    ose.printer.token_data(13)
    assert False

from ..new_net2 import NewNet

def test_new_net():
    """
    Test the NewNet class.
    """
    new_net = NewNet()
    new_net.load_sim("sims/testsim15.py")
    state = new_net.get_state()
    new_net.save_state("test_data/new_state.json")
    new_net.load_state("test_data/new_state.json")
    new_net.printer.summary()
    new_net.printer.tokens()
    new_net.printer.token_data(13)
    assert False