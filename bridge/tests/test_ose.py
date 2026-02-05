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