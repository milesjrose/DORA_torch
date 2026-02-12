from bridge import Bridge

def test_firing_order():
    b = Bridge()
    b.load_both("sims/testsim15.py")
    b.old.network.create_firing_order()
    b.new.dora.create_firing_order()
    assert b.compare_states(), "Mismatch after do_1_to_3"