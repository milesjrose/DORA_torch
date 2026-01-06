# nodes/tests/test_singe_nodes/test_po_pair.py
# Tests for Pairs object

import pytest

from nodes.network.single_nodes import Pairs

def test_add_pairs():
    """test add pairs"""
    pairs = Pairs()
    test_pairs=[
        [1,2],
        [2,1],
        [9,10]
    ]
    for test_pair in test_pairs:
        pairs.add(test_pair[0], test_pair[1])
    
    hashes = [
        12,
        12,
        910
    ]

    print(pairs.pairs)
    print(hashes, test_pairs)

    for i in range(3):
        hsh = hashes[i]
        pr = test_pairs[i]
        pr_pr = pairs.pairs[hsh]
        print(f"{hsh}->{pr} : {pr_pr}")
        assert pr[0] in pr_pr
        assert pr[1] in pr_pr
    
    assert len(pairs.pairs) == 2

def test_get_list():
    """test getting list of pairs"""
    test_pairs = {
        12: (1,2),
        35: (3,5),
        87: (8,7)
    }

    pairs = Pairs()
    pairs.pairs = test_pairs
    p_list = pairs.get_list()
    assert (1,2) in p_list
    assert (3,5) in p_list
    assert (8,7) in p_list
    assert len(p_list) == 3

def test_full():
    pairs = Pairs()
    test_pairs= [
        (1,2),
        (2,1),
        (99,10)
    ]

    for test_p in test_pairs:
        pairs.add(test_p[0], test_p[1])
    
    p_list = pairs.get_list()

    assert p_list == [(2,1), (99,10)]