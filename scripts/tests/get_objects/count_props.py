# Counts the number of P, RB, PO, and semantic nodes in the symProps.

from DORA_tensorised.nodes.enums import *

def add_name(name, names):
     """ Only adds a name if unique"""
     non_exist = "non_exist"
     if name != non_exist:
        if name not in names:
            names.append(name)

def get_set(prop):
    """
    Gets the set of a property.
    """
    mapping = {
        "driver": Set.DRIVER,
        "recipient": Set.RECIPIENT,
        "memory": Set.MEMORY,
        "new_set": Set.NEW_SET
    }
    return mapping[prop['set']]

def count_props(symProps):
    """
    Counts the number of P, RB, PO, and semantic nodes in the symProps.

    Args:
        symProps (list): A list of symProps.

    Returns:
        counts (dict): A dictionary of the counts of the types.
        total (int): The total number of nodes.
    """

    sets = {}
    for set in Set:
        sets[set] = {}
        for type in Type:
            sets[set][type] = []

    for prop in symProps:
        add_name(prop['name'], sets[get_set(prop)][Type.PROP])
        for rb in prop['RBs']:
            add_name(rb['pred_name'] + "_" + rb['object_name'], sets[get_set(prop)][Type.RB])
            add_name(rb['pred_name'], sets[get_set(prop)][Type.PO])
            add_name(rb['object_name'], sets[get_set(prop)][Type.PO])
            for sem in rb['pred_sem']:
                add_name(sem, sets[Set.SEMANTIC][Type.SEMANTIC])
            for sem in rb['object_sem']:
                add_name(sem, sets[Set.SEMANTIC][Type.SEMANTIC])
    
    counts = {
        Type.PROP: len(sets[Set.DRIVER][Type.PROP]),
        Type.RB: len(sets[Set.DRIVER][Type.RB]),
        Type.PO: len(sets[Set.DRIVER][Type.PO]),
        Type.SEMANTIC: len(sets[Set.SEMANTIC][Type.SEMANTIC])
    }

    total = 0
    for set in Set:
        for type in Type:
            total += len(sets[set][type])

    return counts, total
    




