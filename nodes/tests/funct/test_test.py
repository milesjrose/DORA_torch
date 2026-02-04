# expample test file for DORA_bridge

from bridge import Bridge, StatePrinter
from nodes.enums import *
from nodes.utils.printer import Printer as NodePrinter
from nodes.network import Network
bridge = Bridge()

def test_match_new_networks():
    bridge = Bridge()

    bridge.old.load_sim("sims/testsim15.py")
    bridge.new.load_sim("sims/testsim15.py")

    compared = bridge.compare_states()
    print(compared)
    assert compared['match'], "States do not match"
    assert bridge.compare_connections(), "Connections do not match"


def test_update_recipient():
    """
    Test a single update operation on the recipient set.
    
    This test:
    1. Loads the same simulation into both old and new implementations
    2. Sets activation on driver POs to drive semantic activation
    3. Updates semantic inputs/acts in both implementations
    4. Updates recipient inputs/acts in both implementations
    5. Compares the resulting states
    """
    bridge.old.load_sim("sims/testsim15.py")
    bridge.new.load_sim("sims/testsim15.py")

    new_net: Network = bridge.new.network
    memory = bridge.old.memory

    sp = StatePrinter()
    np = NodePrinter()

    import basicRunDORA

    assert compare_states(bridge), "States do not match"
    # =====================================================
    # Step 1: Set driver PO activations in both networks
    # =====================================================
    
    # OLD: Set activations on driver POs (first 2 POs get activation)
    for i, po in enumerate(memory.driver.POs):
        if i < 2:
            po.act = 0.8
        else:
            po.act = 0.0
    
    # NEW: Set activations on driver POs
    # Get driver PO mask and indices
    driver_po_mask = new_net.sets[Set.DRIVER].tensor_op.get_arb_mask({TF.TYPE: Type.PO})
    driver_po_indices = new_net.to_global(driver_po_mask.nonzero().squeeze(1), Set.DRIVER)
    
    # Set first 2 POs to 0.8, rest to 0.0
    for i, idx in enumerate(driver_po_indices):
        if i < 2:
            new_net.token_tensor.tensor[idx, TF.ACT] = 0.8
        else:
            new_net.token_tensor.tensor[idx, TF.ACT] = 0.0
        
    # NOTE: For some reason, we don't generate the links correctly in the new network. 
    # I assume this is because the builder is just going off names, and so when tokens/semantics have the same name in different sets,
    # it will only add the link for the first one it finds.
    # TODO: Fix this in the builder. UPDATED : Now using different builder, so works for now. Still need to fix.
    if False:
        new_net.links.update_link(2, 0, 1.0)
        new_net.links.update_link(2, 1, 1.0)
        new_net.links.update_link(2, 2, 1.0)

        new_net.links.update_link(8, 6, 1.0)
        new_net.links.update_link(8, 7, 1.0)
        new_net.links.update_link(8, 8, 1.0)
    
    assert compare_states(bridge), "States do not match"
    # =====================================================
    # Step 2: Update semantic inputs/acts in both networks
    # =====================================================
    
    # OLD: Update semantic inputs
    for semantic in memory.semantics:
        semantic.update_input(
            memory,
            ho_sem_act_flow=0,  # no higher-order semantic flow
            retrieval_license=False,
            ignore_object_semantics=False,
            ignore_memory_semantics=True
        )
    new_net.update_ops.inputs_sem()
    assert compare_states(bridge), "States do not match"
    
    # OLD: Get max semantic input and update semantic activations
    max_input = basicRunDORA.get_max_sem_input(memory)
    for semantic in memory.semantics:
        semantic.set_max_input(max_input)
        semantic.update_act()
    
    # NEW: Update semantic inputs and acts
    max_sem_input = new_net.update_ops.get_max_sem_input()
    new_net.semantics.set_max_input(max_sem_input)
    new_net.update_ops.acts_sem()

    assert compare_states(bridge), "States do not match"
    # =====================================================
    # Step 3: Update recipient inputs/acts in both networks
    # =====================================================
    # Inputs
    asDORA = new_net.params.as_DORA
    phase_set = new_net.params.phase_set
    lateral_input_level = new_net.params.lateral_input_level
    ignore_object_semantics = new_net.params.ignore_object_semantics
    memory = basicRunDORA.update_recipient_inputs(
        memory, 
        asDORA=asDORA, 
        phase_set=phase_set, 
        lateral_input_level=lateral_input_level, 
        ignore_object_semantics=ignore_object_semantics
    )

    new_net.update_ops.inputs(Set.RECIPIENT)

    assert compare_states(bridge), "States do not match"

    gamma = new_net.params.gamma
    delta = new_net.params.delta
    HebbBias = new_net.params.HebbBias
    
    # OLD: Update recipient acts
    for Group in memory.recipient.Groups:
        Group.update_act(gamma, delta, HebbBias)
    for myP in memory.recipient.Ps:
        myP.update_act(gamma, delta, HebbBias)
    for myRB in memory.recipient.RBs:
        myRB.update_act(gamma, delta, HebbBias)
    for myPO in memory.recipient.POs:
        myPO.update_act(gamma, delta, HebbBias)

    # NEW: Update recipient acts
    new_net.update_ops.acts(Set.RECIPIENT)
    assert compare_states(bridge), "States do not match"


def compare_states(bridge: Bridge):
    bridge.update_states()
    compared = bridge.compare_states()
    match = compared['match']
    if not match:
        diffs = compared['differences']
        names = []
        for diff in diffs:
            names.append(diff[2])
        print("\n=== OLD STATE ===")
        po_names = [l['po_name'] for l in bridge.old.get_state().get('links', {}).get('links_list', []) if l.get('sem_name') in names]
        bridge.old.printer.tokens(names=names+po_names)
        bridge.old.printer.semantics(names=['lover1', 'lover2', 'lover3'])
        bridge.old.printer.links(names=names)
        print("\n=== NEW STATE ===")
        po_names = [l['po_name'] for l in bridge.new.get_state().get('links', {}).get('links_list', []) if l.get('sem_name') in names]
        bridge.new.printer.tokens(names=names+po_names)
        bridge.new.printer.semantics(names=['lover1', 'lover2', 'lover3'])
        bridge.new.printer.links(names=names)
        print("\n=== DIFFERENCES ===")
        bridge.print_diffs(compared['differences'])
    return match

def get_indices(state: dict):
    indices = []
    for type in ['Ps', 'RBs', 'POs']:
        type_tokens = state['tokens'][type]
        for token in type_tokens:
            indices.append(token['index'])
    return sorted(indices)

def map_dict(state, indent=0):
    """"""
    if isinstance(state, dict):
        for key, value in state.items():
            print(f"{' ' * indent}{key}")
            map_dict(value, indent+2)
    elif isinstance(state, list) and len(state) > 0 and isinstance(state[0], dict) and False:
        print(f"{' ' * indent}List:")
        for item in state:
            map_dict(item, indent+2)