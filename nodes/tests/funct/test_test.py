# expample test file for DORA_bridge

from DORA_bridge import Bridge, StatePrinter
from nodes.enums import *
from nodes.utils.printer import Printer as NodePrinter

bridge = Bridge()

def test_match_new_networks():
    bridge = Bridge()

    old_net = bridge.load_sim_old("sims/testsim15.py")
    new_net = bridge.load_sim_new("sims/testsim15.py")

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
    old_net = bridge.load_sim_old("sims/testsim15.py")
    new_net = bridge.load_sim_new("sims/testsim15.py")

    memory = old_net.memory

    sp = StatePrinter()
    np = NodePrinter()

    import basicRunDORA

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
    # it will only add the link for the first one it finds. So for now add links manually.
    # TODO: Fix this in the builder.
    new_net.links.update_link(2, 0, 1.0)
    new_net.links.update_link(2, 1, 1.0)
    new_net.links.update_link(2, 2, 1.0)

    new_net.links.update_link(8, 6, 1.0)
    new_net.links.update_link(8, 7, 1.0)
    new_net.links.update_link(8, 8, 1.0)

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

    
    # OLD: Get max semantic input and update semantic activations
    max_input = basicRunDORA.get_max_sem_input(memory)
    for semantic in memory.semantics:
        semantic.set_max_input(max_input)
        semantic.update_act()
    
    # NEW: Update semantic inputs and acts
    max_sem_input = new_net.update_ops.get_max_sem_input()
    new_net.semantics.set_max_input(max_sem_input)
    new_net.update_ops.acts_sem()

    # =====================================================
    # Step 3: Update recipient inputs/acts in both networks
    # =====================================================
    
    # Inputs
    asDORA = new_net.params.as_DORA
    phase_set = new_net.params.phase_set
    lateral_input_level = new_net.params.lateral_input_level
    ignore_object_semantics = new_net.params.ignore_object_semantics
    old_net.memory = basicRunDORA.update_recipient_inputs(
        old_net.memory, 
        asDORA=asDORA, 
        phase_set=phase_set, 
        lateral_input_level=lateral_input_level, 
        ignore_object_semantics=ignore_object_semantics
    )

    new_net.update_ops.inputs(Set.RECIPIENT)

    print("\n UPDATED INPUTS")
    printer = StatePrinter()
    print("\n=== OLD STATE ===")
    printer.print_tokens(bridge.get_state_old())
    print("\n=== NEW STATE ===")
    printer.print_tokens(bridge.get_state_new())

    gamma = new_net.params.gamma
    delta = new_net.params.delta
    HebbBias = new_net.params.HebbBias
    
    # Acts
    for Group in memory.recipient.Groups:
        Group.update_act(gamma, delta, HebbBias)
    for myP in memory.recipient.Ps:
        myP.update_act(gamma, delta, HebbBias)
    for myRB in memory.recipient.RBs:
        myRB.update_act(gamma, delta, HebbBias)
    for myPO in memory.recipient.POs:
        myPO.update_act(gamma, delta, HebbBias)

    # NEW: Update recipient inputs and acts
    new_net.update_ops.acts(Set.RECIPIENT)

    # =====================================================
    # Step 4: Compare states
    # =====================================================
    compared = bridge.compare_states()
    
    # Debug: Print states if they don't match
    if not compared['match']:
        printer = StatePrinter()
        print("\n=== OLD STATE ===")
        printer.print_tokens(bridge.get_state_old())
        print("\n=== NEW STATE ===")
        printer.print_tokens(bridge.get_state_new())
        print("\n=== DIFFERENCES ===")
        print(compared)
    
    assert compared['match'], f"States do not match: {compared}"

