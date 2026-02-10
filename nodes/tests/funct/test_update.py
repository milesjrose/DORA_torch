from bridge import Bridge, StatePrinter
from nodes.enums import *
from nodes.utils.printer import Printer as NodePrinter
from nodes.network import Network
from nodes.utils.timer import Timer, intervals_table
bridge = Bridge()
from logging import getLogger
logger = getLogger("TEST")

def test_update_recipient():
    """
    Test a series of update operations on the recipient set.
    
    This test:
    1. Loads the same simulation into both old and new implementations
    2. Sets activation on driver POs to drive semantic activation
    3. Updates semantic inputs/acts in both implementations
    4. Updates recipient inputs/acts in both implementations
    5. Compares the resulting states
    """
    # Setup networks
    bridge.old.load_sim("sims/testsim15.py")
    bridge.load_new_from_old()

    new_net: Network = bridge.new.network
    memory = bridge.old.memory

    import basicRunDORA

    match = bridge.compare_states()
    if not match:
        bridge.old.printer.semantics()
        bridge.new.printer.semantics()
        print(new_net.semantics.IDs)
    assert match, "Mismatch after loading sim"
    # Try set the p mode in both networks
    #old
    for myP in memory.Ps:
        myP.get_Pmode()
    #new
    new_net.node_ops.get_pmode()
    assert bridge.compare_states(), "Mismatch after get p mode"

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
    
    assert bridge.compare_states(), "Mismatch after setting driver PO activations"
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
    assert bridge.compare_states(), "Mismatch after update semantic inputs"
    
    # OLD: Get max semantic input and update semantic activations
    max_input = basicRunDORA.get_max_sem_input(memory)
    for semantic in memory.semantics:
        semantic.set_max_input(max_input)
        semantic.update_act()
    
    # NEW: Update semantic inputs and acts
    max_sem_input = new_net.update_ops.get_max_sem_input()
    new_net.semantics.set_max_input(max_sem_input)
    new_net.update_ops.acts_sem()

    assert bridge.compare_states(), "Mismatch after update semantic acts"
    # =====================================================
    # Step 3: Test some update cycles
    # =====================================================

    for i in range(10):
        # Inputs
        # - old
        logger.info(" -------------------------------> UPDATING INPUTS OLD - CYCLE " + str(i))
        memory = basicRunDORA.update_recipient_inputs(
            memory, 
            asDORA=new_net.params.as_DORA, 
            phase_set=new_net.params.phase_set, 
            lateral_input_level=new_net.params.lateral_input_level, 
            ignore_object_semantics=new_net.params.ignore_object_semantics
        )
        # - new
        logger.info(" ------------------------------->UPDATING INPUTS NEW - CYCLE " + str(i))
        new_net.update_ops.inputs(Set.RECIPIENT)
        # Compare
        assert bridge.compare_states(), "Mismatch after update inputs"

        # Acts
        # - old
        logger.info(" -------------------------------> UPDATING ACTS OLD - CYCLE " + str(i))
        gamma = new_net.params.gamma
        delta = new_net.params.delta
        HebbBias = new_net.params.HebbBias
        for Group in memory.recipient.Groups:
            Group.update_act(gamma, delta, HebbBias)
        for myP in memory.recipient.Ps:
            myP.update_act(gamma, delta, HebbBias)
        for myRB in memory.recipient.RBs:
            myRB.update_act(gamma, delta, HebbBias)
        for myPO in memory.recipient.POs:
            myPO.update_act(gamma, delta, HebbBias)

        # - new
        logger.info(" -------------------------------> UPDATING ACTS NEW - CYCLE " + str(i))
        new_net.update_ops.acts(Set.RECIPIENT)
        # Compare
        logger.info(" ===============================> UPDATED ACTS - CYCLE " + str(i))
        assert bridge.compare_states(), "Mismatch after update acts"
            # Try set the p mode in both networks
        #old
        for myP in memory.Ps:
            myP.get_Pmode()
        #new
        new_net.node_ops.get_pmode()
        assert bridge.compare_states(), "Mismatch after get p mode"
        new_net.params.phase_set += 1

def test_update_driver():
    """
    Test a series of update operations on the driver set.
    
    This test:
    1. Loads the same simulation into both old and new implementations
    2. Sets activation on driver Ps to drive input downwards?
    3. Updates semantic inputs/acts in both implementations
    4. Updates recipient inputs/acts in both implementations
    5. Compares the resulting states
    """
    # Setup networks
    bridge.old.load_sim("sims/testsim15.py")
    bridge.load_new_from_old()

    new_net: Network = bridge.new.network
    memory = bridge.old.memory
    import basicRunDORA

    match = bridge.compare_states()
    if not match:
        bridge.old.printer.semantics()
        bridge.new.printer.semantics()
        logger.debug(new_net.semantics.IDs)
    assert match, "States do not match"

    # =====================================================
    # Step 1: Set driver P activations in both networks.
    # =====================================================

    # OLD: Set activations on driver POs (first 2 POs get activation)
    for i, po in enumerate(memory.driver.Ps):
        if i < 2:
            po.act = 0.8
        else:
            po.act = 0.0
    
    # NEW: Set activations on driver POs
    # Get driver P mask and indices
    driver_p_mask = new_net.sets[Set.DRIVER].tensor_op.get_arb_mask({TF.TYPE: Type.P})
    driver_p_indices = new_net.to_global(driver_p_mask.nonzero().squeeze(1), Set.DRIVER)
    
    # Set first 2 POs to 0.8, rest to 0.0
    for i, idx in enumerate(driver_p_indices):
        if i < 2:
            new_net.token_tensor.tensor[idx, TF.ACT] = 0.8
        else:
            new_net.token_tensor.tensor[idx, TF.ACT] = 0.0
    
    assert bridge.compare_states(), "States do not match"
    
    # =====================================================
    # Step 2: Test some update cycles
    # =====================================================
    c = []
    old_timer = Timer()
    new_timer = Timer()
    for i in range(10):
        # Inputs
        # - old
        logger.info(f" -------------------------------> {i} Inputs OLD")
        old_timer.start()
        memory = basicRunDORA.update_driver_inputs(
            memory, 
            asDORA=new_net.params.as_DORA,
            lateral_input_level=new_net.params.lateral_input_level
        )
        old_timer.stop()
        # - new
        logger.info(f" -------------------------------> {i} Inputs NEW")
        new_timer.start()
        new_net.update_ops.inputs(Set.DRIVER)
        new_timer.stop()
        c.append(f"inputs {i}")
        # Compare
        match= bridge.compare_states()
        if not match:
            bridge.old.printer.tokens(ids=[37,38,40,41,48,51,54])
            bridge.new.printer.tokens(ids=[37,38,40,41,48,51,54])
        assert match, "Mismatch after update inputs"

        # Acts
        # - old
        logger.info(f" -------------------------------> {i} Acts OLD")
        gamma = new_net.params.gamma
        delta = new_net.params.delta
        HebbBias = new_net.params.HebbBias
        old_timer.start()
        for Group in memory.driver.Groups:
            Group.update_act(gamma, delta, HebbBias)
        for myP in memory.driver.Ps:
            myP.update_act(gamma, delta, HebbBias)
        for myRB in memory.driver.RBs:
            myRB.update_act(gamma, delta, HebbBias)
        for myPO in memory.driver.POs:
            myPO.update_act(gamma, delta, HebbBias)
        old_timer.stop()
        c.append(f"acts {i}")
        # - new
        logger.info(f" -------------------------------> {i} Acts NEW")
        new_timer.start()
        new_net.update_ops.acts(Set.DRIVER)
        new_timer.stop()
        # Compare
        match= bridge.compare_states()
        assert match, "Mismatch after update acts"
            # Try set the p mode in both networks
        logger.info(f" -------------------------------> {i} Get P mode")
        #old
        for myP in memory.Ps:
            myP.get_Pmode()
        #new
        new_net.node_ops.get_pmode()
        assert bridge.compare_states(), "Mismatch after get p mode"
        new_net.params.phase_set += 1
    intervals_table([old_timer.get_intervals(), new_timer.get_intervals()], comments=c)

def test_update_semantics():
    """ Test the update of the semantics set. """
    # Setup networks
    bridge.old.load_sim("sims/testsim15.py")
    memory = bridge.old.memory
    import basicRunDORA
    from basicRunDORA import get_max_sem_input
    

    # Semantic activation comes from driver PO nodes, so set up some PO activations.

    for i, po in enumerate(memory.driver.POs):
        if i < 2:
            po.act = 0.8
    
    bridge.load_new_from_old()
    network: Network = bridge.new.network

    match = bridge.compare_states()
    assert match, "States do not match"


    # =====================================================
    # Step 2: Test some update cycles
    # =====================================================
    for i in range(10):
        for po_i, po in enumerate(memory.driver.POs):
            if po_i < 2:
                po.act = 0.8
    
        bridge.load_new_from_old()
        network: Network = bridge.new.network

        # Update semantic inputs
        # - old
        logger.info(f" -------------------------------> {i} Inputs OLD")
        ho_sem_act_flow = network.params.ho_sem_act_flow
        retrieval_license = False
        ignore_object_semantics = network.params.ignore_object_semantics
        ignore_memory_semantics = network.params.ignore_memory_semantics
        for semantic in memory.semantics:
            semantic.update_input(
                memory,
                ho_sem_act_flow,
                retrieval_license,
                ignore_object_semantics,
                ignore_memory_semantics,
            )
        logger.info(f" -------------------------------> {i} Inputs NEW")
        # - new
        memory_set = None if network.params.ignore_memory_semantics else network.sets[Set.MEMORY]
        network.semantics.update_input(network.driver(), network.recipient(), memory_set)
        match = bridge.compare_states()
        if not match:
            bridge.old.printer.semantics()
            bridge.new.printer.semantics()
        assert match, f"Inputs {i}, states do not match"
        # Get max semantic input
        logger.info(f" -------------------------------> {i} max inputs OLD")
        # - old
        max_sem_input = get_max_sem_input(memory)
        for semantic in memory.semantics:
            semantic.set_max_input(max_sem_input)
        # - new
        logger.info(f" -------------------------------> {i} max inputs NEW")
        network.update_ops.max_sem_input()

        match = bridge.compare_states()
        if not match:
            bridge.old.printer.semantics()
            bridge.new.printer.semantics()
        assert match, f"Max sem input {i}, states do not match"


        # Update semantic acts
        # - old
        logger.info(f" -------------------------------> {i} Acts OLD")
        for semantic in memory.semantics:
            semantic.update_act()
        # - new
        logger.info(f" -------------------------------> {i} Acts NEW ")
        network.semantics.update_act()
        match = bridge.compare_states()
        if not match:
            bridge.old.printer.semantics()
            bridge.new.printer.semantics()
        assert match, f"Acts {i}, states do not match"

