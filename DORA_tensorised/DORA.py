# DORA_tensorised/DORA.py
# Main DORA file for tensorised version

import nodes
from nodes.enums import *
import logging
import os


class DORA:
    """
    Main DORA class for tensorised version
    """
    def __init__(self):
        pass
    
    # ----------------------------[ LOADING AND SAVING ]-------------------------------
    """ 
    Handles loading and saving of network to disk. 
    NOTE: Currently only can save as .sym file, not as list of props. 
    """

    def load_sim(self, file_name):
        # default path is ./sims/
        default_path = "./sims/"
        file_path = os.path.join(default_path, file_name)
        # if file is sim.py file, use old load in builder.
        # if file is .sym file, use new load in file_ops.
        if file_name.endswith(".py"):
            self.network = nodes.load_network_old(file_path)
        elif file_name.endswith(".sym"):
            self.network = nodes.load_network_new(file_path)
        else:
            raise ValueError("Invalid file type")
    
    def save_sim(self, file_name):
        # default path is ./sims/
        default_path = "./sims/"
        file_path = os.path.join(default_path, file_name)
        # save as .sym file
        nodes.save_network(self.network, file_path)

    # ----------------------------[ INITIALISATION ]-------------------------------
    """
    Functions for initialising the network state. These are called before each phase_set.
    """

    def initialise_run(self):
        """ 1). Bring a prop or props into WM (driver). """
        #NOTE: Doesn't differentiate between make copy of am and in-place am.
        #      Need to ask about this.
        # Copy any analogs with set != memory into AM (driver/recipient)
        self.network.analog_ops.make_AM_copy()
        # Get PO SemNormalizations.
        self.network.node_ops.get_weight_lengths()

    def initialise_network_state(self):
        """ 2). Initialize activations and inputs of all units to 0. """
        self.network.update_ops.initialise_act()
        self.network.update_ops.initialise_act_memory()
        # NOTE: Think this is the same one thats set in the original code. Not 100% tho.
        self.network.routines.rel_form.inferred_new_p = False

    def create_firing_order(self):
        """ 3). Create the firing order """
        if self.network.driver().tensor_ops.get_count(Type.RB) > 0:
            self.network.params.count_by_RBs = True
            self.network.firing_ops.make_firing_order()
        else:
            self.network.params.count_by_RBs = False
            self.network.firing_ops.totally_random()

    def do_1_to_3(self):
        """ Perform steps 1-3 (initialising the network state) """
        self.initialise_run()
        self.initialise_network_state()
        self.create_firing_order()
    
    # ----------------------------[ PHASE SET OPERATIONS ]-------------------------------
    """
        A phase set is each RB firing at least once (i.e., all RBs in firingOrder firing). 
        It is in phase_sets you will do all of DORA's interesting operations (retrieval, mapping, learning, etc.). 
        There is a function for each interesting operation.

        run_phase_sets is the main function for running the phase sets, 
        with each do_routine function running initialisations and post-routine operations.
    """

    def run_phase_sets(self, routine, phase_sets, firing_order):
        """ Runs the phase sets, for a given routine """
        params = self.network.params
        self.do_1_to_3() # Initialise the network.
        for phase_set in range(phase_sets):
            params.phase_set = phase_set
            self.run_phase_set(routine, firing_order)
            self.post_phase_set_operations()
    
    def run_phase_set(self, routine, firing_order):
        """ Runs a single phase set, for a given routine """
        params = self.network.params
        # Set inhibitor based on count_by_RBs.
        if params.count_by_RBs:
            inhibitor = self.network.inhibitor.glbal
        else: # PO
            inhibitor = self.network.inhibitor.local
            params.as_DORA = True # Make sure you are operating as DORA.
        for token in firing_order:
            self.phase_set_iterator = 0
            params.local_inhibitor_fired = False
            if routine == "predication": # clear inferred_new_p flag
                self.network.routines.predication.inferred_new_p = False
            # Fire token
            self.fire_token(routine, token, inhibitor)
            # Token firing is over. Runs once per token.
            self.post_count_by_operations()
    
    def fire_token(self, routine, token, inhibitor):
        """  
        4.1-4.2) Fire the current token in the firingOrder. 
        Update the network in discrete time-steps until the inhibitor fires 
        (i.e., the current active token is inhibited by its inhibitor).
        """
        token = self.network.driver().token_op.get_reference(index=token) # NOTE: can remove if just make the firing_order a list of references instead of indexes 
        while inhibitor == 0: # TODO: Make this a pointer or smt.
            # 4.3.1-4.3.10) update network activations.
            self.network.node_ops.set_value(token, TF.ACT, 1.0)
            self.time_step_activations()
            # 4.3.11) Run routine
            match routine:
                case "map":
                    self.network.routines.map.map_routine()
                case "retrieval":
                    self.network.routines.retrieval.retrieval_routine()
                case "predication":
                    self.network.routines.predication.predication_routine()
                case "rel_form":
                    self.network.routines.rel_form.rel_form_routine()
                case "schematisation":
                    self.network.routines.schematisation.schematisation_routine()
                case "rel_gen":
                    self.network.routines.rel_gen.rel_gen_routine()
            # fire the local inhib if neccessary
            self.time_step_fire_local_inhibitor()
            #if self.network.params.doGUI: # NOTE: Not implemented
            #    self.phase_set_iterator += 1
            #    self.time_step_doGUI()

    def do_map(self):
        """ do mapping """
        params = self.network.params
        # Initialise memory
        self.network.mapping_ops.reset_mappings()
        self.do_1_to_3()
        phase_sets = 3 
        # If there are multiple relations in drive (P>2), switch to LISA mode, and set ignore_object_semantics to True.
        init_dora = params.as_DORA
        init_ios = params.ignore_object_semantics
        d_p_count = self.network.driver().tensor_ops.get_count(Type.P)
        if params.strategic_mapping and d_p_count >= 2:
            params.as_DORA = False
            params.ignore_object_semantics = True
        # Run phase sets
        self.run_phase_sets("map", phase_sets, self.network.firing_ops.firing_order)
        # If changed dora or ios, change them back.
        params.as_DORA = init_dora
        params.ignore_object_semantics = init_ios
        self.post_phase_set_operations()# note this should be in the run_phase_sets function

    def do_retrieval(self):
        """ do retrieval """
        # initialise network
        self.do_1_to_3()
        params = self.network.params
        init_dora = params.as_DORA
        # Run phase sets
        self.run_phase_set("retrieval", self.network.firing_ops.firing_order)
        # Return the .asDORA setting to its pre-firing state.
        params.as_DORA = init_dora
        # phase set is over.
        self.post_phase_set_operations()

    def do_retrieval_v2(self):
        """ retrieval, but limited to 7 or 4 iterations """
        # initialise network
        self.do_1_to_3()
        params = self.network.params
        # NOTE: original code has for loop over phase sets, but sets to 1 before the loop? idk why.
        init_dora = params.as_DORA
        if self.network.driver().tensor_ops.get_count(Type.RB) == 0:
            params.as_DORA = True   # If firing POs, make sure operating as DORA
            no_iterations = 7
        else:
            no_iterations = 4
        for token in self.network.firing_ops.firing_order:
            params.local_inhibitor_fired = False
            # 4.1-4.2) Fire the current token in the firingOrder. 
            # Update the network in discrete time-steps for no_iterations iterations.
            # The point of allowing only a few time steps is to let the most semantically similar POs get active.
            for i in range(no_iterations): # TODO: Make this a pointer or smt.
                # 4.3.1-4.3.10) update network activations.
                token = self.network.driver().token_op.get_reference(index=token)
                self.network.node_ops.set_value(token, TF.ACT, 1.0)
                self.time_step_activations()
                # 4.3.11) Run retrieval routine
                self.network.routines.retrieval.retrieval_routine()
                # fire the local inhib if neccessary
                self.time_step_fire_local_inhibitor()
                # TODO: Implement GUI
            # Token firing is over.
            self.post_count_by_operations()
        # Return the .asDORA setting to its pre-firing state.
        params.as_DORA = init_dora
        # phase set is over.
        self.post_phase_set_operations()

    def do_predication(self):
        """ do predication """
        params = self.network.params
        init_dora = params.as_DORA
        # must be in DORA mode for predication
        params.as_DORA = True
        # initialise network
        self.do_1_to_3() 
        # set firing order (just POs)
        if not params.count_by_RBs:
            firing_order = self.network.firing_ops.get_random_order_of_type(Type.PO)
        else:
            firing_order = self.network.firing_ops.firing_order
        # Run phase set
        self.run_phase_set("predication", firing_order)
        # Post phase set ops.
        params.as_DORA = init_dora
        self.post_phase_set_operations()
        # reset inferences
        self.network.memory_ops.reset_inferences()
        

    def do_rel_form(self):
        """ do rel form """
        params = self.network.params
        # only execute if count_by_rbs is true:
        if params.count_by_RBs:
            # initialise network
            self.do_1_to_3()
            self.network.routines.rel_form.inferred_new_p = False
            # Run phase set
            self.run_phase_set("rel_form", self.network.firing_ops.firing_order)
            # if_inferred_new_p: TODO: Implement this...
            # post phase set ops
            self.post_phase_set_operations() # NOTE: this passes inferred_new_p as true, but doesn't seem to check in the original code.

    def do_schematisation(self):
        " do schematisation "
        params = self.network.params
        init_dora = params.as_DORA
        params.as_DORA = True
        # only execute if count_by_rbs is true:
        if params.count_by_RBs:
            # intitialise network
            self.do_1_to_3()
            self.run_phase_set("schematisation", self.network.firing_ops.firing_order)
            # post phase set ops
            self.post_phase_set_operations()
        # new_set_items_to_analog: NOTE: Not sure if this is needed if the phase set isn't executed?
        self.network.analog_ops.new_set_to_analog()
        # Return the .asDORA setting to its pre-firing state.
        params.as_DORA = init_dora

    def do_rel_gen(self):
        """do rel gen"""
        params = self.network.params
        init_dora = params.as_DORA
        params.as_DORA = True
        # only execute if count_by_rbs is true:
        if params.count_by_RBs:
            # intitialise network
            self.do_1_to_3()
            self.run_phase_set("rel_gen", self.network.firing_ops.firing_order)
            # post phase set ops
            self.post_phase_set_operations()
        # new_set_items_to_analog: NOTE: Not sure if this is needed if the phase set isn't executed?
        self.network.analog_ops.new_set_to_analog()
        # Return the .asDORA setting to its pre-firing state.
        params.as_DORA = init_dora
    
    # ----------------------------[ ENTROPY OPS ]-------------------------------
    """ NOTE: Not implemented yet """

    def do_entropy_ops_within(self):
        pass

    def do_entropy_ops_between(self):
        pass
    # ----------------------------[ COMPRESSION ]-------------------------------
    """ NOTE: Not implemented yet """

    def do_compression(self):
        pass

    def collect_the_rest(self):
        pass

    def bind_others_to_unpacked(self):
        pass

    def do_unpacking(self):
        pass

    # ----------------------------[ TIME-STEP OPERATIONS ]-------------------------------
    """ NOTE: Not implemented yet """

    def time_step_activations(self):
        pass

    def time_step_fire_local_inhibitor(self):
        pass

    def time_step_doGUI(self):
        pass

    # ----------------------------[ POST PHASE SET OPERATIONS ]-------------------------------
    """ NOTE: Not implemented yet """

    def post_phase_set_operations(self):
        pass

    def post_count_by_operations(self):
        pass

    def do_kludge_comparitor(self):
        pass

    def group_recip_maps(self):
        pass
    