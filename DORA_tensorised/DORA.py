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

    # ----------------------------[ INITIALISING ]-------------------------------
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
    """
    
    def do_map(self):
        """ 4). Enter the phase set. """
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
        for phase_set in range(phase_sets):
            params.phase_set = phase_set
            # TODO: Check if this actually changes the inhibitor, or just reads the current value (pretty sure its the latter)
            #       idk how to do pointers in python...
            pre_fire_as_DORA = params.as_DORA # Save this to change back after firing.
            if params.count_by_RBs:
                inhibitor = self.network.inhibitor.glbal
            else: # PO
                inhibitor = self.network.inhibitor.local
                params.as_DORA = True # Make sure you are operating as DORA.
            for token in self.network.firing_ops.firing_order:
                # Initialize phase_set_iterator and flags (local_inhibitor_fired).
                phase_set_iterator = 0
                params.local_inhibitor_fired = False
                # 4.1-4.2) Fire the current token in the firingOrder. 
                # Update the network in discrete time-steps until the globalInhibitor fires 
                # (i.e., the current active token is inhibited by its inhibitor).
                while inhibitor == 0:
                    # 4.3.1-4.3.10) update network activations.
                    # NOTE: Maybe have firing ops as ref tokens to avoid this? I dont think this can be vectorised, so dont need indicies really.
                    token = self.network.driver().token_op.get_reference(index=token)
                    self.network.node_ops.set_value(token, TF.ACT, 1.0)
                    self.time_step_activations()
                    # 4.3.11) Update mapping hypotheses. 
                    self.network.mapping_ops.update_mapping_hyps()
                    # 4.3.12) Fire the local_inhibitor if necessary.
                    self.network.inhibitor.check_fire_local() # TODO: Implement accoriding to time_step_fire_local_inhibitor
                    # 4.3.13) Update GUI.
                    phase_set_iterator += 1
                    # TODO: Implement GUI
                # Token firing is over.
                self.post_count_by_operations()
            # Return the .asDORA setting to its pre-firing state.
            params.as_DORA = pre_fire_as_DORA
            # phase set is over.
            self.post_phase_set_operations()
        # If changed dora or ios, change them back.
        params.as_DORA = init_dora
        params.ignore_object_semantics = init_ios

    def do_retrieval(self):
        pass

    def do_retrieval_v2(self):
        pass

    def do_entropy_ops_within(self):
        pass

    def do_entropy_ops_between(self):
        pass

    def do_predication(self):
        pass

    def do_rel_form(self):
        pass

    def do_schematisation(self):
        pass

    def do_rel_gen(self):
        pass
    
    # ----------------------------[ COMPRESSION ]-------------------------------
    def do_compression(self):
        pass

    def collect_the_rest(self):
        pass

    def bind_others_to_unpacked(self):
        pass

    def do_unpacking(self):
        pass

    # ----------------------------[ TIME-STEP OPERATIONS ]-------------------------------
    def time_step_activations(self):
        pass

    def time_step_fire_local_inhibitor(self):
        pass

    def time_step_doGUI(self):
        pass

    # ----------------------------[ POST PHASE SET OPERATIONS ]-------------------------------
    def post_phase_set_operations(self):
        pass

    def post_count_by_operations(self):
        pass

    def do_kludge_comparitor(self):
        pass

    def group_recip_maps(self):
        pass
    