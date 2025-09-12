# DORA_tensorised/DORA.py
# Main DORA file for tensorised version

import nodes
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
        pass

    def initialise_network_state(self):
        pass

    def create_firing_order(self):
        pass

    def do_1_to_3(self):
        pass
    
    # ----------------------------[ PHASE SET OPERATIONS ]-------------------------------
    def do_map(self):
        pass

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
    