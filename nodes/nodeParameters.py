class NodeParameters(object):
    def __init__(self, parameters: dict, run_on_iphone: bool):
        """
        Holds parameters used by nodeTensors, to streamline passing parameters to nodes.
        """
        self.firingOrderRule = parameters["firingOrderRule"]
        self.firingOrder = None  # initialized to None.
        self.nodes.asDORA = parameters["asDORA"]
        self.gamma = parameters["gamma"]
        self.delta = parameters["delta"]
        self.eta = parameters["eta"]
        self.HebbBias = parameters["HebbBias"]
        self.lateral_input_level = parameters["lateral_input_level"]
        self.strategic_mapping = parameters["strategic_mapping"]
        self.ignore_object_semantics = parameters["ignore_object_semantics"]
        self.ignore_memory_semantics = parameters["ignore_memory_semantics"]
        self.mag_decimal_precision = parameters["mag_decimal_precision"]
        self.exemplar_memory = parameters["exemplar_memory"]
        self.recent_analog_bias = parameters["recent_analog_bias"]
        self.bias_retrieval_analogs = parameters["bias_retrieval_analogs"]
        self.use_relative_act = parameters["use_relative_act"]
        self.ho_sem_act_flow = parameters[
            "ho_sem_act_flow"
        ]  # allows flow of activation; -1: only from regular semantics to higher-order semantics; 1: only from higher-order semantics to regular semantics; 0: in both directions
        self.tokenize = parameters[
            "tokenize"
        ]  # ekaterina: the parameter for unpacking; if tokenize == True: create two copies of unpacked object in memory bound to two roles in two different analogs; if tokenize == False: create one object bound to two unpacked roles in one analog
        self.remove_uncompressed = parameters[
            "remove_uncompressed"
        ]  # ekaterina: allows to choose whether to delete or to leave the original uncompressed structure from LTM after do_compress()
        self.remove_compressed = parameters[
            "remove_compressed"
        ]  # ekaterina: allows to choose whether to delete or to leave the original compressed structure from LTM after do_unpacking()
        if run_on_iphone:
            self.doGUI = False
        else:
            self.doGUI = parameters["doGUI"]
        self.screen = 0
        self.screen_width = parameters["screen_width"]
        self.screen_height = parameters["screen_height"]
        self.GUI_update_rate = parameters["GUI_update_rate"]
        self.starting_iteration = parameters["starting_iteration"]



