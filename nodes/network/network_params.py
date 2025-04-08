class Params(object):
    def __init__(self, parameters: dict, run_on_iphone: bool):
        """
        Holds parameters used by nodeTensors, to streamline passing parameters to nodes.
        """
        # TODO: Document and organise parameters, move parameters not pertaining to nodes to seperate class.
        # ===================[ PARAMETERS ]======================
        self.firingOrderRule = parameters["firingOrderRule"]
        """firingOrderRule (str): The rule for the firing order."""
        self.firingOrder = None
        """firingOrder (list): The firing order. Initialized to None."""
        self.asDORA = parameters["asDORA"]
        """asDORA (bool): Whether to run as DORA, or in LISA mode if False."""
        self.gamma = parameters["gamma"]
        """gamma (float): Effects the increase in act for each unit."""
        self.delta = parameters["delta"]
        """delta (float): Effects the decrease in act for each unit."""
        self.eta = parameters["eta"]
        """eta (float)"""
        self.HebbBias = parameters["HebbBias"]
        """HebbBias (float): The bias for mapping input relative to TD/BU/LATERAL inputs."""
        self.lateral_input_level = parameters["lateral_input_level"]
        """lateral_input_level (float): The lateral input level."""
        self.strategic_mapping = parameters["strategic_mapping"]
        """strategic_mapping"""
        self.ignore_object_semantics = parameters["ignore_object_semantics"]
        """ignore_object_semantics (bool): Whether to ignore object semantics when updating inputs."""
        self.ignore_memory_semantics = parameters["ignore_memory_semantics"]
        """ignore_memory_semantics (bool): Whether to ignore memory semantics."""
        self.mag_decimal_precision = parameters["mag_decimal_precision"]
        """mag_decimal_precision (int): The decimal precision of the magnitude of the nodes."""
        self.exemplar_memory = parameters["exemplar_memory"]
        """exemplar_memory (bool): Whether to use exemplar memory."""
        self.recent_analog_bias = parameters["recent_analog_bias"]
        """recent_analog_bias"""
        self.bias_retrieval_analogs = parameters["bias_retrieval_analogs"]
        """bias_retrieval_analogs"""
        self.use_relative_act = parameters["use_relative_act"]
        """use_relative_act"""
        self.ho_sem_act_flow = parameters["ho_sem_act_flow"]  
        """allows flow of activation;

          -1: only from regular semantics to higher-order semantics; 
          1: only from higher-order semantics to regular semantics; 
          0: in both directions"""
        self.tokenize = parameters["tokenize"]
        """ekaterina: the parameter for unpacking;

        if tokenize == True: 
            create two copies of unpacked object in memory bound to two roles in two different analogs; 
        if tokenize == False: 
            create one object bound to two unpacked roles in one analog
        """
        self.remove_uncompressed = parameters["remove_uncompressed"] 
        """ekaterina: allows to choose whether to delete or to leave the original uncompressed structure from LTM after do_compress()
        """
        self.remove_compressed = parameters["remove_compressed"]  
        """ekaterina: allows to choose whether to delete or to leave the original compressed structure from LTM after do_unpacking()
        """

        # ==============================[ GUI ]====================================
        if run_on_iphone:
            doGUI = False
        else:
            doGUI = parameters["doGUI"]
        self.doGUI = doGUI
        """doGUI (bool): Whether to display the GUI."""

        self.screen = 0
        """screen (int): The screen number."""

        self.screen_width = parameters["screen_width"]
        """screen_width (int): The width of the screen."""

        self.screen_height = parameters["screen_height"]
        """screen_height (int): The height of the screen."""

        self.GUI_update_rate = parameters["GUI_update_rate"]
        """GUI_update_rate (int): The rate at which the GUI updates."""
        #-------------------------------------------------------------------------


        self.starting_iteration = parameters["starting_iteration"]
        """starting_iteration (int): The starting iteration."""

        self.num_phase_sets_to_run = None
        """num_phase_sets_to_run (int): The number of phase sets to run. Initialized to None."""

        self.count_by_RBs = None
        """count_by_RBs: Initialized to None."""

        self.local_inhibitor_fired = False
        """local_inhibitor_fired (bool): Whether the local inhibitor has fired. Initialized to False."""
