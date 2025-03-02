# nodeBuilder.py
# Builds tensorised memory from old memory object

import random, copy, json
import numpy as np
import buildNetwork as bn
import pdb
import nodes as tt
import dataTypes as dt
from nodeEnums import *

def buildNetFromFile(fileName):
    try:
        file = open(fileName, "r")
    except IOError:
        print("\nNo file called", fileName, ".")
        return None
    
    # get props
    simType = ""
    di = {"simType": simType}  # porting from Python2 to Python3
    file.seek(0)  # to get to the beginning of the file.
    exec(file.readline(), di)  # porting from Python2 to Python3
    if di["simType"] == "sym_file":  # porting from Python2 to Python3
        symstring = ""
        for line in file:
            symstring += line
        do_run = True
        symProps = []  # porting from Python2 to Python3
        di = {"symProps": symProps}  # porting from Python2 to Python3
        exec(symstring, di)  # porting from Python2 to Python3
        sym = di["symProps"]  # porting from Python2 to Python3
    elif di["simType"] == "json_sym":  # porting from Python2 to Python3
        # you've loaded a json generated sym file, which means that it's in json format, and thus must be loaded via the json.load() routine.
        # load the second line of the sym file via json.load().
        symProps = json.loads(file.readline())
        sym = symProps
    else:
        print("\nThe sym file you have loaded is formatted incorrectly. \nPlease check your sym file and try again.")
        return None
    
    # build net from props
    sym = bn.interpretSymfile(sym)[0]
    mem = bn.initializeMemorySet()
    mem = bn.buildTheNetwork(sym, mem)
    return mem

class builder(object):
    def __init__(self, mem: dt.memorySet):
        # Memory data
        self.mem = mem
        self.tokenAmounts = [len(self.mem.semantics), 
                             len(self.mem.POs), 
                             len(self.mem.RBs), 
                             len(self.mem.Ps), 
                             len(self.mem.Groups)]
        self.tokenTypes = [dt.POUnit, dt.RBUnit, dt.PUnit, dt.Groups]
        self.sets = ["driver", "recipient", "memory", "newSet"]

        # Intermediate data
        self.IDs = {}                                       # Mapping: node -> ID
        self.names = {}                                     # Mapping: ID -> name
        self.semantics = []                                 # List of all formatted semantics
        self.semDims = {}                                   # Mapping: ID -> dimension (for semantics)
        self.tokens = []                                    # List of all formatted nodes
        self.connections = {}                               # Weighted, directed adjacency list, stores weights and directions of connections
        self.mappings = {}                                  # Directed adjacency list for mappings, stores weights and hypothesis info in buckets

    # TODO: Turn intermediatary format into single object containing tensor data structures
    def tensorise(self):
        memoryTensors = None
        return memoryTensors                                # return tensorMemory object, containing full tensors, and supporting data structures

    # Extract data into intermediatory format
    def formatMemory(self):
        self.identifyNodes()                                # generate IDs for nodes

        # Format semantics
        for sem in self.mem.semantics:
            sm = self.formatSemantic(sem)
            self.semantics.append(sm)
            self.addConnections(sem)

        # Format tokens
        for anumber in range(len(self.mem.analogs)):        # Add one analog at a time
            analog: dt.Analog = self.mem.analogs[anumber]   # for autocomplete
            types = [analog.myPOs, 
                     analog.myRBs, 
                     analog.myPs, 
                     analog.myGroups]
            for tokens in types:                                        
                for token in tokens:                                 
                    tk = self.formatToken(token, anumber)   # Format Token
                    self.tokens.append(tk)   
                    self.addMappings(token)                 # Add Mappings
                    self.addConnections(token)              # Add Connections

    # Returns formatted list of token values
    def formatToken(self, token: dt.TokenUnit, anumber):
        tf = tf
        tk = [None] * len(tf)                       # Empty token            
        # ========        Set shared properties        ========
        self.names[self.IDs.get(token)] = token.name            # Map ID -> name
        # ----------------------[  INTs  ]----------------------
        tk[tf.ID] = self.IDs.get(token)                         # ID
        tk[tf.TYPE] = "TypeNotSet"                              # Type
        tk[tf.SET] = self.sets.index(token.set)                 # Set
        tk[tf.ANALOG] = anumber                                 # analog
        tk[tf.MAX_MAP_UNIT] = self.IDs.get(token.max_map_unit)  # max_map_unit
        tk[tf.MADE_UNIT] = self.IDs.get(token.my_made_unit)     # made_unit
        tk[tf.MAKER_UNIT] = self.IDs.get(token.my_maker_unit)   # maker_unit
        tk[tf.INHIBITOR_THRESHOLD] = token.inhibitorThreshold   # inhibitor_threshold
        # ---------------------[  BOOLs  ]----------------------
        tk[tf.INFERRED] = token.inferred                        # inferred
        tk[tf.RETRIEVED] = token.retrieved                      # retrieved
        tk[tf.COPY_FOR_DR] = token.copy_for_DR                  # copy_for_dr
        tk[tf.COPIED_DR_INDEX] = token.copied_DR_index          # copied_dr_index
        tk[tf.SIM_MADE] = token.sim_made                        # sim_made
        tk[tf.DELETED] = False                                  # isDeleted = False
        # ---------------------[  FLOATS  ]---------------------
        tk[tf.ACT] = token.act                                  # act
        tk[tf.MAX_ACT] = token.max_act                          # max_act
        tk[tf.INHIBITOR_INPUT] = token.inhibitor_input          # inhibitor_input
        tk[tf.INHIBITOR_ACT] = token.inhibitor_act              # inhibitor_act
        tk[tf.MAX_MAP] = token.max_map                          # max_map
        tk[tf.TD_INPUT] = token.td_input                        # td_input
        tk[tf.BU_INPUT] = token.bu_input                        # bu_input
        tk[tf.LATERAL_INPUT] = token.lateral_input              # lateral_input
        tk[tf.MAP_INPUT] = token.map_input                      # map_input
        tk[tf.NET_INPUT] = token.net_input                      # net_input
        # =======         Set per type properties       ========
        match type(token):
            case dt.Groups:                                     # Groups:
                tk[tf.TYPE] = 4                                 #   - token.set
                tk[tf.GROUP_LAYER] = token.group_layer          #   - group layer - INT
            case dt.PUnit:                                      # P:
                tk[tf.TYPE] = 3                                 #   - token.set
                tk[tf.MODE] = token.mode                        #   - p.mode - INT
            case dt.RBUnit:                                     # RB:
                tk[tf.TYPE] = 2                                 #   - token.set
                tk[tf.TIMES_FIRED] = token.timesFired           #   - rb.timesfired - INT
            case dt.POUnit:                                     # PO:
                tk[tf.TYPE] = 1                                 #   - token.set
                tk[tf.SEM_COUNT] = token.semNormalization       #   - po.semNormalizatino - INT
                tk[tf.PRED] = bool(token.predOrObj)             #   - po.predOrObj -> BOOL
                tk[tf.MAX_SEM_WEIGHT] = token.max_sem_weight    #   - max_sem_weight - FLOAT
        return tk
    
    # Return formatted list of semantic values
    def formatSemantic(self, sem: dt.Semantic):
        sf = SemanticFields
        sm = [None] * len(sf)                    # Empty semantic
        ID = self.IDs.get(sem)
        self.semDims[ID] = sem.dimension                        # Add dimension to mapping

        # ---------------------[  FLOATS  ]---------------------
        sm[sf.ID] = self.IDs.get(sem)                           # ID
        sm[sf.AMOUNT] = sem.amount                              # Amount
        sm[sf.MYINPUT] = sem.myinput                            # Input
        sm[sf.MAX_SEM_INPUT] = sem.max_sem_input                # Max_input
        sm[sf.ACT] = sem.act                                    # Act
        # ----------------------[  INTs  ]----------------------
        sm[sf.TYPE] = 0
        match sem.ont_status:                                   # Ont_status: -> (state:0, value:1, SDM:2)
            case "state":               
                sm[sf.ONT_STATUS] = 0
            case "value":               
                sm[sf.ONT_STATUS] = 1
            case "SDM":                 
                sm[sf.ONT_STATUS] = 2
            case _:
                print("Invalid ont_status, ID:", self.IDs.get(sem))
        return sm
    
    # Adds mappings and hypotheses for a given token
    def addMappings(self, token):
        addedMaps = {}

        # Add hypotheses and their corrosponding mapping connection
        for mapHyp in token.mappingHypotheses:              
            mp = []                                         # emtpy mapping
            mapCon = mapHyp.mappingConnection               # get corresponding mapping
            driver = self.IDs[mapCon.driverToken]           # Driver token
            recipient =self.IDs[mapCon.recipientToken]      # recipient token
            mp.append(mapCon.weight)                        # weight
            mp.append(mapHyp.hypothesis)                    # hypothesis
            mp.append(mapHyp.max_hyp)                       # max_hyp
            self.mappings[(driver, recipient)] = mp         # add mapping to list
            addedMaps[mapCon] = True                        # mark mapping as added
        
        # Add any remaining mappings without hypotheses
        # NOTE: need to check when mapping has/hasnt got hypothesis
        for mapCon in token.mappingConnections:        
            if addedMaps[mapCon] != True:              
                mp = []                                     # emtpy mapping
                driver = self.IDs[mapCon.driverToken]       # Driver token
                recipient =self.IDs[mapCon.recipientToken]  # recipient token
                mp.append(mapCon.weight)                    # weight
                mp.append(0.0)                              # hypothesis
                mp.append(0.0)                              # max_hyp
                self.mappings[(driver, recipient)] = mp     # add mapping to list
                addedMaps[mapCon] = True                    # mark mapping as added

    # Adds all connections for token (Only add one direction per node so dont duplicate writes)
    def addConnections(self, token):
        match type(token):                                  # Match token type, as each token stores info differently.
            case dt.POUnit:
                for RB in token.myRBs:                      # PO.myRBs          - connections to tokens RBs
                    self.addCon(token, RB)
                for PO in token.same_RB_POs:                # PO.same_RB_POs    - connections to POs connected to same RB as self
                    self.addCon(token, PO)
                for link in token.mySemantics:              # PO.mySemantics    - Link objects containing connection to my semantics
                    sem = link.mySemantic
                    weight = link.weight
                    self.addCon(token, sem, weight)

            case dt.RBUnit:
                token: dt.RBUnit = token                    # for autocomplete
                for P in token.myParentPs:                  # RB.myParentPs     - connections to tokens Ps
                    self.addCon(token, P)
                for pred in token.myPred:                   # RB.myPred         - connections to my pred unit
                    self.addCon(token, pred)
                for obj in token.myObj:                     # RB.myObj          - connections to my object unit
                    self.addCon(token, obj)
                for P in token.myChildP:                    # RB.myChildP       - connections to my child P unit
                    self.addCon(token, P)
                
            case dt.PUnit:
                token: dt.PUnit = token                     # for autocomplete
                for RB in token.myRBs:                      # P.myRBs           - connections to tokens RBs
                    self.addCon(token, RB)
                for RB in token.myParentRBs:                # P.myParentRBs     - connections to RBs in which I am an argument
                    self.addCon(token, RB)
                for Group in token.myGroups:                # P.myGroups        - connections to groups that I am a part of 
                    self.addCon(Group)
                
            case dt.Groups:
                token: dt.Groups = token                    # for autocomplete
                for group in token.myParentGroups:          # Group.myParentGroups  - connections to groups above me
                    self.addCon(token, group)
                for group in token.myChildGroups:           # Group.myChildGroups   - connections to groups below me
                    self.addCon(token, group)      
                for P in token.myPs:                        # Group.myPs            - connections to my Ps
                    self.addCon(token, P)
                for RB in token.myRBs:                      # Group.myRBs           - connections to my RBs
                    self.addCon(RB)
                for link in token.mySemantics:              # Group.mySemantics     - link objects containing connection to my semantics
                    sem = link.mySemantic
                    weight = link.weight
                    self.addCon(token, sem, weight)
                
            case dt.Semantic:
                token: dt.Semantic = token                  # for autocomplete
                for link in token.myPOs:                    # Semantic.myPOs        - link objects containing connections to my POs
                    sem = link.mySemantic
                    weight = link.weight
                    self.addCon(token, sem, weight)
                for i in range(len(token.semConnect)): 
                    link = token.semConnect[i]              # Semantic.semConnect   - link objects containing connections to other semantics
                    sem = link.mySemantic
                    weight = token.semConnectWeights        # Semantic.semConnectWeights - weights of the semantic-to-semantic connections, stored at same index as the link object in the semConnect list
                    self.addCon(token, sem, weight)
   
    # Creates ID for each node and store in hash map for efficent lookup
    def identifyNodes(self):
        ID = 0
        types = [self.mem.semantics, self.mem.POs, self.mem.RBs, self.mem.Ps, self.mem.Groups]
        for type in types:
            for node in type:
                self.IDs[node] = ID
                ID += 1

    # Takes a pair of nodes, and adds a directed connection from the first to second, with optional weight
    def addCon(self, fromNode, toNode, weight = 1):
        fromID = self.IDs.get(fromNode)
        toID = self.IDs.get(toNode)
        self.connections[(fromID, toID)] = weight
        return

    # Returns 0 if not integer, or val o.w
    def sanitiseInt(self, val):
        if type(val) != int:
            return 0
        return val
    
    # Print formated token in readable way for debugging
    def printNode(self, tk):
        # check for args
        args = None                                 
        if type(tk) == tuple:
            args = tk[1]
            tk = tk[0]
        elif type(tk) == str:
            args = tk

        # set flags
        a, s, sem, tokens = False, False, False, False
        if args != None:
            args = args.lower()
            nodes = self.semantics + self.tokens
            if ("sem" in args):
                nodes = self.semantics
            if ("tok" in args):
                nodes = self.tokens
            if ("sort" in args):
                nodes = sorted(nodes, key=lambda x: x[1])
            # print nodes
            for node in nodes:
                self.printNode(node)
            return
        
        # Labels for properties
        vLabels = [""] * len(labels)
        vLabels[SemanticFields.TYPE] = Type(tk[SemanticFields.TYPE])
        if tk[1] == 0:
            labels = SemanticFields
            vLabels[SemanticFields.ONT_STATUS] = OntStatus(tk[SemanticFields.ONT_STATUS])
            token = False
        else:
            labels = TF
            vLabels[TF.SET] = Set(tk[TF.SET])
            vLabels[TF.MODE] = Mode(tk[TF.MODE])
            token = True

        # generate all values in strings
        values = []
        for i in range(len(labels)):
            values.append(str(tk[i]))
        
        # Header
        #   Columns
        if token:
            ntype   = "Token"
            name    = f" Name : {str(self.names[tk[0]])} "
        else:
            ntype   = "Semantic"
            name    = f" Dimension : {str(self.semDims[tk[0]])} "
        ID = f" ID : {str(tk[0]).ljust(4)}"
        #   Strings
        headerTop =     ("╒" + ("═" * len(ntype)) + "╤" + ("═" * len(ID)) + "╤" + ("═" * len(name)) + "╕")
        headerInfo =    ("│" + ntype              + "│" + ID              + "│" +  name             + "│")
        headerBottom =  ("╘" + ("═" * len(ntype)) + "╧" + ("═" * len(ID)) + "╧" + ("═" * len(name)) + "╛")
        
        # Body
        # index | label | value | label
        maxName = max(len(e.name) for e in labels)
        labelpad = int((maxName - len(" Label ")) / 2) + 2
        top     = ("┌" + ("─" * len(" Ind ")) + "┬" + ("─" * len((" " * labelpad) + " Label " + (" " * labelpad))) + "┬" + ("─" * len(" Value ")) + "┬" + ("─" * len(" Label ")) + "┐")
        columns = ("│" + " Ind "              + "│" + (" " * labelpad)            + " Label " + (" " * labelpad)   + "│" + " Value "              + "│" + " Label "              + "│")
        subhead = ("├" + ("─" * len(" Ind ")) + "┼" + ("─" * len((" " * labelpad) + " Label " + (" " * labelpad))) + "┼" + ("─" * len(" Value ")) + "┼" + ("─" * len(" Label ")) + "┤")
        bottom  = ("└" + ("─" * len(" Ind ")) + "┴" + ("─" * len((" " * labelpad) + " Label " + (" " * labelpad))) + "┴" + ("─" * len(" Value ")) + "┴" + ("─" * len(" Label ")) + "┘")
        
        # Print all
        print(headerTop)
        print(headerInfo)
        print(headerBottom)
        print(top)
        print(columns)
        print(subhead)
        for i in range(len(labels)):
            print(f"│{str(i).ljust(5)}│ {labels(i).name.ljust((labelpad * 2) + 6)}│ {str(tk[i]).ljust(5)} │ {vLabels[i].ljust(6)}│")
        print(bottom)
        
    # Print builder info for debugging
    def print(self):
        print("mem", self.mem)
        # mem details to print number of each node
        # nodes per set

        # builder details to print: number of each node in tokens, number of connections, number of mappings
        # nodes per set