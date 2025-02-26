# fileHandling.py

import random, copy, json
import numpy as np
import buildNetwork as bn
import pdb
import tensorTypes as tt
import dataTypes as dt

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

class tensorBuilder(object):
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
        self.IDs = {}                   # Mapping: node -> ID
        self.names = {}                 # Mapping: ID -> name
        self.semantics = []             # List of all formatted semantics
        self.tokens = []                # List of all formatted nodes
        self.connections = {}           # Weighted, directed adjacency list, stores weights and directions of connections
        self.mappings = {}              # Directed adjacency list for mappings, stores weights and hypothesis info in buckets

    # TODO: Turn intermediatary format into single object containing tensor data structures
    def tensorise(self):
        memoryTensors = None
        return memoryTensors                                            # return tensorMemory object, containing full tensors, and supporting data structures

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
        tk = []                                             # Empty token            
        # =====        Set shared properties        =====
        self.names[self.IDs.get(token)] = token.name        # Map ID -> name
        # --------------------[  INTs  ]--------------------
        tk.append(self.IDs.get(token))                      # ID
        tk.append("TypeNotSet")                             # Type
        tk.append(self.sets.index(token.set))               # Set
        tk.append(anumber)                                  # analog
        tk.append(self.IDs.get(token.max_map_unit))         # max_map_unit
        tk.append(self.IDs.get(token.my_made_unit))         # made_unit
        tk.append(self.IDs.get(token.my_maker_unit))        # maker_unit
        tk.append(self.IDs.get(token.inhibitorThreshold))   # inhibitor_threshold
        # -------------------[  BOOLs  ]--------------------
        tk.append(token.inferred)                           # inferred
        tk.append(token.retrieved)                          # retrieved
        tk.append(token.copy_for_DR)                        # copy_for_dr
        tk.append(token.copied_DR_index)                    # copied_dr_index
        tk.append(token.sim_made)                           # sim_made
        tk.append(False)                                    # isDeleted = False
        # -------------------[  FLOATS  ]-------------------
        tk.append(token.act)                                # act
        tk.append(token.max_act)                            # max_act
        tk.append(token.inhibitor_input)                    # inhibitor_input
        tk.append(token.inhibitor_act)                      # inhibitor_act
        tk.append(token.max_map)                            # max_map
        tk.append(token.td_input)                           # td_input
        tk.append(token.bu_input)                           # bu_input
        tk.append(token.lateral_input)                      # lateral_input
        tk.append(token.map_input)                          # map_input
        tk.append(token.net_input)                          # net_input
        # ====         Set per type properties       =====
        match type(token):
            case dt.Groups:                                 # Groups
                tk[1] = 4                                   # - token.set
                tk.append(token.group_layer)                # - group layer - INT
            case dt.PUnit:                                  # P
                tk[1] = 3                                   # - token.set
                tk.append(token.mode)                       # - p.mode - INT
            case dt.RBUnit:                                 # RB
                tk[1] = 2                                   # - token.set
                tk.append(token.timesFired)                 # - rb.timesfired - INT
            case dt.POUnit:                                 # PO
                tk[1] = 1                                   # - token.set
                tk.append(token.semNormalization)           # - po.semNormalizatino - INT
                tk.append(bool(token.predOrObj))            # - po.predOrObj -> BOOL
                tk.append(token.max_sem_weight)             # - max_sem_weight - FLOAT
        return tk
    
    # Return formatted list of semantic values
    def formatSemantic(self, sem: dt.Semantic):
        sm = []                         # empty semantic

        # ------[  FLOATS  ]-------
        sm.append(self.IDs.get(sem))    # ID
        sm.append(sem.amount)           # Amount
        sm.append(sem.myinput)          # Input
        sm.append(sem.max_sem_input)    # Max_input
        sm.append(sem.act)              # Act

        # --------[  INT  ]--------
        match sem.ont_status:           # Ont_status: -> (state:0, value:1, SDM:2)
            case "state":               
                sm.append(0)
            case "value":               
                sm.append(1)
            case "SDM":                 
                sm.append(2)
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
        match type(token):                          # Match token type, as each token stores info differently.
            case dt.POUnit:
                for RB in token.myRBs:              # PO.myRBs          - connections to tokens RBs
                    self.addCon(token, RB)
                for PO in token.same_RB_POs:        # PO.same_RB_POs    - connections to POs connected to same RB as self
                    self.addCon(token, PO)
                for link in token.mySemantics:      # PO.mySemantics    - Link objects containing connection to my semantics
                    sem = link.mySemantic
                    weight = link.weight
                    self.addCon(token, sem, weight)

            case dt.RBUnit:
                token: dt.RBUnit = token            # for autocomplete
                for P in token.myParentPs:          # RB.myParentPs     - connections to tokens Ps
                    self.addCon(token, P)
                for pred in token.myPred:           # RB.myPred         - connections to my pred unit
                    self.addCon(token, pred)
                for obj in token.myObj:             # RB.myObj          - connections to my object unit
                    self.addCon(token, obj)
                for P in token.myChildP:            # RB.myChildP       - connections to my child P unit
                    self.addCon(token, P)
                
            case dt.PUnit:
                token: dt.PUnit = token             # for autocomplete
                for RB in token.myRBs:              # P.myRBs           - connections to tokens RBs
                    self.addCon(token, RB)
                for RB in token.myParentRBs:        # P.myParentRBs     - connections to RBs in which I am an argument
                    self.addCon(token, RB)
                for Group in token.myGroups:        # P.myGroups        - connections to groups that I am a part of 
                    self.addCon(Group)
                
            case dt.Groups:
                token: dt.Groups = token            # for autocomplete
                for group in token.myParentGroups:  # Group.myParentGroups  - connections to groups above me
                    self.addCon(token, group)
                for group in token.myChildGroups:   # Group.myChildGroups   - connections to groups below me
                    self.addCon(token, group)      
                for P in token.myPs:                # Group.myPs            - connections to my Ps
                    self.addCon(token, P)
                for RB in token.myRBs:              # Group.myRBs           - connections to my RBs
                    self.addCon(RB)
                for link in token.mySemantics:      # Group.mySemantics     - link objects containing connection to my semantics
                    sem = link.mySemantic
                    weight = link.weight
                    self.addCon(token, sem, weight)
                
            case dt.Semantic:
                token: dt.Semantic = token              # for autocomplete
                for link in token.myPOs:                # Semantic.myPOs               - link objects containing connections to my POs
                    sem = link.mySemantic
                    weight = link.weight
                    self.addCon(token, sem, weight)
                for i in range(len(token.semConnect)): 
                    link = token.semConnect[i]          # Semantic.semConnect           - link objects containing connections to other semantics
                    sem = link.mySemantic
                    weight = token.semConnectWeights    # Semantic.semConnectWeights    - weights of the semantic-to-semantic connections, stored at same index as the link object in the semConnect list
                    self.addCon(token, sem, weight)
   
    # creates ID for each node and store in hash map for efficent lookup
    def identifyNodes(self):
        ID = 0
        for type in [self.mem.semantics, self.mem.POs, self.mem.RBs, self.mem.Ps, self.mem.Groups]:
            for node in type:
                self.IDs[node] = ID
                print(node, ID)
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
    def printToken(self, tk):
        # check for args
        args = None                                 
        if type(tk) == tuple:
            args = tk[1]
            tk = tk[0]
        elif type(tk) == str:
            args = tk

        # set flags
        a, s, n = False, False, False 
        if args != None:
            tk = tk.lower()
            if ("-a" in tk) or ("--all" in tk):
                a = True
            if ("-s" in tk) or ("--sort" in tk):
                s = True
            if ("-n" in tk) or ("--name" in tk):
                n = True

            # sort if required
            if s:
                nodes = sorted(self.tokens, key=lambda x: x[1])
            else:
                nodes = self.tokens

            # prind all nodes if required
            if a:
                if n:
                    for node in nodes:
                        self.printToken(node, "-n")
                    return
                else:
                    for node in self.tokens:
                        self.printToken(node)
                    return
        
        # Labels for properties in token list
        properties = [
            "ID",
            "Type",
            "Set",
            "analog",
            "max_map_unit",
            "made_unit",
            "maker_unit",
            "inhibitor_threshold",
            "inferred",
            "retrieved",
            "copy_for_dr",
            "copied_dr_index",
            "sim_made",
            "isDeleted",
            "act",
            "max_act",
            "inhibitor_input",
            "inhibitor_act",
            "max_map",
            "td_input",
            "bu_input",
            "lateral_input",
            "map_input",
            "net_input"
        ]

        # generate all values in strings
        values = []
        for i in range(len(properties)):
            values.append(str(tk[i]))
        
        # TODO: generate value labels
        vLabels = []
        for i in range(len(properties)):
            vLabels.append("None")

        # Header box TODO: add name if -n set
        headerTop = ("╒" + ("═" * len(" Node ")) + "╤" + ("═" * len(" ID : 999 ")) + "╕")
        headerInfo = ("│" + " Node " + "│" + f" ID : {str(tk[0]).ljust(4)}" + "│")
        headerBottom = ("╘" + ("═" * len(" Node ")) + "╧" + ("═" * len(" ID : 999 ")) + "╛")
        
        # index | label | value | label
        labelpad = int((len(max(properties, key=len)) - len(" Label ")) / 2) + 2
        columns = ("│" + " Ind " + "│" + (" " * labelpad) + " Label " + (" " * labelpad) + "│" + " Value " + "│" + " Label " + "│")
        subhead = ("├" + ("─" * len(" Ind ")) + "┼" + ("─" * len((" " * labelpad) + " Label " + (" " * labelpad))) + "┼" + ("─" * len(" Value ")) + "┼" + ("─" * len(" Label ")) + "┤")
        bottom  = ("└" + ("─" * len(" Ind ")) + "┴" + ("─" * len((" " * labelpad) + " Label " + (" " * labelpad))) + "┴" + ("─" * len(" Value ")) + "┴" + ("─" * len(" Label ")) + "┘")
        top     = ("┌" + ("─" * len(" Ind ")) + "┬" + ("─" * len((" " * labelpad) + " Label " + (" " * labelpad))) + "┬" + ("─" * len(" Value ")) + "┬" + ("─" * len(" Label ")) + "┐")

        # print header and column labels
        print(headerTop)
        print(headerInfo)
        print(headerBottom)
        print(top)
        print(columns)
        print(subhead)

        # Print each property aligned
        for i in range(len(properties)):
            print(f"│{str(i).ljust(5)}│ {properties[i].ljust((labelpad * 2) + 6)}│ {str(tk[i]).ljust(5)} │ {vLabels[i].ljust(6)}│")

        print(bottom)
        
    # Print builder info for debugging
    def print(self):
        print("mem", self.mem)
        # mem details to print number of each node
        # nodes per set

        # builder details to print: number of each node in tokens, number of connections, number of mappings
        # nodes per set