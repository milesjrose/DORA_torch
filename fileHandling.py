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
        self.tokenAmounts = [len(self.mem.semantics), len(self.mem.POs), len(self.mem.RBs), len(self.mem.Ps), len(self.mem.Groups)]
        numNodes = sum(self.tokenAmounts)
        self.tokenTypes = [dt.POUnit, dt.RBUnit, dt.PUnit, dt.Groups]
        self.sets = ["driver", "recipient", "memory", "newSet"]
        # Intermediate 
        self.IDs = {}                   # Mapping: node -> ID
        self.names = {}                 # Mapping: ID -> name
        self.nodes = []                 # List of all formatted nodes
        self.connections = {}           # Weighted, directed adjacency list, stores weights and directions of connections
        self.mappings = {}              # Directed adjacency list for mappings, stores weights and hypothesis info in buckets

    # TODO: Turn intermediatary format into single object containing tensor data structures
    def tensorise(self):
        memoryTensors = None
        return memoryTensors                                            # return tensorMemory object, containing full tensors, and supporting data structures

    # Extract data into intermediatory format
    def formatMemory(self):
        nodes = []                                                      # To store all formatted nodes
        self.identifyNodes()
        for anumber in range(len(self.mem.analogs)):                         # iterate through analogs, keeping track of analog number
            analog: dt.Analog = self.mem.analogs[anumber]               # - for autocomplete
            types = [analog.myPOs, analog.myRBs, analog.myPs, analog.myGroups]
            for tokens in types:                                        
                for token in tokens:                                 
                    tk = self.formatToken(token, anumber)               # Format Token
                    self.nodes.append(tk)   
                    self.addMappings(token)                             # Add Mappings
                    self.addConnections(token)                          # Connections

    # Returns formatted list of token values
    def formatToken(self, token: dt.TokenUnit, anumber):
        tk = []                                             # Empty token            
        # =====        Set shared properties        =====
        self.names[self.IDs.get(token)] = token.name        # Map ID -> name
        # -------------------------------------------- INTs
        tk.append(self.IDs.get(token))                      # ID
        tk.append("TypeNotSet")                             # Type
        tk.append(self.sets.index(token.set))               # Set
        tk.append(anumber)                                  # analog
        tk.append(self.IDs.get(token.max_map_unit))         # max_map_unit
        tk.append(self.IDs.get(token.my_made_unit))         # made_unit
        tk.append(self.IDs.get(token.my_maker_unit))        # maker_unit
        tk.append(self.IDs.get(token.inhibitorThreshold))   # inhibitor_threshold
        # ------------------------------------------- BOOLs
        tk.append(token.inferred)                           # inferred
        tk.append(token.retrieved)                          # retrieved
        tk.append(token.copy_for_DR)                        # copy_for_dr
        tk.append(token.copied_DR_index)                    # copied_dr_index
        tk.append(token.sim_made)                           # sim_made
        tk.append(False)                                    # isDeleted = False
        # ------------------------------------------  FLOATs
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
                tk[1] = 4                                   # token.set
                tk.append(token.group_layer)                # - group layer - INT
            case dt.PUnit:                                  # P
                tk[1] = 3                                   # token.set
                tk.append(token.mode)                       # - p.mode - INT
            case dt.RBUnit:                                 # RB
                tk[1] = 2                                   # token.set
                tk.append(token.timesFired)                 # - rb.timesfired - INT
            case dt.POUnit:                                 # PO
                tk[1] = 1                                   # token.set
                tk.append(token.semNormalization)           # - po.semNormalizatino - INT
                tk.append(bool(token.predOrObj))            # - po.predOrObj -> BOOL
                tk.append(token.max_sem_weight)             # - max_sem_weight - FLOAT
        return tk
    
    # TODO: Return formatted list of semantic values
    def formatSemantic(self, sem):
        #sem.ontstatus -> match state-0 value-1 sdm-2 int
        #sem_amount
        #sem_input
        #max_sem_input
        return None
    
    # Adds mappings and hypotheses for a given token
    def addMappings(self, token):
        addedMaps = {}
        for mapHyp in token.mappingHypotheses:              # Add hypotheses
            mp = []                                         # emtpy mapping
            mapCon = mapHyp.mappingConnection               # get corresponding mapping
            driver = self.IDs[mapCon.driverToken]           # Driver token
            recipient =self.IDs[mapCon.recipientToken]      # recipient token
            mp.append(mapCon.weight)                        # weight
            mp.append(mapHyp.hypothesis)                    # hypothesis
            mp.append(mapHyp.max_hyp)                       # max_hyp
            self.mappings[(driver, recipient)] = mp         # add mapping to list
            addedMaps[mapCon] = True                        # mark mapping as added
        for mapCon in token.mappingConnections:        # Add any remaining mappings without hypotheses
            if addedMaps[mapCon] != True:              # NOTE: need to check when mapping has/hasnt got hypothesis
                mp = []                                     # emtpy mapping
                driver = self.IDs[mapCon.driverToken]       # Driver token
                recipient =self.IDs[mapCon.recipientToken]  # recipient token
                mp.append(mapCon.weight)                    # weight
                mp.append(0.0)                              # hypothesis
                mp.append(0.0)                              # max_hyp
                self.mappings[(driver, recipient)] = mp     # add mapping to list
                addedMaps[mapCon] = True                    # mark mapping as added

    # Adds all connections for token (Only add child nodes so dont duplicate writes)
    def addConnections(self, token):
        # TODO add all forward connections for given token
        # Match token type, as each token stores info differently.
        match type(token):
            case dt.POUnit:
                pass
            case dt.RBUnit:
                pass
            case dt.PUnit:
                pass
            case dt.Groups:
                pass
            case dt.Semantic:
                pass
    
    # creates ID for each node and store in hash map for efficent lookup
    def identifyNodes(self):
        ID = 0
        for type in [self.mem.semantics, self.mem.POs, self.mem.RBs, self.mem.Ps, self.mem.Groups]:
            for node in type:
                self.IDs[node] = ID
                print(node, ID)
                ID += 1

    # Returns 0 if not integer, or val o.w
    def sanitiseInt(self, val):
        if type(val) != int:
            return 0
        return val
    
    # Print formated token in readable way for debugging
    def printToken(self, tk):
        # print all nodes
        if tk == "-a":
            for node in self.nodes:
                self.printToken(node)
            return
        # sort nodes by id, then print them all
        if tk == "-s":
            for node in sorted(self.nodes, key=lambda x: x[1]):
                self.printToken(node)
            return

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

        # Find the max length of property names for alignment
        max_key_length = max(len(prop) for prop in properties)

        # Create header box for ID
        id_value = tk[0]  # Assuming the first element corresponds to "ID"
        id_box = f"---=--------- ID : {id_value} ----------"
        print(id_box)

        # Print each property aligned
        for i in range(len(properties)):
            print(f"{str(i).ljust(3)}| {properties[i].ljust(max_key_length)} = {tk[i]}")

        print(("-" * 3) + "=" + ("-" * (len(id_box) - 4)))  # Bottom order

