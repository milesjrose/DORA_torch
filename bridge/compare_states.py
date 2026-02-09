from bridge.state import State
from logging import getLogger, INFO, DEBUG
from nodes.enums import *
from nodes.utils import TablePrinter
logger = getLogger("COMPARE_STATES")

class Diff:
    def __init__(self, diff_type: str,id, val_type: str, old_val, new_val, comment:str = ""):
        self.id = id
        self.val_type = val_type
        self.diff_type = diff_type
        self.old_val = old_val
        self.new_val = new_val
        self.comment = comment
    
    def to_str(self):
        return f"{self.diff_type} {self.id}: {self.val_type} {self.old_val} -> {self.new_val} {self.comment}"

    def to_list(self):
        return [self.diff_type, self.id, self.val_type, self.old_val, self.new_val, self.comment]

class CompareStates:
    def __init__(self, old_state: State, new_state: State):
        self.old_state = old_state
        self.new_state = new_state

    def compare(self, verbose: bool = False) -> tuple[bool, list[Diff]]:
        """ Compare the states data, and return the differences.
        Args:
            verbose: Whether to print detailed comparison results
        Returns:
            Tuple containing:
                - match: bool - Whether states match
                - diffs: list of differences
        """
        if verbose:
            logger.setLevel(DEBUG)
        else:
            logger.setLevel(INFO)
        diffs = []
        diffs.extend(self._compare_tokens())
        diffs.extend(self._compare_semantics())
        if len(diffs) == 0:
            diffs.extend(self._compare_links())
            diffs.extend(self._compare_mappings())
            diffs.extend(self._compare_connections())
            diffs.extend(self._compare_sem_connections())
        else:
            logger.error("Differences found in tokens or semantics, skipping links, mappings, connections, and sem connections.")
        diffs.extend(self._compare_metadata())
        match = (len(diffs) == 0)
        return match, diffs
    
    def print_diffs(self, diffs: list[Diff], table=True):
        """ Print the differences. """
        if len(diffs) == 0:
            logger.info("No differences found.")
            return
        if table:
            columns = ["Class", "Id", "Field", "Old", "New", "Comment"]
            rows = [diff.to_list() for diff in diffs]
            tp = TablePrinter(columns, rows, headers=["Differences"], logger=logger)
            tp.print_table(header=True, column_names=True)
        else:
            for diff in diffs:
                print(diff.to_str(), end="\n")

    def _equal(self, old_val, new_val) -> bool:
        if type(old_val) != type(new_val):
            if isinstance(old_val, int) and isinstance(new_val, float) or isinstance(old_val, float) and isinstance(new_val, int):
                return self._equal(float(old_val), float(new_val))
            if old_val is None:
                n = old_val
                v = new_val
            elif new_val is None:
                n = new_val
                v = old_val 
            else:
                logger.error(f"Type mismatch: {type(old_val).__name__}({old_val}) != {type(new_val).__name__}({new_val})")
                return False
            if v in [None, "N/A", [], 0, float(0)]:
                return True
        
        if isinstance(old_val, float):
            tolerance = 1e-5
            abs_diff = abs(old_val - new_val)
            return abs_diff <= tolerance
        match = old_val == new_val
        if not match and isinstance(old_val, list) and isinstance(new_val, list) and len(old_val) == len(new_val):
            match = set(old_val) == set(new_val)
        return match

    def _compare_tokens(self) -> list[Diff]:
        """ Compare the tokens data, and return the differences. """
        return self._compare_nodes(self.old_state.tokens, self.new_state.tokens, "tokens")

    def _compare_semantics(self) -> list[Diff]:
        """ Compare the semantics data, and return the differences. """
        return self._compare_nodes(self.old_state.semantics, self.new_state.semantics, "sems")
    
    def _compare_nodes(self, old_nodes, new_nodes, n_type: str) -> list[Diff]:
        """ Compare two sets of nodes, and return the differences."""
        diffs: list[Diff] = []
        old_ids = list(old_nodes.keys())
        new_ids = list(new_nodes.keys())
        # Check for count mismatch
        if len(old_ids) != len(new_ids):
            if len(old_ids) == len(new_ids)-4:
                logger.debug(f"Ignoring missing comparative semantics.")
            else:
                logger.debug(f"{n_type} count mismatch: {len(old_ids)} != {len(new_ids)}")
                diffs.append(Diff(n_type, "-", "Count", len(old_ids), len(new_ids), f"{n_type} count mismatch"))
        # Check for duplicate IDs
        if len(old_ids) != len(set(old_ids)):
            logger.error("Duplicate IDs found in old state, ignoring duplicates")
        if len(new_ids) != len(set(new_ids)):
            logger.error("Duplicate IDs found in new state, ignoring duplicates")
        # Compare each node by ID
        all_ids = sorted(list(set(old_ids) | set(new_ids)))
        for id in all_ids:
            if id not in old_ids:
                name = new_nodes[id]['name']
                if name in ["MORE", "LESS", "SAME", "DIFF"]:
                    logger.debug(f"Ignoring missing comparative semantic {name}.")
                    continue
                logger.debug(f"{n_type} {id} missing in old")
                diffs.append(Diff(n_type, id, "ID", None, id, "ID missing in old"))
            elif id not in new_ids:
                logger.debug(f"{n_type} {id} missing in new")
                diffs.append(Diff(n_type, id, "ID", id, None, "ID missing in new"))
            else:
                old_node = old_nodes[id]
                new_node = new_nodes[id]
                for field in old_node.keys():
                    if field not in ["my_index", "same_RB_POs"]: # TODO: skipping rb pos for now, but need to fix at some point.
                        old_val = old_node.get(field, "N/A")
                        new_val = new_node.get(field, "N/A")
                        if not self._equal(old_val, new_val):
                            logger.debug(f"{n_type} {id} {field} mismatch: {old_val} -> {new_val}")
                            diffs.append(Diff(n_type, id, field, old_val, new_val, "Value mismatch"))
        return diffs
    
    def _con_diff(self, id0, id1, old_val, new_val, n_type: str, f_type: str) -> Diff:
        """ Compare a connection between two nodes, and return the differences. """
        ids = f"{id0} → {id1}"
        return Diff(n_type, f"{id0} → {id1}", f_type, old_val, new_val, "Value mismatch")

    def _compare_links(self) -> list[Diff]:
        """ Compare the links data, and return the differences. """
        diffs: list[Diff] = []
        old_links = self.old_state.links
        new_links = self.new_state.links
        for i in range(len(old_links)):
            for j in range(len(old_links[i])):
                old_val = old_links[i][j]
                new_val = new_links[i][j]
                if not self._equal(old_val, new_val):
                    id0 = self.old_state.ids[i]
                    id1 = self.old_state.sem_ids[j]
                    diffs.append(Diff("links", f"{id0} → {id1}", "Weight", old_val, new_val, "Value mismatch"))
        return diffs

    def _compare_mappings(self) -> list[Diff]:
        """ Compare the mappings data, and return the differences. """
        diffs: list[Diff] = []
        old_mappings = self.old_state.mappings
        new_mappings = self.new_state.mappings
        for i in range(len(old_mappings)):
            for j in range(len(old_mappings[i])):
                for field in [MappingFields.WEIGHT, MappingFields.HYPOTHESIS, MappingFields.MAX_HYP]:
                    old_val = old_mappings[i][j].get(field, "N/A")
                    new_val = new_mappings[i][j].get(field, "N/A")
                    if not self._equal(old_val, new_val):
                        diffs.append(self._con_diff(i, j, old_val, new_val, "mappings", field.name.lower()))
        return diffs
    
    def _compare_connections(self) -> list[Diff]:
        """ Compare the connections data, and return the differences. """
        diffs: list[Diff] = []
        old_connections = self.old_state.connections
        new_connections = self.new_state.connections
        for i in range(len(old_connections)):
            for j in range(len(old_connections[i])):
                old_val = old_connections[i][j]
                new_val = new_connections[i][j]
                if not self._equal(old_val, new_val):
                    diffs.append(self._con_diff(i, j, old_val, new_val, "connections", "Weight"))
        return diffs
    
    def _compare_sem_connections(self) -> list[Diff]:
        """ Compare the sem connections data, and return the differences. """
        diffs: list[Diff] = []
        old_sem_connections = self.old_state.sem_connections
        new_sem_connections = self.new_state.sem_connections
        for i in range(len(old_sem_connections)):
            for j in range(len(old_sem_connections[i])):
                old_val = old_sem_connections[i][j]
                new_val = new_sem_connections[i][j]
                if not self._equal(old_val, new_val):
                    diffs.append(self._con_diff(i, j, old_val, new_val, "sem_connections", "Weight"))
        return diffs
    
    def _compare_metadata(self) -> list[Diff]:
        """ Compare the metadata data, and return the differences. """
        diffs: list[Diff] = []
        old = self.old_state.metadata
        new = self.new_state.metadata
        # sim_path
        if old['sim_path'] != new['sim_path']:
            logger.error(f"sim_path mismatch: {old['sim_path']} -> {new['sim_path']}")
        # parameters
        if False: # ignoring for now
            for field in list(old['parameters'].keys()):
                old_val = old['parameters'].get(field, "N/A")
                new_val = new['parameters'].get(field, "N/A")
                if not self._equal(old_val, new_val):
                    diffs.append(Diff("params", "-", field, old_val, new_val, "Params field mismatch"))
        # token_counts
        for field in list(old['token_counts'].keys()):
            old_val = old['token_counts'].get(field, "N/A")
            new_val = new['token_counts'].get(field, "N/A")
            if not self._equal(old_val, new_val) and not (field is Type.SEMANTIC and (old_val == new_val-4)):
                diffs.append(Diff("counts", "-", field.name, old_val, new_val, "Count mismatch"))
        # con_counts
        cons_to_comp = [
            [Type.P, Type.RB, 'child'],
            [Type.P, Type.RB, 'parent'],
            [Type.P, Type.GROUP],
            [Type.RB, Type.P, 'parent'],
            [Type.RB, Type.P, 'child'],
            [Type.RB, Type.PO, 'pred'],
            [Type.RB, Type.PO, 'obj'],
            [Type.PO, Type.RB],
        ]
        for cons in cons_to_comp:
            con_str = f"{cons[0].name}_{cons[1].name}"
            old_val = old['con_counts'][cons[0]][cons[1]]
            new_val = new['con_counts'][cons[0]][cons[1]]
            if len(cons) == 3:
                old_val = old_val[cons[2]]
                new_val = new_val[cons[2]]
                con_str += f"_{cons[2]}"
            if not self._equal(old_val, new_val):
                diffs.append(Diff("con_counts", "-", con_str, old_val, new_val, "Count mismatch"))
        return diffs
        