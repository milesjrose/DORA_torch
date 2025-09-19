# nodes/network/single_nodes/po_pair.py
# Holds a set of po pairs

class Pairs(object):
    """
    A class for holding a pair of po tokens in the same set
    """
    def __init__(self):
        self.pairs: dict[int, (int, int)] = {}
    
    def hash(self, po1: int, po2: int):
        if po1<po2:
            return int(str(po1) + str(po2))
        else:
            return int(str(po2) + str(po1))
    
    def add(self, po1:int, po2:int):
        """
        add a pair of pos
        """
        self.pairs[self.hash(po1, po2)] = (po1, po2)
    
    def get_list(self) -> list[(int, int)]:
        """
        return list of pairs of indices
        """
        return list(self.pairs.values())

        