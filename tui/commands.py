# DORA_tensorised/tui/commands.py
# Commands for the dora tui, folling the old command line interface
try:
    from ..nodes import Network  # package context
    from ..DORA import DORA
    from ..nodes.enums import *
except ImportError:
    from DORA_tensorised.nodes import Network
    from DORA_tensorised.DORA import DORA
    from DORA_tensorised.nodes.enums import *

class Commands:
    def __init__(self, dora: DORA):
        self.dora: DORA = dora
        self.network: Network = dora.network
    
    def cdr(self):
        """ clear driver and recipient """
        self.network.tensor_ops.clear_set(Set.DRIVER)
        self.network.tensor_ops.clear_set(Set.RECIPIENT)
        pass
    
    def cr(self):
        """ clear recipient """
        self.network.tensor_ops.clear_set(Set.RECIPIENT)
    
    def selectTokens(self):
        """ select tokens from memory to place into driver NOTE not implemented yet """
        raise NotImplementedError("Not implemented yet")
    
    def r(self):
        """ retrieve """
        # check if can do retrieval
        if self.network.routines.retrieval.requirements():
            self.dora.do_retrieval_v2()
    
    def w(self):
        """ within entropy operations """
        self.dora.do_entropy_ops_within()
    
    def wp(self):
        """ within entropy operations for preds only """
        self.dora.do_entropy_ops_within(pred_only=True)

    def m(self):
        """ map """
        get_mask = self.dora.network.memory().tensor_ops.get_mask
        if get_mask(Type.P).any() and get_mask(Type.RB).any():
            self.dora.do_map()

    def b(self):
        """ entropy ops between """
        self.dora.do_entropy_ops_between()
    
    def p(self):
        """ predicate """
        if self.network.routines.predication.requirements():
            self.dora.do_predication()
    
    def s(self):
        """ schematize """
        if self.network.routines.schematisation.requirements():
            self.dora.do_schematisation()

    def f(self):
        """ form new relation """
        if self.network.routines.rel_form.requirements():
            self.dora.do_rel_form()
    
    def g(self):
        """ generalize """
        if self.network.routines.rel_gen.requirements():
            self.dora.do_rel_gen()
    
    def co(self):
        """ compression - NOTE: Not implemented yet """
        self.dora.do_compression()
    
    def c(self):
        """ clear : mappings, made units, inferences, driver, recipient, new_set """
        self.network.clear(limited=False)
        raise NotImplementedError("Not implemented yet")
    
    def cl(self):
        """ limited clear : made units, inferences, new_set """
        self.network.clear(limited=True)
    
    def wdr(self):
        """ write driver and recipient TODO """
        raise NotImplementedError("Not implemented yet")
    
    def wn(self):
        """ write network """
        raise NotImplementedError("Not implemented yet")




