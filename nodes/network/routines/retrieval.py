# nodes/network/operations/retrieval_ops.py
# Retrieval operations for Network class

from ...enums import *
import torch
from logging import getLogger
from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps

logger = getLogger(__name__)

if TYPE_CHECKING:
    from ...network import Network

class RetrievalOperations:
    """
    Retrieval finds and brings into the recipient structures in memory that are useful for inference. These are
    found by looking for prominent structures in memory that are co-active with the driver tokens. This can be done
    by looking either at the activations of analogs, or individual tokens by setting params.bias_retrieval_analogs.

    Method:

    First update the inputs and activations in memory. Decide on retrieving tokens or analogs based on 
    params.bias_retrieval_analogs. For analogs, take the sum of the activations of its tokens, and for 
    tokens consider each token type (PO, RB, P.child, P.parent) individually. Select tokens/analogs to 
    retrieve using Luce choice:
        - Get the activations for each token/analog, and the total activation of all tokens/analogs.
                - If retrieving analogs and params.use_relative_act, transform the activations with a sigmoidal function.
        - Give each token/analog a retrieval probability based on its activation/total activations.
        - Generate a number bentween 0 and 1, and select tokens/analogs with a probability greater than this number.
        - Move the selected tokens/analogs to the recipient. If moving tokens, move some of its connected tokens as well.

    Requirements:
        - At least one P (proposition) token must be present in the driver to trigger retrieval
    """

    def __init__(self, network):
        """
        Initialize RetrievalOperations with reference to Network.
        
        Args:
            network: Reference to the Network object
        """
        self.network: 'Network' = network
        self.debug = False
        self.efficient_token_retrieval = True
    
    def requirements(self) -> bool:
        """ 
        Check the requirements for retrieval:
        - At least one P token in the driver.

        Returns:
            bool: True if requirements are met, False o.w.
        """
        p_mask = self.network.driver().tensor_op.get_mask(Type.P)
        if not p_mask.any():
            logger.debug("RetrievalReq Failed: No P tokens in driver")
            return False
        return True


    def retrieval_routine(self):
        """
        Run the model retrieval routine - update input/act in memory, then if bias_retrieval_analogs 
        get the total act for each analog, else track the most active tokens in memory.
        """
        net = self.network
        net.update.inputs(Set.MEMORY)
        net.update.acts(Set.MEMORY)
    
    def retrieve_tokens(self):
        """
        Retrieve tokens from memory.
        """
        net = self.network
        if net.memory().tensor_op.get_count() == 0:
            logger.debug("No tokens in memory, skipping token retrieval.")
            return
        rel_act = net.params.use_relative_act
        analog_bias = net.params.bias_retrieval_analogs
        if analog_bias:
            net.cache_analogs()
            self.retrieve_analogs_biased(rel_act)
        else:
            if rel_act:
                logger.critical("Relative act not implemented for non-bias retrieval, ignoring flag.")
            net.memory().token_op.get_max_acts()
            if self.efficient_token_retrieval:
                self.retrieve_tokens_efficient()
            else:
                self.retrieve_tokens_direct_match()


# ================================[ ANALOG RETRIEVAL LOGIC ]================================
    def retrieve_analogs_biased(self, use_relative_act):
        """ Retrieve analogs from memory, using analog bias"""
        net = self.network
        mem_count = net.memory().tensor_op.get_count()
        if mem_count == 0:
            logger.debug("No tokens in memory, skipping retrieval.")
            return
        # Calc normal act for each analog
        info = net.analog_ops.get_analogs_info(set=Set.MEMORY)
        analogs = info.data[:, info.Cols.NUM]
        counts = info.data[:, info.Cols.COUNT]
        acts = info.data[:, info.Cols.ACT]
        normal_act = acts/counts
        
        # If relative act, transform normal_act with sigmoidal function
        if use_relative_act:
            # take weighted average of normal_acts
            avg_norm =  (torch.mean(normal_act) + torch.max(normal_act))/2
            # transform the norm_acts with
            # 1 / (1 + exp(10 * (norm_act - avg_norm)))
            normal_act = 1 / (1 + torch.exp(10 * (normal_act - avg_norm)))
            sum_normal_act = torch.sum(normal_act)
        else:
            # For non-relative activation, use normal_act directly
            sum_normal_act = torch.sum(normal_act)

        # Retrieve analogs with luce choice
        active_mask = counts > 0                    # Mask 0 act analogs
        if active_mask.sum() > 0 and sum_normal_act > 0:
            # Calc retrieval prob for each analog
            retrieve_prob = normal_act[active_mask]/sum_normal_act
            random_num = torch.rand(analogs.shape[0])
            retrieve_mask = active_mask & (retrieve_prob > random_num)
            # Retrieve analogs NOTE: not vectorised TODO: Add method for moving multiple analogs at once
            analogs_to_retrieve = analogs[retrieve_mask]
            self.retrieve_analogs(analogs_to_retrieve)
            
    def retrieve_analogs(self, analogs: torch.Tensor):
        """
        Move analog(s) from memory to recipient
        """
        # Move analogs to recipient
        self.network.analog.move(analogs, Set.RECIPIENT)
        # Set retrieved to true
        self.network.analog.set_analog_features(analogs, TF.RETRIEVED, B.TRUE)


# ================================[ TOKEN RETRIEVAL LOGIC ]================================
    def luce_choice_retrieval(self, token_sum: float, token_mask: torch.Tensor) -> torch.Tensor:
        """ 
        Decide on retrieval based on luce choice for the tokens in the mask.
        Args:
            token_sum: Sum of the max acts of the tokens in the mask.
            token_mask: Mask of the tokens to decide on retrieval for.
        Returns:
            torch.Tensor: Mask of the tokens that should be retrieved.
        """
        mem = self.network.memory()
        # make sure token_sum > 0
        if token_sum <= 0:
            # No tokens to retrive, return false mask
            return torch.zeros_like(token_mask, dtype=torch.bool)
        # retrieve prob = max_act / token_sum
        retrieve_prob = mem.lcl[token_mask, TF.MAX_ACT] / token_sum
        # if retrieve prob > random num, flag token for retrieval
        random_num = torch.rand_like(retrieve_prob)
        # Create mask for tokens that should be retrieved
        retrieve_mask = torch.zeros_like(token_mask, dtype=torch.bool)
        retrieve_mask[token_mask] = retrieve_prob > random_num
        return retrieve_mask

    def retrieve_tokens_efficient(self):
        """ 
        Uses slightly different logic to retrieve tokens, but way more simple. Depending on how
        the tensor is sructured this may have the same result as the direct match logic. But I did 
        both as I'm not exactly sure how the connections will be structured when running.

        Method:
        Finds the children of each token type recursively, to get all tokens connected under the 
        token selected for retrieval. For P & RB tokens, we get all children recursively. 
        For PO tokens, we also use the child RBs, and get their parents for the parent Ps.

        Potential issues:
        In the case that RB tokens don't just have two children (pred + obj)/(pred + child_p), 
        we are getting all children here. An RBs child P will also have it's own children added 
        to retrieval. Also if an RB is retrieved in the PO retrieval step, all parents of it are 
        retrieved, not just one parent P.
         """
        # Apply luce choice to each token type
        mem = self.network.memory()
        net = self.network

        mem_count = mem.tensor_op.get_count()
        if mem_count == 0:
            logger.debug("No tokens in memory, skipping token retrieval.")
            return

        # Retrieve tokens and their children
        po_ret_mask = None
        for tk_type in [Type.P, Type.RB, Type.PO]:
            type_mask = mem.tensor_op.get_mask(tk_type)
            if not torch.any(type_mask): 
                continue # No tokens of this type to retrieve, skip
            type_sum = mem.lcl[type_mask, TF.MAX_ACT].sum()
            if type_sum == 0: 
                continue # No active tokens to retrieve, skip.
            ret_mask = self.luce_choice_retrieval(type_sum, type_mask)
            glbl_mask = self.lcl_mask_to_glbl(ret_mask) # Mask of mem view -> whole tensor mask
            type_children = mem.tokens.connections.get_children_recursive(glbl_mask)
            ret_mask = ret_mask | type_children
            if tk_type == Type.PO:
                po_ret_mask = ret_mask # Keep the PO ret mask for finding parent Ps.
            ret_type_idxs = torch.where(ret_mask)[0]
            net.node_ops.move_tokens(ret_type_idxs, Set.RECIPIENT)

        # Retrieve parent Ps of RBs retrieved from POs
        if po_ret_mask is not None:
            rbs = mem.tokens.arb_mask({TF.TYPE: Type.RB})
            ret_rbs = rbs & po_ret_mask
            ret_rb_parents = mem.tokens.connections.get_parents(ret_rbs)
            if torch.any(ret_rb_parents):
                net.node_ops.move_tokens(ret_rb_parents, Set.RECIPIENT)

    def retrieve_tokens_direct_match(self):
        """ 
        Attempts to directly emulate the old retrieval logic, but my code for it is pretty 
        inefficient and convoluted. Not sure how robust it is, and might have bugs.
        """
        mem_count = mem.tensor_op.get_count()
        if mem_count == 0:
            logger.debug("No tokens in memory, skipping token retrieval.")
            return
        # Apply luce choice to each token type
        mem = self.network.memory()
        retrieve_masks = {}
        for token_type in [Type.P, Type.RB, Type.PO]:
            token_mask = mem.tensor_op.get_mask(token_type)
            token_sum = mem.lcl[token_mask, TF.MAX_ACT].sum()
            type_retrieve_mask = self.luce_choice_retrieval(token_sum, token_mask)
            retrieve_masks[token_type] = type_retrieve_mask

        # Move tokens to recipient
        self.retrieve_tokens_with_masks(retrieve_masks)

    def retrieve_tokens_with_masks(self, retrieve_masks: dict[Type, torch.Tensor]):
        """
        Move tokens in mask from memory to recipient, including any children of these tokens.

        For P tokens:
            - Move the P token to recipient
            - Add its RBs to list of tokens to retrieve
        For RB tokens:
            - Move the RB token to recipient
            - Add its first Pred to recipient
            - If it has object, add that. O.w add its first child P.
        For PO tokens:
            - Move the PO token to recipient
            - Add its RBs to recipient
            - Add those RBs parent P's to recipient
        """
        # NOTE: trying to match the old code exactly, so may be some inefficiencies here. 
        # The code checks each token type in turn and each has different logic, so if e.g a PO is slated to be retrieved,
        # but while retrieving an RB, it is retrieved through the RB retrieval logic, then it will no longer be retrieved
        # by the PO retrieval logic. This means we need to go through each token type in turn, and generate masks for what
        # will be retrieved with its specific logic - and remove the tokens that are retrieved by a higher tokens logic from
        # the masks for the lower tokens (i.e if PO marked for retrieval, but retrieved RB connects to it, then the PO will
        # be removed from the initial retrieval mask, and instead be retrieved from the RB retrieval logic). The exception is
        # that RBs retrieved in P retrieval act the same as RBs retrieved in RB retrieval, so we don't need to remove them from the
        # PO retrieval mask, but just combine these masks).

        net = self.network
        mem = net.memory()
        cons = net.tokens.connections
        # globalise masks
        for type in [Type.P, Type.RB, Type.PO]:
            retrieve_masks[type] = self.lcl_mask_to_glbl(retrieve_masks[type])
        p_mask = retrieve_masks[Type.P]
        rb_mask = retrieve_masks[Type.RB]
        po_mask = retrieve_masks[Type.PO]
        
        # P retrieval mask logic
        rb_mask |= self.p_retrieval_mask(p_mask)

        # RB retrieval mask logic
        rb_with_obj = self.get_rb_with_obj(rb_mask)
        rb_without_obj = (~rb_with_obj) & rb_mask
        extra_preds = self.rb_pos(rb_with_obj, B.TRUE)
        extra_obj = self.rb_pos(rb_with_obj, B.FALSE)

        rb_parent_ps = self.rb_parent_ps(rb_mask)
        rb_child_ps = self.rb_child_ps(rb_without_obj)

        rb_pos = extra_preds | extra_obj
        rb_ps = rb_parent_ps | rb_child_ps

        # PO retrieval mask logic
        po_mask &= (~rb_pos) # remove POs that are retrieved by RBs
        po_rbs = self.po_rbs(po_mask)
        po_ps = self.po_rb_ps(po_rbs)

        # Combine masks
        Ps = p_mask | rb_ps | po_ps
        RBs = rb_mask | po_rbs
        POs = po_mask | rb_pos
        
        # Retrieve tokens
        all_retrieve_mask = Ps | RBs | POs
        r_idxs = torch.where(all_retrieve_mask)[0]
        self.network.tokens.move(r_idxs, Set.RECIPIENT)
    
    def lcl_mask_to_glbl(self, lcl_mask: torch.Tensor):
        """
        Covert a local memory mask to a global mask. 
        Args:
            lcl_mask: Local mask
        Returns:
            torch.Tensor: Global mask
        """
        net = self.network
        lcl_idxs = torch.where(lcl_mask)[0]
        glbl_idxs = net.to_global(lcl_idxs, Set.MEMORY)
        glbl_mask = torch.zeros_like(net.token_tensor.tensor[:,0], dtype=torch.bool)
        torch.squeeze(glbl_mask) # remeve extra dimension if there is one, might not be tho idk.
        glbl_mask[glbl_idxs] = True
        return glbl_mask

    def p_retrieval_mask(self, p_mask: torch.Tensor):
        """
        Do the mask logic for P retrieval.
        Args:
            p_mask: Global mask of P tokens to retrieve.
        Returns:
            torch.Tensor: Extra RBs to retrieve.
        """
        net = self.network
        cons = net.tokens.connections.tensor
        rbs = net.tokens.arb_mask({TF.TYPE: Type.RB})
        p_children = cons[p_mask, :] == True
        p_rb_children = rbs & p_children
        return p_rb_children
    
    def rb_parent_ps(self, rb_mask: torch.Tensor) -> torch.Tensor:
        """
        Do the mask logic for RB retrieval for parent Ps.
        Args:
            rb_mask: Global mask of RB tokens to retrieve.
        Returns:
            torch.Tensor: Extra P tokens to retrieve.
        """
        net = self.network
        cons = net.tokens.connections.tensor
        parent_p_mask = net.tokens.arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.PARENT})
        rb_parents = cons.tensor[:, rb_mask] == True
        extra_parent_ps = parent_p_mask & rb_parents
        return extra_parent_ps
    
    def rb_child_ps(self, rb_mask: torch.Tensor) -> torch.Tensor:
        """
        Do the mask logic for RB retrieval for child Ps.
        Args:
            rb_mask: Global mask of RB tokens to retrieve.
        Returns:
            torch.Tensor: Extra P tokens to retrieve.
        """
        net = self.network
        cons = net.tokens.connections.tensor
        child_p_mask = net.tokens.arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.CHILD})
        rb_children = cons.tensor[:, rb_mask] == True
        extra_child_ps = child_p_mask & rb_children
        return extra_child_ps
    
    def rb_pos(self, rb_mask: torch.Tensor, pred: B) -> torch.Tensor:
        """
        Do the mask logic for RB retrieval for POs.
        Args:
            rb_mask: Global mask of RB tokens to retrieve.
            pred: True if retrieving preds, False if retrieving objects.
        Returns:
            torch.Tensor: Extra P tokens to retrieve.
        """
        # TODO: Make this not aweful. Currently very inefficient, and ugly
        net = self.network
        cons = net.tokens.connections.tensor
        po_mask = net.tokens.arb_mask({TF.TYPE: Type.PO, TF.PRED: pred})
        
        # Find the RBs that have a po.
        rb_with_po = cons.tensor[rb_mask, po_mask].sum(dim=1) > 0
        tOps.sub_union(rb_mask, rb_with_po)

        # Go through each RB with a PO and add update the mask for its first PO.
        extra_po = torch.zeros_like(rb_mask, dtype=torch.bool)
        for rb_index in rb_with_po.to_list(): # For each RB
            rb_children = cons.tensor[rb_index, :] == True
            rb_children = tOps.sub_union(rb_mask, rb_children)
            rb_po_children = rb_children & po_mask
            if not torch.any(rb_po_children):
                logger.critical("RB has no PO children in mask logic, this should never happen >:(")
                continue # skip to next RB
            rb_pred_idx = torch.where(rb_po_children)[0][0] # Get first pred
            extra_po[rb_pred_idx] = True # Add to extra preds mask
        return extra_po
    
    def get_rb_with_obj(self, rb_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the RBs that have an object.
        Args:
            rb_mask: Global mask of RB tokens to retrieve.
        Returns:
            torch.Tensor: Global mask of RB tokens with an object.
        """
        net = self.network
        cons = net.tokens.connections.tensor
        obj_mask = net.tokens.arb_mask({TF.TYPE: Type.PO, TF.PRED: B.FALSE})
        rb_with_obj = cons.tensor[rb_mask, obj_mask].sum(dim=1) > 0
        return tOps.sub_union(rb_mask, rb_with_obj)
    
    def po_rbs(self, po_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the parent RBs of POs.

        Args:
            po_mask: Global mask of PO tokens to retrieve.
        Returns:
            torch.Tensor: Global mask of RB tokens that are parents of the POs.
        """
        net = self.network
        cons = net.tokens.connections.tensor
        po_parents = cons.tensor[:, po_mask] == True
        rbs = net.tokens.arb_mask({TF.TYPE: Type.RB})
        parent_rbs = rbs & po_parents
        return parent_rbs
    
    def po_rb_ps(self, rb_mask: torch.Tensor) -> torch.Tensor:
        """
        Get the parent Ps of RBs from POs.

        Args:
            po_mask: Global mask of PO tokens to retrieve.
        Returns:
            torch.Tensor: Global mask of P tokens that are parents of the RBs.
        """
        net = self.network
        cons = net.tokens.connections.tensor
        parent_p_mask = net.tokens.arb_mask({TF.TYPE: Type.P, TF.MODE: Mode.PARENT})
        rb_parents = cons.tensor[:, rb_mask] == True
        extra_parent_ps = parent_p_mask & rb_parents
        return extra_parent_ps