# nodes/network/routines/schematisation.py
# Schematisation routines for Network class

from ...enums import *
import logging

from typing import TYPE_CHECKING
from ...utils import tensor_ops as tOps
from ..single_nodes import Token
import torch

if TYPE_CHECKING:
    from ...network import Network

logger = logging.getLogger(__name__)

class SchematisationOperations:
    """
    Schematisation operations for the Network class.

    Does inference into the newSet, to refine the analogs or smt

    Method:
        Do some stuff to schematis the netowkr

    Requirements:
        - All mapping connections must be ≥ 0.7
        - Connected tokens (parents/children) of mapped tokens must also be mapped above threshold
    """
    
    def __init__(self, network):
        """
        Initialize SchematisationOperations with reference to Network.
        """
        self.network: 'Network' = network
        self.debug = False
    
    def requirements(self):
        """
        Check requirments for schematisation:
        - All driver and recepient mapping connections are above threshold (=.7)
        - Parents/Children of these mapped tokens are mapped with weight above threshold
        """
        threshold = 0.7
        # Check recipient nodes
        net = self.network
        cons = net.tokens.connections.tensor
        max_maps = net.tokens.token_tensor.tensor[:, TF.MAX_MAP]

        # Get mask of tokens that map above threshold, and those that don't
        valid_mask = max_maps >= threshold
        invalid_mask = ~valid_mask

        # Check for any nodes with 0 < max_map < threshold:
        if ((max_maps > 0) & (max_maps < threshold)):
            logger.debug("SchematisationReq failed: nodes with 0 < max_map < threshold found")
            return False
        
        # Check valid-invalid connections:
        #   Find all nodes that connect to invalid nodes.
        invalid_child = torch.matmul(cons, invalid_mask.float())
        invalid_parent = torch.matmul(torch.t(cons), invalid_mask.float()) # Transpose to get parent->child connections.
        invalid_connections = (invalid_child > 0) | (invalid_parent > 0)
        #   Get nodes that are valid but connect to invalid nodes.
        fail_nodes = valid_mask & invalid_connections
        if torch.any(fail_nodes):
            logger.debug("SchematisationReq failed: tokens with map>threshold connect to tokens with map<threshold")
            return False
        
        # All checks passed.
        logger.debug("SchematisationReq passed")
        return True
    
    def schematise_p(self, mode):
        """
        Perform schematisation for p tokens with given mode.
        """
        act_thresh = 0.4
        map_thresh = 0.75
        net = self.network
        driver = net.driver()
        logger.debug("Schematisating P tokens")
        p_mask = driver.tensor_op.get_arb_mask({TF.TYPE: Type.P, TF.MODE: mode})

        # Try find most active token.
        active_lcl = driver.token_op.get_most_active_token(local_mask=p_mask)
        if active_lcl is None:
            logger.debug(f"Schematisation failed (P, mode={mode}): no active tokens found")
            return

        # Check if token act is above threshold.
        active_act = driver.token_op.get_feature(active_lcl, TF.ACT)
        if active_act < act_thresh:
            logger.debug(f"Schematisation failed (P, mode={mode}): active token ({driver.lcl.to_global(active_lcl)}) below threshold ({active_act} < {act_thresh})")
            return

        # Check if token has caused a toke to be inferred.
        made = driver.token_op.get_feature(active_lcl, TF.MADE_UNIT)
        if made != null: # Token has caused a token to be inferred, Update made (newSet) unit (act = 1.0, connect to active newSet RBs)
            new_set = net.new_set()
            made = int(made)
            # Set act to 1.0
            net.node_ops.set_feature(made, TF.ACT, 1.0)
            # Get active RBs
            active = new_set.tensor_op.get_active_mask(thresh=0.5)
            rbs = new_set.tensor_op.get_mask(Type.RB)
            active_rbs = active & rbs
            if not (active_rbs).any():
                logger.debug(f"Schematisation(P, mode={mode}): No active RBs found")
                return
            active_rbs_idxs = new_set.lcl.to_global(torch.where(active_rbs)[0]) # Global indices of active RBs.
            # Connect made to active RBs
            if mode == Mode.PARENT: # connect as parent
                logger.debug(f"Schematisation(P, mode={mode}): Connecting {made} to active RBs")
                net.tokens.connections.connect_multiple(parent_idxs=made, child_idxs=active_rbs_idxs)
            else: # connect as child
                logger.debug(f"Schematisation(P, mode={mode}): Connecting active RBs to {made}")
                net.tokens.connections.connect_multiple(parent_idxs=active_rbs_idxs, child_idxs=made)

        else: # Token has not caused a token to be inferred -> infer a new token
            # Check map to rec token above threshold (0.75), infer newSet token
            max_map = driver.token_op.get_feature(active_lcl, TF.MAX_MAP)
            if max_map < map_thresh:
                logger.debug(f"Schematisation(P, mode={mode}): No token inferred")
                return
            # Infer a new token
            logger.debug(f"Schematisation(P, mode={mode}): Inferring new token")
            glbl = driver.lcl.to_global(active_lcl)
            self.infer_token(glbl)


    def schematise_rb(self):
        """
        Perform schematisation for rb tokens.
        """
        act_thresh = 0.4
        map_thresh = 0.75
        logger.debug("Schematising RB tokens")
        net = self.network
        driver = net.driver()
        new_set = net.new_set()

        # Get active RB token in driver
        active_lcl = driver.token_op.get_most_active_token(token_type=Type.RB)
        if active_lcl is None:
            logger.debug("Schematisation failed (RB): no active tokens found")
            return
        
        # check if token has caused a token to be inferred
        made = driver.token_op.get_feature(active_lcl, TF.MADE_UNIT)
        if made != null: # Token has caused a token to be inferred, Update made (newSet) unit (act = 1.0, connect to active newSet POs)
            made = int(made)
            # Set act to 1.0
            net.node_ops.set_feature(made, TF.ACT, 1.0)
            # Get active POs
            active = new_set.tensor_op.get_active_mask(thresh=0.5)
            pos = new_set.tensor_op.get_mask(Type.PO)
            active_pos = active & pos
            if not (active_pos).any():
                logger.debug(f"Schematisation(RB): No active POs found")
                return
            active_pos_idxs = new_set.lcl.to_global(torch.where(active_pos)[0]) # Global indices of active POs.
            # Connect made to active POs
            net.tokens.connections.connect_multiple(parent_idxs=made, child_idxs=active_pos_idxs)
        else: # Token has not caused a token to be inferred -> infer a new token
            # Check active above threshold, and map to recipient token above threshold.
            active_act = driver.token_op.get_feature(active_lcl, TF.ACT)
            max_map = driver.token_op.get_feature(active_lcl, TF.MAX_MAP)
            if active_act < act_thresh or max_map < map_thresh:
                logger.debug(f"Schematisation(RB): No token inferred")
                return
            logger.debug(f"Schematisation(RB): Inferring new token")
            # Infer a new token
            glbl = driver.lcl.to_global(active_lcl)
            self.infer_token(glbl)


    def schematise_po(self):
        """
        Perform schematisation for po tokens.
        """
        act_thresh = 0.4
        map_thresh = 0.75
        logger.debug("Schematising PO tokens")
        net = self.network
        driver = net.driver()
        new_set = net.new_set()

        # Get active PO token in driver
        active_lcl = driver.token_op.get_most_active_token(token_type=Type.PO)
        if active_lcl is None:
            logger.debug("Schematisation failed (PO): no active tokens found")
            return
        
        # Check if token has caused a token to be inferred
        made = driver.token_op.get_feature(active_lcl, TF.MADE_UNIT)
        if made != null: # Token has caused a token to be inferred, Update made (newSet) unit (act = 1.0, update link weights)
            made = int(made)
            # Set act to 1.0
            net.node_ops.set_feature(made, TF.ACT, 1.0)
            # Update link weights
            net.semantics.update_link_weights(made)
        
        else: # Token has not caused a token to be inferred -> infer a new token
            # Check if active above threshold and map to recipient token above threshold.
            active_act = driver.token_op.get_feature(active_lcl, TF.ACT)
            max_map = driver.token_op.get_feature(active_lcl, TF.MAX_MAP)
            if active_act < act_thresh or max_map < map_thresh:
                logger.debug(f"Schematisation(PO): No token inferred")
                return
            logger.debug(f"Schematisation(PO): Inferring new token")
            # Infer a new token
            glbl = driver.lcl.to_global(active_lcl)
            self.infer_token(glbl)
    

    def infer_token(self, maker: int):
        """
        Infer a newSet token (act = 1.0)
        """
        net = self.network
        type = net.node_ops.get_tk_feature(maker, TF.TYPE)
        # Create token
        base_features = {
            TF.SET: Set.NEW_SET,
            TF.INFERRED: B.TRUE,
            TF.ACT: 1.0,
            TF.ANALOG: null,
            TF.MAKER_UNIT: maker,
            TF.MAKER_SET: net.node_ops.get_tk_feature(maker, TF.SET)
        }
        match type:
            case Type.P:
                base_features[TF.MODE] = net.node_ops.get_tk_feature(maker, TF.MODE)
            case Type.PO:
                base_features[TF.PRED] = net.node_ops.get_tk_feature(maker, TF.PRED)
            case Type.RB:
                pass
            case _:
                raise ValueError(f"Invalid token type: {type}")
        new_token = Token(type, base_features)

        made = net.node_ops.add_token(new_token)
        net.node_ops.set_tk_feature(maker, TF.MADE_UNIT, made)
        net.node_ops.set_tk_feature(maker, TF.MADE_SET, new_token.set)
        logger.info(f"inferred token: {made} from {maker}")
        return made

    def schematisation_routine(self):
        """
        Run the schematisation routine.
        """
        logger.info('Running schematisation routine')
        self.schematise_p(Mode.PARENT)
        self.schematise_p(Mode.CHILD)
        self.schematise_rb()
        self.schematise_po()