# nodes/tests/e2e/test_builder.py
# End-to-end tests for the network builder using actual simulation files

import pytest
import os
from pathlib import Path

from nodes.builder_new import build_network
from nodes.enums import Set, Type, TF, B


# =====================[ testsim15.py E2E Tests ]======================

class TestTestsim15:
    """
    End-to-end tests using the testsim15.py simulation file.
    
    testsim15.py structure:
    - 5 propositions total (3 driver, 2 recipient)
    - Each proposition has 2 RBs
    - Each RB has 1 predicate PO and 1 object PO
    - Each PO has 3 semantics
    
    Propositions:
    1. lovesMaryTom (driver): lover->Mary, beloved->Tom
    2. lovesTomJane (driver): lover->Tom, beloved->Jane
    3. jealousMaryJane (driver): jealous_act->Mary, jealous_pat->Jane
    4. lovesJohnKathy (recipient): lover->John, beloved->Kathy
    5. lovesKathySam (recipient): lover->Kathy, beloved->Sam
    """

    @pytest.fixture
    def testsim15_path(self):
        """Get the path to testsim15.py."""
        # Navigate from tests/e2e/ to sims/
        current_dir = Path(__file__).parent
        sims_dir = current_dir.parent.parent.parent / 'sims'
        return str(sims_dir / 'testsim15.py')

    @pytest.fixture
    def network(self, testsim15_path):
        """Build network from testsim15.py."""
        return build_network(file=testsim15_path)

    # ==================[ Token Count Tests ]==================

    def test_total_token_count(self, network):
        """
        Test total token count.
        5 propositions * 7 tokens each (1 P + 2 RB + 4 PO) = 35 tokens
        """
        expected_tokens = 35
        actual_tokens = network.token_tensor.get_count()
        assert actual_tokens == expected_tokens, \
            f"Expected {expected_tokens} tokens, got {actual_tokens}"

    def test_token_count_by_type(self, network):
        """Test token counts by type."""
        tensor = network.token_tensor.tensor
        
        p_count = 0
        rb_count = 0
        po_count = 0
        
        for i in range(network.token_tensor.get_count()):
            token_type = tensor[i, TF.TYPE].item()
            if token_type == Type.P:
                p_count += 1
            elif token_type == Type.RB:
                rb_count += 1
            elif token_type == Type.PO:
                po_count += 1
        
        assert p_count == 5, f"Expected 5 P tokens, got {p_count}"
        assert rb_count == 10, f"Expected 10 RB tokens, got {rb_count}"
        assert po_count == 20, f"Expected 20 PO tokens, got {po_count}"

    def test_driver_token_count(self, network):
        """
        Test driver set token count.
        3 driver propositions * 7 tokens = 21 tokens
        """
        tensor = network.token_tensor.tensor
        driver_count = 0
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.SET].item() == Set.DRIVER:
                driver_count += 1
        
        assert driver_count == 21, f"Expected 21 driver tokens, got {driver_count}"

    def test_recipient_token_count(self, network):
        """
        Test recipient set token count.
        2 recipient propositions * 7 tokens = 14 tokens
        """
        tensor = network.token_tensor.tensor
        recipient_count = 0
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.SET].item() == Set.RECIPIENT:
                recipient_count += 1
        
        assert recipient_count == 14, f"Expected 14 recipient tokens, got {recipient_count}"

    def test_predicate_and_object_counts(self, network):
        """Test predicate vs object PO counts."""
        tensor = network.token_tensor.tensor
        pred_count = 0
        obj_count = 0
        
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.TYPE].item() == Type.PO:
                if tensor[i, TF.PRED].item() == B.TRUE:
                    pred_count += 1
                else:
                    obj_count += 1
        
        # 10 RBs * 1 predicate each = 10 predicates
        assert pred_count == 10, f"Expected 10 predicate POs, got {pred_count}"
        # 10 RBs * 1 object each = 10 objects
        assert obj_count == 10, f"Expected 10 object POs, got {obj_count}"

    # ==================[ Token Names Tests ]==================

    def test_proposition_names_exist(self, network):
        """Test that all proposition names are present."""
        names = list(network.token_tensor.names.values())
        
        expected_props = [
            'lovesMaryTom', 'lovesTomJane', 'jealousMaryJane',
            'lovesJohnKathy', 'lovesKathySam'
        ]
        
        for prop_name in expected_props:
            assert prop_name in names, f"Proposition '{prop_name}' not found in token names"

    def test_predicate_names_exist(self, network):
        """Test that all predicate names are present."""
        names = list(network.token_tensor.names.values())
        
        # Predicates (some appear multiple times)
        expected_preds = ['lover', 'beloved', 'jealous_act', 'jealous_pat']
        
        for pred_name in expected_preds:
            assert pred_name in names, f"Predicate '{pred_name}' not found in token names"

    def test_object_names_exist(self, network):
        """Test that all object names are present."""
        names = list(network.token_tensor.names.values())
        
        expected_objs = ['Mary', 'Tom', 'Jane', 'John', 'Kathy', 'Sam']
        
        for obj_name in expected_objs:
            assert obj_name in names, f"Object '{obj_name}' not found in token names"

    # ==================[ Connection Tests ]==================

    def test_total_connection_count(self, network):
        """
        Test total number of connections.
        - P -> RB: 5 props * 2 RBs = 10 connections
        - RB -> PO: 10 RBs * 2 POs = 20 connections
        - Total: 30 connections
        """
        connections = network.tokens.connections.connections
        total_connections = connections.sum().item()
        
        assert total_connections == 30, \
            f"Expected 30 connections, got {total_connections}"

    def test_p_to_rb_connections(self, network):
        """Test P -> RB connections count."""
        connections = network.tokens.connections.connections
        tensor = network.token_tensor.tensor
        
        p_to_rb_count = 0
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.TYPE].item() == Type.P:
                for j in range(network.token_tensor.get_count()):
                    if tensor[j, TF.TYPE].item() == Type.RB:
                        if connections[i, j]:
                            p_to_rb_count += 1
        
        # 5 P tokens, each with 2 RB children = 10
        assert p_to_rb_count == 10, \
            f"Expected 10 P->RB connections, got {p_to_rb_count}"

    def test_rb_to_po_connections(self, network):
        """Test RB -> PO connections count."""
        connections = network.tokens.connections.connections
        tensor = network.token_tensor.tensor
        
        rb_to_po_count = 0
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.TYPE].item() == Type.RB:
                for j in range(network.token_tensor.get_count()):
                    if tensor[j, TF.TYPE].item() == Type.PO:
                        if connections[i, j]:
                            rb_to_po_count += 1
        
        # 10 RB tokens, each with 2 PO children = 20
        assert rb_to_po_count == 20, \
            f"Expected 20 RB->PO connections, got {rb_to_po_count}"

    def test_each_p_has_two_rb_children(self, network):
        """Test that each P token has exactly 2 RB children."""
        connections = network.tokens.connections.connections
        tensor = network.token_tensor.tensor
        names = network.token_tensor.names
        
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.TYPE].item() == Type.P:
                child_count = 0
                for j in range(network.token_tensor.get_count()):
                    if connections[i, j] and tensor[j, TF.TYPE].item() == Type.RB:
                        child_count += 1
                
                assert child_count == 2, \
                    f"P token '{names[i]}' should have 2 RB children, got {child_count}"

    def test_each_rb_has_two_po_children(self, network):
        """Test that each RB token has exactly 2 PO children (1 pred + 1 obj)."""
        connections = network.tokens.connections.connections
        tensor = network.token_tensor.tensor
        names = network.token_tensor.names
        
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.TYPE].item() == Type.RB:
                pred_children = 0
                obj_children = 0
                for j in range(network.token_tensor.get_count()):
                    if connections[i, j] and tensor[j, TF.TYPE].item() == Type.PO:
                        if tensor[j, TF.PRED].item() == B.TRUE:
                            pred_children += 1
                        else:
                            obj_children += 1
                
                assert pred_children == 1, \
                    f"RB token '{names[i]}' should have 1 pred child, got {pred_children}"
                assert obj_children == 1, \
                    f"RB token '{names[i]}' should have 1 obj child, got {obj_children}"

    def test_no_invalid_connections(self, network):
        """Test that there are no invalid connection types."""
        connections = network.tokens.connections.connections
        tensor = network.token_tensor.tensor
        
        for i in range(network.token_tensor.get_count()):
            for j in range(network.token_tensor.get_count()):
                if connections[i, j]:
                    parent_type = tensor[i, TF.TYPE].item()
                    child_type = tensor[j, TF.TYPE].item()
                    
                    # Valid connections: P->RB, RB->PO
                    valid = (parent_type == Type.P and child_type == Type.RB) or \
                            (parent_type == Type.RB and child_type == Type.PO)
                    
                    assert valid, \
                        f"Invalid connection: type {parent_type} -> type {child_type}"

    # ==================[ Semantic Tests ]==================

    def test_unique_semantic_count(self, network):
        """
        Test number of unique semantics.
        
        Unique semantics in testsim15:
        - lover1, lover2, lover3 (3)
        - beloved1, beloved2, beloved3 (3)
        - jel1, jel2, jel3 (3)
        - jel4, jel5, jel6 (3)
        - mary1, mary2, mary3 (3)
        - tom1, tom2, tome3 (3) - note: tome3 not tom3
        - jane1, jane2 (2) - jane uses mary2 as 3rd semantic
        - john1, john2, john3 (3)
        - kathy1, kathy2, kathy3 (3)
        - sam1, sam2, sam3 (3)
        Total: 29 unique semantics
        
        Note: Network also adds 4 SDM semantics (MORE, LESS, SAME, DIFF)
        """
        sem_names = list(network.semantics.names.values())
        # Filter out SDM semantics
        non_sdm_sems = [s for s in sem_names if s not in ['MORE', 'LESS', 'SAME', 'DIFF']]
        
        assert len(non_sdm_sems) == 29, \
            f"Expected 29 unique semantics, got {len(non_sdm_sems)}"

    def test_semantic_names_present(self, network):
        """Test that expected semantic names are present."""
        sem_names = list(network.semantics.names.values())
        
        expected_sems = [
            'lover1', 'lover2', 'lover3',
            'beloved1', 'beloved2', 'beloved3',
            'jel1', 'jel2', 'jel3',
            'jel4', 'jel5', 'jel6',
            'mary1', 'mary2', 'mary3',
            'tom1', 'tom2', 'tome3',  # note: tome3 not tom3
            'jane1', 'jane2',
            'john1', 'john2', 'john3',
            'kathy1', 'kathy2', 'kathy3',
            'sam1', 'sam2', 'sam3'
        ]
        
        for sem_name in expected_sems:
            assert sem_name in sem_names, \
                f"Semantic '{sem_name}' not found"

    # ==================[ Links Tests ]==================

    def test_total_link_count(self, network):
        """
        Test total number of token-semantic links.
        20 PO tokens * 3 semantics each = 60 links
        """
        links = network.links.adj_matrix
        total_links = (links > 0).sum().item()
        
        assert total_links == 60, \
            f"Expected 60 links, got {total_links}"

    def test_each_po_has_three_links(self, network):
        """Test that each PO token has exactly 3 semantic links."""
        links = network.links.adj_matrix
        tensor = network.token_tensor.tensor
        names = network.token_tensor.names
        
        for i in range(network.token_tensor.get_count()):
            if tensor[i, TF.TYPE].item() == Type.PO:
                link_count = (links[i, :] > 0).sum().item()
                assert link_count == 3, \
                    f"PO token '{names[i]}' should have 3 links, got {link_count}"

    def test_non_po_tokens_have_no_links(self, network):
        """Test that P and RB tokens have no semantic links."""
        links = network.links.adj_matrix
        tensor = network.token_tensor.tensor
        names = network.token_tensor.names
        
        for i in range(network.token_tensor.get_count()):
            token_type = tensor[i, TF.TYPE].item()
            if token_type in [Type.P, Type.RB]:
                link_count = (links[i, :] > 0).sum().item()
                assert link_count == 0, \
                    f"Token '{names[i]}' (type {token_type}) should have no links, got {link_count}"

    def test_all_link_weights_are_one(self, network):
        """Test that all semantic link weights are 1.0."""
        links = network.links.adj_matrix
        non_zero_links = links[links > 0]
        
        assert (non_zero_links == 1.0).all(), \
            "All link weights should be 1.0"

    def test_shared_semantic_links(self, network):
        """Test that shared semantics are linked to multiple POs."""
        links = network.links.adj_matrix
        tensor = network.token_tensor.tensor
        names = network.token_tensor.names
        sem_names = network.semantics.names
        
        # Find 'lover1' semantic - should be linked to 4 predicates
        # (lover in: lovesMaryTom, lovesTomJane, lovesJohnKathy, lovesKathySam)
        lover1_sem_idx = None
        for sem_id, sem_name in sem_names.items():
            if sem_name == 'lover1':
                lover1_sem_idx = network.semantics.IDs[sem_id]
                break
        
        assert lover1_sem_idx is not None, "Semantic 'lover1' not found"
        
        # Count how many tokens link to lover1
        lover1_link_count = (links[:, lover1_sem_idx] > 0).sum().item()
        assert lover1_link_count == 4, \
            f"'lover1' should be linked to 4 tokens, got {lover1_link_count}"

    # ==================[ Specific Structure Tests ]==================

    def test_lovesMaryTom_structure(self, network):
        """Test the exact structure of the lovesMaryTom proposition."""
        connections = network.tokens.connections.connections
        tensor = network.token_tensor.tensor
        names = network.token_tensor.names
        
        # Find lovesMaryTom P token
        p_idx = None
        for idx, name in names.items():
            if name == 'lovesMaryTom' and tensor[idx, TF.TYPE].item() == Type.P:
                p_idx = idx
                break
        
        assert p_idx is not None, "lovesMaryTom P token not found"
        
        # Verify it's in driver set
        assert tensor[p_idx, TF.SET].item() == Set.DRIVER, \
            "lovesMaryTom should be in driver set"
        
        # Find its RB children
        rb_children = []
        for j in range(network.token_tensor.get_count()):
            if connections[p_idx, j]:
                rb_children.append(j)
        
        assert len(rb_children) == 2, \
            f"lovesMaryTom should have 2 RB children, got {len(rb_children)}"
        
        # Verify RBs have correct PO children
        expected_structure = {
            'lover_Mary': ('lover', 'Mary'),
            'beloved_Tom': ('beloved', 'Tom')
        }
        
        for rb_idx in rb_children:
            rb_name = names[rb_idx]
            assert rb_name in expected_structure, \
                f"Unexpected RB: {rb_name}"
            
            pred_name, obj_name = expected_structure[rb_name]
            
            # Find pred and obj children
            found_pred = False
            found_obj = False
            for po_idx in range(network.token_tensor.get_count()):
                if connections[rb_idx, po_idx]:
                    po_name = names[po_idx]
                    if po_name == pred_name and tensor[po_idx, TF.PRED].item() == B.TRUE:
                        found_pred = True
                    elif po_name == obj_name and tensor[po_idx, TF.PRED].item() == B.FALSE:
                        found_obj = True
            
            assert found_pred, f"RB '{rb_name}' missing predicate '{pred_name}'"
            assert found_obj, f"RB '{rb_name}' missing object '{obj_name}'"

    def test_jealousMaryJane_structure(self, network):
        """Test the exact structure of the jealousMaryJane proposition."""
        connections = network.tokens.connections.connections
        tensor = network.token_tensor.tensor
        names = network.token_tensor.names
        
        # Find jealousMaryJane P token
        p_idx = None
        for idx, name in names.items():
            if name == 'jealousMaryJane' and tensor[idx, TF.TYPE].item() == Type.P:
                p_idx = idx
                break
        
        assert p_idx is not None, "jealousMaryJane P token not found"
        
        # Find its RB children
        rb_children = []
        for j in range(network.token_tensor.get_count()):
            if connections[p_idx, j]:
                rb_children.append(j)
        
        assert len(rb_children) == 2
        
        # Verify expected RB names
        rb_names = {names[idx] for idx in rb_children}
        expected_rbs = {'jealous_act_Mary', 'jealous_pat_Jane'}
        
        assert rb_names == expected_rbs, \
            f"Expected RBs {expected_rbs}, got {rb_names}"

    # ==================[ Integration Tests ]==================

    def test_network_sets_accessible(self, network):
        """Test that network sets are properly accessible."""
        assert network.sets[Set.DRIVER] is not None
        assert network.sets[Set.RECIPIENT] is not None
        assert network.sets[Set.MEMORY] is not None
        assert network.sets[Set.NEW_SET] is not None

    def test_network_params_set(self, network):
        """Test that network parameters are properly set."""
        assert network.params is not None
        assert hasattr(network.params, 'eta')
        assert hasattr(network.params, 'gamma')

    def test_sdm_semantics_initialized(self, network):
        """Test that SDM semantics are initialized."""
        assert network.semantics.check_sdm_init(), \
            "SDM semantics should be initialized"

    def test_mapping_dimensions_correct(self, network):
        """Test that mapping tensor has correct dimensions."""
        mapping = network.mappings.adj_matrix
        
        # Driver has 21 tokens, Recipient has 14 tokens
        # Mapping shape: [recipient, driver, fields]
        assert mapping.shape[0] == 14, \
            f"Mapping recipient dimension should be 14, got {mapping.shape[0]}"
        assert mapping.shape[1] == 21, \
            f"Mapping driver dimension should be 21, got {mapping.shape[1]}"

