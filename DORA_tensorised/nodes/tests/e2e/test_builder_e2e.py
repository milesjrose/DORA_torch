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


# =====================[ sim_file and sym_file Format Tests ]======================

class TestSimFileFormat:
    """
    Tests for the 'sim_file' format (testsim.py).
    
    testsim.py structure:
    - 2 propositions total (1 driver, 1 recipient)
    - Each proposition has 2 RBs
    - Each RB has 1 predicate PO and 1 object PO
    
    Propositions:
    1. lovesMaryTom (driver): lover->Mary, beloved->Tom
    2. lovesTomMary (recipient): lover->Tom, beloved->Mary
    """
    
    @pytest.fixture
    def testsim_path(self):
        """Get the path to testsim.py (sim_file format)."""
        current_dir = Path(__file__).parent
        test_sims_dir = current_dir.parent / 'test_sims'
        return str(test_sims_dir / 'testsim.py')
    
    @pytest.fixture
    def network(self, testsim_path):
        """Build network from testsim.py."""
        return build_network(file=testsim_path)
    
    # ==================[ Token Count Tests ]==================
    
    def test_total_token_count(self, network):
        """
        Test total number of tokens.
        - 2 P tokens
        - 4 RB tokens (2 per proposition)
        - 8 PO tokens (4 predicates + 4 objects)
        Total: 14 tokens
        """
        token_count = network.token_tensor.tensor.shape[0]
        assert token_count == 14, f"Expected 14 tokens, got {token_count}"
    
    def test_p_token_count(self, network):
        """Test number of P tokens."""
        p_mask = network.token_tensor.tensor[:, TF.TYPE] == Type.P.value
        p_count = p_mask.sum().item()
        assert p_count == 2, f"Expected 2 P tokens, got {p_count}"
    
    def test_rb_token_count(self, network):
        """Test number of RB tokens."""
        rb_mask = network.token_tensor.tensor[:, TF.TYPE] == Type.RB.value
        rb_count = rb_mask.sum().item()
        assert rb_count == 4, f"Expected 4 RB tokens, got {rb_count}"
    
    def test_po_token_count(self, network):
        """Test number of PO tokens."""
        po_mask = network.token_tensor.tensor[:, TF.TYPE] == Type.PO.value
        po_count = po_mask.sum().item()
        assert po_count == 8, f"Expected 8 PO tokens, got {po_count}"
    
    # ==================[ Set Assignment Tests ]==================
    
    def test_driver_token_count(self, network):
        """Test number of tokens in driver set."""
        driver_mask = network.token_tensor.tensor[:, TF.SET] == Set.DRIVER.value
        driver_count = driver_mask.sum().item()
        # 1 P + 2 RB + 4 PO = 7 driver tokens
        assert driver_count == 7, f"Expected 7 driver tokens, got {driver_count}"
    
    def test_recipient_token_count(self, network):
        """Test number of tokens in recipient set."""
        recipient_mask = network.token_tensor.tensor[:, TF.SET] == Set.RECIPIENT.value
        recipient_count = recipient_mask.sum().item()
        # 1 P + 2 RB + 4 PO = 7 recipient tokens
        assert recipient_count == 7, f"Expected 7 recipient tokens, got {recipient_count}"
    
    # ==================[ Connection Tests ]==================
    
    def test_connections_exist(self, network):
        """Test that connections are created."""
        total_connections = network.tokens.connections.connections.sum().item()
        # 2 P->RB connections per P (4 total) + 2 RB->PO connections per RB (8 total) = 12
        assert total_connections == 12, f"Expected 12 connections, got {total_connections}"
    
    # ==================[ Semantic Tests ]==================
    
    def test_semantics_created(self, network):
        """Test that semantics are created."""
        semantic_count = len(network.semantics.names)
        # Unique semantics from testsim.py
        # lover: lover1, lover2, lover3 (3)
        # beloved: beloved1, beloved2, beloved3 (3)
        # Mary: mary1, mary2 (from driver) + mary3 (from recipient) = mary1, mary2, mary3 (3)
        # Tom: tom1, tom2, tom3 (3)
        # Base: 12 unique semantics + 4 comparative semantics (MORE, LESS, SAME, DIFF)
        # Total: 16 semantics
        assert semantic_count == 16, f"Expected 16 semantics, got {semantic_count}"
    
    # ==================[ Link Tests ]==================
    
    def test_links_created(self, network):
        """Test that PO-semantic links are created."""
        # Links uses adj_matrix tensor
        total_links = (network.links.adj_matrix > 0).sum().item()
        # 8 PO tokens, each has either 2 or 3 semantics
        # Driver lover: 3, Mary: 2, beloved: 3, Tom: 3
        # Recipient lover: 3, Tom: 3, beloved: 3, Mary: 3
        # Total links: 23
        assert total_links == 23, f"Expected 23 links, got {total_links}"


class TestSymFileFormat:
    """
    Tests for the 'sym_file' format (testsym.py).
    
    testsym.py structure:
    - 5 propositions total (3 driver, 2 recipient)
    - Each proposition has 2 RBs
    - Each RB has 1 predicate PO and 1 object PO
    
    Propositions:
    1. lovesMaryTom (driver): lover->Mary, beloved->Tom
    2. lovesTomJane (driver): lover->Tom, beloved->Jane
    3. jealousMaryJane (driver): jealous_act->Mary, jealous_pat->Jane
    4. lovesJohnKathy (recipient): lover->John, beloved->Kathy
    5. lovesKathySam (recipient): lover->Kathy, beloved->Sam
    """
    
    @pytest.fixture
    def testsym_path(self):
        """Get the path to testsym.py (sym_file format)."""
        current_dir = Path(__file__).parent
        test_sims_dir = current_dir.parent / 'test_sims'
        return str(test_sims_dir / 'testsym.py')
    
    @pytest.fixture
    def network(self, testsym_path):
        """Build network from testsym.py."""
        return build_network(file=testsym_path)
    
    # ==================[ Token Count Tests ]==================
    
    def test_total_token_count(self, network):
        """
        Test total number of tokens.
        - 5 P tokens
        - 10 RB tokens (2 per proposition)
        - 20 PO tokens (10 predicates + 10 objects)
        Total: 35 tokens
        """
        token_count = network.token_tensor.tensor.shape[0]
        assert token_count == 35, f"Expected 35 tokens, got {token_count}"
    
    def test_p_token_count(self, network):
        """Test number of P tokens."""
        p_mask = network.token_tensor.tensor[:, TF.TYPE] == Type.P.value
        p_count = p_mask.sum().item()
        assert p_count == 5, f"Expected 5 P tokens, got {p_count}"
    
    def test_rb_token_count(self, network):
        """Test number of RB tokens."""
        rb_mask = network.token_tensor.tensor[:, TF.TYPE] == Type.RB.value
        rb_count = rb_mask.sum().item()
        assert rb_count == 10, f"Expected 10 RB tokens, got {rb_count}"
    
    def test_po_token_count(self, network):
        """Test number of PO tokens."""
        po_mask = network.token_tensor.tensor[:, TF.TYPE] == Type.PO.value
        po_count = po_mask.sum().item()
        assert po_count == 20, f"Expected 20 PO tokens, got {po_count}"
    
    # ==================[ Set Assignment Tests ]==================
    
    def test_driver_token_count(self, network):
        """Test number of tokens in driver set."""
        driver_mask = network.token_tensor.tensor[:, TF.SET] == Set.DRIVER.value
        driver_count = driver_mask.sum().item()
        # 3 P + 6 RB + 12 PO = 21 driver tokens
        assert driver_count == 21, f"Expected 21 driver tokens, got {driver_count}"
    
    def test_recipient_token_count(self, network):
        """Test number of tokens in recipient set."""
        recipient_mask = network.token_tensor.tensor[:, TF.SET] == Set.RECIPIENT.value
        recipient_count = recipient_mask.sum().item()
        # 2 P + 4 RB + 8 PO = 14 recipient tokens
        assert recipient_count == 14, f"Expected 14 recipient tokens, got {recipient_count}"
    
    # ==================[ Connection Tests ]==================
    
    def test_connections_exist(self, network):
        """Test that connections are created."""
        total_connections = network.tokens.connections.connections.sum().item()
        # 5 P tokens with 2 RBs each = 10 P->RB connections
        # 10 RB tokens with 2 POs each = 20 RB->PO connections
        # Total: 30 connections
        assert total_connections == 30, f"Expected 30 connections, got {total_connections}"
    
    # ==================[ Semantic Tests ]==================
    
    def test_semantics_created(self, network):
        """Test that semantics are created."""
        semantic_count = len(network.semantics.names)
        # Unique semantics from testsym.py + 4 comparative semantics (MORE, LESS, SAME, DIFF)
        # Note: some typos in the file (tome3, tome3) - these count as unique
        assert semantic_count > 4, "Should have semantics (at least 4 comparative)"
    
    # ==================[ Link Tests ]==================
    
    def test_links_created(self, network):
        """Test that PO-semantic links are created."""
        # Links uses adj_matrix tensor
        total_links = (network.links.adj_matrix > 0).sum().item()
        # 20 PO tokens, each with 3 semantics = 60 links
        assert total_links == 60, f"Expected 60 links, got {total_links}"


class TestBothFormatsProduceValidNetworks:
    """
    Tests that verify both file formats produce structurally valid networks.
    """
    
    @pytest.fixture
    def sim_file_path(self):
        """Get the path to testsim.py (sim_file format)."""
        current_dir = Path(__file__).parent
        test_sims_dir = current_dir.parent / 'test_sims'
        return str(test_sims_dir / 'testsim.py')
    
    @pytest.fixture
    def sym_file_path(self):
        """Get the path to testsym.py (sym_file format)."""
        current_dir = Path(__file__).parent
        test_sims_dir = current_dir.parent / 'test_sims'
        return str(test_sims_dir / 'testsym.py')
    
    @pytest.fixture
    def sim_network(self, sim_file_path):
        """Build network from sim_file format."""
        return build_network(file=sim_file_path)
    
    @pytest.fixture
    def sym_network(self, sym_file_path):
        """Build network from sym_file format."""
        return build_network(file=sym_file_path)
    
    # ==================[ Structure Validity Tests ]==================
    
    def test_sim_file_has_valid_structure(self, sim_network):
        """Test that sim_file produces a valid network structure."""
        # Has tokens
        assert sim_network.token_tensor is not None
        assert sim_network.token_tensor.tensor.shape[0] > 0
        
        # Has connections
        assert sim_network.tokens.connections is not None
        assert sim_network.tokens.connections.connections.sum() > 0
        
        # Has semantics
        assert sim_network.semantics is not None
        assert len(sim_network.semantics.names) > 0
        
        # Has links
        assert sim_network.links is not None
        total_links = (sim_network.links.adj_matrix > 0).sum().item()
        assert total_links > 0
        
        # Has mappings
        assert sim_network.mappings is not None
    
    def test_sym_file_has_valid_structure(self, sym_network):
        """Test that sym_file produces a valid network structure."""
        # Has tokens
        assert sym_network.token_tensor is not None
        assert sym_network.token_tensor.tensor.shape[0] > 0
        
        # Has connections
        assert sym_network.tokens.connections is not None
        assert sym_network.tokens.connections.connections.sum() > 0
        
        # Has semantics
        assert sym_network.semantics is not None
        assert len(sym_network.semantics.names) > 0
        
        # Has links
        assert sym_network.links is not None
        total_links = (sym_network.links.adj_matrix > 0).sum().item()
        assert total_links > 0
        
        # Has mappings
        assert sym_network.mappings is not None
    
    def test_both_formats_produce_consistent_token_features(self, sim_network, sym_network):
        """Test that both formats produce tokens with the same feature dimensions."""
        sim_features = sim_network.token_tensor.tensor.shape[1]
        sym_features = sym_network.token_tensor.tensor.shape[1]
        assert sim_features == sym_features, \
            f"Both formats should produce tokens with same features: sim={sim_features}, sym={sym_features}"
    
    def test_both_formats_set_token_ids_correctly(self, sim_network, sym_network):
        """Test that both formats assign sequential token IDs starting from 1."""
        for network, name in [(sim_network, 'sim'), (sym_network, 'sym')]:
            ids = [int(x) for x in network.token_tensor.tensor[:, TF.ID].tolist()]
            expected_ids = list(range(1, len(ids) + 1))  # IDs start from 1
            assert ids == expected_ids, \
                f"{name} file should have sequential IDs starting from 1: got {ids[:5]}..."
    
    def test_both_formats_produce_square_connection_matrices(self, sim_network, sym_network):
        """Test that connection matrices are square (n_tokens x n_tokens)."""
        for network, name in [(sim_network, 'sim'), (sym_network, 'sym')]:
            n_tokens = network.token_tensor.tensor.shape[0]
            conn_shape = network.tokens.connections.connections.shape
            assert conn_shape == (n_tokens, n_tokens), \
                f"{name} file connections should be ({n_tokens}, {n_tokens}), got {conn_shape}"
    
    def test_both_formats_produce_correct_link_dimensions(self, sim_network, sym_network):
        """Test that link matrices have correct dimensions (n_tokens x n_semantics)."""
        for network, name in [(sim_network, 'sim'), (sym_network, 'sym')]:
            n_tokens = network.token_tensor.tensor.shape[0]
            n_semantics = len(network.semantics.names)
            link_shape = network.links.adj_matrix.shape
            # Token dimension should match exactly
            assert link_shape[0] == n_tokens, \
                f"{name} file links token dimension should be {n_tokens}, got {link_shape[0]}"
            # Semantics dimension should be at least as large as number of named semantics
            # (may be larger due to pre-allocation during tensor expansion)
            assert link_shape[1] >= n_semantics, \
                f"{name} file links semantics dimension should be >= {n_semantics}, got {link_shape[1]}"


# =====================[ Reference Equality Tests ]======================

class TestObjectReferences:
    """
    Tests that verify objects that should share references actually do.
    This is critical for ensuring changes to one view affect all related objects.
    """

    @pytest.fixture
    def testsim15_path(self):
        """Get the path to testsim15.py."""
        current_dir = Path(__file__).parent
        sims_dir = current_dir.parent.parent.parent / 'sims'
        return str(sims_dir / 'testsim15.py')

    @pytest.fixture
    def network(self, testsim15_path):
        """Build network from testsim15.py."""
        return build_network(file=testsim15_path)

    # ==================[ Token Tensor References ]==================

    def test_network_token_tensor_is_tokens_token_tensor(self, network):
        """
        Test that network.token_tensor is the same object as network.tokens.token_tensor.
        """
        assert network.token_tensor is network.tokens.token_tensor, \
            "network.token_tensor should be the same object as network.tokens.token_tensor"

    def test_connections_shared_between_tokens_and_token_tensor(self, network):
        """
        Test that the connections object is shared between Tokens and Token_Tensor.
        """
        assert network.tokens.connections is network.tokens.token_tensor.connections, \
            "network.tokens.connections should be the same object as network.tokens.token_tensor.connections"

    def test_network_links_is_tokens_links(self, network):
        """
        Test that network.links is the same object as network.tokens.links.
        """
        assert network.links is network.tokens.links, \
            "network.links should be the same object as network.tokens.links"

    def test_network_mappings_is_tokens_mapping(self, network):
        """
        Test that network.mappings is the same object as network.tokens.mapping.
        """
        assert network.mappings is network.tokens.mapping, \
            "network.mappings should be the same object as network.tokens.mapping"

    # ==================[ Set Token Tensor References ]==================

    def test_driver_set_shares_token_tensor(self, network):
        """
        Test that driver set uses the same token_tensor as the network.
        Sets use 'glbl' attribute to reference the global Token_Tensor.
        """
        assert network.sets[Set.DRIVER].glbl is network.token_tensor, \
            "Driver set should share token_tensor with network"

    def test_recipient_set_shares_token_tensor(self, network):
        """
        Test that recipient set uses the same token_tensor as the network.
        """
        assert network.sets[Set.RECIPIENT].glbl is network.token_tensor, \
            "Recipient set should share token_tensor with network"

    def test_memory_set_shares_token_tensor(self, network):
        """
        Test that memory set uses the same token_tensor as the network.
        """
        assert network.sets[Set.MEMORY].glbl is network.token_tensor, \
            "Memory set should share token_tensor with network"

    def test_new_set_shares_token_tensor(self, network):
        """
        Test that new_set uses the same token_tensor as the network.
        """
        assert network.sets[Set.NEW_SET].glbl is network.token_tensor, \
            "New_Set should share token_tensor with network"

    def test_all_sets_share_same_token_tensor(self, network):
        """
        Test that all sets share the exact same token_tensor object.
        """
        token_tensors = [network.sets[s].glbl for s in Set]
        first = token_tensors[0]
        for i, tt in enumerate(token_tensors[1:], 1):
            assert tt is first, \
                f"Set {list(Set)[i]} has different token_tensor than Set {list(Set)[0]}"

    # ==================[ Underlying Tensor References ]==================

    def test_token_tensor_data_shared(self, network):
        """
        Test that the underlying torch tensor is the same object.
        """
        assert network.token_tensor.tensor is network.tokens.token_tensor.tensor, \
            "The underlying tensor data should be the same object"

    def test_connections_tensor_data_shared(self, network):
        """
        Test that the underlying connections torch tensor is the same object.
        """
        assert network.tokens.connections.connections is network.tokens.token_tensor.connections.connections, \
            "The underlying connections tensor should be the same object"

    def test_links_tensor_data_shared(self, network):
        """
        Test that the underlying links torch tensor is the same object.
        """
        assert network.links.adj_matrix is network.tokens.links.adj_matrix, \
            "The underlying links tensor should be the same object"

    def test_mapping_tensor_data_shared(self, network):
        """
        Test that the underlying mapping torch tensor is the same object.
        """
        assert network.mappings.adj_matrix is network.tokens.mapping.adj_matrix, \
            "The underlying mapping tensor should be the same object"

    # ==================[ Mutation Tests ]==================

    def test_mutation_through_tokens_affects_network(self, network):
        """
        Test that modifying data through tokens affects network's view.
        """
        # Get an activation value
        original_act = network.token_tensor.tensor[0, TF.ACT].item()
        
        # Modify through tokens
        network.tokens.token_tensor.tensor[0, TF.ACT] = 999.0
        
        # Check it's visible through network (use pytest.approx for float32 precision)
        assert network.token_tensor.tensor[0, TF.ACT].item() == pytest.approx(999.0, rel=1e-5), \
            "Changes through tokens should be visible through network"
        
        # Restore original value
        network.token_tensor.tensor[0, TF.ACT] = original_act

    def test_mutation_through_network_affects_tokens(self, network):
        """
        Test that modifying data through network affects tokens' view.
        """
        # Get an activation value
        original_act = network.tokens.token_tensor.tensor[0, TF.ACT].item()
        
        # Modify through network
        network.token_tensor.tensor[0, TF.ACT] = 888.0
        
        # Check it's visible through tokens (use pytest.approx for float32 precision)
        assert network.tokens.token_tensor.tensor[0, TF.ACT].item() == pytest.approx(888.0, rel=1e-5), \
            "Changes through network should be visible through tokens"
        
        # Restore original value
        network.tokens.token_tensor.tensor[0, TF.ACT] = original_act

    def test_connection_mutation_affects_both_views(self, network):
        """
        Test that modifying connections through one reference affects both.
        """
        # Store original state
        original_val = network.tokens.connections.connections[0, 1].item()
        
        # Modify through tokens.connections
        network.tokens.connections.connections[0, 1] = not original_val
        
        # Check it's visible through token_tensor.connections
        assert network.tokens.token_tensor.connections.connections[0, 1].item() == (not original_val), \
            "Connection changes should be visible through token_tensor.connections"
        
        # Restore
        network.tokens.connections.connections[0, 1] = original_val

    # ==================[ Cache Reference Tests ]==================

    def test_cache_references_correct_tensor(self, network):
        """
        Test that the cache references the correct tensor.
        """
        assert network.token_tensor.cache.tensor is network.token_tensor.tensor, \
            "Cache should reference the token_tensor's tensor"

    # ==================[ Semantics Links Reference ]==================

    def test_semantics_links_reference(self, network):
        """
        Test that semantics.links is the same as network.links.
        """
        assert network.semantics.links is network.links, \
            "Semantics should share the same links object as network"
