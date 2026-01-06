# nodes/tests/funct/test_printer.py
# Functional tests for the Printer class

import pytest
import torch
import os
from pathlib import Path
from io import StringIO
import sys

from nodes.utils.new_printer import Printer
from nodes.network.tokens.tensor.token_tensor import Token_Tensor
from nodes.enums import Set, Type, TF, B, Mode, null, tensor_type, MappingFields


# =====================[ Output File Path ]======================
OUTPUT_FILE = Path(__file__).parent / "printer_output.txt"


# =====================[ Test Data / Fixtures ]======================

@pytest.fixture
def test_tensor():
    """
    Create a test tensor with various token types and values.
    Creates tokens that exercise different enum types, bools, ints, and floats.
    """
    num_tokens = 10
    num_features = len(TF)
    
    # Create tensor with null values
    tensor = torch.full((num_tokens, num_features), null, dtype=tensor_type)
    
    # Token 0: PO token in DRIVER set (object)
    tensor[0, TF.ID] = 0
    tensor[0, TF.TYPE] = Type.PO
    tensor[0, TF.SET] = Set.DRIVER
    tensor[0, TF.ANALOG] = 0
    tensor[0, TF.ACT] = 0.75
    tensor[0, TF.MAX_ACT] = 0.9
    tensor[0, TF.SEM_COUNT] = 3
    tensor[0, TF.PRED] = B.FALSE
    tensor[0, TF.DELETED] = B.FALSE
    tensor[0, TF.INFERRED] = B.FALSE
    tensor[0, TF.RETRIEVED] = B.FALSE
    
    # Token 1: PO token in DRIVER set (predicate)
    tensor[1, TF.ID] = 1
    tensor[1, TF.TYPE] = Type.PO
    tensor[1, TF.SET] = Set.DRIVER
    tensor[1, TF.ANALOG] = 0
    tensor[1, TF.ACT] = 0.85
    tensor[1, TF.MAX_ACT] = 0.95
    tensor[1, TF.SEM_COUNT] = 2
    tensor[1, TF.PRED] = B.TRUE
    tensor[1, TF.DELETED] = B.FALSE
    tensor[1, TF.INFERRED] = B.FALSE
    tensor[1, TF.RETRIEVED] = B.FALSE
    
    # Token 2: RB token in DRIVER set
    tensor[2, TF.ID] = 2
    tensor[2, TF.TYPE] = Type.RB
    tensor[2, TF.SET] = Set.DRIVER
    tensor[2, TF.ANALOG] = 0
    tensor[2, TF.ACT] = 0.65
    tensor[2, TF.MAX_ACT] = 0.7
    tensor[2, TF.TIMES_FIRED] = 5
    tensor[2, TF.DELETED] = B.FALSE
    tensor[2, TF.INFERRED] = B.FALSE
    tensor[2, TF.RETRIEVED] = B.FALSE
    
    # Token 3: P token in DRIVER set (neutral mode)
    tensor[3, TF.ID] = 3
    tensor[3, TF.TYPE] = Type.P
    tensor[3, TF.SET] = Set.DRIVER
    tensor[3, TF.ANALOG] = 0
    tensor[3, TF.ACT] = 0.55
    tensor[3, TF.MAX_ACT] = 0.6
    tensor[3, TF.MODE] = Mode.NEUTRAL
    tensor[3, TF.DELETED] = B.FALSE
    tensor[3, TF.INFERRED] = B.FALSE
    tensor[3, TF.RETRIEVED] = B.FALSE
    
    # Token 4: PO token in RECIPIENT set (object)
    tensor[4, TF.ID] = 4
    tensor[4, TF.TYPE] = Type.PO
    tensor[4, TF.SET] = Set.RECIPIENT
    tensor[4, TF.ANALOG] = 1
    tensor[4, TF.ACT] = 0.45
    tensor[4, TF.MAX_ACT] = 0.5
    tensor[4, TF.SEM_COUNT] = 4
    tensor[4, TF.PRED] = B.FALSE
    tensor[4, TF.DELETED] = B.FALSE
    tensor[4, TF.INFERRED] = B.TRUE
    tensor[4, TF.RETRIEVED] = B.FALSE
    
    # Token 5: RB token in RECIPIENT set
    tensor[5, TF.ID] = 5
    tensor[5, TF.TYPE] = Type.RB
    tensor[5, TF.SET] = Set.RECIPIENT
    tensor[5, TF.ANALOG] = 1
    tensor[5, TF.ACT] = 0.35
    tensor[5, TF.MAX_ACT] = 0.4
    tensor[5, TF.TIMES_FIRED] = 2
    tensor[5, TF.DELETED] = B.FALSE
    tensor[5, TF.INFERRED] = B.FALSE
    tensor[5, TF.RETRIEVED] = B.TRUE
    
    # Token 6: P token in RECIPIENT set (child mode)
    tensor[6, TF.ID] = 6
    tensor[6, TF.TYPE] = Type.P
    tensor[6, TF.SET] = Set.RECIPIENT
    tensor[6, TF.ANALOG] = 1
    tensor[6, TF.ACT] = 0.25
    tensor[6, TF.MAX_ACT] = 0.3
    tensor[6, TF.MODE] = Mode.CHILD
    tensor[6, TF.DELETED] = B.FALSE
    tensor[6, TF.INFERRED] = B.FALSE
    tensor[6, TF.RETRIEVED] = B.FALSE
    
    # Token 7: PO token in MEMORY set
    tensor[7, TF.ID] = 7
    tensor[7, TF.TYPE] = Type.PO
    tensor[7, TF.SET] = Set.MEMORY
    tensor[7, TF.ANALOG] = 2
    tensor[7, TF.ACT] = 0.15
    tensor[7, TF.MAX_ACT] = 0.2
    tensor[7, TF.SEM_COUNT] = 5
    tensor[7, TF.PRED] = B.TRUE
    tensor[7, TF.DELETED] = B.FALSE
    tensor[7, TF.INFERRED] = B.FALSE
    tensor[7, TF.RETRIEVED] = B.FALSE
    
    # Token 8: GROUP token in NEW_SET
    tensor[8, TF.ID] = 8
    tensor[8, TF.TYPE] = Type.GROUP
    tensor[8, TF.SET] = Set.NEW_SET
    tensor[8, TF.ANALOG] = 3
    tensor[8, TF.ACT] = 0.05
    tensor[8, TF.MAX_ACT] = 0.1
    tensor[8, TF.GROUP_LAYER] = 2
    tensor[8, TF.DELETED] = B.FALSE
    tensor[8, TF.INFERRED] = B.TRUE
    tensor[8, TF.RETRIEVED] = B.FALSE
    
    # Token 9: Deleted token
    tensor[9, TF.ID] = 9
    tensor[9, TF.TYPE] = Type.PO
    tensor[9, TF.SET] = Set.DRIVER
    tensor[9, TF.DELETED] = B.TRUE
    
    return tensor


@pytest.fixture
def test_connections(test_tensor):
    """
    Create mock connections tensor with realistic DORA structure.
    Connections represent parent -> child relationships:
    - P tokens connect to RB tokens
    - RB tokens connect to PO tokens (both pred and obj)
    
    Structure:
    Token 0: Mary_obj (PO object)
    Token 1: lover_pred (PO predicate)
    Token 2: lovesMary_RB (RB) - connects to tokens 0, 1
    Token 3: lovesMary_P (P) - connects to token 2
    Token 4: John_obj (PO object)
    Token 5: lovesJohn_RB (RB) - connects to tokens 4, 1
    Token 6: lovesJohn_P (P) - connects to token 5
    Token 7: Tom_pred (PO predicate)
    Token 8: group1 (GROUP)
    Token 9: deleted_token (DELETED)
    """
    num_tokens = test_tensor.size(0)
    connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
    
    # P -> RB connections
    connections[3, 2] = True  # lovesMary_P -> lovesMary_RB
    connections[6, 5] = True  # lovesJohn_P -> lovesJohn_RB
    
    # RB -> PO connections
    connections[2, 0] = True  # lovesMary_RB -> Mary_obj
    connections[2, 1] = True  # lovesMary_RB -> lover_pred
    connections[5, 4] = True  # lovesJohn_RB -> John_obj
    connections[5, 1] = True  # lovesJohn_RB -> lover_pred (shared)
    
    return connections


@pytest.fixture
def test_names():
    """Create mock names dictionary."""
    return {
        0: "Mary_obj",
        1: "lover_pred",
        2: "lovesMary_RB",
        3: "lovesMary_P",
        4: "John_obj",
        5: "lovesJohn_RB",
        6: "lovesJohn_P",
        7: "Tom_pred",
        8: "group1",
        9: "deleted_token"
    }


@pytest.fixture
def token_tensor(test_tensor, test_connections, test_names):
    """Create a Token_Tensor instance with test data."""
    return Token_Tensor(test_tensor, test_connections, test_names)


def get_token_description(tensor: torch.Tensor, names: dict) -> str:
    """
    Generate a human-readable description of the tokens in the tensor.
    This is used to create a reference for visual comparison.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("TOKEN TENSOR DESCRIPTION (for visual comparison)")
    lines.append("=" * 80)
    lines.append("")
    
    for idx in range(tensor.size(0)):
        deleted = tensor[idx, TF.DELETED].item()
        if deleted == B.TRUE:
            status = "DELETED"
        else:
            status = "ACTIVE"
        
        name = names.get(idx, f"token_{idx}")
        token_type = int(tensor[idx, TF.TYPE].item()) if tensor[idx, TF.TYPE].item() != null else "null"
        token_set = int(tensor[idx, TF.SET].item()) if tensor[idx, TF.SET].item() != null else "null"
        analog = int(tensor[idx, TF.ANALOG].item()) if tensor[idx, TF.ANALOG].item() != null else "null"
        act = tensor[idx, TF.ACT].item() if tensor[idx, TF.ACT].item() != null else "null"
        
        # Convert to enum names for readability
        try:
            type_name = Type(token_type).name if token_type != "null" else "null"
        except (ValueError, TypeError):
            type_name = str(token_type)
        
        try:
            set_name = Set(token_set).name if token_set != "null" else "null"
        except (ValueError, TypeError):
            set_name = str(token_set)
        
        lines.append(f"Token {idx}: {name}")
        lines.append(f"  Status: {status}")
        lines.append(f"  Type: {type_name} (raw: {token_type})")
        lines.append(f"  Set: {set_name} (raw: {token_set})")
        lines.append(f"  Analog: {analog}")
        lines.append(f"  ACT: {act}")
        
        # Add type-specific fields
        if type_name == "PO":
            pred = tensor[idx, TF.PRED].item()
            pred_str = "True (predicate)" if pred == B.TRUE else "False (object)"
            sem_count = int(tensor[idx, TF.SEM_COUNT].item()) if tensor[idx, TF.SEM_COUNT].item() != null else "null"
            lines.append(f"  PRED: {pred_str}")
            lines.append(f"  SEM_COUNT: {sem_count}")
        elif type_name == "RB":
            times_fired = int(tensor[idx, TF.TIMES_FIRED].item()) if tensor[idx, TF.TIMES_FIRED].item() != null else "null"
            lines.append(f"  TIMES_FIRED: {times_fired}")
        elif type_name == "P":
            mode = int(tensor[idx, TF.MODE].item()) if tensor[idx, TF.MODE].item() != null else "null"
            try:
                mode_name = Mode(mode).name if mode != "null" else "null"
            except (ValueError, TypeError):
                mode_name = str(mode)
            lines.append(f"  MODE: {mode_name} (raw: {mode})")
        elif type_name == "GROUP":
            group_layer = int(tensor[idx, TF.GROUP_LAYER].item()) if tensor[idx, TF.GROUP_LAYER].item() != null else "null"
            lines.append(f"  GROUP_LAYER: {group_layer}")
        
        # Add bool fields
        inferred = tensor[idx, TF.INFERRED].item()
        retrieved = tensor[idx, TF.RETRIEVED].item()
        lines.append(f"  INFERRED: {'True' if inferred == B.TRUE else 'False'}")
        lines.append(f"  RETRIEVED: {'True' if retrieved == B.TRUE else 'False'}")
        lines.append("")
    
    return "\n".join(lines)


def capture_printer_output(printer: Printer, token_tensor, **kwargs) -> str:
    """
    Capture the output of the printer to a string.
    Uses a temporary file since tablePrinter writes to log_file, not stdout when print_to_console=False.
    """
    import tempfile
    
    # Create a temporary file to capture output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a new printer with the temp file as log destination
        capture_printer = Printer(
            use_labels=printer.use_labels,
            log_file=tmp_path,
            print_to_console=False
        )
        capture_printer.print_token_tensor(token_tensor, **kwargs)
        
        # Read the captured output
        with open(tmp_path, 'r', encoding='utf-8') as f:
            return f.read()
    finally:
        # Clean up temp file
        import os
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def write_test_output(content: str, section_name: str, output_file: Path = OUTPUT_FILE):
    """
    Append content to the output file with a section header.
    """
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"SECTION: {section_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(content)
        f.write("\n")


# =====================[ Setup / Teardown ]======================

@pytest.fixture(scope="module", autouse=True)
def setup_output_file():
    """Clear and initialize the output file before tests run."""
    # Clear the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("PRINTER TEST OUTPUT\n")
        f.write("Generated by test_printer.py\n")
        f.write("=" * 80 + "\n")
    
    yield
    
    # After all tests, add a footer
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF TEST OUTPUT\n")
        f.write("=" * 80 + "\n")


# =====================[ Tests ]======================

class TestPrinterInit:
    """Tests for Printer initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        printer = Printer()
        assert printer.use_labels == True
        assert printer.log_file is None
        assert printer.print_to_console == True
    
    def test_custom_init(self):
        """Test custom initialization."""
        printer = Printer(use_labels=False, log_file="test.log", print_to_console=False)
        assert printer.use_labels == False
        assert printer.log_file == "test.log"
        assert printer.print_to_console == False


class TestPrintTokenTensor:
    """Tests for printing token tensors."""
    
    def test_print_with_labels(self, token_tensor, test_tensor, test_names):
        """Test printing with labels (enum names, True/False for bools)."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Capture output
        output = capture_printer_output(printer, token_tensor, cols_per_table=8)
        
        # Write to output file for visual inspection
        description = get_token_description(test_tensor, test_names)
        write_test_output(description, "Token Description (Reference)")
        write_test_output(output, "Print with Labels (use_labels=True)")
        
        # Assertions - check that labels appear instead of raw values
        assert "DRIVER" in output or "RECIPIENT" in output or "MEMORY" in output
        assert "PO" in output or "RB" in output or "P" in output
        assert "True" in output or "False" in output
    
    def test_print_without_labels(self, token_tensor):
        """Test printing with raw float values."""
        printer = Printer(use_labels=False, print_to_console=False)
        
        # Capture output
        output = capture_printer_output(printer, token_tensor, cols_per_table=8)
        
        # Write to output file for visual inspection
        write_test_output(output, "Print without Labels (use_labels=False)")
        
        # Assertions - raw values should appear
        # SET.DRIVER = 0, SET.RECIPIENT = 1
        # Type.PO = 0, Type.RB = 1, Type.P = 2
        assert output  # Output should not be empty
    
    def test_print_all_columns_single_table(self, token_tensor):
        """Test printing all columns in a single table."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with cols_per_table=None (all columns)
        output = capture_printer_output(printer, token_tensor, cols_per_table=None)
        
        write_test_output(output, "All Columns in Single Table (cols_per_table=None)")
        
        # Should have all TF feature names in one contiguous section
        assert "ID" in output
        assert "TYPE" in output
        assert "SET" in output
        assert "ACT" in output
    
    def test_print_with_show_deleted(self, token_tensor):
        """Test printing including deleted tokens."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with show_deleted=True
        output = capture_printer_output(printer, token_tensor, show_deleted=True, cols_per_table=8)
        
        write_test_output(output, "Including Deleted Tokens (show_deleted=True)")
        
        # Should include deleted_token
        assert "deleted_token" in output
    
    def test_print_without_deleted(self, token_tensor):
        """Test printing excluding deleted tokens (default)."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with show_deleted=False (default)
        output = capture_printer_output(printer, token_tensor, show_deleted=False, cols_per_table=8)
        
        # Should NOT include deleted_token
        assert "deleted_token" not in output
    
    def test_print_specific_indices(self, token_tensor):
        """Test printing only specific token indices."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print only indices 0, 2, 4
        indices = torch.tensor([0, 2, 4])
        output = capture_printer_output(printer, token_tensor, indices=indices, cols_per_table=8)
        
        write_test_output(output, "Specific Indices [0, 2, 4]")
        
        # Should include tokens at specified indices
        assert "Mary_obj" in output
        assert "lovesMary_RB" in output
        assert "John_obj" in output
        
        # Should NOT include other tokens
        assert "lover_pred" not in output
        assert "lovesJohn_RB" not in output
    
    def test_print_different_column_widths(self, token_tensor):
        """Test printing with different column widths."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with 4 columns per table
        output_4 = capture_printer_output(printer, token_tensor, cols_per_table=4)
        write_test_output(output_4, "4 Columns per Table")
        
        # Print with 16 columns per table
        output_16 = capture_printer_output(printer, token_tensor, cols_per_table=16)
        write_test_output(output_16, "16 Columns per Table")
        
        # Both should have content
        assert len(output_4) > 0
        assert len(output_16) > 0


class TestFormatValue:
    """Tests for the _format_value method."""
    
    def test_format_null_value(self):
        """Test formatting null values."""
        printer = Printer(use_labels=True)
        result = printer._format_value(TF.ACT, null)
        assert result == "null"
    
    def test_format_type_enum(self):
        """Test formatting Type enum values."""
        printer = Printer(use_labels=True)
        
        assert printer._format_value(TF.TYPE, Type.PO) == "PO"
        assert printer._format_value(TF.TYPE, Type.RB) == "RB"
        assert printer._format_value(TF.TYPE, Type.P) == "P"
        assert printer._format_value(TF.TYPE, Type.GROUP) == "GROUP"
    
    def test_format_set_enum(self):
        """Test formatting Set enum values."""
        printer = Printer(use_labels=True)
        
        assert printer._format_value(TF.SET, Set.DRIVER) == "DRIVER"
        assert printer._format_value(TF.SET, Set.RECIPIENT) == "RECIPIENT"
        assert printer._format_value(TF.SET, Set.MEMORY) == "MEMORY"
        assert printer._format_value(TF.SET, Set.NEW_SET) == "NEW_SET"
    
    def test_format_mode_enum(self):
        """Test formatting Mode enum values."""
        printer = Printer(use_labels=True)
        
        assert printer._format_value(TF.MODE, Mode.CHILD) == "CHILD"
        assert printer._format_value(TF.MODE, Mode.NEUTRAL) == "NEUTRAL"
        assert printer._format_value(TF.MODE, Mode.PARENT) == "PARENT"
    
    def test_format_bool_values(self):
        """Test formatting boolean values."""
        printer = Printer(use_labels=True)
        
        assert printer._format_value(TF.DELETED, B.TRUE) == "True"
        assert printer._format_value(TF.DELETED, B.FALSE) == "False"
        assert printer._format_value(TF.INFERRED, B.TRUE) == "True"
        assert printer._format_value(TF.INFERRED, B.FALSE) == "False"
    
    def test_format_int_values(self):
        """Test formatting integer values."""
        printer = Printer(use_labels=True)
        
        result = printer._format_value(TF.ID, 42.0)
        assert result == "42"
        
        result = printer._format_value(TF.ANALOG, 5.0)
        assert result == "5"
    
    def test_format_float_values(self):
        """Test formatting float values."""
        printer = Printer(use_labels=True)
        
        result = printer._format_value(TF.ACT, 0.75)
        assert result == "0.75"
        
        result = printer._format_value(TF.ACT, 0.5)
        assert result == "0.5"
        
        # Should strip trailing zeros
        result = printer._format_value(TF.ACT, 1.0)
        assert result == "1"
    
    def test_format_raw_mode(self):
        """Test formatting with use_labels=False."""
        printer = Printer(use_labels=False)
        
        # Should return raw float representations
        result = printer._format_value(TF.TYPE, Type.PO)
        assert result == "0"  # Type.PO = 0
        
        result = printer._format_value(TF.SET, Set.RECIPIENT)
        assert result == "1"  # Set.RECIPIENT = 1
        
        result = printer._format_value(TF.DELETED, B.TRUE)
        assert result == "1"  # B.TRUE = 1


class TestEmptyTensor:
    """Tests for edge cases with empty or minimal tensors."""
    
    def test_empty_tensor(self, test_connections):
        """Test printing an empty tensor."""
        empty_tensor = torch.full((0, len(TF)), null, dtype=tensor_type)
        empty_names = {}
        token_tensor = Token_Tensor(empty_tensor, test_connections[:0, :0], empty_names)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_printer_output(printer, token_tensor)
        
        assert "Empty tensor" in output
    
    def test_all_deleted_tensor(self, test_connections):
        """Test printing a tensor where all tokens are deleted."""
        tensor = torch.full((5, len(TF)), null, dtype=tensor_type)
        tensor[:, TF.DELETED] = B.TRUE
        names = {i: f"deleted_{i}" for i in range(5)}
        token_tensor = Token_Tensor(tensor, test_connections[:5, :5], names)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_printer_output(printer, token_tensor, show_deleted=False)
        
        assert "No tokens to display" in output


class TestLogFile:
    """Tests for logging to file."""
    
    def test_log_to_file(self, token_tensor, tmp_path):
        """Test that output is logged to file when log_file is set."""
        log_file = tmp_path / "test_log.txt"
        
        printer = Printer(use_labels=True, log_file=str(log_file), print_to_console=False)
        printer.print_token_tensor(token_tensor, cols_per_table=8)
        
        # Check that log file was created and has content
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert len(content) > 0
        
        # Write log file content to output for inspection
        write_test_output(content, f"Log File Output (from {log_file.name})")


# =====================[ Connections Printing Tests ]======================

def capture_connections_output(printer: Printer, token_tensor, method: str = "matrix", **kwargs) -> str:
    """
    Capture the output of the connections printer to a string.
    
    Args:
        printer: The Printer instance.
        token_tensor: The Token_Tensor object.
        method: "matrix" for print_connections, "list" for print_connections_list.
        **kwargs: Additional arguments to pass to the print method.
    """
    import tempfile
    
    # Create a temporary file to capture output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a new printer with the temp file as log destination
        capture_printer = Printer(
            use_labels=printer.use_labels,
            log_file=tmp_path,
            print_to_console=False
        )
        
        if method == "matrix":
            capture_printer.print_connections(token_tensor, **kwargs)
        else:
            capture_printer.print_connections_list(token_tensor, **kwargs)
        
        # Read the captured output
        with open(tmp_path, 'r', encoding='utf-8') as f:
            return f.read()
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_connections_description(connections: torch.Tensor, names: dict) -> str:
    """
    Generate a human-readable description of the connections for visual comparison.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("CONNECTIONS DESCRIPTION (for visual comparison)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Connection format: parent[idx] -> child[idx]")
    lines.append("")
    
    num_tokens = connections.size(0)
    for parent_idx in range(num_tokens):
        parent_name = names.get(parent_idx, f"token_{parent_idx}")
        children = []
        for child_idx in range(num_tokens):
            if connections[parent_idx, child_idx].item():
                child_name = names.get(child_idx, f"token_{child_idx}")
                children.append(f"{child_name}[{child_idx}]")
        
        if children:
            lines.append(f"{parent_name}[{parent_idx}] -> {', '.join(children)}")
    
    lines.append("")
    lines.append(f"Total connections: {connections.sum().item()}")
    
    return "\n".join(lines)


class TestPrintConnections:
    """Tests for printing connections tensor."""
    
    def test_print_connections_matrix(self, token_tensor, test_connections, test_names):
        """Test printing connections as a matrix."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Capture output
        output = capture_connections_output(printer, token_tensor, method="matrix")
        
        # Write description and output to file for visual inspection
        description = get_connections_description(test_connections, test_names)
        write_test_output(description, "Connections Description (Reference)")
        write_test_output(output, "Connections Matrix Output")
        
        # Assertions
        assert "Connections Matrix" in output
        assert "●" in output  # Should have connected chars
        assert "·" in output  # Should have empty chars
    
    def test_print_connections_list(self, token_tensor, test_connections, test_names):
        """Test printing connections as a list."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Capture output
        output = capture_connections_output(printer, token_tensor, method="list")
        
        write_test_output(output, "Connections List Output")
        
        # Assertions
        assert "Connections List" in output
        assert "→" in output  # Should have arrow
        # Should show parent -> children relationships
        assert "Mary_obj" in output or "lover_pred" in output
    
    def test_print_connections_with_indices(self, token_tensor):
        """Test printing connections using indices instead of names."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with use_names=False
        output = capture_connections_output(
            printer, token_tensor, method="matrix", use_names=False
        )
        
        write_test_output(output, "Connections Matrix (Indices Only)")
        
        # Should have numeric indices instead of names
        assert "0" in output
        assert "1" in output
    
    def test_print_connections_custom_chars(self, token_tensor):
        """Test printing connections with custom characters."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with custom characters
        output = capture_connections_output(
            printer, token_tensor, method="matrix",
            connected_char="X", empty_char="-"
        )
        
        write_test_output(output, "Connections Matrix (Custom Chars: X and -)")
        
        # Should have custom characters
        assert "X" in output
        assert "-" in output
    
    def test_print_connections_specific_indices(self, token_tensor):
        """Test printing connections for specific token indices only."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Only show tokens 0, 1, 2, 3 (Mary_obj, lover_pred, lovesMary_RB, lovesMary_P)
        indices = torch.tensor([0, 1, 2, 3])
        output = capture_connections_output(
            printer, token_tensor, method="matrix", indices=indices
        )
        
        write_test_output(output, "Connections Matrix (Indices 0-3 only)")
        
        # Should only contain those tokens
        assert "Mary_obj" in output
        assert "lovesMary_P" in output
        # Should NOT contain tokens outside the range
        assert "John_obj" not in output
        assert "lovesJohn" not in output
    
    def test_print_connections_show_deleted(self, token_tensor):
        """Test printing connections including deleted tokens."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Show all including deleted
        output = capture_connections_output(
            printer, token_tensor, method="matrix", show_deleted=True
        )
        
        write_test_output(output, "Connections Matrix (Including Deleted)")
        
        # Should include the deleted token
        assert "deleted_token" in output
    
    def test_print_connections_list_format(self, token_tensor):
        """Test that list format shows parent -> children properly."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_connections_output(printer, token_tensor, method="list")
        
        # Based on our fixture:
        # lovesMary_RB connects to Mary_obj and lover_pred
        # lovesJohn_RB connects to John_obj and lover_pred
        # The output should show these relationships
        assert "lovesMary_RB" in output
        assert "lovesJohn_RB" in output
    
    def test_print_connections_empty_tensor(self, test_connections):
        """Test printing connections for an empty tensor."""
        empty_tensor = torch.full((0, len(TF)), null, dtype=tensor_type)
        empty_names = {}
        token_tensor = Token_Tensor(empty_tensor, test_connections[:0, :0], empty_names)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_connections_output(printer, token_tensor, method="matrix")
        
        assert "Empty tensor" in output
    
    def test_print_connections_no_connections(self, test_tensor, test_names):
        """Test printing when there are no connections."""
        # Create a token tensor with no connections
        num_tokens = test_tensor.size(0)
        no_connections = torch.zeros((num_tokens, num_tokens), dtype=torch.bool)
        token_tensor = Token_Tensor(test_tensor, no_connections, test_names)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_connections_output(printer, token_tensor, method="matrix")
        
        write_test_output(output, "Connections Matrix (No Connections)")
        
        # Should show 0 connections in the header
        assert "0 connections" in output
        # Should not have any connected chars (only empty chars)
        assert "●" not in output


# =====================[ Links Printing Tests ]======================

@pytest.fixture
def test_links(test_tensor):
    """
    Create mock links tensor representing token-to-semantic connections.
    Shape: [tokens, semantics]
    
    Links (token -> semantic with weight):
    - Token 0 (Mary_obj) -> sem 0 (mary1) weight 1.0, sem 1 (mary2) weight 0.8
    - Token 1 (lover_pred) -> sem 2 (lover1) weight 1.0, sem 3 (lover2) weight 0.9
    - Token 4 (John_obj) -> sem 4 (john1) weight 1.0, sem 5 (john2) weight 0.7
    - Token 7 (Tom_pred) -> sem 6 (tom1) weight 0.5
    """
    num_tokens = test_tensor.size(0)
    num_semantics = 8
    links = torch.zeros((num_tokens, num_semantics), dtype=tensor_type)
    
    # Mary_obj -> mary semantics
    links[0, 0] = 1.0   # Mary_obj -> mary1
    links[0, 1] = 0.8   # Mary_obj -> mary2
    
    # lover_pred -> lover semantics
    links[1, 2] = 1.0   # lover_pred -> lover1
    links[1, 3] = 0.9   # lover_pred -> lover2
    
    # John_obj -> john semantics
    links[4, 4] = 1.0   # John_obj -> john1
    links[4, 5] = 0.7   # John_obj -> john2
    
    # Tom_pred -> tom semantics
    links[7, 6] = 0.5   # Tom_pred -> tom1
    
    return links


@pytest.fixture
def semantic_names():
    """Create mock semantic names dictionary."""
    return {
        0: "mary1",
        1: "mary2",
        2: "lover1",
        3: "lover2",
        4: "john1",
        5: "john2",
        6: "tom1",
        7: "unused_sem"
    }


def capture_links_output(printer: Printer, links, method: str = "matrix", **kwargs) -> str:
    """
    Capture the output of the links printer to a string.
    
    Args:
        printer: The Printer instance.
        links: Links tensor or Links object.
        method: "matrix" for print_links, "list" for print_links_list.
        **kwargs: Additional arguments to pass to the print method.
    """
    import tempfile
    
    # Create a temporary file to capture output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a new printer with the temp file as log destination
        capture_printer = Printer(
            use_labels=printer.use_labels,
            log_file=tmp_path,
            print_to_console=False
        )
        
        if method == "matrix":
            capture_printer.print_links(links, **kwargs)
        else:
            capture_printer.print_links_list(links, **kwargs)
        
        # Read the captured output
        with open(tmp_path, 'r', encoding='utf-8') as f:
            return f.read()
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_links_description(links: torch.Tensor, token_names: dict, semantic_names: dict) -> str:
    """
    Generate a human-readable description of the links for visual comparison.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("LINKS DESCRIPTION (for visual comparison)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Link format: token[idx] -> semantic[idx] (weight)")
    lines.append("")
    
    num_tokens = links.size(0)
    num_semantics = links.size(1)
    
    for tk_idx in range(num_tokens):
        tk_name = token_names.get(tk_idx, f"token_{tk_idx}")
        linked_sems = []
        for sem_idx in range(num_semantics):
            weight = links[tk_idx, sem_idx].item()
            if weight > 0:
                sem_name = semantic_names.get(sem_idx, f"sem_{sem_idx}")
                linked_sems.append(f"{sem_name}[{sem_idx}]({weight:.2f})")
        
        if linked_sems:
            lines.append(f"{tk_name}[{tk_idx}] -> {', '.join(linked_sems)}")
    
    lines.append("")
    lines.append(f"Total links: {(links > 0).sum().item()}")
    
    return "\n".join(lines)


class TestPrintLinks:
    """Tests for printing links tensor."""
    
    def test_print_links_matrix(self, test_links, test_names, semantic_names):
        """Test printing links as a matrix."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Capture output
        output = capture_links_output(
            printer, test_links, method="matrix",
            token_names=test_names, semantic_names=semantic_names
        )
        
        # Write description and output to file for visual inspection
        description = get_links_description(test_links, test_names, semantic_names)
        write_test_output(description, "Links Description (Reference)")
        write_test_output(output, "Links Matrix Output")
        
        # Assertions
        assert "Links Matrix" in output
        assert "links" in output
    
    def test_print_links_list(self, test_links, test_names, semantic_names):
        """Test printing links as a list."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Capture output
        output = capture_links_output(
            printer, test_links, method="list",
            token_names=test_names, semantic_names=semantic_names
        )
        
        write_test_output(output, "Links List Output")
        
        # Assertions
        assert "Links List" in output
        assert "→" in output
    
    def test_print_links_with_weights(self, test_links, test_names, semantic_names):
        """Test printing links showing weight values."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with show_weights=True (default)
        output = capture_links_output(
            printer, test_links, method="matrix",
            token_names=test_names, semantic_names=semantic_names,
            show_weights=True
        )
        
        write_test_output(output, "Links Matrix (With Weights)")
        
        # Should have weight values
        assert "1.0" in output or "0.8" in output or "0.9" in output
    
    def test_print_links_without_weights(self, test_links, test_names, semantic_names):
        """Test printing links without showing weight values (just dots)."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print with show_weights=False
        output = capture_links_output(
            printer, test_links, method="matrix",
            token_names=test_names, semantic_names=semantic_names,
            show_weights=False
        )
        
        write_test_output(output, "Links Matrix (Without Weights)")
        
        # Should have ● for links instead of numbers
        assert "●" in output
    
    def test_print_links_min_weight_filter(self, test_links, test_names, semantic_names):
        """Test filtering links by minimum weight."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print only links with weight >= 0.9
        output = capture_links_output(
            printer, test_links, method="list",
            token_names=test_names, semantic_names=semantic_names,
            min_weight=0.9
        )
        
        write_test_output(output, "Links List (min_weight=0.9)")
        
        # Should not include Tom_pred -> tom1 (weight 0.5)
        # Should include Mary_obj -> mary1 (1.0), lover_pred -> lover1 (1.0), etc.
        assert "Links List" in output
    
    def test_print_links_specific_tokens(self, test_links, test_names, semantic_names):
        """Test printing links for specific token indices only."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Only show tokens 0 and 1 (Mary_obj and lover_pred)
        indices = torch.tensor([0, 1])
        output = capture_links_output(
            printer, test_links, method="matrix",
            token_names=test_names, semantic_names=semantic_names,
            token_indices=indices
        )
        
        write_test_output(output, "Links Matrix (Tokens 0, 1 only)")
        
        # Should include Mary_obj and lover_pred
        assert "Mary_obj" in output
        assert "lover_pred" in output
    
    def test_print_links_specific_semantics(self, test_links, test_names, semantic_names):
        """Test printing links for specific semantic indices only."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Only show semantics 0, 1, 2 (mary1, mary2, lover1)
        sem_indices = torch.tensor([0, 1, 2])
        output = capture_links_output(
            printer, test_links, method="matrix",
            token_names=test_names, semantic_names=semantic_names,
            semantic_indices=sem_indices
        )
        
        write_test_output(output, "Links Matrix (Semantics 0, 1, 2 only)")
        
        # Should include mary1, mary2, lover1
        assert "mary1" in output
        assert "mary2" in output
        assert "lover1" in output
    
    def test_print_links_without_names(self, test_links):
        """Test printing links using indices instead of names."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print without providing names
        output = capture_links_output(
            printer, test_links, method="matrix"
        )
        
        write_test_output(output, "Links Matrix (No Names - Indices Only)")
        
        # Should have T0, T1, S0, S1 etc. instead of names
        assert "T0" in output or "S0" in output
    
    def test_print_links_empty_tensor(self):
        """Test printing an empty links tensor."""
        empty_links = torch.zeros((0, 0), dtype=tensor_type)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_links_output(printer, empty_links, method="matrix")
        
        assert "Empty links tensor" in output
    
    def test_print_links_no_links(self):
        """Test printing when there are no links (all zeros)."""
        no_links = torch.zeros((5, 5), dtype=tensor_type)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_links_output(
            printer, no_links, method="matrix",
            token_indices=torch.arange(5)
        )
        
        write_test_output(output, "Links Matrix (No Links)")
        
        # Should show matrix with "0 links" in the header
        # (semantics are shown since 0.0 >= min_weight of 0.0, but no actual connections)
        assert "0 links" in output
        # Should only have empty chars (·), no connected chars (●) or weights
        assert "●" not in output
    
    def test_print_links_list_with_weights(self, test_links, test_names, semantic_names):
        """Test list format shows weights properly."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_links_output(
            printer, test_links, method="list",
            token_names=test_names, semantic_names=semantic_names,
            show_weights=True
        )
        
        write_test_output(output, "Links List (With Weights)")
        
        # Should show semantic names with weights in parentheses
        # e.g., "mary1(1.0)"
        assert "(" in output and ")" in output


# =====================[ Mappings Printing Tests ]======================

@pytest.fixture
def test_mappings():
    """
    Create mock mapping tensor representing recipient-to-driver mappings.
    Shape: [recipients, drivers, fields]
    Fields: WEIGHT (0), HYPOTHESIS (1), MAX_HYP (2)
    
    Mappings (recipient -> driver):
    - R0 (John_obj) -> D0 (Mary_obj) weight 0.8, hyp 0.5
    - R0 (John_obj) -> D1 (lover_pred) weight 0.3, hyp 0.2
    - R1 (lovesJohn_RB) -> D2 (lovesMary_RB) weight 0.9, hyp 0.7
    - R2 (lovesJohn_P) -> D3 (lovesMary_P) weight 0.95, hyp 0.8
    """
    num_recipients = 5
    num_drivers = 5
    num_fields = len(MappingFields)
    
    mappings = torch.zeros((num_recipients, num_drivers, num_fields), dtype=tensor_type)
    
    # R0 -> D0 (John_obj -> Mary_obj)
    mappings[0, 0, MappingFields.WEIGHT] = 0.8
    mappings[0, 0, MappingFields.HYPOTHESIS] = 0.5
    mappings[0, 0, MappingFields.MAX_HYP] = 0.6
    
    # R0 -> D1 (John_obj -> lover_pred)
    mappings[0, 1, MappingFields.WEIGHT] = 0.3
    mappings[0, 1, MappingFields.HYPOTHESIS] = 0.2
    mappings[0, 1, MappingFields.MAX_HYP] = 0.25
    
    # R1 -> D2 (lovesJohn_RB -> lovesMary_RB)
    mappings[1, 2, MappingFields.WEIGHT] = 0.9
    mappings[1, 2, MappingFields.HYPOTHESIS] = 0.7
    mappings[1, 2, MappingFields.MAX_HYP] = 0.75
    
    # R2 -> D3 (lovesJohn_P -> lovesMary_P)
    mappings[2, 3, MappingFields.WEIGHT] = 0.95
    mappings[2, 3, MappingFields.HYPOTHESIS] = 0.8
    mappings[2, 3, MappingFields.MAX_HYP] = 0.85
    
    return mappings


@pytest.fixture
def driver_names():
    """Create mock driver names dictionary."""
    return {
        0: "Mary_obj",
        1: "lover_pred",
        2: "lovesMary_RB",
        3: "lovesMary_P",
        4: "Tom_pred"
    }


@pytest.fixture
def recipient_names():
    """Create mock recipient names dictionary."""
    return {
        0: "John_obj",
        1: "lovesJohn_RB",
        2: "lovesJohn_P",
        3: "beloved_pred",
        4: "Kathy_obj"
    }


def capture_mappings_output(printer: Printer, mapping, method: str = "matrix", **kwargs) -> str:
    """
    Capture the output of the mappings printer to a string.
    
    Args:
        printer: The Printer instance.
        mapping: Mapping tensor or Mapping object.
        method: "matrix" for print_mappings, "list" for print_mappings_list.
        **kwargs: Additional arguments to pass to the print method.
    """
    import tempfile
    
    # Create a temporary file to capture output
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8') as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a new printer with the temp file as log destination
        capture_printer = Printer(
            use_labels=printer.use_labels,
            log_file=tmp_path,
            print_to_console=False
        )
        
        if method == "matrix":
            capture_printer.print_mappings(mapping, **kwargs)
        else:
            capture_printer.print_mappings_list(mapping, **kwargs)
        
        # Read the captured output
        with open(tmp_path, 'r', encoding='utf-8') as f:
            return f.read()
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_mappings_description(mapping: torch.Tensor, driver_names: dict, recipient_names: dict) -> str:
    """
    Generate a human-readable description of the mappings for visual comparison.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MAPPINGS DESCRIPTION (for visual comparison)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Mapping format: recipient[idx] -> driver[idx] (weight, hyp, max_hyp)")
    lines.append("")
    
    num_recipients = mapping.size(0)
    num_drivers = mapping.size(1)
    
    for rec_idx in range(num_recipients):
        rec_name = recipient_names.get(rec_idx, f"rec_{rec_idx}")
        mapped_drivers = []
        for dri_idx in range(num_drivers):
            weight = mapping[rec_idx, dri_idx, MappingFields.WEIGHT].item()
            if weight > 0:
                dri_name = driver_names.get(dri_idx, f"dri_{dri_idx}")
                hyp = mapping[rec_idx, dri_idx, MappingFields.HYPOTHESIS].item()
                max_hyp = mapping[rec_idx, dri_idx, MappingFields.MAX_HYP].item()
                mapped_drivers.append(f"{dri_name}[{dri_idx}](w={weight:.2f}, h={hyp:.2f}, mh={max_hyp:.2f})")
        
        if mapped_drivers:
            lines.append(f"{rec_name}[{rec_idx}] -> {', '.join(mapped_drivers)}")
    
    lines.append("")
    total_mappings = (mapping[:, :, MappingFields.WEIGHT] > 0).sum().item()
    lines.append(f"Total mappings: {total_mappings}")
    
    return "\n".join(lines)


class TestPrintMappings:
    """Tests for printing mappings tensor."""
    
    def test_print_mappings_matrix_weight(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings as a matrix showing WEIGHT field."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Capture output
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names,
            field=MappingFields.WEIGHT
        )
        
        # Write description and output to file for visual inspection
        description = get_mappings_description(test_mappings, driver_names, recipient_names)
        write_test_output(description, "Mappings Description (Reference)")
        write_test_output(output, "Mappings Matrix [WEIGHT]")
        
        # Assertions
        assert "Mappings [WEIGHT]" in output
        assert "mappings" in output
    
    def test_print_mappings_matrix_hypothesis(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings showing HYPOTHESIS field."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names,
            field=MappingFields.HYPOTHESIS
        )
        
        write_test_output(output, "Mappings Matrix [HYPOTHESIS]")
        
        # Should show HYPOTHESIS in header
        assert "HYPOTHESIS" in output
    
    def test_print_mappings_matrix_max_hyp(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings showing MAX_HYP field."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names,
            field=MappingFields.MAX_HYP
        )
        
        write_test_output(output, "Mappings Matrix [MAX_HYP]")
        
        # Should show MAX_HYP in header
        assert "MAX_HYP" in output
    
    def test_print_mappings_list(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings as a list."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_mappings_output(
            printer, test_mappings, method="list",
            driver_names=driver_names, recipient_names=recipient_names
        )
        
        write_test_output(output, "Mappings List Output")
        
        # Assertions
        assert "Mappings List" in output
        assert "→" in output
    
    def test_print_mappings_with_values(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings showing weight values."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names,
            show_values=True
        )
        
        write_test_output(output, "Mappings Matrix (With Values)")
        
        # Should have weight values
        assert "0.8" in output or "0.9" in output or "0.95" in output
    
    def test_print_mappings_without_values(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings without showing values (just dots)."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names,
            show_values=False
        )
        
        write_test_output(output, "Mappings Matrix (Without Values)")
        
        # Should have ● for mappings instead of numbers
        assert "●" in output
    
    def test_print_mappings_min_value_filter(self, test_mappings, driver_names, recipient_names):
        """Test filtering mappings by minimum value."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print only mappings with weight >= 0.85
        output = capture_mappings_output(
            printer, test_mappings, method="list",
            driver_names=driver_names, recipient_names=recipient_names,
            min_value=0.85
        )
        
        write_test_output(output, "Mappings List (min_value=0.85)")
        
        # Should only include high-value mappings
        assert "Mappings List" in output
    
    def test_print_mappings_specific_recipients(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings for specific recipient indices only."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Only show recipients 0 and 1
        indices = torch.tensor([0, 1])
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names,
            recipient_indices=indices
        )
        
        write_test_output(output, "Mappings Matrix (Recipients 0, 1 only)")
        
        # Should include John_obj and lovesJohn_RB
        assert "John_obj" in output
        assert "lovesJohn_RB" in output
    
    def test_print_mappings_specific_drivers(self, test_mappings, driver_names, recipient_names):
        """Test printing mappings for specific driver indices only."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Only show drivers 0, 1, 2
        dri_indices = torch.tensor([0, 1, 2])
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names,
            driver_indices=dri_indices
        )
        
        write_test_output(output, "Mappings Matrix (Drivers 0, 1, 2 only)")
        
        # Should include Mary_obj, lover_pred, lovesMary_RB
        assert "Mary_obj" in output
        assert "lover_pred" in output
        assert "lovesMary_RB" in output
    
    def test_print_mappings_without_names(self, test_mappings):
        """Test printing mappings using indices instead of names."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Print without providing names
        output = capture_mappings_output(
            printer, test_mappings, method="matrix"
        )
        
        write_test_output(output, "Mappings Matrix (No Names - Indices Only)")
        
        # Should have R0, R1, D0, D1 etc. instead of names
        assert "R0" in output or "D0" in output
    
    def test_print_mappings_empty_tensor(self):
        """Test printing an empty mapping tensor."""
        empty_mappings = torch.zeros((0, 0, len(MappingFields)), dtype=tensor_type)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_mappings_output(printer, empty_mappings, method="matrix")
        
        assert "Empty mapping tensor" in output
    
    def test_print_mappings_no_mappings(self):
        """Test printing when there are no mappings (all zeros)."""
        no_mappings = torch.zeros((5, 5, len(MappingFields)), dtype=tensor_type)
        
        printer = Printer(use_labels=True, print_to_console=False)
        output = capture_mappings_output(
            printer, no_mappings, method="matrix",
            recipient_indices=torch.arange(5),
            driver_indices=torch.arange(5)
        )
        
        write_test_output(output, "Mappings Matrix (No Mappings)")
        
        # Should show "0 mappings" in the header
        assert "0 mappings" in output
    
    def test_print_mappings_list_with_values(self, test_mappings, driver_names, recipient_names):
        """Test list format shows values properly."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        output = capture_mappings_output(
            printer, test_mappings, method="list",
            driver_names=driver_names, recipient_names=recipient_names,
            show_values=True
        )
        
        write_test_output(output, "Mappings List (With Values)")
        
        # Should show driver names with values in parentheses
        assert "(" in output and ")" in output
    
    def test_print_mappings_default_field_is_weight(self, test_mappings, driver_names, recipient_names):
        """Test that default field is WEIGHT."""
        printer = Printer(use_labels=True, print_to_console=False)
        
        # Don't specify field - should default to WEIGHT
        output = capture_mappings_output(
            printer, test_mappings, method="matrix",
            driver_names=driver_names, recipient_names=recipient_names
        )
        
        # Should show WEIGHT in header
        assert "WEIGHT" in output


# =====================[ Integration Test ]======================

class TestIntegration:
    """Integration tests comparing labels vs raw output."""
    
    def test_labels_vs_raw_comparison(self, token_tensor):
        """Compare output with labels vs without labels side by side."""
        printer_labels = Printer(use_labels=True, print_to_console=False)
        printer_raw = Printer(use_labels=False, print_to_console=False)
        
        output_labels = capture_printer_output(printer_labels, token_tensor, cols_per_table=6)
        output_raw = capture_printer_output(printer_raw, token_tensor, cols_per_table=6)
        
        # Create comparison output
        comparison = []
        comparison.append("WITH LABELS (use_labels=True):")
        comparison.append("-" * 40)
        comparison.append(output_labels)
        comparison.append("")
        comparison.append("WITHOUT LABELS (use_labels=False):")
        comparison.append("-" * 40)
        comparison.append(output_raw)
        
        write_test_output("\n".join(comparison), "Labels vs Raw Comparison")
        
        # Both should produce output
        assert len(output_labels) > 0
        assert len(output_raw) > 0
        
        # Labels version should have enum names
        assert "DRIVER" in output_labels or "RECIPIENT" in output_labels
        
        # Raw version should have numeric values
        # (it won't have "DRIVER" as a word since it shows 0, 1, 2, etc.)

