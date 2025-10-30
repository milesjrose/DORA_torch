"""
Network Overview Segment
Displays network structure and statistics in a tree view
"""

from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import Static, Tree, Label, DataTable, Input, Button
from textual.reactive import reactive
from textual.message import Message
from textual.screen import ModalScreen
try:
    from ...nodes.enums import Set
    from ...nodes.network.network_params import Params
except ImportError:
    try:
        from DORA_tensorised.nodes.enums import Set
        from DORA_tensorised.nodes.network.network_params import Params
    except ImportError:
        class Set:
            DRIVER = "DRIVER"
            RECIPIENT = "RECIPIENT"
            MEMORY = "MEMORY"
            NEW_SET = "NEW_SET"


class NetworkOverviewSegment(Static):
    """Segment for displaying network overview and statistics"""
    
    current_network = reactive(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network_tree = None
        self.stats_table = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the network overview."""
        yield Label("Network Overview", classes="segment-title")
        
        with Horizontal():
            with Vertical():
                yield Label("Network Structure", classes="subsection-title")
                yield Tree("Network", id="network_tree")
            
            with Vertical():
                yield Label("Statistics", classes="subsection-title")
                yield DataTable(id="stats_table")
        with Horizontal():
            with Vertical():
                yield Label("Parameters", classes="subsection-title")
                yield DataTable(id="params_table")
                yield Horizontal(
                    Button("Save", variant="primary", id="save_params"),
                    Button("Reset", variant="default", id="reset_params"),
                )
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.network_tree = self.query_one("#network_tree", Tree)
        self.stats_table = self.query_one("#stats_table", DataTable)
        self.params_table = self.query_one("#params_table", DataTable)
        # Setup stats table
        self.stats_table.add_columns("Component", "Count", "Details")
        self.params_table.add_columns("Parameter", "Value")
        # Initialize with empty state
        self.update_empty_state()
    
    def watch_current_network(self, network) -> None:
        """Called when current_network changes."""
        if network:
            self.update_network(network)
        else:
            self.update_empty_state()
    
    def update_network(self, network) -> None:
        """Update the display with network information."""
        self.current_network = network
        self.update_tree_view(network)
        self.update_statistics(network)
        self.update_params(network)

    def update_params(self, network) -> None:
        """Update the parameters table with network information."""
            
        self.params_table.clear()
        
        if not hasattr(network, 'params'):
            self.params_table.add_row("Parameters", "Not loaded")
            return
        params: Params = network.params
        for key, value in params.get_params_dict().items():
            self.params_table.add_row(key, value)

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        """Handle row selection in the parameters table."""
        if event.data_table.id == "params_table":
            self.notify(f"params_rows: {self.params_table.get_row_at(event.coordinate.row)}")
            row = self.params_table.get_row_at(event.coordinate.row)
            row_key = row[0]
            row_value = row[1]
            dialog = ParameterDialog(param_name=row_key)
            dialog.params_value_input.value = str(row_value)
            self.app.push_screen(dialog)
            

    def update_empty_state(self) -> None:
        """Update display for empty state (no network loaded)."""
        # Clear tree
        self.network_tree.clear()
        root = self.network_tree.root
        root.label = "No Network Loaded"
        
        # Clear stats table
        self.stats_table.clear()
        self.stats_table.add_row("Status", "No network loaded", "Load a sim file to view network details")
    
    def update_tree_view(self, network) -> None:
        """Update the tree view with network structure."""
        self.network_tree.clear()
        root = self.network_tree.root
        root.label = "Network"
        
        # Add semantics
        semantics_node = root.add("Semantics", expand=True)
        if hasattr(network, 'semantics') and network.semantics:
            semantics_count = getattr(network.semantics, 'get_count', lambda: 0)()
            semantics_node.add(f"Nodes: {semantics_count}")
        
        # Add sets
        sets_node = root.add("Sets", expand=True)
        if hasattr(network, 'sets') and network.sets:
            for set_type in [Set.DRIVER, Set.RECIPIENT, Set.MEMORY, Set.NEW_SET]:
                if set_type in network.sets:
                    set_obj = network.sets[set_type]
                    set_count = getattr(set_obj, 'get_count', lambda: 0)()
                    set_node = sets_node.add(f"{set_type.name}: {set_count} tokens")
                    
                    # Add more details if available
                    if hasattr(set_obj, 'IDs') and set_obj.IDs:
                        set_node.add(f"IDs: {len(set_obj.IDs)}")
                    if hasattr(set_obj, 'names') and set_obj.names:
                        set_node.add(f"Named: {len(set_obj.names)}")
        
        # Add connections
        connections_node = root.add("Connections", expand=True)
        if hasattr(network, 'links') and network.links:
            connections_node.add("Links: Available")
        if hasattr(network, 'mappings') and network.mappings:
            try:
                rec_count = network.mappings.size(0)
                drv_count = network.mappings.size(1)
                connections_node.add(f"Mappings: {rec_count}×{drv_count}")
            except Exception:
                connections_node.add("Mappings: Available")
    
    def update_statistics(self, network) -> None:
        """Update the statistics table with network information."""
        self.stats_table.clear()
        
        try:
            # Total node count
            total_count = getattr(network, 'get_count', lambda: 0)()
            self.stats_table.add_row("Total Nodes", str(total_count), "All nodes in network")
            
            # Semantics count
            if hasattr(network, 'semantics') and network.semantics:
                sem_count = getattr(network.semantics, 'get_count', lambda: 0)()
                self.stats_table.add_row("Semantics", str(sem_count), "Semantic nodes")
            
            # Set statistics
            if hasattr(network, 'sets') and network.sets:
                for set_type in [Set.DRIVER, Set.RECIPIENT, Set.MEMORY, Set.NEW_SET]:
                    if set_type in network.sets:
                        set_obj = network.sets[set_type]
                        set_count = getattr(set_obj, 'get_count', lambda: 0)()
                        named_count = len(getattr(set_obj, 'names', {}) or {})
                        self.stats_table.add_row(
                            f"{set_type.name} Set", 
                            str(set_count), 
                            f"{named_count} named tokens"
                        )
            
            # Connection statistics
            if hasattr(network, 'links') and network.links:
                self.stats_table.add_row("Links", "Available", "Inter-set connections")
            
            if hasattr(network, 'mappings') and network.mappings:
                try:
                    rec_count = network.mappings.size(0)
                    drv_count = network.mappings.size(1)
                    self.stats_table.add_row("Mappings", f"{rec_count}×{drv_count}", "recipient × driver")
                except Exception:
                    self.stats_table.add_row("Mappings", "Available", "Set mappings")
            
            # Parameters
            if hasattr(network, 'params') and network.params:
                self.stats_table.add_row("Parameters", "Loaded", "Network configuration")
            
        except Exception as e:
            self.stats_table.add_row("Error", "Failed to load stats", str(e))


class ParameterDialog(ModalScreen):
    """Modal dialog for editing parameters"""
    
    class UpdateParam(Message):
        """Message sent when save is requested"""
        def __init__(self, param_name: str, param_value: str) -> None:
            self.param_name = param_name
            self.param_value = param_value
            super().__init__()
    
    def __init__(self, param_name: str, **kwargs):
        super().__init__(**kwargs)
        self.params_value_input = Input(placeholder="Enter parameter value")
        self.param_name = param_name
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the save dialog."""
        yield Vertical(
            Label("Edit Parameters", classes="dialog-title"),
            self.params_value_input,
            Button("Save", variant="primary", id="save_btn"),
            Button("Cancel", variant="default", id="cancel_btn"),
            classes="save-dialog"
        )
    
    def on_mount(self) -> None:
        """Called when the dialog is mounted."""
        self.params_value_input.focus()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save_btn":
            param_value = self.params_value_input.value.strip()
            # Convert param_value to appropriate type
            try:
                if param_value.isdigit():
                    param_value = int(param_value)
                elif param_value.replace('.', '').isdigit():
                    param_value = float(param_value)
                elif param_value.lower() in ['true', 'false', '1', '0', 'yes', 'no']:
                    param_value = param_value.lower() in ['true', '1', 'yes']
                else:
                    param_value = str(param_value)
            except ValueError:
                param_value = str(param_value)
            self.post_message(self.UpdateParam(self.param_name, param_value))
            self.dismiss()
        elif event.button.id == "cancel_btn":
            self.dismiss()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        self.on_button_pressed(Button.Pressed(Button("Save", id="save_btn")))
