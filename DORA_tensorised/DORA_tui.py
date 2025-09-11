# DORA_tensorised/DORA_cli.py
# Textual command line interface for DORA

import os
import sys
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Header, Footer, Static, Input, DirectoryTree, ListView, ListItem, Label
from textual.binding import Binding
from textual.message import Message
from DORA import DORA


class FileListItem(ListItem):
    """Custom list item for file selection."""
    
    def __init__(self, filename: str, filepath: str) -> None:
        self.filename = filename
        self.filepath = filepath
        super().__init__(Label(f"📄 {filename}"))


class DORACLI(App):
    """Textual application for DORA network operations."""
    
    CSS = """
    Screen {
        layout: vertical;
    }
    
    .main-container {
        layout: horizontal;
        height: 1fr;
    }
    
    .left-panel {
        width: 1fr;
        border: solid $primary;
        margin: 1;
    }
    
    .right-panel {
        width: 1fr;
        border: solid $primary;
        margin: 1;
    }
    
    .status-bar {
        height: 3;
        background: $surface;
        border: solid $primary;
        margin: 1;
    }
    
    .button-container {
        height: auto;
        layout: horizontal;
        margin: 1;
    }
    
    Button {
        margin: 1;
    }
    
    Input {
        margin: 1;
    }
    
    ListView {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("l", "load_network", "Load Network"),
        Binding("s", "save_network", "Save Network"),
        Binding("r", "refresh_files", "Refresh Files"),
    ]
    
    def __init__(self):
        super().__init__()
        self.dora = DORA()
        self.sims_dir = "./sims"
        self.current_network = None
        self.available_files = []
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with Container(classes="main-container"):
            with Container(classes="left-panel"):
                yield Static("Available Networks", classes="title")
                yield ListView(id="file-list")
            
            with Container(classes="right-panel"):
                yield Static("Network Operations", classes="title")
                yield Static("Status: No network loaded", id="status")
                
                with Container(classes="button-container"):
                    yield Button("Load Network", id="load-btn", variant="primary")
                    yield Button("Save Network", id="save-btn", variant="success")
                    yield Button("Refresh Files", id="refresh-btn", variant="default")
                
                yield Input(placeholder="Enter filename to save...", id="save-input")
        
        yield Static("DORA CLI - Press 'q' to quit, 'l' to load, 's' to save, 'r' to refresh", classes="status-bar")
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when app starts."""
        self.refresh_file_list()
    
    def refresh_file_list(self) -> None:
        """Refresh the list of available network files."""
        self.available_files = self.get_available_files()
        file_list = self.query_one("#file-list", ListView)
        file_list.clear()
        
        for filename in self.available_files:
            filepath = os.path.join(self.sims_dir, filename)
            file_list.append(FileListItem(filename, filepath))
    
    def get_available_files(self) -> list:
        """Get list of available network files in sims directory."""
        if not os.path.exists(self.sims_dir):
            return []
        
        files = []
        for file in os.listdir(self.sims_dir):
            if file.endswith(('.py', '.sym')):
                files.append(file)
        
        return sorted(files)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "load-btn":
            self.action_load_network()
        elif event.button.id == "save-btn":
            self.action_save_network()
        elif event.button.id == "refresh-btn":
            self.action_refresh_files()
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle file selection from list."""
        if isinstance(event.item, FileListItem):
            filename = event.item.filename
            self.load_network_file(filename)
    
    def load_network_file(self, filename: str) -> None:
        """Load a network from the specified file."""
        try:
            self.notify(f"Loading network from: {filename}")
            self.dora.load_sim(filename)
            self.current_network = filename
            self.update_status(f"Status: Loaded network '{filename}'")
            self.notify(f"✓ Successfully loaded network: {filename}", severity="information")
        except Exception as e:
            self.notify(f"✗ Error loading network: {str(e)}", severity="error")
    
    def save_network_file(self, filename: str) -> None:
        """Save the current network to the specified file."""
        if self.current_network is None:
            self.notify("No network currently loaded. Please load a network first.", severity="warning")
            return
        
        if not filename:
            self.notify("Filename cannot be empty.", severity="warning")
            return
        
        # Ensure .sym extension
        if not filename.endswith('.sym'):
            filename += '.sym'
        
        try:
            self.notify(f"Saving network to: {filename}")
            self.dora.save_sim(filename)
            self.update_status(f"Status: Saved network as '{filename}'")
            self.notify(f"✓ Successfully saved network: {filename}", severity="information")
            self.refresh_file_list()  # Refresh to show new file
        except Exception as e:
            self.notify(f"✗ Error saving network: {str(e)}", severity="error")
    
    def update_status(self, message: str) -> None:
        """Update the status display."""
        status_widget = self.query_one("#status", Static)
        status_widget.update(message)
    
    def action_load_network(self) -> None:
        """Action to load a network."""
        if not self.available_files:
            self.notify("No network files found in ./sims directory.", severity="warning")
            return
        
        # If there's only one file, load it directly
        if len(self.available_files) == 1:
            self.load_network_file(self.available_files[0])
        else:
            self.notify("Select a file from the list to load", severity="information")
    
    def action_save_network(self) -> None:
        """Action to save a network."""
        save_input = self.query_one("#save-input", Input)
        filename = save_input.value.strip()
        self.save_network_file(filename)
        save_input.value = ""  # Clear input after saving
    
    def action_refresh_files(self) -> None:
        """Action to refresh the file list."""
        self.refresh_file_list()
        self.notify("File list refreshed", severity="information")
    
    def action_quit(self) -> None:
        """Action to quit the application."""
        self.exit()


def main():
    """Main entry point for the Textual application."""
    try:
        app = DORACLI()
        app.run()
    except Exception as e:
        print(f"Error starting DORA CLI: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
