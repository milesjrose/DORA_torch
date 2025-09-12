#!/usr/bin/env python3
"""
DORA TUI Application
Main application file for the DORA Textual User Interface
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import DORA
sys.path.append(str(Path(__file__).parent.parent))

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Static
from textual.binding import Binding

from DORA import DORA
from segments.params_loader import ParamsLoaderSegment
from segments.file_loader import FileLoaderSegment
from segments.file_saver import FileSaverSegment
from segments.network_overview import NetworkOverviewSegment, ParameterDialog


class DORATUI(App):
    """Main DORA TUI Application"""
    
    CSS = """
    Screen {
        layout: horizontal;
    }
    
    .left-file-panel {
        width: 100%;
        min-height: 30%;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    .left-params-panel {
        width: 100%;
        min-height: 30%;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    .right-panel {
        width: 70%;
        max-height: 97%;
        border: solid $primary;
        margin: 1;
        padding: 1;
    }
    
    .segment {
        height: 1fr;
        border: solid $secondary;
        margin: 1;
        padding: 1;
    }
    
    .segment-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    .subsection-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    
    .save-instruction {
        color: $text-muted;
        text-align: center;
        margin: 1;
    }
    
    .save-dialog {
        width: 50%;
        height: auto;
        border: solid $primary;
        padding: 2;
        background: $surface;
    }
    
    .dialog-title {
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 2;
    }
    
    #file_list {
        height: 15;
    }
    
    #network_tree {
        height: 20;
    }
    
    #stats_table {
        height: 20;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+s", "save_file", "Save Network"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]
    
    def __init__(self):
        super().__init__()
        self.dora = DORA()
        self.current_network = None
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        with Horizontal():
            with Vertical():
                with Vertical(classes="left-file-panel"):
                    yield FileLoaderSegment(id="file_loader")
                    yield FileSaverSegment(id="file_saver")
                
                with Vertical(classes="left-params-panel"):
                    yield ParamsLoaderSegment(id="params_loader")
            
            with Vertical(classes="right-panel"):
                yield NetworkOverviewSegment(id="network_overview")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "DORA TUI"
        self.sub_title = "Tensorised DORA Interface"
        
        # Debug: Check if params loader is created
        try:
            params_loader = self.query_one("#params_loader", ParamsLoaderSegment)
            self.notify("Params loader segment created successfully", severity="information")
        except Exception as e:
            self.notify(f"Failed to find params loader: {str(e)}", severity="error")
    
    def action_save_file(self) -> None:
        """Handle Ctrl+S to save file."""
        file_saver = self.query_one("#file_saver", FileSaverSegment)
        file_saver.show_save_dialog()
    
    def action_refresh(self) -> None:
        """Handle Ctrl+R to refresh the interface."""
        if self.current_network:
            network_overview = self.query_one("#network_overview", NetworkOverviewSegment)
            network_overview.update_network(self.current_network)
    
    def on_file_loader_segment_file_selected(self, event: FileLoaderSegment.FileSelected) -> None:
        """Handle file selection from file loader."""
        self.load_network(event.file_path)
    
    def on_params_loader_segment_params_selected(self, event: ParamsLoaderSegment.ParamsSelected) -> None:
        """Handle loading params file"""
        if self.current_network is None:
            self.notify("No network to load params to.")
            return
        else:
            self.current_network.load_json_params(event.params)
            # update network overview
            network_overview = self.query_one("#network_overview", NetworkOverviewSegment)
            network_overview.update_network(self.current_network)
    
    def on_file_saver_segment_save_requested(self, event: FileSaverSegment.SaveRequested) -> None:
        """Handle save request from file saver."""
        self.save_network(event.filename)
    
    def on_parameter_dialog_update_param(self, event: ParameterDialog.UpdateParam) -> None:
        """Handle parameter update from parameter dialog."""
        if not self.current_network or not hasattr(self.current_network, 'params'):
            self.notify("No network loaded to update parameters", severity="warning")
            return
        
        try:
            # Update the parameter in the network
            params = self.current_network.params
            param_attr = event.param_name
            
            if hasattr(params, param_attr):
                setattr(params, param_attr, event.param_value)
                self.notify(f"Updated {event.param_name} to {event.param_value}", severity="information")
                
                # Refresh the network overview to show updated parameters
                network_overview = self.query_one("#network_overview", NetworkOverviewSegment)
                network_overview.update_network(self.current_network)
            else:
                self.notify(f"Parameter {event.param_name} not found", severity="error")
                
        except Exception as e:
            self.notify(f"Failed to update parameter: {str(e)}", severity="error")
    
    def load_network(self, file_path: str) -> None:
        """Load a network from file and update all segments."""
        try:
            # Extract just the filename for DORA.load_sim
            filename = os.path.basename(file_path)
            self.dora.load_sim(file_path)
            self.current_network = self.dora.network
            
            # Update network overview
            network_overview = self.query_one("#network_overview", NetworkOverviewSegment)
            network_overview.update_network(self.current_network)
            
            self.notify(f"Loaded network from {filename}", severity="information")
            
        except Exception as e:
            self.notify(f"Failed to load network: {str(e)}", severity="error")
    
    def save_network(self, filename: str) -> None:
        """Save the current network to file."""
        if not self.current_network:
            self.notify("No network loaded to save", severity="warning")
            return
        
        try:
            self.dora.save_sim(filename)
            self.notify(f"Network saved as {filename}", severity="information")
        except Exception as e:
            self.notify(f"Failed to save network: {str(e)}", severity="error")


def main():
    """Main entry point for the DORA TUI application."""
    app = DORATUI()
    app.run()


if __name__ == "__main__":
    main()
