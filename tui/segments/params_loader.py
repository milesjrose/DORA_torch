"""
File Loader Segment
Displays available sim files and allows loading them
"""

import os
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.notifications import SeverityLevel
from textual.widgets import Static, ListView, ListItem, Label
from textual.message import Message
from textual.reactive import reactive

#from ...nodes import Params

class ParamsLoaderSegment(Static):
    """Segment for loading params files from the params directory"""
    
    class ParamsSelected(Message):
        """Message sent when a file is selected for loading"""
        def __init__(self, params: dict) -> None:
            self.params = params
            super().__init__()
    
    params_files = reactive(list)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params_dir = Path(__file__).parent.parent.parent / "params"
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the file loader."""
        yield Label("Available Params Files", classes="segment-title")
        yield ListView(id="file_list")
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.refresh_file_list()
        # Delay the update to ensure ListView is ready
        self.set_timer(0.1, self.update_file_list)
    
    def watch_params_files(self, params_files: list) -> None:
        """Called when params_files changes."""
        self.update_file_list()
    
    def refresh_file_list(self) -> None:
        """Refresh the list of available params files."""
        if not self.params_dir.exists():
            self.params_files = []
            self.notify("Params dir cannot be found")
            return
        
        files = []
        for file_path in self.params_dir.iterdir():
            if file_path.is_file() and file_path.suffix == '.json':
                files.append(str(file_path))
        
        # Sort files by name
        files.sort()
        self.params_files = files
        self.notify(f"Found {len(files)} params files", severity="information")
    
    def update_file_list(self) -> None:
        """Update the ListView with current params files."""
        try:
            file_list = self.query_one("#file_list", ListView)
            file_list.clear()
            
            for file_path in self.params_files:
                filename = os.path.basename(file_path)
                item = ListItem(Label(f"{filename}"))
                item.file_path = file_path
                file_list.append(item)
        except Exception as e:
            # ListView might not be ready yet, this is normal during initialization
            pass
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle file selection."""
        if hasattr(event.item, 'file_path'):
            self.post_message(self.ParamsSelected(event.item.file_path))
    
    def action_refresh(self) -> None:
        """Refresh the file list."""
        self.refresh_file_list()
