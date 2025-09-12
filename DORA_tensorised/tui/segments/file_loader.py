"""
File Loader Segment
Displays available sim files and allows loading them
"""

import os
from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, ListView, ListItem, Label
from textual.message import Message
from textual.reactive import reactive


class FileLoaderSegment(Static):
    """Segment for loading sim files from the sims directory"""
    
    class FileSelected(Message):
        """Message sent when a file is selected for loading"""
        def __init__(self, file_path: str) -> None:
            self.file_path = file_path
            super().__init__()
    
    sim_files = reactive(list)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sims_dir = Path(__file__).parent.parent.parent / "sims"
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the file loader."""
        yield Label("Available Sim Files", classes="segment-title")
        yield ListView(id="file_list")
    
    def on_mount(self) -> None:
        """Called when the widget is mounted."""
        self.refresh_file_list()
    
    def watch_sim_files(self, sim_files: list) -> None:
        """Called when sim_files changes."""
        self.update_file_list()
    
    def refresh_file_list(self) -> None:
        """Refresh the list of available sim files."""
        if not self.sims_dir.exists():
            self.sim_files = []
            return
        
        files = []
        for file_path in self.sims_dir.iterdir():
            if file_path.is_file() and file_path.suffix in ['.py', '.sym']:
                files.append(str(file_path))
        
        # Sort files by name
        files.sort()
        self.sim_files = files
    
    def update_file_list(self) -> None:
        """Update the ListView with current sim files."""
        file_list = self.query_one("#file_list", ListView)
        file_list.clear()
        
        for file_path in self.sim_files:
            filename = os.path.basename(file_path)
            item = ListItem(Label(f"{filename}"))
            item.file_path = file_path
            file_list.append(item)
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle file selection."""
        if hasattr(event.item, 'file_path'):
            self.post_message(self.FileSelected(event.item.file_path))
    
    def action_refresh(self) -> None:
        """Refresh the file list."""
        self.refresh_file_list()
