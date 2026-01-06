"""
File Saver Segment
Handles saving networks to files with Ctrl+S functionality
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, Label, Input, Button
from textual.message import Message
from textual.screen import ModalScreen


class SaveDialog(ModalScreen):
    """Modal dialog for saving files"""
    
    class SaveRequested(Message):
        """Message sent when save is requested"""
        def __init__(self, filename: str) -> None:
            self.filename = filename
            super().__init__()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.filename_input = Input(placeholder="Enter filename (e.g., my_network.sym)")
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the save dialog."""
        yield Vertical(
            Label("💾 Save Network", classes="dialog-title"),
            self.filename_input,
            Button("Save", variant="primary", id="save_btn"),
            Button("Cancel", variant="default", id="cancel_btn"),
            classes="save-dialog"
        )
    
    def on_mount(self) -> None:
        """Called when the dialog is mounted."""
        self.filename_input.focus()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save_btn":
            filename = self.filename_input.value.strip()
            if filename:
                # Ensure .sym extension if not provided
                if not filename.endswith('.sym'):
                    filename += '.sym'
                self.post_message(self.SaveRequested(filename))
                self.dismiss()
        elif event.button.id == "cancel_btn":
            self.dismiss()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        self.on_button_pressed(Button.Pressed(Button("Save", id="save_btn")))


class FileSaverSegment(Static):
    """Segment for saving networks to files"""
    
    class SaveRequested(Message):
        """Message sent when save is requested"""
        def __init__(self, filename: str) -> None:
            self.filename = filename
            super().__init__()
    
    def compose(self) -> ComposeResult:
        """Create child widgets for the file saver."""
        yield Label("💾 Save Network", classes="segment-title")
        yield Static("Press Ctrl+S to save the current network", classes="save-instruction")
    
    def show_save_dialog(self) -> None:
        """Show the save dialog."""
        self.app.push_screen(SaveDialog(), self.handle_save_request)
    
    def handle_save_request(self, filename: str) -> None:
        """Handle save request from dialog."""
        if filename:
            self.post_message(self.SaveRequested(filename))
