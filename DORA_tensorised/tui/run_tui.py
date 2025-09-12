#!/usr/bin/env python3
"""
DORA TUI Launcher
Simple launcher script for the DORA TUI application
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add the parent directory to access DORA
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

if __name__ == "__main__":
    try:
        from dora_tui import main
        main()
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Make sure you have installed the requirements:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error running DORA TUI: {e}")
        sys.exit(1)
