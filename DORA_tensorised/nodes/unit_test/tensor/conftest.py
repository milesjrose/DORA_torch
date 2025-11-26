# conftest.py for unit_test/tensor
# Adds DORA_tensorised to Python path so 'nodes' module can be imported

import sys
from pathlib import Path

# Get the DORA_tensorised directory (parent of nodes)
dora_tensorised_dir = Path(__file__).parent.parent.parent.parent

# Add to Python path if not already there
if str(dora_tensorised_dir) not in sys.path:
    sys.path.insert(0, str(dora_tensorised_dir))


