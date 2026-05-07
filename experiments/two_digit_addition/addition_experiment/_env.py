"""Environment tweaks needed by plotting and pyvene code in this package."""

import os
from pathlib import Path
import sys
import tempfile


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "causal_abstractions_gw_mplconfig"))

for parent in Path(__file__).resolve().parents:
    if (parent / "experiments").is_dir():
        repo_root = str(parent)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        break
