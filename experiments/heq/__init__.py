"""Hierarchical equality experiment entry points."""

from pathlib import Path
import sys


_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
