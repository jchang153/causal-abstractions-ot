"""Fixed-C1 binary-addition experiment."""

from pathlib import Path
import sys


_PACKAGE_DIR = Path(__file__).resolve().parent
_TWO_DIGIT_ADDITION_ROOT = _PACKAGE_DIR.parent / "two_digit_addition"
for _path in (_PACKAGE_DIR, _TWO_DIGIT_ADDITION_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
