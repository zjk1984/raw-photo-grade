#!/usr/bin/env python3
"""Evaluate DSLR/mirrorless RAW files with M4 GPU / MPS acceleration.

Shim — implementation is in ../../shared/scripts/eval_photo.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared" / "scripts"
if not (_SHARED / "eval_photo.py").exists():
    sys.exit(
        "shared/ not found next to this skill.\n"
        "Install shared/ alongside camera-raw-grade/ "
        "— see the repo README's install section."
    )

sys.path.insert(0, str(_SHARED))
from eval_photo import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
