#!/usr/bin/env python3
"""Develop a phone DNG into a graded JPEG/TIFF.

Thin CLI over the shared engine in ../../shared/scripts/raw_develop.py —
this file only owns the phone-tuned LOOKS presets and accepted suffixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared" / "scripts"
if not (_SHARED / "raw_develop.py").exists():
    sys.exit(
        "shared/ not found next to this skill.\n"
        "Install shared/ alongside phone-dng-grade/ (and camera-raw-grade/ if you use it) "
        "— see the repo README's install section."
    )
sys.path.insert(0, str(_SHARED))
from looks import phone_looks  # noqa: E402
from raw_develop import run  # noqa: E402

LOOKS = phone_looks()

SUFFIXES = {".dng", ".DNG"}

if __name__ == "__main__":
    raise SystemExit(run(LOOKS, "sony-st", SUFFIXES, "Develop phone DNG into a graded JPEG/TIFF"))
