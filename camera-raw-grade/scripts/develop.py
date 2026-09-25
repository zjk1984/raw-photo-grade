#!/usr/bin/env python3
"""Develop a DSLR/mirrorless RAW into a graded JPEG/TIFF.

Thin CLI over the shared engine in ../../shared/scripts/raw_develop.py —
this file only owns the camera-tuned LOOKS presets and accepted suffixes.
Defaults assume a larger sensor than a phone: less noise reduction by
default, a bit more shadow headroom, no computational tone-mapping baked
into the file already.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared" / "scripts"
if not (_SHARED / "raw_develop.py").exists():
    sys.exit(
        "shared/ not found next to this skill.\n"
        "Install shared/ alongside camera-raw-grade/ (and phone-dng-grade/ if you use it) "
        "— see the repo README's install section."
    )
sys.path.insert(0, str(_SHARED))
from looks import camera_looks  # noqa: E402
from raw_develop import run  # noqa: E402

LOOKS = camera_looks()

SUFFIXES = {
    ".nef", ".NEF",
    ".cr2", ".CR2", ".cr3", ".CR3",
    ".arw", ".ARW",
    ".raf", ".RAF",
    ".orf", ".ORF",
    ".rw2", ".RW2",
    ".pef", ".PEF",
    ".srw", ".SRW",
    ".dng", ".DNG",
}

if __name__ == "__main__":
    raise SystemExit(run(LOOKS, "sony-st", SUFFIXES, "Develop DSLR/mirrorless RAW into a graded JPEG/TIFF"))
