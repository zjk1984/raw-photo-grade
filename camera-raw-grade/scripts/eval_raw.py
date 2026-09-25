#!/usr/bin/env python3
"""Evaluate DSLR/mirrorless RAW with sensor-aware scoring.

Camera profile is chosen from EXIF Make/Model (e.g. ILCE-7C → sony_a7c_imx410).
Pass --sensor-profile only to force a profile.
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
    if "--raw-aware" not in sys.argv and "--no-raw-aware" not in sys.argv:
        sys.argv.append("--raw-aware")
    # Default: auto-detect from EXIF (do not force a body)
    if "--sensor-profile" not in sys.argv:
        sys.argv.extend(["--sensor-profile", "auto"])
    raise SystemExit(main())
