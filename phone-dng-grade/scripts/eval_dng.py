#!/usr/bin/env python3
"""Evaluate phone DNG / ProRAW with sensor-aware scoring.

Profile is chosen from EXIF (e.g. Model=iPhone 17 Pro Max → iphone_17_promax).
Pass --sensor-profile only to force a profile.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared" / "scripts"
if not (_SHARED / "eval_photo.py").exists():
    sys.exit(
        "shared/ not found next to this skill.\n"
        "Install shared/ alongside phone-dng-grade/ "
        "— see the repo README's install section."
    )
sys.path.insert(0, str(_SHARED))
from eval_photo import main  # noqa: E402

if __name__ == "__main__":
    if "--raw-aware" not in sys.argv and "--no-raw-aware" not in sys.argv:
        sys.argv.append("--raw-aware")
    if "--sensor-profile" not in sys.argv:
        sys.argv.extend(["--sensor-profile", "auto"])
    raise SystemExit(main())
