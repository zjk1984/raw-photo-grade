#!/usr/bin/env python3
"""Shared RAW inspection engine — metadata + raw shape, no develop.

Not a skill by itself. Each skill's own scripts/inspect_*.py imports `run()`
from here and supplies a `classify(exif) -> (family, hint)` function that
knows about its own camera family (phone brands vs DSLR/mirrorless brands).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw_common import collect_inputs, dump_json, require_rawpy  # noqa: E402

EXIFTOOL_KEYS = [
    "Make",
    "Model",
    "Lens",
    "LensModel",
    "LensID",
    "Software",
    "DateTimeOriginal",
    "ISO",
    "ExposureTime",
    "FNumber",
    "FocalLength",
    "FocalLengthIn35mmFormat",
    "Orientation",
    "ColorMatrix1",
    "ColorMatrix2",
    "AsShotNeutral",
    "UniqueCameraModel",
    "CameraModelName",
    "CameraModel",
    "Manufacturer",
    "HostComputer",
    "DNGVersion",
    "ImageWidth",
    "ImageHeight",
    "DefaultCropSize",
    "SemanticName",
]


def exiftool_tags(path: Path) -> dict:
    """Read camera identity + exposure tags. Prefer exiftool; fall back to Pillow."""
    path = Path(path)
    if shutil.which("exiftool"):
        try:
            raw = subprocess.check_output(
                ["exiftool", "-json", "-n"] + [f"-{k}" for k in EXIFTOOL_KEYS] + [str(path)],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            arr = json.loads(raw)
            if arr:
                row = arr[0]
                row.pop("SourceFile", None)
                return row
        except Exception:
            pass
    # Pillow fallback (JPEG / some DNG)
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(path) as im:
            raw_exif = im.getexif()
            if not raw_exif:
                return {}
            out = {}
            for tag_id, value in raw_exif.items():
                name = TAGS.get(tag_id, str(tag_id))
                if name in EXIFTOOL_KEYS or name in {"Make", "Model", "Software", "LensModel"}:
                    out[name] = value
            # IFD 0x8769 often holds EXIF sub-IFD; skip deep walk for speed
            return out
    except Exception:
        return {}


def inspect_one(path: Path, classify: Callable[[dict], tuple[str, str]]) -> dict:
    require_rawpy()
    import rawpy

    info = {
        "path": str(path.resolve()),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "suffix": path.suffix,
    }
    try:
        with rawpy.imread(str(path)) as raw:
            sizes = raw.sizes
            info["raw"] = {
                "raw_width": sizes.raw_width,
                "raw_height": sizes.raw_height,
                "width": sizes.width,
                "height": sizes.height,
                "iwidth": sizes.iwidth,
                "iheight": sizes.iheight,
            }
            try:
                info["num_colors"] = int(raw.num_colors)
            except Exception:
                pass
            try:
                info["camera_whitebalance"] = [float(x) for x in raw.camera_whitebalance]
            except Exception:
                pass
            try:
                info["daylight_whitebalance"] = [float(x) for x in raw.daylight_whitebalance]
            except Exception:
                pass
            try:
                info["black_level"] = int(raw.black_level_per_channel[0])
            except Exception:
                pass
            color_desc = getattr(raw, "color_desc", None)
            if color_desc:
                info["color_desc"] = (
                    color_desc.decode() if isinstance(color_desc, bytes) else str(color_desc)
                )
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"

    extra = exiftool_tags(path)
    if extra:
        info["exif"] = extra

    family, hint = classify(extra)
    info["family"] = family
    if hint:
        info["hint"] = hint

    return info


def print_report(r: dict) -> None:
    print(f"file        {r['path']}")
    print(f"size        {r['size_mb']} MB")
    print(f"family      {r.get('family', '?')}")
    if "raw" in r:
        raw = r["raw"]
        print(f"raw size    {raw['raw_width']}x{raw['raw_height']}")
        print(f"active      {raw['width']}x{raw['height']}")
    if r.get("camera_whitebalance"):
        print(f"camera WB   {r['camera_whitebalance']}")
    if "exif" in r:
        ex = r["exif"]
        for key in (
            "Make",
            "Model",
            "Lens",
            "LensModel",
            "Software",
            "DateTimeOriginal",
            "ISO",
            "ExposureTime",
            "FNumber",
            "FocalLength",
            "FocalLengthIn35mmFormat",
            "Orientation",
        ):
            if key in ex:
                print(f"{key:<12}{ex[key]}")
    if r.get("hint"):
        print(f"hint        {r['hint']}")
    if r.get("error"):
        print(f"error       {r['error']}")


def run(suffixes: set[str], description: str, classify: Callable[[dict], tuple[str, str]]) -> int:
    """Entry point a skill's own inspect_*.py calls with its classify()."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("inputs", nargs="+", help="RAW file or folder")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    files = collect_inputs(args.inputs, suffixes)
    reports = [inspect_one(p, classify) for p in files]

    if args.json or len(reports) > 1:
        dump_json(reports if len(reports) > 1 else reports[0])
    else:
        print_report(reports[0])

    return 1 if any(r.get("error") for r in reports) else 0
