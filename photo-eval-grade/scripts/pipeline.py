#!/usr/bin/env python3
"""End-to-End Automated Pipeline: Evaluate on M4 GPU -> Filter S/A Keepers -> Develop & Crop.

Ties photo evaluation together with camera-raw-grade / phone-dng-grade develop engines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared" / "scripts"
if not (_SHARED / "eval_photo.py").exists() or not (_SHARED / "raw_develop.py").exists():
    sys.exit("Error: shared/scripts not found. Make sure skills are installed together.")

sys.path.insert(0, str(_SHARED))
from crop import inscribe_rect, tilt_angle_deg  # noqa: E402
from eval_photo import PhotoEvaluator, format_table  # noqa: E402
from raw_develop import apply_grade, apply_orientation, decode_raw, resize_long_edge, save_image  # noqa: E402
from PIL import Image
import numpy as np


# Standard default camera looks
DEFAULT_LOOKS = {
    "natural": {
        "exposure": 0.1,
        "contrast": 12,
        "highlights": -12,
        "shadows": 16,
        "whites": 4,
        "blacks": -6,
        "temperature": 2,
        "tint": -1,
        "vibrance": 10,
        "saturation": 2,
        "clarity": 10,
        "vignette": -4,
        "sharpen": 20,
        "noise_luma": 4,
    },
    "warm-golden": {
        "exposure": 0.15,
        "contrast": 14,
        "highlights": -18,
        "shadows": 18,
        "whites": 2,
        "blacks": -8,
        "temperature": 16,
        "tint": 4,
        "vibrance": 14,
        "saturation": 6,
        "clarity": 8,
        "vignette": -8,
        "sharpen": 18,
        "noise_luma": 4,
    },
    "portrait": {
        "exposure": 0.2,
        "contrast": 8,
        "highlights": -10,
        "shadows": 22,
        "whites": 2,
        "blacks": -4,
        "temperature": 8,
        "tint": 3,
        "vibrance": 8,
        "saturation": -2,
        "clarity": 4,
        "vignette": -6,
        "sharpen": 12,
        "noise_luma": 6,
    },
    "cool-cinematic": {
        "exposure": -0.05,
        "contrast": 18,
        "highlights": -15,
        "shadows": 10,
        "whites": -4,
        "blacks": -14,
        "temperature": -12,
        "tint": -2,
        "vibrance": 6,
        "saturation": -4,
        "clarity": 12,
        "vignette": -12,
        "sharpen": 16,
        "noise_luma": 5,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M4 GPU Auto Pipeline: Evaluate -> Filter S/A Keepers -> Develop -> Crop"
    )
    parser.add_argument("inputs", nargs="+", help="RAW/photo directory or file list")
    parser.add_argument("--tiers", default="S,A", help="Tiers to develop (default: S,A)")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--preset", default="general", help="Evaluation preset")
    parser.add_argument("--look", default="natural", choices=sorted(DEFAULT_LOOKS.keys()))
    parser.add_argument("--out-dir", required=True, help="Directory to save final developed photos")
    parser.add_argument("--straighten", action="store_true", help="Auto-level horizon")
    parser.add_argument("--preview", action="store_true", help="Output 1600px preview instead of full-res")
    parser.add_argument("--quality", type=int, default=92, help="JPEG export quality (default: 92)")
    args = parser.parse_args()

    out_path = Path(args.out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"==> Step 1: Evaluating photos using M4/Metal GPU (device: {args.device})...", file=sys.stderr)
    evaluator = PhotoEvaluator(device_name=args.device)

    # Collect files
    from eval_photo import ALL_SUPPORTED_SUFFIXES
    files = []
    for inp in args.inputs:
        p = Path(inp).expanduser()
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.suffix in ALL_SUPPORTED_SUFFIXES and not f.name.startswith(".")))
        elif p.is_file() and p.suffix in ALL_SUPPORTED_SUFFIXES:
            files.append(p)

    if not files:
        sys.stderr.write("No supported photos found.\n")
        return 1

    evals = []
    for f in files:
        try:
            ev = evaluator.evaluate(f)
            evals.append(ev)
        except Exception as e:
            sys.stderr.write(f"Error evaluating {f.name}: {e}\n")

    evals.sort(key=lambda x: x.overall_score, reverse=True)
    print(format_table(evals), file=sys.stderr)

    # Filter tiers
    target_tiers = {t.strip().upper() for t in args.tiers.split(",")}
    keepers = [ev for ev in evals if ev.tier in target_tiers]

    print(f"\n==> Step 2: Selected {len(keepers)} keepers in tier(s) {args.tiers}.", file=sys.stderr)
    if not keepers:
        print("No photos matched the selected tiers.", file=sys.stderr)
        return 0

    print(f"==> Step 3: Developing and finishing with look '{args.look}'...", file=sys.stderr)
    params = dict(DEFAULT_LOOKS[args.look])

    developed_records = []
    for ev in keepers:
        src = Path(ev.path)
        dest_name = f"{src.stem}_graded.jpg"
        dest_file = out_path / dest_name

        print(f"Developing {src.name} [{ev.tier} | {ev.overall_score:.1f}pts] -> {dest_name}", file=sys.stderr)

        # 1. Decode RAW
        if src.suffix.lower() in {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2"}:
            rgb = decode_raw(src, bright=1.0, no_auto_bright=False)
            rgb = apply_orientation(rgb, "auto", src)
        else:
            with Image.open(src) as im:
                im = im.convert("RGB")
                rgb = np.asarray(im, dtype=np.float32) / 255.0

        # Downsample if preview
        if args.preview:
            rgb = resize_long_edge(rgb, 1600)

        # 2. Grade
        graded = apply_grade(rgb, params)

        # 3. Straighten if requested
        if args.straighten:
            im_graded = Image.fromarray((np.clip(graded, 0, 1) * 255.0 + 0.5).astype(np.uint8))
            angle = tilt_angle_deg(np.asarray(im_graded))
            if abs(angle) >= 0.15:
                im_graded = im_graded.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
                x0, y0, x1, y1 = inscribe_rect(im_graded.width, im_graded.height, angle)
                im_graded = im_graded.crop((x0, y0, x1, y1))
                graded = np.asarray(im_graded, dtype=np.float32) / 255.0

        # 4. Save
        save_image(graded, dest_file, quality=args.quality, tiff=False)
        developed_records.append({
            "source": str(src),
            "tier": ev.tier,
            "overall_score": ev.overall_score,
            "output": str(dest_file),
        })

    # Save manifest
    manifest_path = out_path / "pipeline_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "look": args.look,
            "target_tiers": list(target_tiers),
            "total_evaluated": len(evals),
            "total_developed": len(developed_records),
            "developed": developed_records,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline complete! Output saved to: {out_path}", file=sys.stderr)
    print(f"Manifest written to: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
