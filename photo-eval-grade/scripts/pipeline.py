#!/usr/bin/env python3
"""End-to-End Automated Pipeline: Evaluate on M4 GPU -> Filter S/A Keepers -> Develop & Crop.

Ties photo evaluation together with camera-raw-grade / phone-dng-grade develop engines.
Supports classic looks, Sony Creative Look inspired presets, and --look auto scene selection.
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
from look_select import suggest_look  # noqa: E402
from looks import ALL_LOOKS, get_look, look_choices, load_prefs  # noqa: E402
from raw_develop import apply_grade, apply_orientation, decode_raw, resize_long_edge, save_image  # noqa: E402
from PIL import Image
import numpy as np


def main() -> int:
    look_opts = look_choices() + ["auto"]
    parser = argparse.ArgumentParser(
        description="M4 GPU Auto Pipeline: Evaluate -> Filter S/A Keepers -> Develop -> Crop"
    )
    parser.add_argument("inputs", nargs="+", help="RAW/photo directory or file list")
    parser.add_argument("--tiers", default="S,A", help="Tiers to develop (default: S,A)")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--preset", default="general", help="Evaluation preset")
    parser.add_argument(
        "--look",
        default="auto",
        choices=look_opts,
        help="Grade look, or 'auto' to pick brand look from scene heuristics",
    )
    parser.add_argument(
        "--brand",
        default=None,
        choices=["sony", "fuji", "nikon"],
        help="When --look auto: prefer Sony / Fuji / Nikon master looks (default: prefs)",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to save final developed photos")
    parser.add_argument("--straighten", action="store_true", help="Auto-level horizon")
    parser.add_argument("--preview", action="store_true", help="Output 1600px preview instead of full-res")
    parser.add_argument("--quality", type=int, default=92, help="JPEG export quality (default: 92)")
    args = parser.parse_args()

    out_path = Path(args.out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    prefs = load_prefs()

    print(f"==> Step 1: Evaluating photos using M4/Metal GPU (device: {args.device})...", file=sys.stderr)
    evaluator = PhotoEvaluator(device_name=args.device)

    from eval_photo import ALL_SUPPORTED_SUFFIXES
    skip_dirs = {"curated", "edited", "selected", "PhotoGrade_Export", "PhotoGrade_Curated"}
    files = []
    seen = set()
    for inp in args.inputs:
        p = Path(inp).expanduser()
        if p.is_dir():
            candidates = sorted(
                f for f in p.iterdir()
                if f.is_file()
                and f.suffix in ALL_SUPPORTED_SUFFIXES
                and not f.name.startswith(".")
            )
        elif p.is_file() and p.suffix in ALL_SUPPORTED_SUFFIXES:
            candidates = [p]
        else:
            candidates = []
        for f in candidates:
            key = f.resolve()
            if key in seen:
                continue
            if any(part in skip_dirs for part in f.parts):
                continue
            seen.add(key)
            files.append(f)

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

    target_tiers = {t.strip().upper() for t in args.tiers.split(",")}
    keepers = [ev for ev in evals if ev.tier in target_tiers]

    print(f"\n==> Step 2: Selected {len(keepers)} keepers in tier(s) {args.tiers}.", file=sys.stderr)
    if not keepers:
        print("No photos matched the selected tiers.", file=sys.stderr)
        return 0

    print(f"==> Step 3: Developing (look={args.look})...", file=sys.stderr)

    developed_records = []
    for ev in keepers:
        src = Path(ev.path)
        dest_name = f"{src.stem}_graded.jpg"
        dest_file = out_path / dest_name

        print(f"Developing {src.name} [{ev.tier} | {ev.overall_score:.1f}pts] -> {dest_name}", file=sys.stderr)

        if src.suffix.lower() in {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}:
            rgb = decode_raw(src, bright=1.0, no_auto_bright=False)
            rgb = apply_orientation(rgb, "auto", src)
        else:
            with Image.open(src) as im:
                im = im.convert("RGB")
                rgb = np.asarray(im, dtype=np.float32) / 255.0

        if args.preview:
            rgb = resize_long_edge(rgb, 1600)

        look_name, scene_tag = suggest_look(
            rgb,
            ev.details,
            forced=None if args.look == "auto" else args.look,
            auto=(args.look == "auto"),
            brand=args.brand,
        )
        if look_name not in ALL_LOOKS:
            look_name = prefs.get("default_look", "sony-st")
        params = get_look(look_name)

        graded = apply_grade(rgb, params)

        if args.straighten:
            im_graded = Image.fromarray((np.clip(graded, 0, 1) * 255.0 + 0.5).astype(np.uint8))
            angle = tilt_angle_deg(np.asarray(im_graded))
            if abs(angle) >= 0.15:
                im_graded = im_graded.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
                x0, y0, x1, y1 = inscribe_rect(im_graded.width, im_graded.height, angle)
                im_graded = im_graded.crop((x0, y0, x1, y1))
                graded = np.asarray(im_graded, dtype=np.float32) / 255.0

        save_image(graded, dest_file, quality=args.quality, tiff=False)
        developed_records.append({
            "source": str(src),
            "tier": ev.tier,
            "overall_score": ev.overall_score,
            "output": str(dest_file),
            "look": look_name,
            "scene_tag": scene_tag,
        })
        print(f"  look={look_name} scene={scene_tag}", file=sys.stderr)

    manifest_path = out_path / "pipeline_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "look_mode": args.look,
            "brand": args.brand or prefs.get("brand"),
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
