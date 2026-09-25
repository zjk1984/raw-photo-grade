#!/usr/bin/env python3
"""End-to-End Automated Pipeline: Evaluate -> Filter S/A Keepers -> Develop & Crop.

Auto look selection uses ordered scene pools + secondary cues (skill-aligned),
optional sticky lock (one shoot, one grade), and --look-compare contact sheets.
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
from look_select import build_look_compare_sheet, suggest_look_detail  # noqa: E402
from looks import ALL_LOOKS, get_look, look_choices, load_prefs  # noqa: E402
from raw_develop import apply_grade, apply_orientation, decode_raw, resize_long_edge, save_image  # noqa: E402
from PIL import Image
import numpy as np


def _load_rgb(src: Path) -> np.ndarray:
    if src.suffix.lower() in {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}:
        rgb = decode_raw(src, bright=1.0, no_auto_bright=False)
        return apply_orientation(rgb, "auto", src)
    with Image.open(src) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _maybe_straighten(graded: np.ndarray) -> np.ndarray:
    im_graded = Image.fromarray((np.clip(graded, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    angle = tilt_angle_deg(np.asarray(im_graded))
    if abs(angle) < 0.15:
        return graded
    im_graded = im_graded.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    x0, y0, x1, y1 = inscribe_rect(im_graded.width, im_graded.height, angle)
    im_graded = im_graded.crop((x0, y0, x1, y1))
    return np.asarray(im_graded, dtype=np.float32) / 255.0


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
        help="Grade look, or 'auto' for pool + secondary cue pick",
    )
    parser.add_argument(
        "--brand",
        default=None,
        choices=["sony", "fuji", "nikon"],
        help="When --look auto: brand look pools (default: prefs)",
    )
    parser.add_argument(
        "--look-compare",
        action="store_true",
        help="Also export top pool alternates + contact sheet (skill: preview before final)",
    )
    parser.add_argument(
        "--no-sticky",
        action="store_true",
        help="Disable one-shoot sticky look lock (default: sticky on for auto)",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to save final developed photos")
    parser.add_argument("--straighten", action="store_true", help="Auto-level horizon")
    parser.add_argument("--preview", action="store_true", help="Output 1600px preview instead of full-res")
    parser.add_argument("--quality", type=int, default=92, help="JPEG export quality (default: 92)")
    args = parser.parse_args()

    out_path = Path(args.out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)
    prefs = load_prefs()
    sticky = (not args.no_sticky) and bool(prefs.get("sticky_look", True))

    # Resume sticky lock from prior run in same out-dir (one shoot, one grade)
    locked_look: str | None = None
    manifest_path = out_path / "pipeline_manifest.json"
    if sticky and args.look == "auto" and manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            locked_look = prev.get("locked_look") or None
            if locked_look and locked_look not in ALL_LOOKS:
                locked_look = None
            if locked_look:
                print(f"==> Sticky lock from prior manifest: {locked_look}", file=sys.stderr)
        except Exception:
            locked_look = None

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

    print(
        f"==> Step 3: Developing (look={args.look}, sticky={sticky}, compare={args.look_compare})...",
        file=sys.stderr,
    )

    developed_records = []
    compare_dir = out_path / "look_compare"
    for ev in keepers:
        src = Path(ev.path)
        dest_name = f"{src.stem}_graded.jpg"
        dest_file = out_path / dest_name

        print(f"Developing {src.name} [{ev.tier} | {ev.overall_score:.1f}pts] -> {dest_name}", file=sys.stderr)

        rgb = _load_rgb(src)
        if args.preview or args.look_compare:
            rgb = resize_long_edge(rgb, 1600)

        suggestion = suggest_look_detail(
            rgb,
            ev.details,
            forced=None if args.look == "auto" else args.look,
            auto=(args.look == "auto"),
            brand=args.brand,
            locked_look=locked_look if args.look == "auto" else None,
            sticky=sticky if args.look == "auto" else False,
        )
        look_name = suggestion.look
        if look_name not in ALL_LOOKS:
            look_name = prefs.get("default_look", "sony-st")
            suggestion.look = look_name

        # Establish sticky lock on first auto pick of this run
        if sticky and args.look == "auto" and locked_look is None:
            locked_look = look_name
            print(f"  sticky lock set -> {locked_look}", file=sys.stderr)

        params = get_look(look_name)
        graded = apply_grade(rgb, params)
        if args.straighten:
            graded = _maybe_straighten(graded)
        save_image(graded, dest_file, quality=args.quality, tiff=False)

        compare_outputs: list[str] = []
        sheet_path = None
        if args.look_compare and suggestion.candidates:
            compare_dir.mkdir(parents=True, exist_ok=True)
            panels: list[tuple[str, np.ndarray]] = []
            for cand in suggestion.candidates[:3]:
                if cand not in ALL_LOOKS:
                    continue
                alt = apply_grade(rgb, get_look(cand))
                tag = "PRIMARY" if cand == look_name else "ALT"
                label = f"{tag}: {cand}"
                alt_path = compare_dir / f"{src.stem}__{cand}.jpg"
                save_image(alt, alt_path, quality=88, tiff=False)
                compare_outputs.append(str(alt_path))
                panels.append((label, alt))
            if panels:
                sheet_path = compare_dir / f"{src.stem}__compare.jpg"
                build_look_compare_sheet(panels, sheet_path)
                print(f"  compare sheet -> {sheet_path.name}", file=sys.stderr)

        developed_records.append({
            "source": str(src),
            "tier": ev.tier,
            "overall_score": ev.overall_score,
            "output": str(dest_file),
            "look": look_name,
            "scene_tag": suggestion.scene_tag,
            "candidates": suggestion.candidates,
            "reason": suggestion.reason,
            "sticky": suggestion.sticky or (sticky and locked_look == look_name and args.look == "auto"),
            "compare_outputs": compare_outputs,
            "compare_sheet": str(sheet_path) if sheet_path else None,
        })
        print(
            f"  look={look_name} scene={suggestion.scene_tag} reason={suggestion.reason} "
            f"candidates={suggestion.candidates}",
            file=sys.stderr,
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "look_mode": args.look,
            "brand": args.brand or prefs.get("brand"),
            "sticky_look": sticky,
            "locked_look": locked_look if args.look == "auto" else None,
            "look_compare": args.look_compare,
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
