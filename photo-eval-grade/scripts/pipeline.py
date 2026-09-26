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


def _grade_context(src: Path, details: dict | None = None) -> tuple[str | None, dict]:
    """Resolve look brand + fill details with EXIF camera cues."""
    details = dict(details or {})
    try:
        from camera_grade import resolve_grade_adapter
        from raw_inspect import exiftool_tags

        exif = exiftool_tags(src)
        adapter, meta = resolve_grade_adapter(src, exif)
        details.setdefault("look_brand", meta["brand"])
        details.setdefault("camera_make", meta.get("camera_make") or "")
        details.setdefault("camera_model", meta.get("camera_model") or "")
        details["grade_adapter"] = meta["adapter_id"]
        details["grade_adapter_label"] = meta["adapter_label"]
        details["look_brand_reason"] = meta.get("brand_reason", "")
        return adapter.brand, details
    except Exception:
        return details.get("look_brand"), details


def _load_rgb(src: Path) -> np.ndarray:
    if src.suffix.lower() in {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}:
        rgb = decode_raw(src, bright=1.0, no_auto_bright=False)
        return apply_orientation(rgb, "auto", src)
    with Image.open(src) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _maybe_straighten(graded: np.ndarray) -> np.ndarray:
    # Detect tilt on a downscale — full-res Hough is slow and same angle within 0.05°
    preview = resize_long_edge(graded, 1200)
    im_p = Image.fromarray((np.clip(preview, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    angle = tilt_angle_deg(np.asarray(im_p))
    if abs(angle) < 0.15:
        return graded
    im_graded = Image.fromarray((np.clip(graded, 0, 1) * 255.0 + 0.5).astype(np.uint8))
    im_graded = im_graded.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    x0, y0, x1, y1 = inscribe_rect(im_graded.width, im_graded.height, angle)
    im_graded = im_graded.crop((x0, y0, x1, y1))
    return np.asarray(im_graded, dtype=np.float32) / 255.0


def _evals_from_json(path: Path):
    from eval_photo import ImageEvaluation

    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or data
    evals = []
    for r in results:
        evals.append(
            ImageEvaluation(
                path=r["path"],
                filename=r.get("filename") or Path(r["path"]).name,
                overall_score=float(r["overall_score"]),
                tier=str(r["tier"]).upper(),
                sharpness=float(r.get("sharpness") or 0),
                dynamic_range=float(r.get("dynamic_range") or 0),
                noise_control=float(r.get("noise_control") or 0),
                color_harmony=float(r.get("color_harmony") or 0),
                composition=float(r.get("composition") or 0),
                flags=list(r.get("flags") or []),
                details=dict(r.get("details") or {}),
            )
        )
    return evals


def main() -> int:
    look_opts = look_choices() + ["auto"]
    parser = argparse.ArgumentParser(
        description="M4 GPU Auto Pipeline: Evaluate -> Filter S/A Keepers -> Develop -> Crop"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="RAW/photo directory or file list (default: prefs photo_root)",
    )
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
        choices=["auto", "sony", "fuji", "nikon", "apple", "canon"],
        help="Look pools: auto=EXIF Make/Model (default), or force a brand",
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
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save developed photos (default: <photo_root>/edited)",
    )
    parser.add_argument("--straighten", action="store_true", help="Auto-level horizon")
    parser.add_argument("--preview", action="store_true", help="Primary export long-edge 1600px (default: full-res)")
    parser.add_argument("--quality", type=int, default=92, help="JPEG export quality (default: 92)")
    parser.add_argument(
        "--face-finish",
        action="store_true",
        help="Enable LR People refine after Auto Basic + look (auto-on for face under/hot)",
    )
    parser.add_argument(
        "--face-finish-amount",
        type=float,
        default=1.0,
        help="Face finish strength 0–1.5 (default 1.0)",
    )
    parser.add_argument(
        "--eval-json",
        default=None,
        help="Reuse a prior eval.py --json result (skip re-eval; much faster)",
    )
    args = parser.parse_args()

    from looks import edited_dir, photo_root

    prefs = load_prefs()
    root = photo_root(prefs)
    if not args.inputs and not args.eval_json:
        args.inputs = [str(root)]
        print(f"==> Using default photo root: {root}", file=sys.stderr)
    out_path = Path(args.out_dir).expanduser() if args.out_dir else edited_dir(prefs)
    out_path.mkdir(parents=True, exist_ok=True)
    sticky = (not args.no_sticky) and bool(prefs.get("sticky_look", True))

    # Resume sticky lock from prior run in same out-dir (one shoot, one grade)
    locked_look: str | None = None
    locked_brand: str | None = None
    manifest_path = out_path / "pipeline_manifest.json"
    if sticky and args.look == "auto" and manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            locked_look = prev.get("locked_look") or None
            locked_brand = prev.get("locked_brand") or None
            if locked_look and locked_look not in ALL_LOOKS:
                locked_look = None
            if locked_look:
                print(
                    f"==> Sticky lock from prior manifest: {locked_look}"
                    + (f" (brand={locked_brand})" if locked_brand else ""),
                    file=sys.stderr,
                )
        except Exception:
            locked_look = None
            locked_brand = None

    if args.eval_json:
        eval_path = Path(args.eval_json).expanduser()
        print(f"==> Step 1: Loading eval from {eval_path} (skip re-eval)...", file=sys.stderr)
        evals = _evals_from_json(eval_path)
        print(format_table(evals), file=sys.stderr)
    else:
        print(f"==> Step 1: Evaluating photos using M4/Metal GPU (device: {args.device})...", file=sys.stderr)
        evaluator = PhotoEvaluator(device_name=args.device)
        evaluator.preset = args.preset

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

        from batch_rank import apply_batch_relative_ranking, build_verdict_reason

        evals = apply_batch_relative_ranking(evals)
        for ev in evals:
            try:
                ev.details["verdict_reason"] = build_verdict_reason(ev)
            except Exception:
                pass
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
    force_brand = None if (not args.brand or args.brand == "auto") else args.brand

    from camera_grade import look_params_for_camera
    from face_recover import should_face_finish
    from lr_stack import develop_lr_stack

    for ev in keepers:
        src = Path(ev.path)
        dest_name = f"{src.stem}_graded.jpg"
        dest_file = out_path / dest_name

        print(f"Developing {src.name} [{ev.tier} | {ev.overall_score:.1f}pts] -> {dest_name}", file=sys.stderr)

        rgb = _load_rgb(src)
        # Full-res by default; --preview only shrinks the primary export.
        # look_compare uses a separate downscaled copy so contact sheets stay cheap.
        rgb_full = rgb
        if args.preview:
            rgb = resize_long_edge(rgb, 1600)
        # Suggest look always on ≤1600 — same sticky pick, far cheaper than full-res cues
        rgb_for_suggest = resize_long_edge(rgb_full if not args.preview else rgb, 1600)
        rgb_compare = rgb_for_suggest if args.look_compare else rgb_for_suggest

        exif_brand, details = _grade_context(src, ev.details)
        brand_for_file = force_brand or exif_brand

        # Sticky only within the same camera brand (mixed shoots re-pick)
        use_lock = (
            locked_look
            if (
                sticky
                and args.look == "auto"
                and locked_look
                and (locked_brand is None or locked_brand == brand_for_file)
            )
            else None
        )

        suggestion = suggest_look_detail(
            rgb_for_suggest,
            details,
            forced=None if args.look == "auto" else args.look,
            auto=(args.look == "auto"),
            brand=brand_for_file,
            locked_look=use_lock,
            sticky=sticky if args.look == "auto" else False,
        )
        look_name = suggestion.look
        if look_name not in ALL_LOOKS:
            look_name = prefs.get("default_look", "sony-st")
            suggestion.look = look_name

        # Establish sticky lock on first auto pick of this run / brand
        if sticky and args.look == "auto" and (locked_look is None or locked_brand != brand_for_file):
            locked_look = look_name
            locked_brand = brand_for_file
            print(f"  sticky lock set -> {locked_look} (brand={locked_brand})", file=sys.stderr)

        params, grade_meta = look_params_for_camera(look_name, path=src)
        flags = list(ev.flags or [])
        do_people = bool(args.face_finish) or should_face_finish(flags)
        # Lightroom-ordered stack: Auto Basic → Look/Detail → People → Selective
        graded, lr_report = develop_lr_stack(
            rgb,
            params,
            flags=flags,
            face_mean=(ev.details or {}).get("face_mean_luma"),
            people_amount=float(args.face_finish_amount) if do_people else 0.0,
            selective_amount=1.0 if do_people else 0.65,
            do_people=do_people,
            do_selective=True,
        )
        if do_people:
            print(
                f"  lr-stack stages={lr_report.get('stages')} "
                f"people={lr_report.get('people', {}).get('mode')} "
                f"face_ev={lr_report.get('people', {}).get('face_ev')}",
                file=sys.stderr,
            )
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
                alt_params, _ = look_params_for_camera(cand, path=src)
                alt, _ = develop_lr_stack(
                    rgb_compare,
                    alt_params,
                    flags=flags,
                    face_mean=(ev.details or {}).get("face_mean_luma"),
                    people_amount=0.0,
                    selective_amount=0.5,
                    do_people=False,
                    do_selective=True,
                )
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
            "brand": grade_meta.get("brand") or brand_for_file,
            "grade_adapter": grade_meta.get("adapter_id"),
            "camera_make": grade_meta.get("camera_make"),
            "face_finish": do_people,
            "lr_stack": lr_report.get("stages"),
            "camera_model": grade_meta.get("camera_model"),
            "sticky": suggestion.sticky or (sticky and locked_look == look_name and args.look == "auto"),
            "compare_outputs": compare_outputs,
            "compare_sheet": str(sheet_path) if sheet_path else None,
        })
        print(
            f"  look={look_name} brand={grade_meta.get('brand')} adapter={grade_meta.get('adapter_id')} "
            f"scene={suggestion.scene_tag} reason={suggestion.reason} "
            f"candidates={suggestion.candidates}",
            file=sys.stderr,
        )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "look_mode": args.look,
            "brand": force_brand or locked_brand or prefs.get("brand"),
            "sticky_look": sticky,
            "locked_look": locked_look if args.look == "auto" else None,
            "locked_brand": locked_brand if args.look == "auto" else None,
            "look_compare": args.look_compare,
            "eval_json": args.eval_json,
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
