#!/usr/bin/env python3
"""Shared RAW develop engine — decode, grade, crop-friendly export.

Not a skill by itself. Each skill's own scripts/develop.py imports `run()`
from here and supplies its own LOOKS presets, default look, and accepted
file suffixes. The image-processing math (exposure, tone, color, clarity,
noise reduction, sharpen, vignette, fade, HSL zones, 3D LUT) is
camera-agnostic; only the LOOKS numbers and the RAW decode call should
differ per camera family.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from looks import SLIDER_KEYS, STRING_KEYS, complete_params  # noqa: E402
from lut3d import apply_lut, load_cube  # noqa: E402
from raw_common import collect_inputs, linear_to_srgb, luma, require_rawpy, srgb_to_linear  # noqa: E402

# Re-export for callers that imported SLIDER_KEYS from here
__all__ = [
    "SLIDER_KEYS",
    "apply_grade",
    "decode_raw",
    "run",
    "merge_params",
    "parse_args",
]


def parse_args(looks: dict, default_look: str, description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("inputs", nargs="+", help="RAW file(s) or folder")
    p.add_argument("-o", "--output", help="Output file (single input only)")
    p.add_argument("--out-dir", help="Output directory for batch")
    p.add_argument("--look", default=default_look, choices=sorted(looks))
    p.add_argument("--params", help="JSON with slider overrides")
    p.add_argument("--preview", action="store_true", help="Long edge 1600 JPEG")
    p.add_argument("--full", action="store_true", help="Full resolution")
    p.add_argument("--long-edge", type=int, default=0)
    p.add_argument("--tiff", action="store_true")
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--bright", type=float, default=1.0, help="rawpy bright")
    p.add_argument("--no-auto-bright", action="store_true")
    p.add_argument("--orient", default="auto", help="auto|0|90|180|270")
    p.add_argument("--lut", default=None, help="Path to .cube 3D LUT")
    p.add_argument("--lut-amount", type=int, default=None, help="0-100 LUT blend")
    p.add_argument(
        "--no-camera-adapt",
        action="store_true",
        help="Disable EXIF body grade adapter (α7C / iPhone deltas)",
    )
    p.add_argument(
        "--face-finish",
        action="store_true",
        help="After look: local face lift (exposure/contrast/clarity on skin mask)",
    )
    p.add_argument(
        "--face-finish-amount",
        type=float,
        default=1.0,
        help="Face finish strength 0–1.5 (default 1.0)",
    )
    for key in SLIDER_KEYS:
        if key == "lut_amount":
            continue
        typ = float if key == "exposure" else int
        p.add_argument(f"--{key.replace('_', '-')}", type=typ, default=None)
    return p.parse_args()


def merge_params(args: argparse.Namespace, looks: dict) -> dict:
    params = complete_params(looks[args.look])
    params["look"] = args.look
    if args.params:
        with open(args.params, encoding="utf-8") as fh:
            extra = json.load(fh)
        if isinstance(extra, dict):
            params.update({k: extra[k] for k in extra if k in SLIDER_KEYS or k in STRING_KEYS or k == "look"})
    for key in SLIDER_KEYS:
        if key == "lut_amount":
            cli = getattr(args, "lut_amount", None)
        else:
            cli = getattr(args, key, None)
        if cli is not None:
            params[key] = cli
    if getattr(args, "lut", None):
        params["lut"] = args.lut
    return complete_params(params)


def decode_raw(path: Path, bright: float, no_auto_bright: bool) -> np.ndarray:
    require_rawpy()
    import rawpy

    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            bright=bright,
            no_auto_bright=no_auto_bright,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
            highlight_mode=rawpy.HighlightMode.Blend,
            four_color_rgb=False,
        )
    return rgb.astype(np.float32) / 65535.0


def apply_orientation(img: np.ndarray, how: str, path: Path) -> np.ndarray:
    if how == "0":
        return img
    degrees = None
    if how in {"90", "180", "270"}:
        degrees = int(how)
    elif how == "auto":
        degrees = _exif_orientation_degrees(path)
    if not degrees:
        return img
    k = {90: 3, 180: 2, 270: 1}.get(degrees, 0)
    return np.rot90(img, k) if k else img


def _exif_orientation_degrees(path: Path) -> int:
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif()
            orient = exif.get(274, 1) if exif else 1
        return {3: 180, 6: 90, 8: 270}.get(int(orient), 0)
    except Exception:
        return 0


def apply_grade(rgb: np.ndarray, p: dict) -> np.ndarray:
    p = complete_params(p)
    lin = srgb_to_linear(np.clip(rgb, 0, 1))

    if p["exposure"]:
        lin *= 2.0 ** float(p["exposure"])

    temp = float(p["temperature"]) / 100.0
    tint = float(p["tint"]) / 100.0
    if temp or tint:
        lin = lin.copy()
        lin[..., 0] *= 1.0 + 0.35 * temp + 0.12 * tint
        lin[..., 1] *= 1.0 - 0.08 * temp - 0.22 * tint
        lin[..., 2] *= 1.0 - 0.40 * temp + 0.12 * tint

    lin = np.clip(lin, 0, None)
    y = luma(lin)

    hi = float(p["highlights"]) / 100.0
    sh = float(p["shadows"]) / 100.0
    if hi or sh:
        hi_mask = np.clip((y - 0.45) / 0.55, 0, 1)[..., None]
        sh_mask = np.clip((0.40 - y) / 0.40, 0, 1)[..., None]
        if hi < 0:
            lin = lin / (1.0 + hi_mask * (-hi) * 1.4)
        elif hi > 0:
            lin = lin * (1.0 + hi_mask * hi * 0.35)
        if sh:
            lin = lin + sh_mask * sh * 0.12
            lin *= 1.0 + sh_mask * sh * 0.15

    wh = float(p["whites"]) / 100.0
    bl = float(p["blacks"]) / 100.0
    if wh:
        lin = lin * (1.0 + wh * 0.25)
    if bl:
        lin = lin + bl * 0.04
        lin = np.clip(lin, 0, None)

    contrast = float(p["contrast"]) / 100.0
    if contrast:
        mid = 0.18
        lin = (lin - mid) * (1.0 + contrast * 0.9) + mid
        lin = np.clip(lin, 0, None)

    srgb = np.clip(linear_to_srgb(lin), 0, 1)

    # Fade (Sony Creative Look): lift blacks + soft matte ceiling
    fade = float(p.get("fade", 0)) / 100.0
    if fade:
        black_lift = fade * 0.14
        white_crush = fade * 0.08
        srgb = black_lift + srgb * (1.0 - black_lift - white_crush)
        srgb = np.clip(srgb, 0, 1)

    sat = float(p["saturation"]) / 100.0
    vib = float(p["vibrance"]) / 100.0
    if sat or vib:
        gray = luma(srgb)[..., None]
        if sat:
            srgb = gray + (srgb - gray) * (1.0 + sat)
        if vib:
            mx = srgb.max(axis=2, keepdims=True)
            mn = srgb.min(axis=2, keepdims=True)
            already = np.clip((mx - mn) * 1.6, 0, 1)
            srgb = gray + (srgb - gray) * (1.0 + vib * (1.0 - already))

    srgb = np.clip(srgb, 0, 1)
    srgb = _hsl_zones(srgb, p)

    lut_path = str(p.get("lut") or "").strip()
    lut_amount = float(p.get("lut_amount", 0)) / 100.0
    if lut_path and lut_amount > 0:
        try:
            table, size = load_cube(lut_path)
            srgb = apply_lut(srgb, table, size, amount=lut_amount)
        except Exception as exc:
            print(f"warning: LUT skipped ({lut_path}): {exc}", file=sys.stderr)

    if p["clarity"]:
        srgb = _clarity(srgb, float(p["clarity"]) / 100.0)
    if p["noise_luma"]:
        srgb = _luma_nr(srgb, float(p["noise_luma"]) / 100.0)
    if p["vignette"]:
        srgb = _vignette(srgb, float(p["vignette"]) / 100.0)
    if p["sharpen"]:
        srgb = _sharpen(srgb, float(p["sharpen"]) / 100.0, int(p.get("sharpen_range", 2) or 2))

    return np.clip(srgb, 0, 1)


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    v = mx
    s = np.where(mx > 1e-8, diff / np.maximum(mx, 1e-8), 0.0)
    h = np.zeros_like(mx)
    mask = diff > 1e-8
    rc = ((mx - r) / np.maximum(diff, 1e-8))
    gc = ((mx - g) / np.maximum(diff, 1e-8))
    bc = ((mx - b) / np.maximum(diff, 1e-8))
    h = np.where(mask & (mx == r), (bc - gc) / 6.0 % 1.0, h)
    h = np.where(mask & (mx == g), (2.0 + rc - bc) / 6.0 % 1.0, h)
    h = np.where(mask & (mx == b), (4.0 + gc - rc) / 6.0 % 1.0, h)
    return h, s, v


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def _hue_weight(h: np.ndarray, center: float, width: float) -> np.ndarray:
    """Soft circular distance weight around hue center in [0,1]."""
    d = np.abs(h - center)
    d = np.minimum(d, 1.0 - d)
    return np.clip(1.0 - d / max(width, 1e-6), 0.0, 1.0)


def _hsl_zones(srgb: np.ndarray, p: dict) -> np.ndarray:
    skin_h = float(p.get("hsl_skin_hue", 0))
    skin_s = float(p.get("hsl_skin_sat", 0))
    sky_h = float(p.get("hsl_sky_hue", 0))
    sky_s = float(p.get("hsl_sky_sat", 0))
    green_h = float(p.get("hsl_green_hue", 0))
    green_s = float(p.get("hsl_green_sat", 0))
    if not any([skin_h, skin_s, sky_h, sky_s, green_h, green_s]):
        return srgb

    h, s, v = _rgb_to_hsv(np.clip(srgb, 0, 1))
    # skin ~ orange-red, sky ~ cyan-blue, green ~ foliage
    w_skin = np.maximum(_hue_weight(h, 0.04, 0.08), _hue_weight(h, 0.96, 0.06))
    w_sky = _hue_weight(h, 0.58, 0.12)
    w_green = _hue_weight(h, 0.33, 0.10)

    dh = (w_skin * skin_h + w_sky * sky_h + w_green * green_h) / 100.0 * 0.08
    ds = (w_skin * skin_s + w_sky * sky_s + w_green * green_s) / 100.0
    h = (h + dh) % 1.0
    s = np.clip(s * (1.0 + ds), 0.0, 1.0)
    return np.clip(_hsv_to_rgb(h, s, v), 0, 1)


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return img
    pad = radius
    x = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(x, axis=0)
    v = c[2 * radius :, :, :] - c[: -2 * radius, :, :]
    c = np.cumsum(v, axis=1)
    h = c[:, 2 * radius :, :] - c[:, : -2 * radius, :]
    area = float((2 * radius) * (2 * radius))
    return h / area


def _clarity(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    # Cap radius: at 24MP *0.012≈48 is ~3s with little extra look vs 16
    radius = max(3, min(16, int(min(h, w) * 0.012)))
    blur = _box_blur(img, radius)
    return img + (img - blur) * amount * 1.4


def _sharpen(img: np.ndarray, amount: float, sharpen_range: int = 2) -> np.ndarray:
    # Sony Sharpness Range ≈ kernel radius (1..5)
    radius = int(np.clip(sharpen_range, 1, 5))
    if min(img.shape[:2]) < 1200:
        radius = max(1, radius - 1)
    blur = _box_blur(img, radius)
    return img + (img - blur) * amount * 1.8


def _luma_nr(img: np.ndarray, amount: float) -> np.ndarray:
    y = luma(img)
    radius = 1 if amount < 0.15 else 2
    y3 = np.repeat(y[..., None], 3, axis=2)
    yb = _box_blur(y3, radius)[..., 0]
    chroma = img - y[..., None]
    y_mix = y * (1.0 - amount * 0.7) + yb * (amount * 0.7)
    return np.clip(chroma + y_mix[..., None], 0, 1)


def _vignette(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r, 0, 1.6) / 1.6
    falloff = r * r
    return np.clip(img * (1.0 + amount * falloff[..., None]), 0, 1)


def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    from PIL import Image

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return img
    scale = long_edge / m
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8), mode="RGB")
    pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
    return np.asarray(pil).astype(np.float32) / 255.0


def save_image(img: np.ndarray, dest: Path, quality: int, tiff: bool) -> None:
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    if tiff or dest.suffix.lower() in {".tif", ".tiff"}:
        pil.save(dest, format="TIFF", compression="tiff_deflate")
    else:
        pil.save(dest, format="JPEG", quality=quality, optimize=True, subsampling=2)


def dest_path(src: Path, args: argparse.Namespace) -> Path:
    ext = ".tif" if args.tiff else ".jpg"
    if args.output and not args.out_dir:
        return Path(args.output).expanduser()
    folder = Path(args.out_dir).expanduser() if args.out_dir else src.parent / "edited"
    tag = "preview" if args.preview and not args.full else "edit"
    return folder / f"{src.stem}_{tag}{ext}"


def process_one(src: Path, dest: Path, args: argparse.Namespace, params: dict) -> dict:
    rgb = decode_raw(src, args.bright, args.no_auto_bright)
    rgb = apply_orientation(rgb, args.orient, src)
    long_edge = args.long_edge
    if args.preview and not args.full and not long_edge:
        long_edge = 1600
    if long_edge:
        rgb = resize_long_edge(rgb, long_edge)
    # Per-file EXIF body adapter (α7C / iPhone / …) unless disabled
    file_params = params
    grade_meta: dict = {}
    if not getattr(args, "no_camera_adapt", False):
        try:
            from camera_grade import apply_adapter, resolve_grade_adapter
            from raw_inspect import exiftool_tags

            exif = exiftool_tags(src)
            adapter, grade_meta = resolve_grade_adapter(src, exif)
            file_params = apply_adapter(params, adapter)
        except Exception as exc:
            grade_meta = {"adapter_error": str(exc)}
    # Prefer protective highlights when finishing faces
    if getattr(args, "face_finish", False) and float(file_params.get("highlights", 0) or 0) > -20:
        file_params = dict(file_params)
        file_params["highlights"] = int(min(-20, float(file_params.get("highlights", 0)) - 8))
    graded = apply_grade(rgb, file_params)
    face_finish_meta: dict = {}
    if getattr(args, "face_finish", False):
        from face_recover import apply_face_finish

        amt = float(getattr(args, "face_finish_amount", 1.0) or 1.0)
        graded = apply_face_finish(graded, preset="portrait", amount=amt)
        face_finish_meta = {"face_finish": True, "face_finish_amount": amt}
    save_image(graded, dest, args.quality, args.tiff)
    return {
        "input": str(src),
        "output": str(dest),
        "pixels": list(graded.shape[:2]),
        "params": file_params,
        "grade_adapter": grade_meta.get("adapter_id"),
        "camera_model": grade_meta.get("camera_model"),
        "brand": grade_meta.get("brand"),
        **face_finish_meta,
    }


def run(looks: dict, default_look: str, suffixes: set[str], description: str) -> int:
    """Entry point a skill's own develop.py calls with its LOOKS/suffixes."""
    args = parse_args(looks, default_look, description)
    files = collect_inputs(args.inputs, suffixes)
    if args.output and len(files) > 1:
        sys.stderr.write("-o only works for a single file. Use --out-dir for batches.\n")
        return 1
    params = merge_params(args, looks)
    reports = []
    for src in files:
        dest = dest_path(src, args)
        print(f"develop  {src.name}  ->  {dest}", file=sys.stderr)
        report = process_one(src, dest, args, params)
        if report.get("grade_adapter"):
            print(
                f"  adapter={report['grade_adapter']} brand={report.get('brand')} "
                f"model={report.get('camera_model')}",
                file=sys.stderr,
            )
        reports.append(report)
    json.dump(reports if len(reports) > 1 else reports[0], sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0
