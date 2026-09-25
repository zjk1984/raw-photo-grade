#!/usr/bin/env python3
"""RAW-aware metrics: headroom from sensor levels, linear demosaic focus, ISO-normalized noise."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from sensor_profiles import SensorProfile, expected_noise


def _exif_iso(exif: dict[str, Any] | None) -> float:
    if not exif:
        return 0.0
    for key in ("ISO", "ISOSpeedRatings", "PhotographicSensitivity", "iso"):
        v = exif.get(key)
        if v is None:
            continue
        try:
            if isinstance(v, (list, tuple)) and v:
                return float(v[0])
            return float(v)
        except (TypeError, ValueError):
            continue
    return 0.0


def extract_raw_context(
    path: Path,
    profile: SensorProfile,
    exif: dict[str, Any] | None = None,
    *,
    focus_patch: dict[str, float] | None = None,
    high_res_focus: bool = False,
) -> dict[str, Any]:
    """Decode RAW sensor stats + a linear 16-bit demosaic crop for focus.

    focus_patch: normalized {x0,y0,x1,y1} from RGB plane pick (P2).
    high_res_focus: keep longer edge ~2200 on critical-band recheck.
    """
    import rawpy

    exif = exif or {}
    iso = _exif_iso(exif) or float(profile.base_iso)

    with rawpy.imread(str(path)) as raw:
        black = float(np.mean(raw.black_level_per_channel))
        white = float(raw.white_level)
        visible = raw.raw_image_visible.astype(np.float64)
        span = max(white - black, 1.0)
        lin_mosaic = np.clip((visible - black) / span, 0.0, 1.0)

        # Highlight / shadow on mosaic (green-ish average is fine for headroom)
        hi_pct = float((lin_mosaic > 0.98).mean() * 100.0)
        sh_pct = float((lin_mosaic < 0.02).mean() * 100.0)
        p99 = float(np.quantile(lin_mosaic, 0.99))
        p50 = float(np.quantile(lin_mosaic, 0.50))
        # Stops of highlight headroom below clip at the 99th percentile
        headroom_ev = float(-math.log2(max(p99, 1e-4))) if p99 < 1.0 else 0.0
        # Midtone placement vs mid-grey ~0.18 linear
        mid_ev = float(math.log2(max(p50, 1e-4) / 0.18))

        # Linear demosaic for focus (no auto bright — preserves true edges)
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=16,
            bright=1.0,
            output_color=rawpy.ColorSpace.sRGB,
        )
        rgb = rgb16.astype(np.float32) / 65535.0

    h, w = rgb.shape[:2]
    if focus_patch and all(k in focus_patch for k in ("x0", "y0", "x1", "y1")):
        # Expand patch slightly for RAW micro-contrast stability
        x0n = float(focus_patch["x0"])
        y0n = float(focus_patch["y0"])
        x1n = float(focus_patch["x1"])
        y1n = float(focus_patch["y1"])
        cx, cy = 0.5 * (x0n + x1n), 0.5 * (y0n + y1n)
        bw, bh = max(0.12, x1n - x0n), max(0.12, y1n - y0n)
        x0n, x1n = max(0.0, cx - bw * 0.65), min(1.0, cx + bw * 0.65)
        y0n, y1n = max(0.0, cy - bh * 0.65), min(1.0, cy + bh * 0.65)
        y0, y1 = int(y0n * h), max(int(y1n * h), int(y0n * h) + 1)
        x0, x1 = int(x0n * w), max(int(x1n * w), int(x0n * w) + 1)
    else:
        # Fallback: center 50%
        y0, y1 = h // 4, 3 * h // 4
        x0, x1 = w // 4, 3 * w // 4
    crop = rgb[y0:y1, x0:x1]
    # Downsample crop — higher res on critical-band recheck
    target = 2200 if high_res_focus else 1600
    ch, cw = crop.shape[:2]
    m = max(ch, cw)
    if m > target:
        scale = target / m
        nh, nw = max(1, int(ch * scale)), max(1, int(cw * scale))
        from PIL import Image

        pil = Image.fromarray((np.clip(crop, 0, 1) * 255).astype(np.uint8), mode="RGB")
        pil = pil.resize((nw, nh), Image.Resampling.LANCZOS)
        crop = np.asarray(pil, dtype=np.float32) / 255.0

    gray = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    mag = np.sqrt(gx * gx + gy * gy)
    edge95 = float(np.quantile(mag, 0.95))

    residual = (
        np.pad(gray, 1, mode="edge")[0:-2, 1:-1]
        + np.pad(gray, 1, mode="edge")[2:, 1:-1]
        + np.pad(gray, 1, mode="edge")[1:-1, 0:-2]
        + np.pad(gray, 1, mode="edge")[1:-1, 2:]
        - 4.0 * gray
    )
    noise_mad = float(np.median(np.abs(residual)))

    return {
        "iso": iso,
        "black_level": black,
        "white_level": white,
        "raw_highlight_pct": round(hi_pct, 3),
        "raw_shadow_pct": round(sh_pct, 3),
        "raw_p99": round(p99, 4),
        "raw_p50": round(p50, 4),
        "headroom_ev": round(headroom_ev, 2),
        "midtone_ev": round(mid_ev, 2),
        "raw_edge95": round(edge95, 4),
        "raw_noise_mad": round(noise_mad, 5),
        "expected_noise": round(expected_noise(profile, iso), 5),
        "profile_id": profile.id,
        "focus_patch": focus_patch,
        "crop_rgb": crop,  # for optional re-score; caller may drop before JSON
    }


def score_raw_sharpness(edge95: float, profile: SensorProfile) -> float:
    ref = max(profile.sharp_ref, 1e-4)
    return float(np.clip((edge95 / ref) * 85.0, 0.0, 100.0))


def score_raw_dynamic_range(
    ctx: dict[str, Any],
    profile: SensorProfile,
    *,
    preset: str = "general",
    exposure_ctx: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    """RAW edit latitude (recoverability). Prefer score_edit_latitude for new callers."""
    from edit_latitude import score_edit_latitude

    score, _details, flags = score_edit_latitude(
        ctx, profile, preset=preset, exposure_ctx=exposure_ctx
    )
    return score, flags


def score_raw_noise(ctx: dict[str, Any], profile: SensorProfile) -> float:
    mad = float(ctx.get("raw_noise_mad", 0.0))
    expect = float(ctx.get("expected_noise") or expected_noise(profile, ctx.get("iso") or profile.base_iso))
    # ratio 1.0 → ~90; quieter than expected → higher
    ratio = mad / max(expect, 1e-6)
    score = 100.0 - (ratio - 0.7) * 55.0
    return float(np.clip(score, 15.0, 100.0))


def merge_scores(
    rgb_scores: dict[str, float],
    raw_sharp: float,
    raw_dr: float,
    raw_noise: float,
    profile: SensorProfile,
) -> dict[str, float]:
    """Blend RGB preview with RAW-aware scores.

    Dynamic range / latitude uses elevated trust (RAW headroom beats as-shot look).
    """
    t = float(np.clip(profile.raw_trust, 0.0, 1.0))
    t_dr = float(np.clip(max(t, 0.92), 0.0, 1.0))
    return {
        "sharpness": round((1 - t) * rgb_scores["sharpness"] + t * raw_sharp, 1),
        "dynamic_range": round((1 - t_dr) * rgb_scores["dynamic_range"] + t_dr * raw_dr, 1),
        "noise_control": round((1 - t) * rgb_scores["noise_control"] + t * raw_noise, 1),
        "color_harmony": rgb_scores["color_harmony"],
        "composition": rgb_scores["composition"],
    }
