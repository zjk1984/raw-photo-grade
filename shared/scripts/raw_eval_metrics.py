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


def extract_raw_context(path: Path, profile: SensorProfile, exif: dict[str, Any] | None = None) -> dict[str, Any]:
    """Decode RAW sensor stats + a linear 16-bit demosaic crop for focus."""
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
    # Attention-ish center crop (50%) — subject focus, ignore corners / bokeh rim
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    crop = rgb[y0:y1, x0:x1]
    # Downsample crop long edge to ~1600 for speed while keeping micro-contrast
    ch, cw = crop.shape[:2]
    m = max(ch, cw)
    if m > 1600:
        scale = 1600 / m
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
        "crop_rgb": crop,  # for optional re-score; caller may drop before JSON
    }


def score_raw_sharpness(edge95: float, profile: SensorProfile) -> float:
    ref = max(profile.sharp_ref, 1e-4)
    return float(np.clip((edge95 / ref) * 85.0, 0.0, 100.0))


def score_raw_dynamic_range(ctx: dict[str, Any], profile: SensorProfile) -> tuple[float, list[str]]:
    """Score exposure recoverability from sensor levels."""
    flags: list[str] = []
    headroom = float(ctx.get("headroom_ev", 0.0))
    hi = float(ctx.get("raw_highlight_pct", 0.0))
    sh = float(ctx.get("raw_shadow_pct", 0.0))
    mid_ev = float(ctx.get("midtone_ev", 0.0))

    # Ideal: ~1–3 stops headroom below clip, midtones near 0 EV from mid-grey
    hr_score = 100.0 - abs(headroom - 1.8) * 18.0
    if profile.computational:
        # Phone ProRAW already compressed; less headroom is normal
        hr_score = 100.0 - abs(headroom - 0.8) * 22.0
    hr_score = float(np.clip(hr_score, 20.0, 100.0))

    hi_pen = float(np.clip(hi * 2.5, 0.0, 45.0))
    sh_pen = float(np.clip(max(0.0, sh - 5.0) * 1.2, 0.0, 30.0))
    mid_pen = float(np.clip(max(0.0, abs(mid_ev) - 1.2) * 12.0, 0.0, 25.0))

    if hi > 2.5:
        flags.append("raw_highlight_clip")
    if hi > 8.0:
        flags.append("clipped_highlights")
    if sh > 25.0:
        flags.append("crushed_shadows")
    if mid_ev < -2.5:
        flags.append("underexposed")
    elif mid_ev > 2.5:
        flags.append("overexposed")

    score = float(np.clip(hr_score - hi_pen - sh_pen - mid_pen, 0.0, 100.0))
    # Cap by sensor DR reputation (phone lower ceiling)
    score = min(score, 55.0 + profile.base_dr_ev * 3.0)
    return round(score, 1), flags


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
    """Blend RGB preview scores with RAW-aware scores using profile.raw_trust."""
    t = float(np.clip(profile.raw_trust, 0.0, 1.0))
    return {
        "sharpness": round((1 - t) * rgb_scores["sharpness"] + t * raw_sharp, 1),
        "dynamic_range": round((1 - t) * rgb_scores["dynamic_range"] + t * raw_dr, 1),
        "noise_control": round((1 - t) * rgb_scores["noise_control"] + t * raw_noise, 1),
        "color_harmony": rgb_scores["color_harmony"],
        "composition": rgb_scores["composition"],
    }
