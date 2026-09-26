#!/usr/bin/env python3
"""RAW edit latitude — recoverability score (skill: exposure/ISO often fixable in post).

Track A (capture integrity): blurry / hard sensor clip — may veto S/A.
Track B (edit latitude): headroom, shadow floor, tone placement, ISO vs profile.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sensor_profiles import SensorProfile, expected_noise


# Preset multipliers on ISO noise drag inside latitude (higher = more forgiving)
PRESET_LATITUDE_ISO: dict[str, float] = {
    "general": 1.0,
    "portrait": 1.05,
    "landscape": 0.95,
    "street": 1.12,
    "night": 1.22,
}


def shadow_floor_score(raw_shadow_pct: float, raw_p50: float, profile: SensorProfile) -> float:
    """High when shadows still have code values (recoverable); low when crushed dead."""
    sh = float(raw_shadow_pct)
    p50 = float(raw_p50)
    # Soft underexposure with midtones still lifting → good floor
    if sh < 8.0 and p50 > 0.04:
        return 92.0
    if sh < 15.0 and p50 > 0.02:
        return 80.0
    if sh < 25.0 and p50 > 0.01:
        return 65.0
    # Large % near black but some mid left
    if p50 > 0.015:
        return float(np.clip(55.0 - (sh - 25.0) * 0.8, 25.0, 55.0))
    return float(np.clip(35.0 - sh * 0.5, 10.0, 40.0))


def headroom_score(headroom_ev: float, profile: SensorProfile) -> float:
    """Prefer 1–3 stops (camera) or ~0.5–1.5 (computational phone)."""
    hr = float(headroom_ev)
    if profile.computational:
        target, width = 0.9, 22.0
    else:
        target, width = 2.0, 16.0
    # More headroom than target is still good (room to lift) until absurd
    if hr >= target:
        return float(np.clip(95.0 - max(0.0, hr - target - 2.5) * 8.0, 50.0, 100.0))
    return float(np.clip(100.0 - (target - hr) * width, 20.0, 100.0))


def placement_score(midtone_ev: float) -> float:
    """As-shot mid placement: off-center is OK if latitude remains (soft curve)."""
    mid = abs(float(midtone_ev))
    # Within ±1.5 EV of mid-grey: excellent for latitude narrative
    if mid <= 1.5:
        return 90.0
    if mid <= 2.5:
        return 75.0
    if mid <= 3.5:
        return 55.0
    return float(np.clip(40.0 - (mid - 3.5) * 8.0, 15.0, 40.0))


def iso_latitude_score(
    ctx: dict[str, Any],
    profile: SensorProfile,
    *,
    preset: str = "general",
) -> float:
    """Noise relative to ISO expectation — high ISO narrows latitude, doesn't zero it."""
    mad = float(ctx.get("raw_noise_mad") or 0.0)
    iso = float(ctx.get("iso") or profile.base_iso)
    expect = float(ctx.get("expected_noise") or expected_noise(profile, iso))
    tol = float(PRESET_LATITUDE_ISO.get((preset or "general").lower(), 1.0))
    ratio = mad / max(expect * tol, 1e-6)
    # ratio 1 → ~88; worse noise eats latitude
    return float(np.clip(100.0 - (ratio - 0.65) * 50.0, 20.0, 100.0))


def analyze_sky_highlights(np_rgb: np.ndarray) -> dict[str, Any]:
    """Detect bright low-sat upper-frame mass (clouds / white sky) vs subject clip.

    Skill: cloudy white sky is normal — not a cull/score defect.
    """
    img = np.clip(np_rgb, 0, 1)
    h, w = img.shape[:2]
    upper = img[: max(1, h // 3)]
    lower = img[h // 3 :]
    r, g, b = upper[..., 0], upper[..., 1], upper[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    bright = luma > 0.88
    bright_pct = float(bright.mean() * 100.0)
    sat_bright = float(sat[bright].mean()) if bright.any() else 0.0
    # Texture: soft cloud gradients keep some luma std; dead clip is flat
    std_bright = float(luma[bright].std()) if bright.sum() > 32 else 0.0

    lr, lg, lb = lower[..., 0], lower[..., 1], lower[..., 2]
    ll = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb
    lower_hi_pct = float((ll > 0.97).mean() * 100.0)

    # Cloud / white sky: mostly upper bright + desaturated + not chroma-blown
    is_cloud = (
        bright_pct >= 12.0
        and sat_bright <= 0.12
        and lower_hi_pct < 6.0
    )
    # Soft texture optional — flat overcast still OK if low sat
    if bright_pct >= 25.0 and sat_bright <= 0.08 and lower_hi_pct < 8.0:
        is_cloud = True

    return {
        "sky_bright_pct": round(bright_pct, 2),
        "sky_sat_bright": round(sat_bright, 3),
        "sky_luma_std": round(std_bright, 4),
        "lower_hi_pct": round(lower_hi_pct, 2),
        "cloud_sky": bool(is_cloud),
    }


def score_edit_latitude(
    ctx: dict[str, Any],
    profile: SensorProfile,
    *,
    preset: str = "general",
    exposure_ctx: dict[str, Any] | None = None,
    face: dict[str, Any] | None = None,
    sky: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any], list[str]]:
    """Track B: recoverability 0–100 + hard/soft flags.

    Face window (optional): as-shot hot faces with channel headroom stay recoverable;
    global sky clip alone should not nuke portrait latitude.
    Cloud / white sky (optional): not a penalty — natural bright clouds.
    """
    flags: list[str] = []
    headroom = float(ctx.get("headroom_ev", 0.0))
    hi = float(ctx.get("raw_highlight_pct", 0.0))
    sh = float(ctx.get("raw_shadow_pct", 0.0))
    mid_ev = float(ctx.get("midtone_ev", 0.0))
    p50 = float(ctx.get("raw_p50", 0.18))
    face = face or {}
    sky = sky or {}
    cloud_sky = bool(sky.get("cloud_sky"))
    lower_hi = float(sky.get("lower_hi_pct") or 0.0)
    preset_l = (preset or "general").lower()
    personish = (
        preset_l == "portrait"
        or float(face.get("face_skin_frac") or 0) >= 0.06
        or bool(face.get("face_hot_as_shot"))
        or bool(face.get("face_recoverable"))
    )

    h_sc = headroom_score(headroom, profile)
    s_sc = shadow_floor_score(sh, p50, profile)
    m_sc = placement_score(mid_ev)
    n_sc = iso_latitude_score(ctx, profile, preset=preset)

    # Portrait: blend in face headroom so recoverable hot faces keep latitude
    face_hr = face.get("face_headroom")
    if face_hr is not None and personish:
        # Map face linear headroom ~0–0.2 → score-ish
        f_sc = float(np.clip(40.0 + float(face_hr) * 280.0, 25.0, 95.0))
        if face.get("face_dead"):
            f_sc = min(f_sc, 25.0)
        h_sc = 0.55 * h_sc + 0.45 * f_sc

    # Weights: headroom + shadow floor dominate (true edit space)
    lat = 0.35 * h_sc + 0.30 * s_sc + 0.20 * m_sc + 0.15 * n_sc

    # Sensor DR ceiling
    ceiling = 55.0 + float(profile.base_dr_ev) * 3.0
    if profile.computational:
        ceiling = min(ceiling, 88.0)
    lat = min(lat, ceiling)

    # Global sensor highlight clip — clouds / white sky are NOT a defect
    face_ok = bool(face.get("face_recoverable")) and not bool(face.get("face_dead"))
    if cloud_sky and lower_hi < 6.0:
        flags.append("cloud_sky")
        # Bright sky reduces measured headroom; restore latitude narrative for clouds
        if headroom < 1.0:
            lat = min(100.0, lat + min(12.0, (1.0 - headroom) * 10.0))
    elif hi > 4.0:
        if personish and face_ok and float(face.get("face_all_clip_pct") or 0) < 5.0:
            # Non-cloud background clip; face still editable — informational
            flags.append("preview_highlights")
            lat -= min(8.0, (hi - 4.0) * 0.8)
        else:
            flags.append("raw_highlight_clip")
            lat -= min(25.0, (hi - 4.0) * 3.0)
    if hi > 12.0 and not cloud_sky and not (personish and face_ok):
        flags.append("no_latitude")
        lat -= 15.0
    elif hi > 12.0 and personish and face_ok and not cloud_sky:
        lat -= 6.0

    if face.get("face_dead"):
        flags.append("face_dead_highlights")
        lat -= 18.0
        if "no_latitude" not in flags:
            flags.append("no_latitude")
    elif face.get("face_hot_as_shot"):
        flags.append("face_hot_as_shot")
        if face_ok:
            flags.append("face_recoverable")
            # Mild — recoverable with highlights pull + local finish
            lat = min(100.0, lat + 3.0)

    # Dead shadows (little code left) — not the same as dark-but-liftable
    if sh > 35.0 and p50 < 0.012:
        flags.append("crushed_shadows")
        lat -= 12.0
    elif sh > 20.0 and p50 >= 0.02:
        # Dark file with recoverable floor — informational
        flags.append("underexposed_as_shot")

    if mid_ev < -2.0 and "underexposed_as_shot" not in flags and hi < 3.0:
        flags.append("underexposed_as_shot")
    elif mid_ev > 2.2 and hi < 2.0 and "face_hot_as_shot" not in flags and not cloud_sky:
        flags.append("overexposed_as_shot")

    # Both sides exhausted (ignore when bright mass is just clouds)
    if headroom < 0.35 and s_sc < 40.0 and not cloud_sky and not (personish and face_ok):
        if "no_latitude" not in flags:
            flags.append("no_latitude")
        lat -= 10.0

    # Mild ISO context from exposure_ctx
    if exposure_ctx:
        ratio = float(exposure_ctx.get("iso_ratio") or 1.0)
        if ratio >= 16 and n_sc < 55:
            lat -= 4.0

    lat = float(np.clip(lat, 0.0, 100.0))
    details = {
        "edit_latitude": round(lat, 1),
        "latitude_headroom": round(h_sc, 1),
        "latitude_shadow_floor": round(s_sc, 1),
        "latitude_placement": round(m_sc, 1),
        "latitude_iso": round(n_sc, 1),
        "shadow_floor": round(s_sc, 1),
        "headroom_ev": round(headroom, 2),
        **{k: v for k, v in face.items() if k != "face_box"},
        **{k: v for k, v in sky.items()},
    }
    if face.get("face_box"):
        details["face_box"] = face["face_box"]
    return round(lat, 1), details, flags


def score_as_shot_preview(
    entropy: float,
    clipped_hi: float,
    clipped_sh: float,
    mean_luma: float,
    *,
    sky: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float], list[str]]:
    """Track C: weak as-shot look score — does not drive hard vetoes alone.

    Cloudy white sky does not add highlight penalty or preview_highlights.
    """
    flags: list[str] = []
    sky = sky or {}
    cloud_sky = bool(sky.get("cloud_sky"))
    entropy_score = float(np.clip((entropy / 5.4) * 85.0, 20.0, 95.0))
    # Soft penalties only (preview ≠ sensor). Clouds: ignore upper bright mass.
    effective_hi = 0.0 if cloud_sky else clipped_hi
    if cloud_sky and float(sky.get("lower_hi_pct") or 0) >= 6.0:
        # Subject/lower still hot — keep a fraction of penalty
        effective_hi = min(clipped_hi, float(sky["lower_hi_pct"]) / 100.0)
    hi_penalty = float(np.clip(effective_hi * 80.0, 0.0, 20.0))
    sh_penalty = float(np.clip(clipped_sh * 50.0, 0.0, 15.0))
    exp_dev = abs(mean_luma - 0.42)
    # Don't treat overall-bright cloudy frames as overexposed
    if cloud_sky:
        exp_dev = max(0.0, exp_dev - 0.12)
    exp_penalty = float(np.clip(max(0.0, exp_dev - 0.20) * 40.0, 0.0, 18.0))
    score = float(np.clip(entropy_score - hi_penalty - sh_penalty - exp_penalty + 8.0, 0.0, 100.0))

    if cloud_sky:
        flags.append("cloud_sky")
    elif effective_hi > 0.12:
        flags.append("preview_highlights")
    if clipped_sh > 0.28:
        flags.append("preview_shadows")
    if mean_luma < 0.10:
        flags.append("underexposed_as_shot")
    elif mean_luma > 0.88 and not cloud_sky:
        flags.append("overexposed_as_shot")

    details = {
        "as_shot_score": round(score, 1),
        "entropy": round(entropy, 2),
        "clipped_highlights_pct": round(clipped_hi * 100.0, 2),
        "clipped_shadows_pct": round(clipped_sh * 100.0, 2),
        "mean_luma": round(mean_luma, 3),
        **{k: v for k, v in sky.items()},
    }
    return round(score, 1), details, flags
