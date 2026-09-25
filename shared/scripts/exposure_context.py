#!/usr/bin/env python3
"""EXIF exposure-triangle context (F / Tv / ISO) for skill-aligned score calibration.

Calibrates expectations (blur_cut, noise tolerance, info flags) — does not replace
image evidence. Focus is King still uses S_plane vs blur_cut'.
"""

from __future__ import annotations

import math
from typing import Any

from focus_sharpness import parse_fnumber


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        if isinstance(v, (list, tuple)) and v:
            return float(v[0])
        # exiftool may return "1/125"
        if isinstance(v, str) and "/" in v:
            a, b = v.split("/", 1)
            return float(a) / float(b)
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def parse_exposure_time(exif: dict[str, Any] | None) -> float | None:
    """Shutter time in seconds."""
    if not exif:
        return None
    for key in ("ExposureTime", "ShutterSpeedValue", "exposure_time"):
        t = _as_float(exif.get(key))
        if t is not None and t > 0:
            # ShutterSpeedValue is APEX Tv = -log2(t); if value looks like APEX (>20 unlikely as seconds)
            if key == "ShutterSpeedValue" and t > 2.5:
                return float(2.0 ** (-t))
            return t
    return None


def parse_iso(exif: dict[str, Any] | None) -> float | None:
    if not exif:
        return None
    for key in ("ISO", "ISOSpeedRatings", "PhotographicSensitivity", "iso"):
        v = _as_float(exif.get(key))
        if v is not None and v > 0:
            return v
    return None


def parse_focal_35mm(exif: dict[str, Any] | None) -> float | None:
    if not exif:
        return None
    for key in (
        "FocalLengthIn35mmFormat",
        "FocalLengthIn35mmFilm",
        "FocalLength35efl",
    ):
        v = _as_float(exif.get(key))
        if v is not None and v > 0:
            return v
    # Fallback: raw focal length (not 35mm-eq) — still better than nothing
    v = _as_float(exif.get("FocalLength"))
    return v if v and v > 0 else None


def scene_ev100(fnumber: float | None, exposure_s: float | None, iso: float | None) -> float | None:
    """Approximate metered scene EV at ISO 100: log2(N²/t) − log2(ISO/100)."""
    if not fnumber or not exposure_s or fnumber <= 0 or exposure_s <= 0:
        return None
    iso = float(iso or 100.0)
    try:
        return float(math.log2((fnumber * fnumber) / exposure_s) - math.log2(iso / 100.0))
    except (ValueError, ZeroDivisionError):
        return None


def safe_shutter_s(
    *,
    focal_35: float | None,
    family: str = "camera",
    ibis_stops: float = 1.0,
) -> float:
    """Reciprocal-rule safe shutter (seconds). IBIS adds stops of forgiveness."""
    if family == "phone":
        base = 1.0 / 125.0
    else:
        fl = float(focal_35) if focal_35 and focal_35 > 0 else 50.0
        base = 1.0 / max(fl, 24.0)
    # Each IBIS stop doubles allowable time
    return float(base * (2.0 ** max(0.0, ibis_stops)))


def build_exposure_context(
    exif: dict[str, Any] | None,
    *,
    profile: Any | None = None,
    preset: str = "general",
) -> dict[str, Any]:
    """E0: parse F/Tv/ISO + derived shake_risk / scene_ev (no scoring)."""
    exif = dict(exif or {})
    fnumber = parse_fnumber(exif)
    exposure_s = parse_exposure_time(exif)
    iso = parse_iso(exif)
    focal_35 = parse_focal_35mm(exif)

    family = "camera"
    base_iso = 100.0
    ibis = 1.0
    if profile is not None:
        family = str(getattr(profile, "family", "camera") or "camera")
        base_iso = float(getattr(profile, "base_iso", 100) or 100)
        # Modern ILC bodies: assume ~1 stop IBIS; phones: stronger OIS + multi-frame
        ibis = 2.0 if family == "phone" else 1.0
        # EXIF stabilization hints (when present)
        for key in ("ImageStabilization", "Stabilization", "SensorStabilization"):
            v = str(exif.get(key) or "").lower()
            if v and v not in {"off", "none", "0", "false", "no"}:
                ibis = max(ibis, 2.5 if family == "phone" else 1.5)
                break

    t_safe = safe_shutter_s(focal_35=focal_35, family=family, ibis_stops=ibis)
    shake_risk = None
    if exposure_s and exposure_s > 0 and t_safe > 0:
        shake_risk = float(math.log2(exposure_s / t_safe))

    iso_v = float(iso) if iso else base_iso
    iso_ratio = iso_v / max(base_iso, 1.0)
    ev = scene_ev100(fnumber, exposure_s, iso_v)

    return {
        "fnumber": round(fnumber, 2) if fnumber else None,
        "exposure_time": round(exposure_s, 6) if exposure_s else None,
        "iso": round(iso_v, 1) if iso else None,
        "focal_35mm": round(focal_35, 1) if focal_35 else None,
        "t_safe": round(t_safe, 6),
        "shake_risk": round(shake_risk, 2) if shake_risk is not None else None,
        "iso_ratio": round(iso_ratio, 2),
        "scene_ev100": round(ev, 2) if ev is not None else None,
        "preset": preset,
        "base_iso": base_iso,
        "family": family,
    }


# Preset noise tolerance (>1 = more forgiving of high-ISO MAD)
PRESET_NOISE_TOLERANCE: dict[str, float] = {
    "general": 1.0,
    "portrait": 1.05,
    "landscape": 0.92,
    "street": 1.18,
    "night": 1.28,
}


def blur_cuts_for_exposure(
    base_blur: float,
    base_soft: float,
    ctx: dict[str, Any],
    *,
    preset: str = "general",
) -> tuple[float, float, dict[str, float]]:
    """E1: blur_cut' from F / Tv / ISO / preset / phone. Cap total relief so Focus is King holds."""
    blur, soft = float(base_blur), float(base_soft)
    deltas = {"f": 0.0, "tv": 0.0, "iso": 0.0, "preset": 0.0, "phone": 0.0}

    family = str(ctx.get("family") or "camera").lower()
    f = ctx.get("fnumber")
    if f is not None:
        f = float(f)
        if f <= 2.0:
            deltas["f"] = -4.0
        elif f <= 2.8:
            deltas["f"] = -2.0
        elif f >= 11.0:
            deltas["f"] = 2.0
        elif f >= 8.0:
            deltas["f"] = 1.0
    elif family == "phone":
        # ProRAW often omits F; typical main/ultrawide ~1.5–2.2
        deltas["f"] = -2.0

    if family == "phone":
        # Computational stack + OIS: slightly more forgiving plane cut
        deltas["phone"] = -2.0

    shake = ctx.get("shake_risk")
    if shake is not None:
        shake = float(shake)
        if shake > 2.0:
            deltas["tv"] = -2.0
        elif shake > 1.0:
            deltas["tv"] = -1.0
        elif shake < -1.0:
            deltas["tv"] = 0.5  # very safe shutter — slightly stricter

    ratio = float(ctx.get("iso_ratio") or 1.0)
    if ratio >= 32:
        deltas["iso"] = -3.0
    elif ratio >= 16:
        deltas["iso"] = -2.0
    elif ratio >= 8:
        deltas["iso"] = -1.0

    p = (preset or ctx.get("preset") or "general").lower()
    if p == "night":
        deltas["preset"] = -2.0
    elif p == "street":
        deltas["preset"] = -1.0
    elif p == "landscape":
        deltas["preset"] = 1.0
    elif p == "portrait" and (f is not None and float(f) <= 2.8 or family == "phone"):
        deltas["preset"] = -1.0

    total = sum(deltas.values())
    # Phones may take a bit more relief; still cap so Focus is King
    lo = -10.0 if family == "phone" else -8.0
    total = max(lo, min(4.0, total))
    blur_out = blur + total
    soft_out = soft + total * 0.75
    # Keep ordering soft > blur (phone soft band slightly wider)
    soft_gap = 10.0 if family == "phone" else 8.0
    if soft_out <= blur_out:
        soft_out = blur_out + soft_gap
    return blur_out, soft_out, deltas


def exposure_info_flags(ctx: dict[str, Any]) -> list[str]:
    """Informational flags (no hard S/A veto by themselves)."""
    flags: list[str] = []
    shake = ctx.get("shake_risk")
    if shake is not None and float(shake) > 1.0:
        flags.append("motion_risk")
    ratio = float(ctx.get("iso_ratio") or 1.0)
    if ratio >= 16:
        flags.append("high_iso")
    return flags


def adjust_noise_for_preset(
    noise_score: float,
    ctx: dict[str, Any],
    *,
    preset: str = "general",
) -> float:
    """E2: preset×ISO tolerance — pull high-ISO street/night scores up slightly."""
    import numpy as np

    tol = float(PRESET_NOISE_TOLERANCE.get((preset or "general").lower(), 1.0))
    ratio = float(ctx.get("iso_ratio") or 1.0)
    score = float(noise_score)
    if tol > 1.0 and ratio > 4.0:
        # Forgiving presets: recover up to ~10 pts when ISO is elevated
        boost = min(10.0, (tol - 1.0) * 40.0 * math.log2(max(ratio / 4.0, 1.0)))
        score += boost
    elif tol < 1.0 and ratio > 8.0:
        score -= min(6.0, (1.0 - tol) * 25.0 * math.log2(max(ratio / 8.0, 1.0)))
    return float(np.clip(score, 0.0, 100.0))


def exposure_mismatch_penalty(
    midtone_ev: float | None,
    ctx: dict[str, Any],
) -> tuple[float, str | None]:
    """E2: light penalty when scene EV and RAW midtones disagree badly."""
    ev = ctx.get("scene_ev100")
    if ev is None or midtone_ev is None:
        return 0.0, None
    ev = float(ev)
    mid = float(midtone_ev)
    # Bright metering but dark file / dark metering but hot file
    if ev >= 10.0 and mid <= -2.0:
        return 6.0, "exposure_mismatch"
    if ev <= 4.0 and mid >= 1.8:
        return 6.0, "exposure_mismatch"
    return 0.0, None
