#!/usr/bin/env python3
"""Shared look presets (including Sony Creative Look inspired recipes).

Used by camera-raw-grade, phone-dng-grade, and photo-eval-grade pipeline/UI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lut3d import ensure_builtin_luts

# All numeric grade knobs (plus optional lut path / amount).
SLIDER_KEYS = [
    "exposure",
    "contrast",
    "highlights",
    "shadows",
    "whites",
    "blacks",
    "temperature",
    "tint",
    "vibrance",
    "saturation",
    "clarity",
    "vignette",
    "sharpen",
    "noise_luma",
    "fade",
    "sharpen_range",
    "hsl_skin_hue",
    "hsl_skin_sat",
    "hsl_sky_hue",
    "hsl_sky_sat",
    "hsl_green_hue",
    "hsl_green_sat",
    "lut_amount",
]

STRING_KEYS = ["lut"]


def _base(**kw: Any) -> dict[str, Any]:
    p: dict[str, Any] = {
        "exposure": 0.0,
        "contrast": 0,
        "highlights": 0,
        "shadows": 0,
        "whites": 0,
        "blacks": 0,
        "temperature": 0,
        "tint": 0,
        "vibrance": 0,
        "saturation": 0,
        "clarity": 0,
        "vignette": 0,
        "sharpen": 10,
        "noise_luma": 0,
        "fade": 0,
        "sharpen_range": 2,
        "hsl_skin_hue": 0,
        "hsl_skin_sat": 0,
        "hsl_sky_hue": 0,
        "hsl_sky_sat": 0,
        "hsl_green_hue": 0,
        "hsl_green_sat": 0,
        "lut": "",
        "lut_amount": 0,
    }
    p.update(kw)
    return p


def _lut(name: str) -> str:
    ensure_builtin_luts()
    return str(Path(__file__).resolve().parent.parent / "luts" / name)


# --- Classic looks (camera defaults; phone may raise noise_luma) ---

CLASSIC_LOOKS: dict[str, dict[str, Any]] = {
    "neutral": _base(sharpen=10),
    "natural": _base(
        exposure=0.1,
        contrast=12,
        highlights=-12,
        shadows=16,
        whites=4,
        blacks=-6,
        temperature=2,
        tint=-1,
        vibrance=10,
        saturation=2,
        clarity=10,
        vignette=-4,
        sharpen=20,
        noise_luma=4,
    ),
    "warm-golden": _base(
        exposure=0.15,
        contrast=14,
        highlights=-18,
        shadows=18,
        whites=2,
        blacks=-8,
        temperature=16,
        tint=4,
        vibrance=14,
        saturation=6,
        clarity=8,
        vignette=-8,
        sharpen=18,
        noise_luma=4,
    ),
    "cool-cinematic": _base(
        exposure=-0.05,
        contrast=18,
        highlights=-15,
        shadows=10,
        whites=-4,
        blacks=-14,
        temperature=-12,
        tint=-2,
        vibrance=6,
        saturation=-4,
        clarity=12,
        vignette=-12,
        sharpen=16,
        noise_luma=5,
        fade=12,
    ),
    "portrait": _base(
        exposure=0.2,
        contrast=8,
        highlights=-10,
        shadows=22,
        whites=2,
        blacks=-4,
        temperature=8,
        tint=3,
        vibrance=8,
        saturation=-2,
        clarity=4,
        vignette=-6,
        sharpen=12,
        noise_luma=6,
        fade=6,
        hsl_skin_sat=-6,
        hsl_skin_hue=2,
    ),
    "food": _base(
        exposure=0.15,
        contrast=16,
        highlights=-8,
        shadows=12,
        whites=8,
        blacks=-8,
        temperature=10,
        tint=2,
        vibrance=18,
        saturation=8,
        clarity=16,
        vignette=-6,
        sharpen=24,
        noise_luma=3,
    ),
    "travel": _base(
        exposure=0.12,
        contrast=16,
        highlights=-16,
        shadows=16,
        whites=6,
        blacks=-10,
        temperature=6,
        tint=0,
        vibrance=16,
        saturation=6,
        clarity=14,
        vignette=-6,
        sharpen=22,
        noise_luma=4,
        hsl_sky_sat=8,
        hsl_green_sat=6,
    ),
    "night": _base(
        exposure=0.2,
        contrast=10,
        highlights=-25,
        shadows=10,
        whites=-6,
        blacks=-12,
        temperature=-4,
        tint=-2,
        vibrance=8,
        saturation=2,
        clarity=6,
        vignette=-8,
        sharpen=12,
        noise_luma=12,
    ),
    "editorial-flat": _base(
        exposure=0.2,
        contrast=-8,
        highlights=-6,
        shadows=20,
        whites=-6,
        blacks=8,
        vibrance=4,
        saturation=-6,
        sharpen=10,
        noise_luma=3,
        fade=18,
    ),
}

# Sony Creative Look inspired presets (JPEG-side grammar → our sliders + HSL + LUT).
SONY_LOOKS: dict[str, dict[str, Any]] = {
    "sony-st": _base(
        exposure=0.08,
        contrast=10,
        highlights=-8,
        shadows=12,
        whites=2,
        blacks=-4,
        temperature=1,
        vibrance=8,
        saturation=2,
        clarity=8,
        sharpen=16,
        sharpen_range=2,
        noise_luma=3,
    ),
    "sony-pt": _base(
        exposure=0.18,
        contrast=4,
        highlights=-10,
        shadows=22,
        whites=2,
        blacks=-2,
        temperature=7,
        tint=3,
        vibrance=4,
        saturation=-4,
        clarity=2,
        vignette=-4,
        sharpen=8,
        sharpen_range=1,
        noise_luma=5,
        fade=10,
        hsl_skin_hue=3,
        hsl_skin_sat=-10,
    ),
    "sony-nt": _base(
        exposure=0.05,
        contrast=-4,
        highlights=-4,
        shadows=10,
        vibrance=-4,
        saturation=-10,
        clarity=0,
        sharpen=6,
        sharpen_range=1,
        fade=6,
        noise_luma=3,
    ),
    "sony-vv": _base(
        exposure=0.1,
        contrast=18,
        highlights=-14,
        shadows=12,
        whites=4,
        blacks=-8,
        vibrance=22,
        saturation=8,
        clarity=12,
        sharpen=20,
        sharpen_range=3,
        hsl_sky_sat=14,
        hsl_green_sat=12,
        noise_luma=3,
    ),
    "sony-vv2": _base(
        exposure=0.12,
        contrast=14,
        highlights=-12,
        shadows=16,
        whites=6,
        blacks=-6,
        vibrance=26,
        saturation=6,
        clarity=18,
        sharpen=18,
        sharpen_range=3,
        hsl_sky_sat=16,
        hsl_green_sat=10,
        lut=_lut("sony_vv2_approx.cube"),
        lut_amount=35,
        noise_luma=3,
    ),
    "sony-fl": _base(
        exposure=0.05,
        contrast=22,
        highlights=-16,
        shadows=8,
        whites=-2,
        blacks=-12,
        temperature=-6,
        tint=-2,
        vibrance=4,
        saturation=-8,
        clarity=10,
        vignette=-8,
        sharpen=16,
        sharpen_range=3,
        fade=20,
        hsl_sky_hue=-14,
        hsl_sky_sat=12,
        hsl_green_hue=8,
        hsl_green_sat=10,
        hsl_skin_sat=-12,
        lut=_lut("sony_fl_approx.cube"),
        lut_amount=55,
        noise_luma=4,
    ),
    "sony-fl2": _base(
        exposure=0.0,
        contrast=20,
        highlights=-18,
        shadows=6,
        blacks=-10,
        temperature=-4,
        tint=-1,
        vibrance=2,
        saturation=-12,
        clarity=8,
        vignette=-10,
        sharpen=14,
        fade=28,
        hsl_sky_hue=-10,
        hsl_sky_sat=8,
        hsl_green_sat=6,
        hsl_skin_sat=-14,
        lut=_lut("sony_fl_approx.cube"),
        lut_amount=65,
        noise_luma=4,
    ),
    "sony-in": _base(
        exposure=0.12,
        contrast=-14,
        highlights=-6,
        shadows=18,
        whites=-8,
        blacks=14,
        vibrance=-8,
        saturation=-14,
        clarity=0,
        sharpen=8,
        sharpen_range=1,
        fade=48,
        noise_luma=4,
    ),
    "sony-sh": _base(
        exposure=0.35,
        contrast=-8,
        highlights=-10,
        shadows=28,
        whites=12,
        blacks=6,
        temperature=4,
        vibrance=12,
        saturation=6,
        clarity=2,
        sharpen=10,
        sharpen_range=1,
        fade=22,
        noise_luma=4,
    ),
}

# Fujifilm Film Simulation inspired presets.
FUJI_LOOKS: dict[str, dict[str, Any]] = {
    "fuji-provia": _base(
        exposure=0.08,
        contrast=10,
        highlights=-10,
        shadows=12,
        whites=2,
        blacks=-4,
        temperature=2,
        vibrance=8,
        saturation=2,
        clarity=8,
        sharpen=16,
        sharpen_range=2,
        noise_luma=3,
    ),
    "fuji-astia": _base(
        exposure=0.16,
        contrast=4,
        highlights=-12,
        shadows=20,
        whites=2,
        blacks=-2,
        temperature=6,
        tint=2,
        vibrance=6,
        saturation=-2,
        clarity=2,
        sharpen=10,
        sharpen_range=1,
        fade=8,
        hsl_skin_hue=2,
        hsl_skin_sat=-8,
        noise_luma=4,
    ),
    "fuji-velvia": _base(
        exposure=0.05,
        contrast=20,
        highlights=-18,
        shadows=10,
        whites=4,
        blacks=-10,
        vibrance=24,
        saturation=12,
        clarity=14,
        sharpen=20,
        sharpen_range=3,
        hsl_sky_sat=18,
        hsl_sky_hue=-8,
        hsl_green_sat=20,
        hsl_green_hue=6,
        lut=_lut("fuji_velvia_approx.cube"),
        lut_amount=50,
        noise_luma=3,
    ),
    "fuji-classic-chrome": _base(
        exposure=0.06,
        contrast=12,
        highlights=-14,
        shadows=10,
        whites=-2,
        blacks=-8,
        temperature=-4,
        tint=-2,
        vibrance=-2,
        saturation=-10,
        clarity=6,
        vignette=-4,
        sharpen=14,
        fade=14,
        hsl_sky_hue=-10,
        hsl_sky_sat=6,
        hsl_skin_sat=-6,
        lut=_lut("fuji_classic_chrome_approx.cube"),
        lut_amount=55,
        noise_luma=3,
    ),
    "fuji-classic-neg": _base(
        exposure=0.1,
        contrast=8,
        highlights=-8,
        shadows=16,
        whites=-4,
        blacks=4,
        temperature=4,
        tint=1,
        vibrance=4,
        saturation=-6,
        clarity=4,
        sharpen=12,
        fade=22,
        hsl_green_hue=-4,
        hsl_green_sat=-4,
        hsl_skin_sat=-4,
        lut=_lut("fuji_classic_chrome_approx.cube"),
        lut_amount=30,
        noise_luma=3,
    ),
    "fuji-nostalgic-neg": _base(
        exposure=0.15,
        contrast=6,
        highlights=-10,
        shadows=18,
        whites=2,
        blacks=6,
        temperature=12,
        tint=4,
        vibrance=8,
        saturation=2,
        clarity=2,
        sharpen=10,
        fade=26,
        hsl_sky_sat=-6,
        hsl_skin_hue=4,
        lut=_lut("fuji_nostalgic_neg_approx.cube"),
        lut_amount=60,
        noise_luma=4,
    ),
    "fuji-eterna": _base(
        exposure=0.12,
        contrast=-10,
        highlights=-8,
        shadows=18,
        whites=-8,
        blacks=10,
        temperature=-2,
        vibrance=-4,
        saturation=-12,
        clarity=0,
        sharpen=8,
        fade=28,
        noise_luma=3,
    ),
    "fuji-acros": _base(
        exposure=0.08,
        contrast=16,
        highlights=-12,
        shadows=8,
        whites=2,
        blacks=-12,
        vibrance=0,
        saturation=-100,
        clarity=10,
        sharpen=18,
        sharpen_range=3,
        fade=6,
        noise_luma=4,
    ),
}

# Nikon Picture Control inspired presets.
NIKON_LOOKS: dict[str, dict[str, Any]] = {
    "nikon-standard": _base(
        exposure=0.08,
        contrast=12,
        highlights=-10,
        shadows=12,
        whites=2,
        blacks=-6,
        vibrance=10,
        saturation=2,
        clarity=10,
        sharpen=18,
        sharpen_range=2,
        noise_luma=3,
    ),
    "nikon-neutral": _base(
        exposure=0.05,
        contrast=-2,
        highlights=-4,
        shadows=10,
        vibrance=0,
        saturation=-6,
        clarity=2,
        sharpen=8,
        sharpen_range=1,
        fade=4,
        noise_luma=3,
    ),
    "nikon-vivid": _base(
        exposure=0.08,
        contrast=18,
        highlights=-14,
        shadows=10,
        whites=4,
        blacks=-10,
        vibrance=22,
        saturation=10,
        clarity=14,
        sharpen=22,
        sharpen_range=3,
        hsl_sky_sat=12,
        hsl_green_sat=10,
        noise_luma=3,
    ),
    "nikon-portrait": _base(
        exposure=0.18,
        contrast=6,
        highlights=-12,
        shadows=22,
        whites=2,
        blacks=-2,
        temperature=6,
        tint=2,
        vibrance=6,
        saturation=-2,
        clarity=2,
        sharpen=10,
        sharpen_range=1,
        fade=6,
        hsl_skin_hue=2,
        hsl_skin_sat=-8,
        noise_luma=5,
    ),
    "nikon-rich-tone-portrait": _base(
        exposure=0.12,
        contrast=8,
        highlights=-18,
        shadows=18,
        whites=-2,
        blacks=-4,
        temperature=5,
        tint=2,
        vibrance=10,
        saturation=2,
        clarity=4,
        sharpen=12,
        fade=4,
        hsl_skin_hue=1,
        hsl_skin_sat=-4,
        noise_luma=4,
    ),
    "nikon-landscape": _base(
        exposure=0.08,
        contrast=16,
        highlights=-16,
        shadows=12,
        whites=4,
        blacks=-8,
        vibrance=16,
        saturation=6,
        clarity=14,
        sharpen=20,
        sharpen_range=3,
        hsl_sky_sat=16,
        hsl_sky_hue=-6,
        hsl_green_sat=14,
        hsl_green_hue=4,
        lut=_lut("nikon_landscape_approx.cube"),
        lut_amount=45,
        noise_luma=3,
    ),
    "nikon-flat": _base(
        exposure=0.15,
        contrast=-12,
        highlights=-6,
        shadows=20,
        whites=-8,
        blacks=12,
        vibrance=-4,
        saturation=-10,
        clarity=0,
        sharpen=8,
        fade=20,
        noise_luma=3,
    ),
    "nikon-monochrome": _base(
        exposure=0.06,
        contrast=14,
        highlights=-10,
        shadows=8,
        whites=2,
        blacks=-10,
        vibrance=0,
        saturation=-100,
        clarity=8,
        sharpen=16,
        fade=4,
        noise_luma=4,
    ),
}

ALL_LOOKS: dict[str, dict[str, Any]] = {
    **CLASSIC_LOOKS,
    **SONY_LOOKS,
    **FUJI_LOOKS,
    **NIKON_LOOKS,
}

# Scene tag → preferred look (overridable via prefs).
DEFAULT_SCENE_MAP: dict[str, str] = {
    "portrait": "sony-pt",
    "landscape": "sony-fl",
    "vivid": "sony-vv2",
    "matte": "sony-in",
    "highkey": "sony-sh",
    "night": "night",
    "neutral_grade": "sony-nt",
    "general": "sony-st",
}

BRAND_SCENE_MAPS: dict[str, dict[str, str]] = {
    "sony": DEFAULT_SCENE_MAP,
    "fuji": {
        "portrait": "fuji-astia",
        "landscape": "fuji-velvia",
        "vivid": "fuji-velvia",
        "matte": "fuji-classic-chrome",
        "highkey": "fuji-astia",
        "night": "fuji-eterna",
        "neutral_grade": "fuji-provia",
        "general": "fuji-provia",
    },
    "nikon": {
        "portrait": "nikon-portrait",
        "landscape": "nikon-landscape",
        "vivid": "nikon-vivid",
        "matte": "nikon-flat",
        "highkey": "nikon-rich-tone-portrait",
        "night": "night",
        "neutral_grade": "nikon-neutral",
        "general": "nikon-standard",
    },
}

BRAND_DEFAULT_LOOK: dict[str, str] = {
    "sony": "sony-st",
    "fuji": "fuji-provia",
    "nikon": "nikon-standard",
}

PREFS_PATH = Path.home() / ".photograde" / "look_prefs.json"


def scene_map_for_brand(brand: str) -> dict[str, str]:
    return dict(BRAND_SCENE_MAPS.get(brand, DEFAULT_SCENE_MAP))


def load_prefs() -> dict[str, Any]:
    fallback = {
        "default_look": "sony-st",
        "auto_look": True,
        "brand": "sony",
        "scene_map": dict(DEFAULT_SCENE_MAP),
    }
    if not PREFS_PATH.exists():
        return dict(fallback)
    try:
        with PREFS_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return dict(fallback)
        brand = str(data.get("brand") or "sony").lower()
        if brand not in BRAND_SCENE_MAPS:
            brand = "sony"
        data["brand"] = brand
        data.setdefault("default_look", BRAND_DEFAULT_LOOK.get(brand, "sony-st"))
        data.setdefault("auto_look", True)
        sm = scene_map_for_brand(brand)
        # Only overlay explicit user scene_map keys that still exist
        user_sm = data.get("scene_map") or {}
        if isinstance(user_sm, dict):
            sm.update({k: v for k, v in user_sm.items() if isinstance(v, str)})
        data["scene_map"] = sm
        return data
    except Exception:
        return dict(fallback)


def save_prefs(prefs: dict[str, Any]) -> None:
    PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREFS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(prefs, fh, indent=2, ensure_ascii=False)


def complete_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill missing slider keys so apply_grade never KeyErrors."""
    out = _base()
    if params:
        for k, v in params.items():
            if k in SLIDER_KEYS or k in STRING_KEYS or k == "look":
                out[k] = v
    return out


def get_look(name: str, looks: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    catalog = looks or ALL_LOOKS
    if name not in catalog:
        raise KeyError(f"Unknown look: {name}")
    p = complete_params(catalog[name])
    p["look"] = name
    return p


def phone_looks() -> dict[str, dict[str, Any]]:
    """Phone-tuned: slightly higher noise reduction, softer clarity."""
    out: dict[str, dict[str, Any]] = {}
    for name, base in ALL_LOOKS.items():
        p = dict(base)
        p["noise_luma"] = int(p.get("noise_luma", 0)) + 4
        p["clarity"] = max(0, int(p.get("clarity", 0)) - 2)
        if name == "night":
            p["noise_luma"] = 22
            p["exposure"] = 0.25
            p["highlights"] = -30
        out[name] = p
    return out


def camera_looks() -> dict[str, dict[str, Any]]:
    return {k: dict(v) for k, v in ALL_LOOKS.items()}


def look_choices() -> list[str]:
    return sorted(ALL_LOOKS.keys())
