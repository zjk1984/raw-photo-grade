#!/usr/bin/env python3
"""Heuristic scene tagging → brand look selection (Sony / Fuji / Nikon)."""

from __future__ import annotations

from typing import Any

import numpy as np

from looks import BRAND_DEFAULT_LOOK, BRAND_SCENE_MAPS, DEFAULT_SCENE_MAP, load_prefs, scene_map_for_brand


def classify_scene(np_rgb: np.ndarray, details: dict[str, Any] | None = None) -> str:
    """Return a scene tag: portrait|landscape|vivid|matte|highkey|night|neutral_grade|general."""
    details = details or {}
    img = np.clip(np_rgb, 0, 1)
    h, w = img.shape[:2]
    step = max(1, min(h, w) // 256)
    small = img[::step, ::step]
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mean_l = float(luma.mean())
    contrast = float(luma.std())
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    mean_sat = float(sat.mean())
    colorfulness = float(details.get("colorfulness", mean_sat * 1.2))

    upper = small[: max(1, small.shape[0] // 3)]
    blue_bias = float(upper[..., 2].mean() - upper[..., 0].mean())
    green_bias = float(small[..., 1].mean() - small[..., 0].mean())

    cy0, cy1 = small.shape[0] // 4, 3 * small.shape[0] // 4
    cx0, cx1 = small.shape[1] // 4, 3 * small.shape[1] // 4
    center = small[cy0:cy1, cx0:cx1]
    cr, cg, cb = center[..., 0], center[..., 1], center[..., 2]
    skin = (cr > cg) & (cg > cb * 0.85) & ((cr - cb) > 0.05) & ((cr - cg) < 0.25)
    skin_frac = float(skin.mean()) if skin.size else 0.0

    subj = details.get("subject_center") or [0.5, 0.5]
    subj_y = float(subj[1]) if len(subj) > 1 else 0.5

    if mean_l < 0.22 and contrast < 0.22:
        return "night"
    if mean_l > 0.62 and contrast < 0.18:
        return "highkey"
    if mean_sat < 0.08 and contrast < 0.16:
        return "matte"
    if skin_frac > 0.12 and 0.25 < subj_y < 0.75:
        return "portrait"
    if blue_bias > 0.04 or (green_bias > 0.03 and blue_bias > 0.01):
        return "landscape"
    if colorfulness > 0.28 or mean_sat > 0.22:
        return "vivid"
    if mean_sat < 0.12 and contrast < 0.20:
        return "neutral_grade"
    return "general"


def suggest_look(
    np_rgb: np.ndarray,
    details: dict[str, Any] | None = None,
    *,
    forced: str | None = None,
    auto: bool | None = None,
    brand: str | None = None,
) -> tuple[str, str]:
    """Return (look_name, scene_tag). Honors ~/.photograde/look_prefs.json."""
    prefs = load_prefs()
    if forced and forced != "auto":
        return forced, "forced"
    use_auto = prefs.get("auto_look", True) if auto is None else auto
    brand_key = (brand or prefs.get("brand") or "sony").lower()
    if brand_key not in BRAND_SCENE_MAPS:
        brand_key = "sony"
    if not use_auto:
        default = prefs.get("default_look") or BRAND_DEFAULT_LOOK.get(brand_key, "sony-st")
        return str(default), "prefs"
    tag = classify_scene(np_rgb, details)
    # Explicit --brand overrides stored scene_map so auto picks that brand's looks
    if brand is not None:
        scene_map = scene_map_for_brand(brand_key)
    else:
        scene_map = prefs.get("scene_map") or scene_map_for_brand(brand_key) or DEFAULT_SCENE_MAP
    look = scene_map.get(tag) or BRAND_DEFAULT_LOOK.get(brand_key) or "sony-st"
    return str(look), tag
