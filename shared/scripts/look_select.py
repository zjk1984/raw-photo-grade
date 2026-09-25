#!/usr/bin/env python3
"""Scene tagging → ordered look pools → deterministic secondary pick (skill-aligned).

Flow matches camera-raw-grade / photo-eval-grade skills:
  classify scene → ordered candidate pool → cue-based pick → optional sticky lock.
Never random. Batch sticky keeps one shoot, one grade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from looks import (
    ALL_LOOKS,
    BRAND_DEFAULT_LOOK,
    BRAND_SCENE_POOLS,
    DEFAULT_SCENE_MAP,
    load_prefs,
    scene_pools_for_brand,
)


@dataclass
class LookSuggestion:
    look: str
    scene_tag: str
    candidates: list[str] = field(default_factory=list)
    reason: str = ""
    sticky: bool = False
    brand: str = "sony"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_scene(np_rgb: np.ndarray, details: dict[str, Any] | None = None) -> str:
    """Return a scene tag: portrait|landscape|vivid|matte|highkey|night|neutral_grade|general."""
    cues = extract_cues(np_rgb, details)
    if cues["mean_l"] < 0.22 and cues["contrast"] < 0.22:
        return "night"
    if cues["mean_l"] > 0.62 and cues["contrast"] < 0.18:
        return "highkey"
    if cues["mean_sat"] < 0.08 and cues["contrast"] < 0.16:
        return "matte"
    if cues["skin_frac"] > 0.12 and 0.25 < cues["subj_y"] < 0.75:
        return "portrait"
    if cues["blue_bias"] > 0.04 or (cues["green_bias"] > 0.03 and cues["blue_bias"] > 0.01):
        return "landscape"
    if cues["colorfulness"] > 0.28 or cues["mean_sat"] > 0.22:
        return "vivid"
    if cues["mean_sat"] < 0.12 and cues["contrast"] < 0.20:
        return "neutral_grade"
    return "general"


def extract_cues(np_rgb: np.ndarray, details: dict[str, Any] | None = None) -> dict[str, float]:
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
    warm_bias = float(small[..., 0].mean() - small[..., 2].mean())

    cy0, cy1 = small.shape[0] // 4, 3 * small.shape[0] // 4
    cx0, cx1 = small.shape[1] // 4, 3 * small.shape[1] // 4
    center = small[cy0:cy1, cx0:cx1]
    cr, cg, cb = center[..., 0], center[..., 1], center[..., 2]
    skin = (cr > cg) & (cg > cb * 0.85) & ((cr - cb) > 0.05) & ((cr - cg) < 0.25)
    skin_frac = float(skin.mean()) if skin.size else 0.0

    subj = details.get("subject_center") or [0.5, 0.5]
    subj_y = float(subj[1]) if len(subj) > 1 else 0.5
    hi_clip = float(details.get("clipped_highlights_pct") or details.get("highlight_clip_pct") or 0.0)

    return {
        "mean_l": mean_l,
        "contrast": contrast,
        "mean_sat": mean_sat,
        "colorfulness": colorfulness,
        "blue_bias": blue_bias,
        "green_bias": green_bias,
        "warm_bias": warm_bias,
        "skin_frac": skin_frac,
        "subj_y": subj_y,
        "hi_clip": hi_clip,
    }


def _score_candidate(name: str, scene: str, cues: dict[str, float]) -> tuple[float, str]:
    """Higher score wins. Reasons are short tags for manifests / agent Read."""
    n = name.lower()
    score = 0.0
    reasons: list[str] = []

    # Pool order baseline: earlier candidates get a tiny prior (skill: safe first)
    # applied by caller via index; here only cue fit.

    if scene == "landscape":
        if cues["colorfulness"] > 0.28 or cues["mean_sat"] > 0.20:
            if any(k in n for k in ("velvia", "vv2", "vv", "vivid", "landscape")):
                score += 3.0
                reasons.append("high_colorfulness")
        if cues["mean_sat"] < 0.14 or cues["contrast"] < 0.18:
            if any(k in n for k in ("classic-chrome", "fl", "chrome", "flat")):
                score += 2.5
                reasons.append("muted_documentary")
        if cues["blue_bias"] > 0.05 or cues["green_bias"] > 0.04:
            if any(k in n for k in ("velvia", "landscape", "vv", "fl")):
                score += 1.5
                reasons.append("sky_green_bias")

    elif scene == "portrait":
        if cues["skin_frac"] > 0.15:
            if any(k in n for k in ("pt", "astia", "portrait")) and "rich" not in n:
                score += 2.5
                reasons.append("skin_protect")
        if cues["hi_clip"] > 1.5 or cues["mean_l"] > 0.55:
            if any(k in n for k in ("rich-tone", "astia", "sh", "pt")):
                score += 2.0
                reasons.append("highlight_protect")
        if cues["warm_bias"] > 0.04 and cues["mean_l"] > 0.45:
            if any(k in n for k in ("nostalgic", "rich-tone", "warm")):
                score += 1.5
                reasons.append("warm_skin")

    elif scene == "vivid":
        if any(k in n for k in ("velvia", "vv2", "vv", "vivid", "landscape")):
            score += 3.0
            reasons.append("vivid_punch")
        if cues["warm_bias"] > 0.05 and "classic-neg" in n:
            score += 1.5
            reasons.append("warm_urban")

    elif scene == "matte":
        if any(k in n for k in ("in", "classic-chrome", "flat", "eterna", "fl2", "editorial")):
            score += 3.0
            reasons.append("matte_fade")

    elif scene == "highkey":
        if any(k in n for k in ("sh", "astia", "rich-tone", "provia")):
            score += 2.5
            reasons.append("highkey_soft")

    elif scene == "night":
        if n == "night" or any(k in n for k in ("eterna", "flat", "fl", "nt", "neutral")):
            score += 2.5
            reasons.append("night_safe")

    elif scene == "neutral_grade":
        if any(k in n for k in ("nt", "neutral", "flat", "eterna", "provia", "editorial")):
            score += 2.5
            reasons.append("grade_later")

    else:  # general
        if any(k in n for k in ("st", "provia", "standard", "natural")):
            score += 2.0
            reasons.append("safe_standard")
        if cues["colorfulness"] > 0.25 and any(k in n for k in ("fl", "chrome", "astia")):
            score += 1.0
            reasons.append("mild_character")

    return score, "+".join(reasons) if reasons else "pool_default"


def pick_from_pool(
    pool: list[str],
    scene: str,
    cues: dict[str, float],
) -> tuple[str, list[str], str]:
    """Deterministic pick inside an ordered pool. Returns (look, candidates, reason)."""
    cleaned = [x for x in pool if x in ALL_LOOKS]
    if not cleaned:
        cleaned = ["natural"] if "natural" in ALL_LOOKS else list(ALL_LOOKS.keys())[:1]
    # Keep at most 3 for compare sheets (skill: 2–3 alternates, not full scan)
    candidates = cleaned[:3]

    best_name = candidates[0]
    best_score = -1e9
    best_reason = "pool_default"
    for idx, name in enumerate(candidates):
        cue_score, reason = _score_candidate(name, scene, cues)
        # Earlier pool entries get slight prior (safe-first)
        score = cue_score + (len(candidates) - idx) * 0.15
        if score > best_score:
            best_score = score
            best_name = name
            best_reason = reason

    # Put winner first in candidates list for compare UX
    ordered = [best_name] + [c for c in candidates if c != best_name]
    return best_name, ordered, best_reason


def resolve_pool(
    brand: str,
    scene: str,
    prefs: dict[str, Any] | None = None,
    *,
    brand_override: bool = False,
) -> list[str]:
    prefs = prefs or {}
    if brand_override or not prefs.get("scene_pools"):
        pools = scene_pools_for_brand(brand)
    else:
        pools = prefs.get("scene_pools") or scene_pools_for_brand(brand)
    pool = list(pools.get(scene) or pools.get("general") or scene_pools_for_brand(brand)["general"])
    return pool


def suggest_look_detail(
    np_rgb: np.ndarray,
    details: dict[str, Any] | None = None,
    *,
    forced: str | None = None,
    auto: bool | None = None,
    brand: str | None = None,
    locked_look: str | None = None,
    sticky: bool | None = None,
) -> LookSuggestion:
    """Full suggestion with candidates + reason. Honors sticky lock when set."""
    prefs = load_prefs()
    brand_key = (brand or prefs.get("brand") or "sony").lower()
    if brand_key not in BRAND_SCENE_POOLS:
        brand_key = "sony"
    use_sticky = prefs.get("sticky_look", True) if sticky is None else sticky

    if forced and forced != "auto":
        return LookSuggestion(
            look=forced,
            scene_tag="forced",
            candidates=[forced],
            reason="forced",
            sticky=False,
            brand=brand_key,
        )

    use_auto = prefs.get("auto_look", True) if auto is None else auto
    if not use_auto:
        default = str(prefs.get("default_look") or BRAND_DEFAULT_LOOK.get(brand_key, "sony-st"))
        return LookSuggestion(
            look=default,
            scene_tag="prefs",
            candidates=[default],
            reason="prefs_default",
            sticky=False,
            brand=brand_key,
        )

    # One shoot, one grade: sticky lock wins after first auto pick
    if use_sticky and locked_look and locked_look in ALL_LOOKS:
        tag = classify_scene(np_rgb, details)
        pool = resolve_pool(brand_key, tag, prefs, brand_override=(brand is not None))
        return LookSuggestion(
            look=locked_look,
            scene_tag=tag,
            candidates=[locked_look] + [c for c in pool[:3] if c != locked_look][:2],
            reason="sticky_lock",
            sticky=True,
            brand=brand_key,
        )

    tag = classify_scene(np_rgb, details)
    cues = extract_cues(np_rgb, details)
    pool = resolve_pool(brand_key, tag, prefs, brand_override=(brand is not None))
    look, candidates, reason = pick_from_pool(pool, tag, cues)
    return LookSuggestion(
        look=look,
        scene_tag=tag,
        candidates=candidates,
        reason=reason,
        sticky=False,
        brand=brand_key,
    )


def suggest_look(
    np_rgb: np.ndarray,
    details: dict[str, Any] | None = None,
    *,
    forced: str | None = None,
    auto: bool | None = None,
    brand: str | None = None,
    locked_look: str | None = None,
    sticky: bool | None = None,
) -> tuple[str, str]:
    """Backward-compatible (look_name, scene_tag)."""
    s = suggest_look_detail(
        np_rgb,
        details,
        forced=forced,
        auto=auto,
        brand=brand,
        locked_look=locked_look,
        sticky=sticky,
    )
    return s.look, s.scene_tag


def build_look_compare_sheet(
    panels: list[tuple[str, np.ndarray]],
    dest,
    *,
    long_edge: int = 900,
) -> None:
    """Horizontal contact sheet of (label, rgb float) panels for agent Read."""
    from pathlib import Path

    from PIL import Image, ImageDraw, ImageFont

    dest = Path(dest)
    labeled: list[Image.Image] = []
    for text, rgb in panels:
        arr = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        im = Image.fromarray(arr, mode="RGB")
        w, h = im.size
        m = max(w, h)
        if m > long_edge:
            scale = long_edge / m
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (im.width, im.height + 36), (16, 16, 16))
        canvas.paste(im, (0, 36))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        draw.text((10, 8), text, fill=(240, 240, 240), font=font)
        labeled.append(canvas)

    if not labeled:
        return
    h = min(im.height for im in labeled)
    resized = []
    for im in labeled:
        if im.height != h:
            im = im.resize((int(im.width * h / im.height), h), Image.Resampling.LANCZOS)
        resized.append(im)
    gap = 12
    width = sum(im.width for im in resized) + gap * (len(resized) - 1)
    sheet = Image.new("RGB", (width, h), (8, 8, 8))
    x = 0
    for im in resized:
        sheet.paste(im, (x, 0))
        x += im.width + gap
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, format="JPEG", quality=88, optimize=True)
