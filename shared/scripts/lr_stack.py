#!/usr/bin/env python3
"""Lightroom-ordered develop stack.

Mirrors LR edit order (simplified, skill-aligned):

  1. Geometry          — straighten / crop (caller)
  2. Global Basic Auto — exposure, contrast, highlights, shadows, WB presence
  3. Look / color      — creative look via apply_grade (LUT, HSL, fade)
  4. Detail            — sharpen / NR (inside apply_grade)
  5. People refine     — face skin / eyes / teeth / clothes (subtle, relative)
  6. Selective masks   — subject / sky / background / radial / linear
  7. Effects           — vignette already in look; keep selective light

The old "force faces to mid-grey with large soft ovals" path is intentionally
replaced: global auto carries most of the backlight lift; people/sky masks
only do small coordinated deltas so skin, clothes, and scene stay consistent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from face_recover import _feather
from looks import complete_params
from raw_common import luma
from raw_develop import apply_grade


def _mid_luma(rgb: np.ndarray) -> float:
    y = luma(np.clip(rgb, 0, 1))
    return float(np.quantile(y, 0.50))


def _p_luma(rgb: np.ndarray, q: float) -> float:
    y = luma(np.clip(rgb, 0, 1))
    return float(np.quantile(y, q))


def auto_basic_params(
    rgb: np.ndarray,
    base: dict,
    *,
    flags: list[str] | None = None,
    face_mean: float | None = None,
) -> dict:
    """Stage 2 — LR Basic auto: modest whole-frame exposure/tone (not face-only).

    Backlit people: lift exposure/shadows enough for the *frame*, protect sky
    highlights. Face-specific work waits for People refine.
    """
    flags = flags or []
    out = dict(complete_params(base))
    mid = _mid_luma(rgb)
    hi = _p_luma(rgb, 0.95)
    lo = _p_luma(rgb, 0.10)
    under = (
        "underexposed_as_shot" in flags
        or "face_underexposed_as_shot" in flags
        or mid < 0.32
        or (face_mean is not None and face_mean < 0.32)
    )
    # Target mid-grey for the whole frame (LR Auto-ish)
    target_mid = 0.42 if under else 0.42
    if under and face_mean is not None and face_mean < 0.28:
        # Severe backlight: bias global mid up so People refine stays subtle
        target_mid = 0.46
    if mid > 1e-4:
        ev = float(np.log2(target_mid / mid))
        ev_max = 1.15 if (under and face_mean is not None and face_mean < 0.28) else (0.85 if under else 0.45)
        ev = float(np.clip(ev, -0.35, ev_max))
    else:
        ev = 0.0
    out["exposure"] = float(out.get("exposure", 0) or 0) + ev

    # Highlights: protect bright sky when p95 is high
    if hi > 0.85 or under:
        out["highlights"] = int(min(-34, float(out.get("highlights", 0) or 0) - 14))
    # Shadows: open underexposure without HDR mush
    if under or lo < 0.12:
        sh_add = 24 if (face_mean is not None and face_mean < 0.28) else 18
        out["shadows"] = int(min(48, float(out.get("shadows", 0) or 0) + sh_add))
        blacks = float(out.get("blacks", 0) or 0)
        if blacks < -4:
            out["blacks"] = int(min(-2, blacks + 8))
    # Open shade: slight warm (people under blue sky)
    if under:
        out["temperature"] = float(out.get("temperature", 0) or 0) + 6
        out["clarity"] = int(min(8, float(out.get("clarity", 0) or 0)))
    return out


def _apply_masked_ev(
    rgb: np.ndarray,
    mask: np.ndarray,
    ev: float,
    *,
    warm: float = 0.0,
) -> np.ndarray:
    """Relative EV on mask (LR-style local exposure), restrained gain."""
    if abs(ev) < 1e-4 and abs(warm) < 1e-4:
        return rgb
    m = np.clip(mask, 0, 1).astype(np.float32)
    if float(m.max()) < 1e-4:
        return rgb
    out = np.clip(rgb, 0, 1).astype(np.float32).copy()
    gain = float(np.clip(2.0 ** ev, 0.65, 1.75))
    g = 1.0 + (gain - 1.0) * m
    out = np.clip(out * g[..., None], 0, 1)
    if abs(warm) > 1e-4:
        y = luma(out)
        shade = np.clip((0.50 - y) / 0.45, 0, 1) * m
        w = warm * shade
        out[..., 0] = np.clip(out[..., 0] * (1.0 + 0.06 * w) + 0.008 * w, 0, 1)
        out[..., 1] = np.clip(out[..., 1] * (1.0 + 0.015 * w), 0, 1)
        out[..., 2] = np.clip(out[..., 2] * (1.0 - 0.04 * w), 0, 1)
    return out


def _apply_masked_mid_lift(
    rgb: np.ndarray,
    mask: np.ndarray,
    target: float,
    *,
    strength: float = 1.0,
) -> np.ndarray:
    """Soft residual midtone nudge — never a hard chase to portrait targets."""
    m = np.clip(mask, 0, 1).astype(np.float32) * float(np.clip(strength, 0, 1.5))
    if float(m.max()) < 1e-4:
        return rgb
    out = np.clip(rgb, 0, 1).astype(np.float32).copy()
    y = luma(out)
    room = np.clip((0.65 - y) / 0.65, 0, 1)
    delta = (float(target) - y) * room * m * 0.45
    delta = np.clip(delta, 0.0, 0.10)
    return np.clip(out + delta[..., None], 0, 1)


def correct_skin_tone(
    rgb: np.ndarray,
    skin_mask: np.ndarray,
    *,
    amount: float = 1.0,
    warmth: float = 0.22,
    magenta: float = 0.08,
    sat: float = 0.05,
) -> np.ndarray:
    """Gentle open-shade skin nudge — stay in ambient light, avoid orange faces."""
    amount = float(np.clip(amount, 0.0, 1.5))
    m = np.clip(skin_mask, 0, 1).astype(np.float32) * amount
    if float(m.max()) < 1e-4:
        return rgb
    out = np.clip(rgb, 0, 1).astype(np.float32).copy()
    y = luma(out)
    mid = np.clip(1.0 - np.abs(y - 0.42) / 0.38, 0, 1) * m

    w = warmth * mid
    t = magenta * mid
    out[..., 0] = np.clip(out[..., 0] * (1.0 + 0.07 * w + 0.02 * t) + 0.006 * w, 0, 1)
    out[..., 1] = np.clip(out[..., 1] * (1.0 + 0.01 * w - 0.03 * t), 0, 1)
    out[..., 2] = np.clip(out[..., 2] * (1.0 - 0.05 * w - 0.01 * t), 0, 1)

    if sat > 0:
        gray = luma(out)[..., None]
        already = np.clip((out.max(axis=2, keepdims=True) - out.min(axis=2, keepdims=True)) * 2.0, 0, 1)
        vib = sat * mid[..., None] * (1.0 - already)
        out = np.clip(gray + (out - gray) * (1.0 + vib), 0, 1)
    return out


def _sky_mask(rgb: np.ndarray) -> np.ndarray:
    """Rough sky: bright + blue-ish, biased to upper frame."""
    h, w = rgb.shape[:2]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = luma(rgb)
    blue = (b > r * 1.05) & (b > g * 0.98) & (y > 0.35)
    bright = y > 0.72
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    upper = np.clip(1.2 - yy * 1.6, 0, 1)
    m = ((blue | bright).astype(np.float32) * upper).astype(np.float32)
    return _feather(m, radius=max(6, int(min(h, w) * 0.01)))


def _linear_top_mask(h: int, w: int, start: float = 0.0, end: float = 0.45) -> np.ndarray:
    """LR linear gradient from top (sky strip)."""
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    if end <= start:
        return np.zeros((h, w), dtype=np.float32)
    m = np.clip(1.0 - (yy - start) / max(end - start, 1e-3), 0, 1)
    return np.broadcast_to(m, (h, w)).copy()


def _radial_mask(
    h: int, w: int, cx: float, cy: float, rx: float, ry: float, feather: float = 0.45
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((xx / max(w - 1, 1) - cx) / max(rx, 1e-3)) ** 2 + ((yy / max(h - 1, 1) - cy) / max(ry, 1e-3)) ** 2)
    return np.clip(1.0 - (dist - (1.0 - feather)) / max(feather, 1e-3), 0, 1).astype(np.float32)


def people_masks(rgb: np.ndarray) -> dict[str, Any]:
    """Build LR People-like masks from organic face + person contours (not rectangles)."""
    try:
        from person_segment import build_people_contours

        return build_people_contours(rgb)
    except Exception:
        # Minimal empty fallback
        h, w = rgb.shape[:2]
        empty = np.zeros((h, w), dtype=np.float32)
        return {
            "faces": [],
            "face_skin": empty,
            "eyes": empty,
            "teeth": empty,
            "clothes": empty,
            "body_skin": empty,
            "subject": empty,
        }


def refine_people(
    rgb: np.ndarray,
    *,
    amount: float = 1.0,
    as_shot_face_mean: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stage 5 — LR People: match subject to scene, not studio skin targets.

    Face/body/clothes share one coordinated lift anchored to non-sky midtones so
    people stay in ambient contrast/color (open shade stays cool, no orange faces).
    """
    amount = float(np.clip(amount, 0.0, 1.5))
    masks = people_masks(rgb)
    face_m = masks.get("face_skin")
    has_skin = face_m is not None and float(np.max(face_m)) > 0.15
    meta: dict[str, Any] = {
        "faces": masks.get("faces") or [],
        "backend": masks.get("backend"),
        "mode": "people_refine" if has_skin and amount > 0 else "skip",
    }
    if not has_skin or amount <= 0:
        return rgb, meta

    y = luma(rgb)
    subject = masks.get("subject")
    sky = _sky_mask(rgb)
    if subject is not None:
        sky = np.clip(sky * (1.0 - subject * 0.85), 0, 1)
    # Ambient anchor: midtones of non-sky, non-subject (mountains / ground)
    env = (sky < 0.25).astype(np.float32)
    if subject is not None:
        env = env * (1.0 - np.clip(subject, 0, 1) * 0.9)
    if float(env.sum()) > 200:
        scene_ref = float(np.quantile(y[env > 0.5], 0.55)) if (env > 0.5).any() else float(np.median(y))
    else:
        scene_ref = float(np.quantile(y, 0.45))
    # Dark mountains shouldn't crush people; bright snow shouldn't pull them up to sky
    scene_ref = float(np.clip(scene_ref, 0.26, 0.45))
    meta["scene_ref"] = round(scene_ref, 3)

    face_now = float(y[face_m > 0.30].mean()) if (face_m > 0.30).any() else float(y.mean())
    body_m = masks.get("body_skin")
    body_now = (
        float(y[body_m > 0.30].mean())
        if body_m is not None and (body_m > 0.30).any()
        else face_now
    )
    clothes = masks.get("clothes")
    clothes_now = (
        float(y[clothes > 0.30].mean())
        if clothes is not None and (clothes > 0.30).any()
        else face_now
    )
    meta["face_luma_in"] = round(face_now, 3)
    meta["body_luma_in"] = round(body_now, 3)

    # Face slightly above scene mid; open shade stays below sky — readable, not flash-lit
    want_face = float(np.clip(scene_ref + 0.04, 0.32, 0.42))
    if as_shot_face_mean is not None and as_shot_face_mean < 0.22:
        want_face = float(np.clip(want_face + 0.02, 0.32, 0.44))
    want_body = want_face * 0.97
    want_clothes = float(np.clip(scene_ref + 0.02, 0.30, want_face))

    face_ev = float(np.clip(np.log2(want_face / max(face_now, 0.06)), 0.0, 0.65)) * amount
    body_ev = float(np.clip(np.log2(want_body / max(body_now, 0.06)), 0.0, 0.58)) * amount
    clothes_ev = float(np.clip(np.log2(want_clothes / max(clothes_now, 0.06)), 0.0, 0.48)) * amount
    # Keep body/clothes within ~0.1 EV of face so no spotlight head
    body_ev = float(np.clip(body_ev, face_ev - 0.08, face_ev + 0.02))
    clothes_ev = float(np.clip(clothes_ev, face_ev - 0.12, face_ev * 0.85))
    eye_ev = min(0.10, face_ev * 0.18) * amount
    teeth_ev = 0.06 * amount
    meta["face_ev"] = round(face_ev, 3)
    meta["body_ev"] = round(body_ev, 3)
    meta["clothes_ev"] = round(clothes_ev, 3)
    meta["want_face"] = round(want_face, 3)

    out = rgb
    # Broad subject first (hair/clothes/body), then face — shared lighting family
    hair = masks.get("hair")
    if hair is not None and float(hair.max()) > 0.1:
        out = _apply_masked_ev(out, hair, min(clothes_ev, face_ev * 0.55), warm=0.0)
    if clothes is not None:
        out = _apply_masked_ev(out, clothes, clothes_ev, warm=0.0)
    if body_m is not None:
        out = _apply_masked_ev(out, body_m, body_ev, warm=0.03 * amount)
    out = _apply_masked_ev(out, face_m, face_ev, warm=0.04 * amount)
    eyes = masks.get("eyes")
    teeth = masks.get("teeth")
    if eyes is not None:
        out = _apply_masked_ev(out, eyes, eye_ev, warm=0.0)
    if teeth is not None:
        out = _apply_masked_ev(out, teeth, teeth_ev, warm=-0.02)

    skin_union = np.clip(face_m + (body_m if body_m is not None else 0), 0, 1)
    # Tiny residual only if still clearly below scene-relative target
    y_mid = luma(out)
    face_mid = float(y_mid[face_m > 0.30].mean()) if (face_m > 0.30).any() else face_now
    if face_mid < want_face - 0.04:
        out = _apply_masked_mid_lift(out, face_m, want_face, strength=0.55 * amount)
        if body_m is not None:
            out = _apply_masked_mid_lift(out, body_m, want_body, strength=0.45 * amount)

    # Cool mountain ambient: light skin nudge, never heavy warm/magenta
    out = correct_skin_tone(
        out,
        skin_union,
        amount=amount * 0.7,
        warmth=0.20 if (as_shot_face_mean is not None and as_shot_face_mean < 0.30) else 0.12,
        magenta=0.06,
        sat=0.04,
    )

    y2 = luma(out)
    meta["face_luma_out"] = round(
        float(y2[face_m > 0.30].mean()) if (face_m > 0.30).any() else float(y2.mean()),
        3,
    )
    if body_m is not None and (body_m > 0.30).any():
        meta["body_luma_out"] = round(float(y2[body_m > 0.30].mean()), 3)
    return out, meta


def selective_masks(
    rgb: np.ndarray,
    people: dict[str, Any] | None = None,
    *,
    amount: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stage 6 — Subject / Sky / Background / Linear (subtle, scene-matched)."""
    amount = float(np.clip(amount, 0.0, 1.5))
    people = people or people_masks(rgb)
    subject = people.get("subject")
    if subject is None or float(np.max(subject)) < 1e-4:
        h, w = rgb.shape[:2]
        subject = np.zeros((h, w), dtype=np.float32)
    sky = _sky_mask(rgb)
    sky = np.clip(sky * (1.0 - subject * 0.85), 0, 1)
    h, w = rgb.shape[:2]
    linear = _linear_top_mask(h, w, 0.0, 0.42) * (1.0 - subject * 0.7)

    meta = {
        "sky_mean": round(float(sky.mean()), 4),
        "subject_mean": round(float(subject.mean()), 4),
    }
    out = rgb
    out = _apply_masked_ev(out, sky * amount, -0.10 * amount)
    out = _apply_masked_ev(out, linear * amount, -0.06 * amount)
    y = luma(out)
    if float(subject.max()) > 0 and (subject > 0.25).any():
        sub_y = float(y[subject > 0.25].mean())
        sky_y = float(y[sky > 0.3].mean()) if (sky > 0.3).any() else sub_y
        # Only a light subject fill when silhouette is crushed vs sky — no second skin bake
        if sub_y < sky_y - 0.28:
            fill = min(0.22, (sky_y - sub_y) * 0.22) * amount
            out = _apply_masked_ev(out, subject, fill, warm=0.02)
    meta["applied"] = True
    return out, meta


def develop_lr_stack(
    rgb: np.ndarray,
    look_params: dict,
    *,
    flags: list[str] | None = None,
    face_mean: float | None = None,
    people_amount: float = 1.0,
    selective_amount: float = 1.0,
    do_people: bool = True,
    do_selective: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Full LR-ordered stack: Auto Basic → Look → People → Selective.

    Geometry (straighten) remains the caller's responsibility (before or after).
    """
    flags = flags or []
    report: dict[str, Any] = {"stages": []}

    # 2) Global auto on raw demosaic (before creative look)
    basic = auto_basic_params(rgb, look_params, flags=flags, face_mean=face_mean)
    report["auto_basic"] = {
        "exposure": basic.get("exposure"),
        "shadows": basic.get("shadows"),
        "highlights": basic.get("highlights"),
        "temperature": basic.get("temperature"),
    }
    report["stages"].append("auto_basic")

    # 3–4) Look + detail (existing engine)
    graded = apply_grade(rgb, basic)
    report["stages"].append("look_detail")

    people_meta: dict[str, Any] = {}
    if do_people and (
        people_amount > 0
        and (
            "face_underexposed_as_shot" in flags
            or "face_hot_as_shot" in flags
            or "underexposed_as_shot" in flags
            or people_amount >= 1.0
        )
    ):
        graded, people_meta = refine_people(
            graded, amount=people_amount, as_shot_face_mean=face_mean
        )
        report["people"] = people_meta
        report["stages"].append("people_refine")

    if do_selective and selective_amount > 0:
        # Rebuild masks on current image for sky/subject
        pmasks = people_masks(graded) if do_people else people_masks(graded)
        graded, sel_meta = selective_masks(graded, pmasks, amount=selective_amount)
        report["selective"] = sel_meta
        report["stages"].append("selective")

    report["mid_luma"] = round(_mid_luma(graded), 3)
    return graded, report
