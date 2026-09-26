#!/usr/bin/env python3
"""Focal-plane sharpness: max-patch / attention window vs full-field (skill-aligned).

S_plane drives blurry veto; S_field is informational (shallow_dof).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def edge95_from_gray(gray: np.ndarray) -> float:
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.size == 0:
        return 0.0
    return float(np.quantile(mag, 0.95))


def score_from_edge95(edge95: float) -> float:
    """Same mapping as legacy compute_sharpness: typical sharp ~0.15–0.35 → ~40–100."""
    return float(np.clip((edge95 / 0.28) * 80.0, 0.0, 100.0))


def _luma(np_rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * np_rgb[..., 0]
        + 0.7152 * np_rgb[..., 1]
        + 0.0722 * np_rgb[..., 2]
    )


def _clamp_box(
    cx: float, cy: float, half_w: float, half_h: float
) -> tuple[float, float, float, float]:
    x0 = max(0.0, cx - half_w)
    y0 = max(0.0, cy - half_h)
    x1 = min(1.0, cx + half_w)
    y1 = min(1.0, cy + half_h)
    if x1 - x0 < 0.08:
        x0, x1 = max(0.0, cx - 0.04), min(1.0, cx + 0.04)
    if y1 - y0 < 0.08:
        y0, y1 = max(0.0, cy - 0.04), min(1.0, cy + 0.04)
    return x0, y0, x1, y1


def attention_window(
    subject_center: list[float] | tuple[float, float] | None,
    *,
    preset: str = "general",
) -> tuple[float, float, float, float]:
    """Normalized (x0,y0,x1,y1) around attention / portrait eye band."""
    if subject_center and len(subject_center) >= 2:
        sx, sy = float(subject_center[0]), float(subject_center[1])
    else:
        sx, sy = 0.5, 0.5
    # Portrait: bias window upward toward eyes
    if preset == "portrait":
        sy = max(0.15, sy * 0.78)
        half = 0.16
    else:
        half = 0.15
    return _clamp_box(sx, sy, half, half)


def _box_iou(a: dict[str, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = float(a["x0"]), float(a["y0"]), float(a["x1"]), float(a["y1"])
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def compute_plane_field_sharpness(
    np_rgb: np.ndarray,
    *,
    subject_center: list[float] | tuple[float, float] | None = None,
    preset: str = "general",
    grid: int = 5,
    fnumber: float | None = None,
) -> dict[str, Any]:
    """Return S_plane / S_field and winning focus patch (normalized box).

    Prefer top patches that overlap the attention window (subject plane),
    not a random sharp corner.
    """
    h, w = np_rgb.shape[:2]
    gray = _luma(np_rgb)

    grid = max(3, int(grid))
    ph, pw = h // grid, w // grid
    scored: list[dict[str, float]] = []
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = gy * ph, (gy + 1) * ph if gy < grid - 1 else h
            x0, x1 = gx * pw, (gx + 1) * pw if gx < grid - 1 else w
            if y1 - y0 < 8 or x1 - x0 < 8:
                continue
            e = edge95_from_gray(gray[y0:y1, x0:x1])
            scored.append(
                {
                    "edge95": e,
                    "score": score_from_edge95(e),
                    "x0": x0 / w,
                    "y0": y0 / h,
                    "x1": x1 / w,
                    "y1": y1 / h,
                }
            )

    scored.sort(key=lambda p: p["score"], reverse=True)
    if scored:
        mid = scored[len(scored) // 2]["score"]
        sharp_field = round(float(mid), 1)
        field_edge = float(scored[len(scored) // 2]["edge95"])
    else:
        field_edge = edge95_from_gray(gray)
        sharp_field = round(score_from_edge95(field_edge), 1)

    ax0, ay0, ax1, ay1 = attention_window(subject_center, preset=preset)
    att_box = (ax0, ay0, ax1, ay1)
    ay0i, ay1i = int(ay0 * h), max(int(ay1 * h), int(ay0 * h) + 1)
    ax0i, ax1i = int(ax0 * w), max(int(ax1 * w), int(ax0 * w) + 1)
    att_edge = edge95_from_gray(gray[ay0i:ay1i, ax0i:ax1i])
    att_score = score_from_edge95(att_edge)

    # Prefer sharpest among top-K patches that overlap attention (IoU > 0.12)
    top_k = scored[: max(5, len(scored) // 3)] if scored else []
    overlapping = [p for p in top_k if _box_iou(p, att_box) >= 0.12]
    pick_mode = "global_max"
    if overlapping:
        best = max(overlapping, key=lambda p: p["score"])
        pick_mode = "attention_intersect"
        plane_edge = best["edge95"]
        plane_score = round(float(best["score"]), 1)
    elif scored:
        wide = fnumber is not None and float(fnumber) <= 2.8
        if wide and len(scored) >= 2:
            plane_edge = 0.5 * (scored[0]["edge95"] + scored[1]["edge95"])
            plane_score = round(0.5 * (scored[0]["score"] + scored[1]["score"]), 1)
            best = scored[0]
            pick_mode = "top2_wide"
        else:
            best = scored[0]
            plane_edge = best["edge95"]
            plane_score = round(best["score"], 1)
    else:
        best = {
            "x0": ax0,
            "y0": ay0,
            "x1": ax1,
            "y1": ay1,
            "edge95": att_edge,
            "score": att_score,
        }
        plane_edge = att_edge
        plane_score = round(att_score, 1)
        pick_mode = "attention_only"

    # Attention window itself if clearly sharper than intersect pick
    if att_score >= plane_score * 1.05:
        best = {
            "edge95": att_edge,
            "score": att_score,
            "x0": ax0,
            "y0": ay0,
            "x1": ax1,
            "y1": ay1,
        }
        plane_edge = att_edge
        plane_score = round(att_score, 1)
        pick_mode = "attention_window"

    if preset == "landscape" and scored:
        central = [
            p
            for p in scored
            if 0.2 <= (p["x0"] + p["x1"]) / 2 <= 0.8 and 0.2 <= (p["y0"] + p["y1"]) / 2 <= 0.8
        ]
        if central:
            c_scores = sorted((p["score"] for p in central), reverse=True)
            p90 = c_scores[max(0, int(len(c_scores) * 0.1))]
            plane_score = round(0.65 * plane_score + 0.35 * p90, 1)

    # People: focal plane = face. Underexposed faces: measure after luma normalize.
    face_meta: dict[str, Any] = {}
    try:
        from face_recover import measure_face_plane_sharpness

        face_meta = measure_face_plane_sharpness(
            np_rgb, subject_center=subject_center, preset=preset
        )
        if face_meta.get("use_face_plane"):
            f_norm = float(face_meta["face_sharp_norm"])
            f_as = float(face_meta["face_sharp_as_shot"])
            # Prefer normalized face score when dark-as-shot under-reads edges
            if face_meta.get("face_underexposed_as_shot") and f_norm >= f_as:
                plane_score = round(max(plane_score, f_norm), 1)
                plane_edge = float(face_meta.get("face_edge95_norm") or plane_edge)
                pick_mode = "face_plane_norm"
            elif f_norm >= plane_score * 0.92 or f_as >= plane_score * 0.95:
                plane_score = round(max(plane_score, f_norm, f_as), 1)
                pick_mode = "face_plane"
            box = face_meta.get("face_box") or {}
            if box and pick_mode.startswith("face_plane"):
                best = {
                    "x0": box["x0"],
                    "y0": box["y0"],
                    "x1": box["x1"],
                    "y1": box["y1"],
                    "edge95": plane_edge,
                    "score": plane_score,
                }
    except Exception:
        face_meta = {}

    sharp_plane = plane_score
    focus_patch = {
        "x0": round(float(best["x0"]), 4),
        "y0": round(float(best["y0"]), 4),
        "x1": round(float(best["x1"]), 4),
        "y1": round(float(best["y1"]), 4),
    }

    out = {
        "sharp_plane": sharp_plane,
        "sharp_field": sharp_field,
        "edge95_plane": round(float(plane_edge), 4),
        "edge95_field": round(field_edge, 4),
        "focus_patch": focus_patch,
        "attention_score": round(att_score, 1),
        "top_patch_score": round(float(scored[0]["score"]), 1) if scored else sharp_field,
        "focus_pick_mode": pick_mode,
    }
    for k, v in face_meta.items():
        if k != "face_box":
            out[k] = v
    return out


def blur_cut_adjusted(
    base_blur: float,
    base_soft: float,
    *,
    fnumber: float | None = None,
    exposure_ctx: dict[str, Any] | None = None,
    preset: str = "general",
) -> tuple[float, float]:
    """Backward-compatible wrapper; prefer blur_cuts_for_exposure via exposure_ctx."""
    from exposure_context import blur_cuts_for_exposure, build_exposure_context

    if exposure_ctx is not None:
        b, s, _ = blur_cuts_for_exposure(base_blur, base_soft, exposure_ctx, preset=preset)
        return b, s
    ctx = {"fnumber": fnumber, "iso_ratio": 1.0, "shake_risk": None, "preset": preset}
    b, s, _ = blur_cuts_for_exposure(base_blur, base_soft, ctx, preset=preset)
    return b, s


def should_flag_shallow_dof(
    sharp_plane: float,
    sharp_field: float,
    *,
    blur_cut: float,
    soft_cut: float,
    fnumber: float | None = None,
) -> bool:
    if sharp_plane < blur_cut:
        return False
    delta = sharp_plane - sharp_field
    if delta >= 18.0:
        return True
    if sharp_plane >= blur_cut and sharp_field < soft_cut and delta >= 10.0:
        return True
    if fnumber is not None and float(fnumber) <= 2.8 and delta >= 12.0:
        return True
    return False


def parse_fnumber(exif: dict[str, Any] | None) -> float | None:
    if not exif:
        return None
    for key in ("FNumber", "Aperture", "ApertureValue", "fnumber"):
        v = exif.get(key)
        if v is None:
            continue
        try:
            if isinstance(v, (list, tuple)) and v:
                return float(v[0])
            return float(v)
        except (TypeError, ValueError):
            continue
    return None
