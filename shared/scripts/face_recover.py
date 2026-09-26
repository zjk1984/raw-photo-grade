#!/usr/bin/env python3
"""Face / subject-plane highlight recoverability (skill: as-shot hot ≠ unrecoverable).

Separates preview-bright faces from true dead sensor clip in the attention/skin window.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from focus_sharpness import attention_window, edge95_from_gray, score_from_edge95


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Bright / normal skin
    bright = (r > g) & (g > b * 0.85) & ((r - b) > 0.05) & ((r - g) < 0.28)
    # Underexposed skin: ratios still hold at low luma
    dark = (r > g * 0.95) & (g >= b * 0.9) & (r > 0.04) & (r < 0.45) & ((r - b) > 0.02)
    # Cool-grade / shade skin (R may drop toward G after FL/teal looks)
    cool = (
        (r > b * 0.92)
        & (g > b * 0.90)
        & ((r - b) > 0.02)
        & (np.abs(r - g) < 0.16)
        & (r > 0.06)
        & (r < 0.70)
        & ((0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.62)
    )
    return bright | dark | cool


def skin_anchor(
    rgb: np.ndarray,
    *,
    min_frac: float = 0.012,
) -> dict[str, Any] | None:
    """Centroid + tight bbox of skin pixels (norm coords). None if too little skin."""
    skin = _skin_mask(rgb)
    frac = float(skin.mean())
    if frac < min_frac:
        return None
    ys, xs = np.where(skin)
    h, w = rgb.shape[:2]
    cx = float(xs.mean() / max(w - 1, 1))
    cy = float(ys.mean() / max(h - 1, 1))
    # Prefer upper-quantile of skin blob (faces sit above torso skin)
    y_face = float(np.quantile(ys, 0.25) / max(h - 1, 1))
    x_face = float(xs[ys <= np.quantile(ys, 0.45)].mean() / max(w - 1, 1)) if ys.size else cx
    pad_x = max(0.10, 0.55 * (xs.max() - xs.min()) / max(w, 1))
    pad_y = max(0.11, 0.45 * (ys.max() - ys.min()) / max(h, 1))
    pad_x = float(min(pad_x, 0.22))
    pad_y = float(min(pad_y, 0.24))
    return {
        "center": [round(x_face, 4), round(y_face, 4)],
        "skin_centroid": [round(cx, 4), round(cy, 4)],
        "skin_frac": round(frac, 4),
        "face_box": {
            "x0": round(max(0.0, x_face - pad_x), 4),
            "y0": round(max(0.0, y_face - pad_y), 4),
            "x1": round(min(1.0, x_face + pad_x), 4),
            "y1": round(min(1.0, y_face + pad_y * 1.15), 4),
        },
    }


def resolve_face_center(
    rgb: np.ndarray,
    subject_center: list[float] | tuple[float, float] | None = None,
    *,
    face_box_hint: dict[str, float] | None = None,
) -> tuple[list[float], dict[str, float] | None, str]:
    """Pick a person/face anchor. Prefer detector faces > skin > saliency."""
    try:
        from face_detect import detect_faces

        faces = detect_faces(rgb)
        if faces:
            # Primary = largest face; if 2+, use midpoint of top faces for window
            if len(faces) >= 2:
                cx = float(np.mean([f["center"][0] for f in faces[:3]]))
                cy = float(np.mean([f["center"][1] for f in faces[:3]]))
                box = {
                    "x0": min(f["x0"] for f in faces),
                    "y0": min(f["y0"] for f in faces),
                    "x1": max(f["x1"] for f in faces),
                    "y1": max(f["y1"] for f in faces),
                }
                return [cx, cy], box, "yunet_multi"
            f0 = faces[0]
            return list(f0["center"]), {
                "x0": f0["x0"], "y0": f0["y0"], "x1": f0["x1"], "y1": f0["y1"]
            }, "yunet"
    except Exception:
        pass

    anchor = skin_anchor(rgb)
    sx, sy = 0.5, 0.5
    if subject_center and len(subject_center) >= 2:
        sx, sy = float(subject_center[0]), float(subject_center[1])

    if anchor is not None:
        ax, ay = anchor["center"]
        dist = abs(ax - sx) + abs(ay - sy)
        h, w = rgb.shape[:2]
        ix, iy = int(np.clip(sx, 0, 1) * (w - 1)), int(np.clip(sy, 0, 1) * (h - 1))
        on_skin = bool(_skin_mask(rgb[iy : iy + 1, ix : ix + 1])[0, 0])
        if (not on_skin and dist > 0.12) or ay > sy + 0.08:
            return list(anchor["center"]), dict(anchor["face_box"]), "skin_anchor"
        if dist > 0.06:
            cx = 0.35 * sx + 0.65 * ax
            cy = 0.35 * sy + 0.65 * ay
            return [cx, cy], dict(anchor["face_box"]), "skin_blend"

    if face_box_hint and all(k in face_box_hint for k in ("x0", "y0", "x1", "y1")):
        cx = 0.5 * (float(face_box_hint["x0"]) + float(face_box_hint["x1"]))
        cy = 0.5 * (float(face_box_hint["y0"]) + float(face_box_hint["y1"]))
        return [cx, cy], dict(face_box_hint), "hint_box"

    return [sx, sy], None, "subject_center"


def face_box_norm(
    subject_center: list[float] | tuple[float, float] | None = None,
    *,
    preset: str = "general",
) -> dict[str, float]:
    ax0, ay0, ax1, ay1 = attention_window(subject_center, preset=preset)
    # Portrait: bias upward toward eyes; enlarge slightly for full face
    cx, cy = 0.5 * (ax0 + ax1), 0.5 * (ay0 + ay1)
    if preset == "portrait":
        cy = max(0.18, cy * 0.92)
        half_w, half_h = 0.16, 0.18
    else:
        half_w = max(0.14, 0.55 * (ax1 - ax0))
        half_h = half_w * 1.05
    x0 = max(0.0, cx - half_w)
    x1 = min(1.0, cx + half_w)
    y0 = max(0.0, cy - half_h)
    y1 = min(1.0, cy + half_h)
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def measure_face_plane_sharpness(
    np_rgb: np.ndarray,
    *,
    subject_center: list[float] | tuple[float, float] | None = None,
    preset: str = "general",
) -> dict[str, Any]:
    """Sharpness on the face plane, with luma normalize for underexposed faces.

    Skill: people focal plane = face. Dark faces look soft as-shot but edges
    remain after a midtone lift — measure after normalize, flag underexposure.
    """
    h, w = np_rgb.shape[:2]
    center, box_override, mode = resolve_face_center(np_rgb, subject_center)
    box = box_override or face_box_norm(center, preset=preset)
    yi0, yi1 = int(box["y0"] * h), max(int(box["y1"] * h), int(box["y0"] * h) + 1)
    xi0, xi1 = int(box["x0"] * w), max(int(box["x1"] * w), int(box["x0"] * w) + 1)
    crop = np.clip(np_rgb[yi0:yi1, xi0:xi1], 0, 1)
    if crop.size < 64:
        return {
            "face_sharp_as_shot": 0.0,
            "face_sharp_norm": 0.0,
            "face_edge95_norm": 0.0,
            "face_mean_luma": 0.0,
            "face_underexposed_as_shot": False,
            "face_skin_frac": 0.0,
            "face_box": box,
            "face_anchor_mode": mode,
            "use_face_plane": False,
        }

    gray = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    skin = _skin_mask(crop)
    skin_frac = float(skin.mean())
    if skin_frac >= 0.03:
        # Dilate skin a bit via bounding rows/cols for edge context
        ys, xs = np.where(skin)
        pad = 2
        y0s, y1s = max(0, int(ys.min()) - pad), min(gray.shape[0], int(ys.max()) + pad + 1)
        x0s, x1s = max(0, int(xs.min()) - pad), min(gray.shape[1], int(xs.max()) + pad + 1)
        g_as = gray[y0s:y1s, x0s:x1s]
    else:
        g_as = gray

    mean_l = float(g_as.mean()) if g_as.size else float(gray.mean())
    e_as = edge95_from_gray(g_as)
    sharp_as = score_from_edge95(e_as)

    # Normalize dark faces toward mid-grey so underexposure ≠ soft focus
    under = mean_l < 0.32
    if under and mean_l > 1e-4:
        scale = 0.48 / mean_l
        g_n = np.clip(g_as * scale, 0, 1)
    else:
        g_n = g_as
    e_n = edge95_from_gray(g_n)
    sharp_n = score_from_edge95(e_n)

    # Prefer face plane when skin present, portrait, or clearly dark subject window
    use_face = (
        preset == "portrait"
        or skin_frac >= 0.04
        or (under and sharp_n >= sharp_as * 1.15)
        or (mean_l < 0.40 and skin_frac >= 0.02)
    )

    return {
        "face_sharp_as_shot": round(float(sharp_as), 1),
        "face_sharp_norm": round(float(sharp_n), 1),
        "face_edge95_norm": round(float(e_n), 4),
        "face_mean_luma": round(mean_l, 3),
        "face_underexposed_as_shot": bool(under),
        "face_skin_frac": round(skin_frac, 3),
        "face_box": {
            "x0": round(box["x0"], 4),
            "y0": round(box["y0"], 4),
            "x1": round(box["x1"], 4),
            "y1": round(box["y1"], 4),
        },
        "face_anchor_mode": mode,
        "face_center": [round(center[0], 4), round(center[1], 4)],
        "use_face_plane": bool(use_face),
    }


def analyze_face_region(
    np_rgb: np.ndarray,
    *,
    subject_center: list[float] | tuple[float, float] | None = None,
    preset: str = "general",
    face_box_hint: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score face/attention-window exposure recoverability + plane sharpness hooks."""
    h, w = np_rgb.shape[:2]
    center, box_override, mode = resolve_face_center(
        np_rgb, subject_center, face_box_hint=face_box_hint
    )
    box = box_override or face_box_norm(center, preset=preset)
    x0, y0, x1, y1 = box["x0"], box["y0"], box["x1"], box["y1"]
    yi0, yi1 = int(y0 * h), max(int(y1 * h), int(y0 * h) + 1)
    xi0, xi1 = int(x0 * w), max(int(x1 * w), int(x0 * w) + 1)
    crop = np.clip(np_rgb[yi0:yi1, xi0:xi1], 0, 1)
    plane = measure_face_plane_sharpness(
        np_rgb, subject_center=center, preset=preset
    )
    if crop.size < 64:
        return {
            "face_mean_luma": 0.0,
            "face_hi_pct": 0.0,
            "face_all_clip_pct": 0.0,
            "face_p99": 0.0,
            "face_headroom": 1.0,
            "face_skin_frac": 0.0,
            "face_recoverable": True,
            "face_dead": False,
            "face_hot_as_shot": False,
            "face_underexposed_as_shot": False,
            "face_box": box,
            "face_anchor_mode": mode,
            "face_center": center,
            **{k: v for k, v in plane.items() if k not in {"face_box", "face_anchor_mode", "face_center"}},
        }

    luma = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    skin = _skin_mask(crop)
    skin_frac = float(skin.mean())
    # Prefer skin pixels when enough; else full attention crop
    if skin_frac >= 0.04:
        region = crop[skin]
        region_y = luma[skin]
    else:
        region = crop.reshape(-1, 3)
        region_y = luma.reshape(-1)

    mean_l = float(region_y.mean()) if region_y.size else 0.0
    p99 = float(np.quantile(region_y, 0.99)) if region_y.size else 0.0
    hi_pct = float((region_y > 0.95).mean() * 100.0) if region_y.size else 0.0
    all_clip = (
        (region[..., 0] > 0.985) & (region[..., 1] > 0.985) & (region[..., 2] > 0.985)
    )
    all_clip_pct = float(all_clip.mean() * 100.0) if all_clip.size else 0.0
    any_clip = (region.max(axis=-1) > 0.98).mean() * 100.0 if region.size else 0.0
    headroom = float(max(0.0, 1.0 - p99))
    # Shadow floor: dark but not crushed (code values remain)
    p10 = float(np.quantile(region_y, 0.10)) if region_y.size else 0.0
    face_under = mean_l < 0.32 and p10 > 0.01
    face_crushed = mean_l < 0.12 and p10 < 0.008

    face_dead = all_clip_pct >= 8.0 and headroom < 0.02
    face_recoverable = (not face_dead) and (
        headroom >= 0.015 or all_clip_pct < 3.0 or (any_clip > all_clip_pct * 1.5) or face_under
    )
    face_hot = (mean_l >= 0.62 or hi_pct >= 12.0) and face_recoverable and not face_dead

    out = {
        "face_mean_luma": round(mean_l, 3),
        "face_hi_pct": round(hi_pct, 2),
        "face_all_clip_pct": round(all_clip_pct, 2),
        "face_any_clip_pct": round(float(any_clip), 2),
        "face_p99": round(p99, 3),
        "face_p10": round(p10, 3),
        "face_headroom": round(headroom, 3),
        "face_skin_frac": round(max(skin_frac, plane.get("face_skin_frac", 0)), 3),
        "face_recoverable": bool(face_recoverable and not face_crushed),
        "face_dead": bool(face_dead),
        "face_hot_as_shot": bool(face_hot),
        "face_underexposed_as_shot": bool(face_under and not face_crushed),
        "face_crushed_shadows": bool(face_crushed),
        "face_box": {
            "x0": round(x0, 4),
            "y0": round(y0, 4),
            "x1": round(x1, 4),
            "y1": round(y1, 4),
        },
        "face_anchor_mode": mode,
        "face_center": [round(center[0], 4), round(center[1], 4)],
    }
    for k in (
        "face_sharp_as_shot",
        "face_sharp_norm",
        "face_edge95_norm",
        "use_face_plane",
    ):
        if k in plane:
            out[k] = plane[k]
    return out


def face_flags_from_analysis(face: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if face.get("face_dead"):
        flags.append("face_dead_highlights")
    elif face.get("face_hot_as_shot"):
        flags.append("face_hot_as_shot")
    if face.get("face_underexposed_as_shot"):
        flags.append("face_underexposed_as_shot")
    if face.get("face_crushed_shadows"):
        flags.append("face_crushed_shadows")
    if face.get("face_recoverable") and (
        face.get("face_hot_as_shot")
        or face.get("face_underexposed_as_shot")
        or float(face.get("face_hi_pct") or 0) >= 8.0
    ):
        flags.append("face_recoverable")
    return flags


def should_face_finish(flags: list[str] | None) -> bool:
    """Auto face-finish for recoverable hot *or* underexposed faces."""
    flags = flags or []
    if "face_recoverable" not in flags and "face_underexposed_as_shot" not in flags:
        return False
    return (
        "face_hot_as_shot" in flags
        or "face_underexposed_as_shot" in flags
    )


def face_finish_exposure_for_mean(
    mean_luma: float,
    *,
    target: float = 0.48,
    hot: bool = False,
) -> float:
    """EV lift so face midtones approach target; hot faces get near-zero lift."""
    mean_luma = float(max(mean_luma, 1e-4))
    if hot and mean_luma >= 0.55:
        return 0.04
    if mean_luma >= target - 0.02:
        return 0.12
    ev = float(np.log2(target / mean_luma))
    return float(np.clip(ev, 0.15, 1.60))


def _feather(mask: np.ndarray, radius: int = 8) -> np.ndarray:
    """Box-blur feather for soft person masks."""
    if radius <= 0:
        return mask
    r = int(radius)
    pad = np.pad(mask, r, mode="edge")
    # Separable cumulative sum blur
    c = np.cumsum(pad, axis=0)
    v = c[2 * r :, :] - c[: -2 * r, :]
    c = np.cumsum(v, axis=1)
    h = c[:, 2 * r :] - c[:, : -2 * r]
    area = float((2 * r) * (2 * r))
    return (h / area).astype(np.float32)


def _region_luma(rgb: np.ndarray, box: dict[str, float], skin_only: bool = True) -> float:
    h, w = rgb.shape[:2]
    y0, y1 = int(box["y0"] * h), max(int(box["y1"] * h), int(box["y0"] * h) + 1)
    x0, x1 = int(box["x0"] * w), max(int(box["x1"] * w), int(box["x0"] * w) + 1)
    crop = np.clip(rgb[y0:y1, x0:x1], 0, 1)
    y = 0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]
    if skin_only:
        sk = _skin_mask(crop)
        if float(sk.mean()) >= 0.02:
            return float(y[sk].mean())
    return float(y.mean())


def _soft_rect_mask(h: int, w: int, box: dict[str, float], feather: float = 0.12) -> np.ndarray:
    """Axis-aligned soft rectangle (better than dual spotlights for multi-person)."""
    y0, y1 = box["y0"] * (h - 1), box["y1"] * (h - 1)
    x0, x1 = box["x0"] * (w - 1), box["x1"] * (w - 1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Signed distance outside the rect (0 inside)
    dx = np.maximum(np.maximum(x0 - xx, xx - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - yy, yy - y1), 0.0)
    dist = np.sqrt(dx * dx + dy * dy)
    fw = max(feather * max(x1 - x0, y1 - y0, 1.0), 2.0)
    return np.clip(1.0 - dist / fw, 0.0, 1.0).astype(np.float32)


def _ellipse_mask(h: int, w: int, box: dict[str, float], soft: float = 0.35) -> np.ndarray:
    """Soft ellipse inside a normalized box."""
    y0, y1 = box["y0"] * h, box["y1"] * h
    x0, x1 = box["x0"] * w, box["x1"] * w
    cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
    ry = max(0.5 * (y1 - y0), 1.0)
    rx = max(0.5 * (x1 - x0), 1.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    return np.clip(1.0 - (dist - (1.0 - soft)) / max(soft, 1e-3), 0.0, 1.0).astype(np.float32)


def apply_people_finish(
    rgb: np.ndarray,
    *,
    amount: float = 1.0,
    target_luma: float = 0.52,
    as_shot_mean: float | None = None,
    warm_shadows: float = 0.12,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Multi-face + body coordinated lift for backlit people.

    Detects every face, builds torso regions, then applies a *shared* target luma
    so adult/child skin and nearby body stay consistent. Soft mask + mild warm
    in lifted shadows keeps the grade looking like one exposure, not patches.
    """
    amount = float(np.clip(amount, 0.0, 1.5))
    meta: dict[str, Any] = {"faces": [], "target_luma": target_luma, "mode": "none"}
    if amount <= 0:
        return rgb, meta

    try:
        from face_detect import people_regions
    except ImportError:
        return rgb, meta

    regions = people_regions(rgb)
    faces = regions.get("faces") or []
    bodies = regions.get("bodies") or []
    meta["faces"] = faces
    if not faces:
        return rgb, meta

    out = np.clip(rgb, 0, 1).astype(np.float32).copy()
    h, w = out.shape[:2]

    # Per-face skin luma → shared target (pull everyone toward same midtone)
    face_lumas = [_region_luma(out, f, skin_only=True) for f in faces]
    body_lumas = [_region_luma(out, b, skin_only=False) for b in bodies]
    meta["face_lumas"] = [round(v, 3) for v in face_lumas]
    meta["body_lumas"] = [round(v, 3) for v in body_lumas]

    under = (
        (as_shot_mean is not None and float(as_shot_mean) < 0.35)
        or (min(face_lumas) < 0.36 if face_lumas else False)
    )
    # Target: at least mid skin; if one face already brighter, match toward it
    bright = max(face_lumas) if face_lumas else target_luma
    target = float(target_luma)
    if under:
        target = max(target_luma, min(0.58, bright + 0.08 if bright < 0.45 else bright))
        target = max(target, 0.50)
    meta["target_luma"] = round(target, 3)
    meta["mode"] = "multi_face"

    # Soft union: one people slab (rects) so face+torso stay connected
    mask = np.zeros((h, w), dtype=np.float32)
    for face, body, fl in zip(faces, bodies, face_lumas):
        boost = 1.0 + max(0.0, (target - fl) * 0.8)
        mask = np.maximum(mask, _soft_rect_mask(h, w, face, feather=0.35) * 0.95 * boost)
        mask = np.maximum(mask, _soft_rect_mask(h, w, body, feather=0.40) * 0.92 * boost)

    if regions.get("union"):
        u = dict(regions["union"])
        pad_x, pad_y = 0.05, 0.08
        u = {
            "x0": max(0.0, u["x0"] - pad_x),
            "y0": max(0.0, u["y0"] - pad_y),
            "x1": min(1.0, u["x1"] + pad_x),
            "y1": min(1.0, u["y1"] + pad_y * 1.6),
        }
        mask = np.maximum(mask, _soft_rect_mask(h, w, u, feather=0.45) * 0.75)
        y0, y1 = int(u["y0"] * h), int(u["y1"] * h)
        x0, x1 = int(u["x0"] * w), int(u["x1"] * w)
        crop = out[y0:y1, x0:x1]
        sk = _skin_mask(crop).astype(np.float32)
        if float(sk.mean()) >= 0.01:
            pad = np.pad(sk, 5, mode="edge")
            dil = np.zeros_like(sk)
            ch, cw = sk.shape
            for dy in range(11):
                for dx in range(11):
                    dil = np.maximum(dil, pad[dy : dy + ch, dx : dx + cw])
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], dil * 0.95)

    mask = _feather(np.clip(mask, 0, 1), radius=max(8, int(min(h, w) * 0.014)))
    mask = np.clip(mask * amount, 0, 1)

    # Local adaptive gain toward shared target (face+body consistent)
    luma = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    local = _feather(luma, radius=max(5, int(min(h, w) * 0.01)))
    # Gate: do not brighten sky/clouds that intersect face ellipses (kills hat halos)
    shadow_gate = np.clip((0.58 - local) / 0.38, 0.0, 1.0)
    mask = mask * (0.15 + 0.85 * shadow_gate)
    mask = _feather(mask, radius=max(4, int(min(h, w) * 0.008)))

    gain = target / np.maximum(local, 0.04)
    gain = np.clip(gain, 1.0, 3.0 if under else 2.2)
    gain_m = 1.0 + (gain - 1.0) * mask
    lifted = np.clip(out * gain_m[..., None], 0, 1)

    # Mild warm in lifted open-shade shadows (blue cast from sky)
    if warm_shadows > 0 and under:
        cool = np.clip((0.45 - luma) / 0.45, 0, 1) * mask
        warm = warm_shadows * cool
        lifted = lifted.copy()
        lifted[..., 0] = np.clip(lifted[..., 0] * (1.0 + 0.10 * warm) + 0.02 * warm, 0, 1)
        lifted[..., 1] = np.clip(lifted[..., 1] * (1.0 + 0.03 * warm), 0, 1)
        lifted[..., 2] = np.clip(lifted[..., 2] * (1.0 - 0.06 * warm), 0, 1)

    # Soft local mid contrast so faces don't look flat after heavy lift
    mid = 0.50
    c = 0.08 * mask
    lifted = np.clip((lifted - mid) * (1.0 + c[..., None]) + mid, 0, 1)

    meta["mask_mean"] = round(float(mask.mean()), 4)
    meta["gain_p95"] = round(float(np.quantile(gain_m[mask > 0.2], 0.95)) if (mask > 0.2).any() else 1.0, 3)
    return lifted.astype(np.float32), meta


def apply_face_finish(
    rgb: np.ndarray,
    *,
    subject_center: list[float] | tuple[float, float] | None = None,
    preset: str = "portrait",
    exposure: float | None = None,
    contrast: float = 12.0,
    clarity: float = 4.0,
    amount: float = 1.0,
    target_luma: float = 0.52,
    as_shot_mean: float | None = None,
    face_box_hint: dict[str, float] | None = None,
) -> np.ndarray:
    """After global look: prefer YuNet multi-face finish; fall back to skin oval."""
    amount = float(np.clip(amount, 0.0, 1.5))
    if amount <= 0:
        return rgb

    finished, meta = apply_people_finish(
        rgb,
        amount=amount,
        target_luma=target_luma,
        as_shot_mean=as_shot_mean,
        warm_shadows=0.14,
    )
    if meta.get("mode") == "multi_face" and meta.get("faces"):
        return finished

    # Fallback: previous single-region skin/oval path
    h, w = rgb.shape[:2]
    center, box_override, _mode = resolve_face_center(
        rgb, subject_center, face_box_hint=face_box_hint
    )
    face = analyze_face_region(
        rgb,
        subject_center=center,
        preset=preset,
        face_box_hint=box_override or face_box_hint,
    )
    mean_now = float(face.get("face_mean_luma") or 0.0)
    hot = bool(face.get("face_hot_as_shot")) or mean_now >= 0.62
    under_hint = (
        (as_shot_mean is not None and float(as_shot_mean) < 0.35)
        or mean_now < 0.36
        or bool(face.get("face_underexposed_as_shot"))
    )
    box = box_override or face["face_box"]
    y0, y1 = int(box["y0"] * h), max(int(box["y1"] * h), int(box["y0"] * h) + 1)
    x0, x1 = int(box["x0"] * w), max(int(box["x1"] * w), int(box["x0"] * w) + 1)
    if exposure is None:
        if under_hint and mean_now < 0.62:
            exposure = max(
                0.70,
                face_finish_exposure_for_mean(max(mean_now, 0.06), target=0.55, hot=False),
            )
        elif mean_now > 0 and mean_now < target_luma - 0.02:
            exposure = face_finish_exposure_for_mean(mean_now, target=target_luma, hot=False)
        elif hot:
            exposure = 0.04
        else:
            exposure = 0.12
    exposure = float(exposure)
    if under_hint or mean_now < 0.40:
        contrast = min(float(contrast), 5.0)

    out = np.clip(rgb, 0, 1).astype(np.float32).copy()
    crop = out[y0:y1, x0:x1]
    skin = _skin_mask(crop).astype(np.float32)
    ch, cw = crop.shape[:2]
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    cy, cx = (ch - 1) / 2.0, (cw - 1) / 2.0
    dist = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    fall = np.clip(1.0 - dist * 0.75, 0.0, 1.0)
    skin_frac = float(skin.mean())
    if skin_frac >= 0.03:
        pad = np.pad(skin, 2, mode="edge")
        dil = np.zeros_like(skin)
        for dy in range(5):
            for dx in range(5):
                dil = np.maximum(dil, pad[dy : dy + ch, dx : dx + cw])
        skin_w = 0.55 if under_hint else 0.85
        mask = np.clip((1.0 - skin_w) * fall + skin_w * dil * fall, 0.0, 1.0) * amount
    else:
        mask = fall * amount
    mask = _feather(mask, radius=6)
    mask3 = mask[..., None]
    if under_hint and mean_now < 0.22:
        exposure = max(exposure, 1.45)
    elif under_hint and mean_now < 0.32:
        exposure = max(exposure, 1.15)

    lift = crop * (2.0 ** (exposure * mask3[..., 0]))[..., None]
    mid = 0.50 if under_hint else 0.55
    c = (contrast / 100.0) * 0.7
    lift = (lift - mid) * (1.0 + c * mask3) + mid
    lift = np.clip(lift, 0, 1)
    if clarity > 0:
        pad = np.pad(lift, ((1, 1), (1, 1), (0, 0)), mode="edge")
        blur = (
            pad[0:-2, 1:-1]
            + pad[2:, 1:-1]
            + pad[1:-1, 0:-2]
            + pad[1:-1, 2:]
        ) * 0.25
        cl = (clarity / 100.0) * 0.55
        lift = np.clip(lift + (lift - blur) * cl * mask3, 0, 1)
    out[y0:y1, x0:x1] = np.clip(crop * (1.0 - mask3) + lift * mask3, 0, 1)
    return out


def bump_params_for_face_under(params: dict, flags: list[str] | None) -> dict:
    """Global assist so people finish blends with the look (not a floating patch)."""
    flags = flags or []
    if "face_underexposed_as_shot" not in flags and "underexposed_as_shot" not in flags:
        return params
    out = dict(params)
    # Stronger shadow lift + slight exposure; protect highlights for backlight sky
    out["shadows"] = int(min(55, float(out.get("shadows", 0) or 0) + 28))
    out["exposure"] = float(out.get("exposure", 0) or 0) + 0.28
    out["highlights"] = int(min(-25, float(out.get("highlights", 0) or 0) - 8))
    blacks = float(out.get("blacks", 0) or 0)
    if blacks < -2:
        out["blacks"] = int(min(0, blacks + 10))
    # Slight warm to counter open-shade blue on people
    out["temperature"] = float(out.get("temperature", 0) or 0) + 6
    return out
