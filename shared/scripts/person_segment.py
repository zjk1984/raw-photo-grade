#!/usr/bin/env python3
"""Portrait masks via dedicated MediaPipe selfie multiclass model.

Class map (selfie_multiclass_256x256):
  0 background | 1 hair | 2 body-skin | 3 face-skin | 4 clothes | 5 others

YuNet boxes remain optional seeds for eyes/teeth bands; silhouettes come from
the segmenter confidence maps (not rectangles / GrabCut).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_MULTICLASS = _MODEL_DIR / "selfie_multiclass_256x256.tflite"
_BINARY = _MODEL_DIR / "selfie_segmenter.tflite"

# Confidence indices for multiclass
CLS_BG, CLS_HAIR, CLS_BODY, CLS_FACE, CLS_CLOTHES, CLS_OTHER = 0, 1, 2, 3, 4, 5


@lru_cache(maxsize=1)
def _segmenter():
    if not _MULTICLASS.is_file():
        return None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        BaseOptions = mp_python.BaseOptions
        kwargs: dict[str, Any] = {"model_asset_path": str(_MULTICLASS)}
        if hasattr(BaseOptions, "Delegate"):
            kwargs["delegate"] = BaseOptions.Delegate.CPU  # avoid Metal crash
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(**kwargs),
            output_category_mask=True,
            output_confidence_masks=True,
        )
        return vision.ImageSegmenter.create_from_options(opts)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _binary_segmenter():
    if not _BINARY.is_file():
        return None
    try:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        BaseOptions = mp_python.BaseOptions
        kwargs: dict[str, Any] = {"model_asset_path": str(_BINARY)}
        if hasattr(BaseOptions, "Delegate"):
            kwargs["delegate"] = BaseOptions.Delegate.CPU
        opts = vision.ImageSegmenterOptions(
            base_options=BaseOptions(**kwargs),
            output_category_mask=False,
            output_confidence_masks=True,
        )
        return vision.ImageSegmenter.create_from_options(opts)
    except Exception:
        return None


def _to_u8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _conf_list(result) -> list[np.ndarray]:
    masks = []
    for m in result.confidence_masks or []:
        a = np.asarray(m.numpy_view(), dtype=np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        masks.append(np.clip(a, 0, 1))
    return masks


def segment_portrait(rgb: np.ndarray, *, max_side: int = 1280) -> dict[str, np.ndarray] | None:
    """Run selfie multiclass. Returns soft maps at full `rgb` resolution."""
    import mediapipe as mp

    seg = _segmenter()
    if seg is None:
        return None

    h, w = rgb.shape[:2]
    work = rgb
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        try:
            import cv2

            work = cv2.resize(_to_u8(rgb), (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        except Exception:
            from PIL import Image as PILImage

            work = (
                np.asarray(
                    PILImage.fromarray(_to_u8(rgb)).resize((nw, nh), resample=PILImage.BILINEAR)
                ).astype(np.float32)
                / 255.0
            )

    u8 = _to_u8(work)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=u8)
    result = seg.segment(mp_image)
    conf = _conf_list(result)
    if len(conf) < 6:
        return None

    def up(a: np.ndarray) -> np.ndarray:
        if a.shape[0] == h and a.shape[1] == w:
            return a
        try:
            import cv2

            return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        except Exception:
            from PIL import Image as PILImage

            return (
                np.asarray(
                    PILImage.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize(
                        (w, h), resample=PILImage.BILINEAR
                    )
                ).astype(np.float32)
                / 255.0
            )

    face = up(conf[CLS_FACE])
    body = up(conf[CLS_BODY])
    clothes = up(conf[CLS_CLOTHES])
    hair = up(conf[CLS_HAIR])
    other = up(conf[CLS_OTHER])
    person = np.clip(face + body + clothes + hair + other, 0, 1)
    return {
        "face_skin": face,
        "body_skin": body,
        "clothes": clothes,
        "hair": hair,
        "other": other,
        "subject": person,
        "background": up(conf[CLS_BG]),
    }


def _soft_from_binary(binary: np.ndarray, feather_px: int = 12) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return binary.astype(np.float32)
    b = (binary > 0.5).astype(np.uint8)
    if int(b.sum()) < 8:
        return np.zeros_like(binary, dtype=np.float32)
    dist_out = cv2.distanceTransform(1 - b, cv2.DIST_L2, 3)
    soft = np.where(b > 0, 1.0, np.clip(1.0 - dist_out / max(feather_px, 1), 0, 1))
    return soft.astype(np.float32)


def face_contour_mask(rgb: np.ndarray, face: dict[str, float], *, pad: float = 0.18) -> np.ndarray:
    """Fallback organic face from skin when segmenter unavailable."""
    try:
        import cv2
        from face_recover import _skin_mask
    except ImportError:
        return np.zeros(rgb.shape[:2], dtype=np.float32)

    h, w = rgb.shape[:2]
    fw = float(face["x1"] - face["x0"])
    fh = float(face["y1"] - face["y0"])
    x0 = max(0, int((face["x0"] - pad * fw) * w))
    x1 = min(w, int((face["x1"] + pad * fw) * w))
    y0 = max(0, int((face["y0"] - pad * fh * 1.2) * h))
    y1 = min(h, int((face["y1"] + pad * fh * 0.5) * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.zeros((h, w), dtype=np.float32)
    crop = np.clip(rgb[y0:y1, x0:x1], 0, 1)
    seed = _skin_mask(crop).astype(np.uint8)
    if int(seed.sum()) < 20:
        ch, cw = seed.shape
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        ell = (((yy - ch * 0.48) / max(ch * 0.48, 1)) ** 2 + ((xx - cw * 0.5) / max(cw * 0.42, 1)) ** 2) <= 1.0
        seed = ell.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    local = np.zeros(seed.shape, dtype=np.uint8)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(local, [cv2.convexHull(c)], -1, 1, thickness=-1)
    out = np.zeros((h, w), dtype=np.float32)
    out[y0:y1, x0:x1] = local
    return out


def person_grabcut_mask(rgb: np.ndarray, faces: list[dict[str, Any]], **kwargs: Any) -> np.ndarray:
    """Legacy GrabCut fallback (kept for environments without MediaPipe model)."""
    try:
        import cv2
    except ImportError:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    if not faces:
        return np.zeros(rgb.shape[:2], dtype=np.float32)

    h, w = rgb.shape[:2]
    max_side = int(kwargs.get("max_side", 720))
    iters = int(kwargs.get("iters", 4))
    scale = 1.0
    work = rgb
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        work = cv2.resize(_to_u8(rgb), (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    wh, ww = work.shape[:2]
    bgr = np.ascontiguousarray(_to_u8(work)[:, :, ::-1])
    gc = np.full((wh, ww), cv2.GC_BGD, dtype=np.uint8)
    y = 0.2126 * work[..., 0] + 0.7152 * work[..., 1] + 0.0722 * work[..., 2]
    skyish = (work[..., 2] > work[..., 0] * 1.05) & (y > 0.45)
    top = np.zeros((wh, ww), dtype=bool)
    top[: max(1, int(0.22 * wh)), :] = True
    gc[skyish | top] = cv2.GC_PR_BGD
    for f in faces:
        fx0, fy0 = int(f["x0"] * ww), int(f["y0"] * wh)
        fx1, fy1 = int(f["x1"] * ww), int(f["y1"] * wh)
        fw, fh = max(fx1 - fx0, 1), max(fy1 - fy0, 1)
        gc[
            max(0, fy0 + int(0.12 * fh)) : min(wh, fy1 - int(0.10 * fh)),
            max(0, fx0 + int(0.15 * fw)) : min(ww, fx1 - int(0.15 * fw)),
        ] = cv2.GC_FGD
        py0, py1 = max(0, fy0 - int(0.45 * fh)), min(wh, fy1 + int(2.8 * fh))
        px0, px1 = max(0, fx0 - int(0.55 * fw)), min(ww, fx1 + int(0.55 * fw))
        roi = gc[py0:py1, px0:px1]
        roi[roi == cv2.GC_BGD] = cv2.GC_PR_FGD
        gc[py0:py1, px0:px1] = roi
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except Exception:
        mask = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    else:
        mask = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.float32)
    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(mask, 0, 1).astype(np.float32)


def build_people_contours(rgb: np.ndarray) -> dict[str, Any]:
    """Soft portrait masks from dedicated segmenter (+ YuNet for face landmarks bands)."""
    h, w = rgb.shape[:2]
    empty = np.zeros((h, w), dtype=np.float32)
    out: dict[str, Any] = {
        "faces": [],
        "face_skin": empty.copy(),
        "eyes": empty.copy(),
        "teeth": empty.copy(),
        "clothes": empty.copy(),
        "body_skin": empty.copy(),
        "hair": empty.copy(),
        "subject": empty.copy(),
        "face_binary": empty.copy(),
        "subject_binary": empty.copy(),
        "backend": "none",
    }

    faces: list[dict[str, Any]] = []
    try:
        from face_detect import detect_faces

        faces = detect_faces(rgb)
    except Exception:
        faces = []
    out["faces"] = faces

    seg = segment_portrait(rgb)
    if seg is not None:
        out["backend"] = "mediapipe_selfie_multiclass"
        out["face_skin"] = seg["face_skin"]
        out["body_skin"] = seg["body_skin"]
        out["clothes"] = seg["clothes"]
        out["hair"] = seg["hair"]
        out["subject"] = seg["subject"]
        out["face_binary"] = (seg["face_skin"] > 0.35).astype(np.float32)
        out["subject_binary"] = (seg["subject"] > 0.35).astype(np.float32)
    else:
        # Fallback: contour face + GrabCut body
        out["backend"] = "grabcut_fallback"
        face_bin = empty.copy()
        for f in faces:
            face_bin = np.maximum(face_bin, face_contour_mask(rgb, f))
        person = person_grabcut_mask(rgb, faces) if faces else face_bin
        person = np.maximum(person, face_bin)
        from face_recover import _skin_mask

        skin = _skin_mask(rgb).astype(np.float32)
        out["face_skin"] = _soft_from_binary(face_bin, 8)
        out["subject"] = _soft_from_binary(person, 12)
        out["body_skin"] = _soft_from_binary(skin * person * (1.0 - face_bin * 0.85), 8)
        out["clothes"] = _soft_from_binary(np.clip(person - face_bin * 0.9, 0, 1), 12)
        out["face_binary"] = face_bin
        out["subject_binary"] = person

    # Eyes / teeth: YuNet face boxes ∩ face-skin confidence
    face_m = out["face_skin"]
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    eyes = empty.copy()
    teeth = empty.copy()
    for f in faces:
        y0, y1 = int(f["y0"] * h), max(int(f["y1"] * h), int(f["y0"] * h) + 1)
        x0, x1 = int(f["x0"] * w), max(int(f["x1"] * w), int(f["x0"] * w) + 1)
        eh = max(1, int(0.38 * (y1 - y0)))
        eyes[y0 : y0 + eh, x0:x1] = np.maximum(eyes[y0 : y0 + eh, x0:x1], face_m[y0 : y0 + eh, x0:x1])
        th0, th1 = y0 + int(0.55 * (y1 - y0)), y0 + int(0.80 * (y1 - y0))
        mx0, mx1 = x0 + int(0.22 * (x1 - x0)), x0 + int(0.78 * (x1 - x0))
        band = face_m[th0:th1, mx0:mx1] * np.clip((y[th0:th1, mx0:mx1] - 0.32) / 0.28, 0, 1)
        teeth[th0:th1, mx0:mx1] = np.maximum(teeth[th0:th1, mx0:mx1], band)
    # If no YuNet faces, approximate eyes on upper face-skin mass
    if not faces and float(face_m.max()) > 0.2:
        ys, xs = np.where(face_m > 0.4)
        if len(ys):
            y_cut = int(np.quantile(ys, 0.35))
            eyes[:y_cut, :] = np.maximum(eyes[:y_cut, :], face_m[:y_cut, :] * 0.8)

    out["eyes"] = eyes.astype(np.float32)
    out["teeth"] = teeth.astype(np.float32)
    return out
