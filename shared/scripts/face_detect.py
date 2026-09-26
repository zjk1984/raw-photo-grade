#!/usr/bin/env python3
"""Multi-face detection (OpenCV YuNet) for portrait finish / eval anchors.

Returns normalized face boxes. Optional body boxes extend below each face so
skin + torso can share one exposure target (backlit family portraits).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_MODEL = (
    Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
)


@lru_cache(maxsize=1)
def _yunet_path() -> str | None:
    return str(_MODEL) if _MODEL.is_file() else None


def detect_faces(
    rgb: np.ndarray,
    *,
    score_thresh: float = 0.45,
    nms_thresh: float = 0.3,
    max_side: int = 1280,
) -> list[dict[str, Any]]:
    """Detect faces. Each item: x0,y0,x1,y1 (norm), score, center.

    Runs on a downscaled copy for speed; boxes map back to full-frame norms.
    """
    try:
        import cv2
    except ImportError:
        return []

    path = _yunet_path()
    if not path:
        return []

    img = np.clip(rgb, 0, 1)
    h, w = img.shape[:2]
    scale = 1.0
    work = img
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        try:
            import cv2

            work = cv2.resize(
                (np.clip(work, 0, 1) * 255.0).astype(np.uint8),
                (nw, nh),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32) / 255.0
        except Exception:
            from PIL import Image as PILImage

            work = (
                np.asarray(
                    PILImage.fromarray((np.clip(work, 0, 1) * 255).astype(np.uint8)).resize(
                        (nw, nh), resample=PILImage.BILINEAR
                    )
                ).astype(np.float32)
                / 255.0
            )
    wh, ww = work.shape[:2]
    u8 = (np.clip(work, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    bgr = np.ascontiguousarray(u8[:, :, ::-1])

    try:
        det = cv2.FaceDetectorYN.create(path, "", (ww, wh), float(score_thresh), float(nms_thresh), 5000)
        det.setInputSize((ww, wh))
        _, faces = det.detect(bgr)
    except Exception:
        return []

    if faces is None or len(faces) == 0:
        return []

    out: list[dict[str, Any]] = []
    for f in faces:
        x, y, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        score = float(f[-1])
        # map to full-frame normalized
        x0 = max(0.0, x / ww)
        y0 = max(0.0, y / wh)
        x1 = min(1.0, (x + fw) / ww)
        y1 = min(1.0, (y + fh) / wh)
        if x1 - x0 < 0.02 or y1 - y0 < 0.02:
            continue
        out.append(
            {
                "x0": round(x0, 4),
                "y0": round(y0, 4),
                "x1": round(x1, 4),
                "y1": round(y1, 4),
                "score": round(score, 3),
                "center": [round(0.5 * (x0 + x1), 4), round(0.5 * (y0 + y1), 4)],
                "area": round((x1 - x0) * (y1 - y0), 5),
            }
        )
    out.sort(key=lambda d: d["area"], reverse=True)
    return out


def face_to_body_box(face: dict[str, float], *, body_scale: float = 2.4) -> dict[str, float]:
    """Expand a face box downward/wide to cover upper torso + hat brim."""
    x0, y0, x1, y1 = float(face["x0"]), float(face["y0"]), float(face["x1"]), float(face["y1"])
    fw, fh = x1 - x0, y1 - y0
    cx = 0.5 * (x0 + x1)
    # Pad up for hats; down for torso
    top = max(0.0, y0 - 0.55 * fh)
    bottom = min(1.0, y1 + body_scale * fh)
    half_w = max(fw * 0.85, 0.08)
    return {
        "x0": round(max(0.0, cx - half_w), 4),
        "y0": round(top, 4),
        "x1": round(min(1.0, cx + half_w), 4),
        "y1": round(bottom, 4),
    }


def people_regions(rgb: np.ndarray, **detect_kw: Any) -> dict[str, Any]:
    """Detect faces and derive body regions + union bbox."""
    faces = detect_faces(rgb, **detect_kw)
    bodies = [face_to_body_box(f) for f in faces]
    union = None
    if bodies:
        union = {
            "x0": min(b["x0"] for b in bodies),
            "y0": min(b["y0"] for b in bodies),
            "x1": max(b["x1"] for b in bodies),
            "y1": max(b["y1"] for b in bodies),
        }
    return {"faces": faces, "bodies": bodies, "union": union, "count": len(faces)}
