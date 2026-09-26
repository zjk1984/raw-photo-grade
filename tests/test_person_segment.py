"""Tests for organic face / person contour masks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))


def test_face_contour_not_rectangle():
    from person_segment import face_contour_mask

    h, w = 200, 160
    img = np.full((h, w, 3), 0.1, dtype=np.float32)
    # Skin-ish oval (not a rect)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    face = (((yy - 90) / 45) ** 2 + ((xx - 80) / 35) ** 2) <= 1.0
    img[face, 0], img[face, 1], img[face, 2] = 0.55, 0.40, 0.32
    box = {"x0": 0.25, "y0": 0.2, "x1": 0.75, "y1": 0.7}
    m = face_contour_mask(img, box)
    assert m.sum() > 100
    # Contour should leave corners of the YuNet box empty (not fill the rectangle)
    y0, y1 = int(box["y0"] * h), int(box["y1"] * h)
    x0, x1 = int(box["x0"] * w), int(box["x1"] * w)
    corner = m[y0 : y0 + 3, x0 : x0 + 3].mean()
    center = m[85:95, 75:85].mean()
    assert center > 0.5
    assert corner < center


def test_build_people_contours_on_117_if_present():
    from pathlib import Path
    from person_segment import build_people_contours
    from raw_develop import apply_orientation, decode_raw, resize_long_edge

    src = Path("/Users/zjk/Documents/photo/DSC00117.ARW")
    if not src.exists():
        return
    rgb = resize_long_edge(apply_orientation(decode_raw(src, 1.0, False), "auto", src), 960)
    m = build_people_contours(rgb)
    assert m["backend"] in {"mediapipe_selfie_multiclass", "grabcut_fallback"}
    assert float(m["face_skin"].max()) > 0.3
    assert float(m["subject"].max()) > 0.3
    assert float(m["subject"].sum()) > float(m["face_skin"].sum()) * 1.1


def test_segment_portrait_multiclass_if_model():
    from pathlib import Path
    from person_segment import _MULTICLASS, segment_portrait
    from raw_develop import apply_orientation, decode_raw, resize_long_edge

    if not _MULTICLASS.is_file():
        return
    src = Path("/Users/zjk/Documents/photo/DSC00117.ARW")
    if not src.exists():
        return
    rgb = resize_long_edge(apply_orientation(decode_raw(src, 1.0, False), "auto", src), 640)
    seg = segment_portrait(rgb)
    assert seg is not None
    assert set(seg) >= {"face_skin", "body_skin", "clothes", "hair", "subject"}
    # Person area should be meaningful on this portrait
    assert float(seg["subject"].mean()) > 0.05
