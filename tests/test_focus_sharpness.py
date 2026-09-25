"""Tests for focal-plane sharpness (max-patch / attention / shallow_dof)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from focus_sharpness import (  # noqa: E402
    blur_cut_adjusted,
    compute_plane_field_sharpness,
    should_flag_shallow_dof,
)


def _sharp_center_soft_border(h: int = 128, w: int = 128) -> np.ndarray:
    """Synthetic shallow DOF: sharp checker in center, smooth elsewhere."""
    img = np.full((h, w, 3), 0.35, dtype=np.float32)
    # Soft gradient border
    yy, xx = np.mgrid[0:h, 0:w]
    img[..., 0] = 0.3 + 0.1 * (xx / w)
    img[..., 1] = 0.32
    img[..., 2] = 0.34
    # Sharp high-contrast patch in center
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 3, 2 * w // 3
    tile = ((xx[y0:y1, x0:x1] // 3) + (yy[y0:y1, x0:x1] // 3)) % 2
    img[y0:y1, x0:x1, :] = np.where(tile[..., None] > 0, 0.95, 0.05)
    return img


def _all_soft(h: int = 128, w: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 0.4 + 0.05 * np.sin(xx / 40.0) * np.sin(yy / 40.0)
    img = np.stack([v, v * 0.98, v * 1.02], axis=-1).astype(np.float32)
    return np.clip(img, 0, 1)


def test_plane_beats_field_on_bokeh_synth():
    img = _sharp_center_soft_border()
    out = compute_plane_field_sharpness(img, subject_center=[0.5, 0.5])
    assert out["sharp_plane"] > out["sharp_field"]
    assert out["sharp_plane"] >= 40.0
    assert should_flag_shallow_dof(
        out["sharp_plane"], out["sharp_field"], blur_cut=24.0, soft_cut=38.0
    )


def test_all_soft_stays_below_blur_cut():
    img = _all_soft()
    out = compute_plane_field_sharpness(img)
    assert out["sharp_plane"] < 28.0


def test_blur_cut_relaxes_at_wide_aperture():
    b, s = blur_cut_adjusted(24.0, 38.0, fnumber=1.8)
    assert b < 24.0
    assert s < 38.0
    b2, _ = blur_cut_adjusted(24.0, 38.0, fnumber=8.0)
    assert b2 >= 24.0  # small aperture → equal or slightly stricter
