"""Unit tests for fade / HSL / 3D LUT grade path and Sony looks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from look_select import classify_scene, suggest_look  # noqa: E402
from looks import ALL_LOOKS, get_look  # noqa: E402
from lut3d import apply_lut, ensure_builtin_luts, load_cube, make_film_approx_lut  # noqa: E402
from raw_develop import apply_grade  # noqa: E402


def _ramp() -> np.ndarray:
    xs = np.linspace(0, 1, 64, dtype=np.float32)
    r, g, b = np.meshgrid(xs, xs, [0.2, 0.5, 0.8][:1], indexing="ij")
    # simpler: solid color gradient image
    img = np.zeros((48, 64, 3), dtype=np.float32)
    img[..., 0] = np.linspace(0.1, 0.9, 64)[None, :]
    img[..., 1] = 0.4
    img[..., 2] = np.linspace(0.8, 0.2, 64)[None, :]
    return img


def test_sony_looks_registered():
    for name in ("sony-st", "sony-fl", "sony-vv2", "sony-in", "sony-pt"):
        assert name in ALL_LOOKS
        p = get_look(name)
        assert "fade" in p
        assert "hsl_sky_sat" in p


def test_fuji_nikon_looks_registered():
    for name in (
        "fuji-provia",
        "fuji-velvia",
        "fuji-classic-chrome",
        "fuji-nostalgic-neg",
        "fuji-acros",
        "nikon-standard",
        "nikon-landscape",
        "nikon-flat",
        "nikon-monochrome",
    ):
        assert name in ALL_LOOKS
        out = apply_grade(_ramp(), get_look(name))
        assert out.shape == _ramp().shape


def test_brand_auto_suggest():
    img = np.full((64, 64, 3), 0.85, dtype=np.float32)
    look_f, _ = suggest_look(img, brand="fuji", auto=True)
    look_n, _ = suggest_look(img, brand="nikon", auto=True)
    assert look_f.startswith("fuji-") or look_f == "night"
    assert look_n.startswith("nikon-") or look_n == "night"


def test_fade_lifts_blacks():
    img = np.zeros((32, 32, 3), dtype=np.float32)
    out = apply_grade(img, {"fade": 50, "exposure": 0, "contrast": 0, "highlights": 0,
                            "shadows": 0, "whites": 0, "blacks": 0, "temperature": 0,
                            "tint": 0, "vibrance": 0, "saturation": 0, "clarity": 0,
                            "vignette": 0, "sharpen": 0, "noise_luma": 0})
    assert float(out.mean()) > 0.02


def test_lut_roundtrip_and_apply():
    lut_dir = ensure_builtin_luts()
    path = lut_dir / "sony_fl_approx.cube"
    assert path.exists()
    assert (lut_dir / "fuji_velvia_approx.cube").exists()
    assert (lut_dir / "nikon_landscape_approx.cube").exists()
    table, size = load_cube(path)
    assert size == 33
    img = _ramp()
    mapped = apply_lut(img, table, size, amount=1.0)
    assert mapped.shape == img.shape
    assert float(np.abs(mapped - img).mean()) > 0.001


def test_apply_grade_with_sony_fl():
    img = _ramp()
    out = apply_grade(img, get_look("sony-fl"))
    assert out.shape == img.shape
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_scene_suggest_highkey():
    bright = np.full((64, 64, 3), 0.85, dtype=np.float32)
    tag = classify_scene(bright)
    assert tag in {"highkey", "general", "vivid", "matte"}
    look, scene = suggest_look(bright, forced="sony-fl")
    assert look == "sony-fl"
    assert scene == "forced"


def test_procedural_lut_shape():
    t = make_film_approx_lut(17)
    assert t.shape == (17, 17, 17, 3)
