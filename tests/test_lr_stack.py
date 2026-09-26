"""Tests for Lightroom-ordered develop stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from lr_stack import auto_basic_params, develop_lr_stack, refine_people  # noqa: E402
from looks import complete_params  # noqa: E402


def test_auto_basic_lifts_dark_frame():
    img = np.full((64, 64, 3), 0.12, dtype=np.float32)
    base = complete_params({"exposure": 0.0, "shadows": 0, "highlights": 0})
    out = auto_basic_params(img, base, flags=["underexposed_as_shot", "face_underexposed_as_shot"], face_mean=0.15)
    assert out["exposure"] > 0.2
    assert out["shadows"] >= 15
    assert out["highlights"] <= -20


def test_refine_people_relative_not_extreme():
    """Without skin masks, refine is no-op; with segments face_ev stays scene-matched."""
    h, w = 120, 120
    img = np.full((h, w, 3), 0.35, dtype=np.float32)
    out, meta = refine_people(img, amount=1.0, as_shot_face_mean=0.2)
    assert out.shape == img.shape
    assert meta["mode"] in {"skip", "people_refine"}
    if meta["mode"] == "people_refine":
        assert float(meta.get("face_ev") or 0) <= 0.65


def test_correct_skin_tone_warms_cool_skin():
    from lr_stack import correct_skin_tone

    rgb = np.zeros((40, 40, 3), dtype=np.float32)
    rgb[..., 2] = 0.42
    rgb[..., 1] = 0.32
    rgb[..., 0] = 0.28
    mask = np.ones((40, 40), dtype=np.float32)
    out = correct_skin_tone(rgb, mask, amount=1.0, warmth=0.7, magenta=0.25, sat=0.12)
    assert float(out[..., 0].mean()) > float(rgb[..., 0].mean())
    assert float(out[..., 2].mean()) < float(rgb[..., 2].mean())


def test_develop_lr_stack_runs():
    img = np.clip(np.random.rand(80, 60, 3).astype(np.float32) * 0.25 + 0.05, 0, 1)
    params = complete_params({"exposure": 0.05, "shadows": 8, "highlights": -15, "contrast": 10, "clarity": 0, "sharpen": 8})
    out, report = develop_lr_stack(
        img,
        params,
        flags=["underexposed_as_shot"],
        face_mean=0.18,
        people_amount=0.0,
        selective_amount=0.8,
        do_people=False,
        do_selective=True,
    )
    assert out.shape == img.shape
    assert "auto_basic" in report["stages"]
    assert "look_detail" in report["stages"]
    assert "selective" in report["stages"]
