"""Tests for RAW edit latitude (recoverability) scoring."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from edit_latitude import score_as_shot_preview, score_edit_latitude  # noqa: E402
from sensor_profiles import SONY_A7C_IMX410  # noqa: E402


def test_underexposed_with_headroom_has_latitude():
    ctx = {
        "headroom_ev": 3.5,
        "raw_highlight_pct": 0.1,
        "raw_shadow_pct": 18.0,
        "midtone_ev": -2.2,
        "raw_p50": 0.05,
        "raw_noise_mad": SONY_A7C_IMX410.noise_ref_base,
        "expected_noise": SONY_A7C_IMX410.noise_ref_base,
        "iso": 100,
    }
    score, details, flags = score_edit_latitude(ctx, SONY_A7C_IMX410, preset="general")
    assert score >= 55.0
    assert "underexposed_as_shot" in flags
    assert "no_latitude" not in flags
    assert details["edit_latitude"] == score


def test_hard_clip_hurts_latitude():
    ctx = {
        "headroom_ev": 0.1,
        "raw_highlight_pct": 15.0,
        "raw_shadow_pct": 5.0,
        "midtone_ev": 0.5,
        "raw_p50": 0.2,
        "raw_noise_mad": SONY_A7C_IMX410.noise_ref_base,
        "expected_noise": SONY_A7C_IMX410.noise_ref_base,
        "iso": 100,
    }
    score, _d, flags = score_edit_latitude(ctx, SONY_A7C_IMX410)
    assert "raw_highlight_clip" in flags
    assert score < 70.0


def test_as_shot_soft_flags():
    score, details, flags = score_as_shot_preview(5.0, 0.15, 0.05, 0.08)
    assert details["as_shot_score"] == score
    assert "underexposed_as_shot" in flags
    assert "clipped_highlights" not in flags
