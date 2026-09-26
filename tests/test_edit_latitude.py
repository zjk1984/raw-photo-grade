"""Tests for RAW edit latitude (recoverability) scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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


def test_cloud_sky_not_penalized():
    from edit_latitude import analyze_sky_highlights

    h, w = 120, 160
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Upper third: bright desaturated clouds
    img[:40] = 0.94
    img[:40, ..., 2] = 0.96  # slight cool
    # Lower: normal scene
    img[40:] = 0.35
    sky = analyze_sky_highlights(img)
    assert sky["cloud_sky"] is True
    score, details, flags = score_as_shot_preview(5.0, 0.35, 0.02, 0.55, sky=sky)
    assert "cloud_sky" in flags
    assert "preview_highlights" not in flags
    # Without cloud handling, hi=0.35 would cost ~20; with clouds penalty ~0
    assert score >= 70.0


def test_latitude_cloud_sky_skips_raw_clip():
    ctx = {
        "headroom_ev": 0.3,
        "raw_highlight_pct": 15.0,
        "raw_shadow_pct": 5.0,
        "midtone_ev": 0.2,
        "raw_p50": 0.18,
        "raw_noise_mad": SONY_A7C_IMX410.noise_ref_base,
        "expected_noise": SONY_A7C_IMX410.noise_ref_base,
        "iso": 100,
    }
    sky = {"cloud_sky": True, "lower_hi_pct": 2.0, "sky_bright_pct": 40.0}
    score, _d, flags = score_edit_latitude(ctx, SONY_A7C_IMX410, sky=sky)
    assert "cloud_sky" in flags
    assert "raw_highlight_clip" not in flags
    assert "no_latitude" not in flags
    assert score >= 50.0
