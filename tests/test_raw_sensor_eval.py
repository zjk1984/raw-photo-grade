"""Tests for sensor profiles and RAW-aware metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from sensor_profiles import (  # noqa: E402
    IPHONE_17_PROMAX,
    SONY_A7C_IMX410,
    detect_sensor_profile,
    expected_noise,
)
from raw_eval_metrics import merge_scores, score_raw_dynamic_range, score_raw_noise, score_raw_sharpness  # noqa: E402


def test_detect_sony_a7c():
    p = detect_sensor_profile(
        Path("DSC00009.ARW"),
        {"Make": "SONY", "Model": "ILCE-7C"},
    )
    assert p.id == "sony_a7c_imx410"


def test_detect_iphone_17_promax():
    p = detect_sensor_profile(
        Path("IMG_0001.DNG"),
        {"Make": "Apple", "Model": "iPhone 17 Pro Max"},
    )
    assert p.id == "iphone_17_promax"


def test_force_prefer():
    p = detect_sensor_profile(Path("x.jpg"), {}, prefer="iphone_17_promax")
    assert p is IPHONE_17_PROMAX


def test_expected_noise_scales_with_iso():
    n100 = expected_noise(SONY_A7C_IMX410, 100)
    n1600 = expected_noise(SONY_A7C_IMX410, 1600)
    assert n1600 > n100
    phone = expected_noise(IPHONE_17_PROMAX, 800)
    assert phone > n100


def test_score_helpers():
    sharp = score_raw_sharpness(SONY_A7C_IMX410.sharp_ref, SONY_A7C_IMX410)
    assert 80 <= sharp <= 100
    ctx = {
        "headroom_ev": 1.8,
        "raw_highlight_pct": 0.2,
        "raw_shadow_pct": 3.0,
        "midtone_ev": 0.0,
        "raw_noise_mad": SONY_A7C_IMX410.noise_ref_base,
        "expected_noise": SONY_A7C_IMX410.noise_ref_base,
        "iso": 100,
    }
    dr, flags = score_raw_dynamic_range(ctx, SONY_A7C_IMX410)
    assert dr > 70
    assert not flags
    assert score_raw_noise(ctx, SONY_A7C_IMX410) > 80


def test_merge_trusts_raw():
    merged = merge_scores(
        {
            "sharpness": 20.0,
            "dynamic_range": 50.0,
            "noise_control": 50.0,
            "color_harmony": 80.0,
            "composition": 80.0,
        },
        raw_sharp=90.0,
        raw_dr=90.0,
        raw_noise=90.0,
        profile=SONY_A7C_IMX410,
    )
    # high raw_trust → sharpness pulled up from 20 toward 90
    assert merged["sharpness"] > 70
    assert merged["color_harmony"] == 80.0


@pytest.mark.skipif(
    not Path("/Users/zjk/Documents/photo/DSC00009.ARW").exists(),
    reason="sample A7C ARW not present",
)
def test_live_a7c_raw_aware():
    from eval_photo import PhotoEvaluator

    ev = PhotoEvaluator(
        device_name="cpu",
        raw_aware=True,
        sensor_profile="sony_a7c_imx410",
        max_edge=1280,
    )
    result = ev.evaluate(Path("/Users/zjk/Documents/photo/DSC00009.ARW"))
    assert result.details.get("sensor_profile") == "sony_a7c_imx410"
    assert "headroom_ev" in result.details or "raw_aware_error" not in result.details
