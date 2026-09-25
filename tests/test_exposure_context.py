"""Tests for EXIF exposure-triangle context (F / Tv / ISO)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from exposure_context import (  # noqa: E402
    adjust_noise_for_preset,
    blur_cuts_for_exposure,
    build_exposure_context,
    exposure_info_flags,
    exposure_mismatch_penalty,
    scene_ev100,
)
from sensor_profiles import SONY_A7C_IMX410  # noqa: E402


def test_build_exposure_context_triangle():
    exif = {
        "FNumber": 1.8,
        "ExposureTime": 1 / 60,
        "ISO": 3200,
        "FocalLengthIn35mmFormat": 50,
    }
    ctx = build_exposure_context(exif, profile=SONY_A7C_IMX410, preset="night")
    assert ctx["fnumber"] == 1.8
    assert ctx["iso"] == 3200
    assert ctx["exposure_time"] is not None
    assert ctx["shake_risk"] is not None
    assert ctx["iso_ratio"] == 32.0
    assert ctx["scene_ev100"] is not None


def test_blur_cut_combines_f_tv_iso():
    ctx = {
        "fnumber": 1.8,
        "shake_risk": 1.5,
        "iso_ratio": 16.0,
        "preset": "night",
    }
    b, s, deltas = blur_cuts_for_exposure(24.0, 38.0, ctx, preset="night")
    assert b < 24.0
    assert sum(deltas.values()) < 0
    # Cap: cannot drop more than 8
    assert 24.0 - b <= 8.0 + 1e-6


def test_motion_and_high_iso_flags():
    flags = exposure_info_flags({"shake_risk": 1.5, "iso_ratio": 20.0})
    assert "motion_risk" in flags
    assert "high_iso" in flags


def test_noise_preset_boost_night():
    ctx = {"iso_ratio": 16.0}
    base = 70.0
    night = adjust_noise_for_preset(base, ctx, preset="night")
    land = adjust_noise_for_preset(base, ctx, preset="landscape")
    assert night >= base
    assert land <= base


def test_exposure_mismatch():
    pen, flag = exposure_mismatch_penalty(-2.5, {"scene_ev100": 12.0})
    assert pen > 0 and flag == "exposure_mismatch"
    pen2, flag2 = exposure_mismatch_penalty(0.0, {"scene_ev100": 12.0})
    assert pen2 == 0 and flag2 is None


def test_scene_ev100_sunny16ish():
    # F/16, 1/100s, ISO 100 ≈ EV 15
    ev = scene_ev100(16.0, 1 / 100.0, 100.0)
    assert ev is not None and 14.0 <= ev <= 16.0


def test_phone_blur_cut_relief_without_fnumber():
    from sensor_profiles import IPHONE_17_PROMAX

    ctx = build_exposure_context(
        {"ISO": 125, "ExposureTime": 1 / 120},
        profile=IPHONE_17_PROMAX,
        preset="general",
    )
    assert ctx["family"] == "phone"
    assert ctx["fnumber"] is None
    b, s, deltas = blur_cuts_for_exposure(
        float(IPHONE_17_PROMAX.blur_sharp),
        float(IPHONE_17_PROMAX.soft_sharp),
        ctx,
        preset="general",
    )
    assert deltas.get("phone", 0) < 0
    assert deltas.get("f", 0) < 0  # assumed wide when F missing
    assert b < float(IPHONE_17_PROMAX.blur_sharp)
    assert s > b
    # Phone may relieve up to 10
    assert float(IPHONE_17_PROMAX.blur_sharp) - b <= 10.0 + 1e-6


def test_phone_ibis_relaxes_handhold_expectation():
    from sensor_profiles import IPHONE_PRORAW_GENERIC

    # 1/30s is near phone t_safe (~1/31 with 2-stop OIS on 1/125 base)
    ctx = build_exposure_context(
        {"ExposureTime": 1 / 30, "ISO": 100},
        profile=IPHONE_PRORAW_GENERIC,
    )
    assert ctx["family"] == "phone"
    assert ctx["t_safe"] >= (1.0 / 125.0) * (2.0**2) - 1e-6
    assert ctx["shake_risk"] is not None
    assert ctx["shake_risk"] < 0.5  # not flagged as severe motion
