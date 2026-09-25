"""Tests for EXIF-based camera → sensor profile matching."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from sensor_profiles import detect_sensor_profile_with_reason  # noqa: E402


def test_exif_sony_a7c():
    p, reason = detect_sensor_profile_with_reason(
        Path("DSC00009.ARW"),
        {"Make": "SONY", "Model": "ILCE-7C"},
    )
    assert p.id == "sony_a7c_imx410"
    assert "α7C" in reason or "7C" in reason or "IMX410" in reason


def test_exif_sony_a7c_ii_not_a7c():
    p, reason = detect_sensor_profile_with_reason(
        Path("x.ARW"),
        {"Make": "SONY", "Model": "ILCE-7C2"},
    )
    assert p.id == "sony_imx410_family"
    assert "7C II" in reason or "IMX410" in reason


def test_exif_iphone_17_promax():
    p, reason = detect_sensor_profile_with_reason(
        Path("IMG_0001.DNG"),
        {"Make": "Apple", "Model": "iPhone 17 Pro Max"},
    )
    assert p.id == "iphone_17_promax"
    assert "exif" in reason


def test_exif_iphone_15_generic():
    p, reason = detect_sensor_profile_with_reason(
        Path("IMG_0001.DNG"),
        {"Make": "Apple", "Model": "iPhone 15 Pro"},
    )
    assert p.id == "iphone_proraw"
    assert "exif" in reason


def test_exif_nikon_generic():
    p, reason = detect_sensor_profile_with_reason(
        Path("DSC_0001.NEF"),
        {"Make": "NIKON CORPORATION", "Model": "NIKON Z 6"},
    )
    assert p.id == "generic_camera"
    assert "Nikon" in reason or "nikon" in reason.lower() or "NEF" in reason or "camera" in reason


def test_no_exif_fallback():
    p, reason = detect_sensor_profile_with_reason(Path("unknown.bin"), {})
    assert p.id == "generic_camera"
    assert "fallback" in reason or "exif" in reason
