"""Tests for EXIF → look brand + body grade adapters."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from camera_grade import (  # noqa: E402
    apply_adapter,
    detect_look_brand,
    look_params_for_camera,
    resolve_grade_adapter,
)
from looks import get_look  # noqa: E402


def test_brand_sony_a7c():
    brand, reason = detect_look_brand({"Make": "SONY", "Model": "ILCE-7C"}, path=Path("x.ARW"))
    assert brand == "sony"
    assert "SONY" in reason or "sony" in reason.lower() or "ILCE" in reason


def test_brand_fuji_raf():
    brand, _ = detect_look_brand({"Make": "FUJIFILM", "Model": "X-T5"}, path=Path("x.RAF"))
    assert brand == "fuji"


def test_brand_iphone():
    brand, _ = detect_look_brand(
        {"Make": "Apple", "Model": "iPhone 17 Pro Max"}, path=Path("x.DNG")
    )
    assert brand == "apple"


def test_adapter_a7c_tint():
    adapter, meta = resolve_grade_adapter(
        Path("DSC.ARW"), {"Make": "SONY", "Model": "ILCE-7C"}
    )
    assert adapter.id == "sony_a7c_imx410"
    assert meta["brand"] == "sony"
    base = get_look("sony-st")
    adapted = apply_adapter(base, adapter)
    assert adapted["tint"] == int(base["tint"]) + 3
    assert adapted["noise_luma"] == max(0, int(base["noise_luma"]) - 2)


def test_adapter_iphone17_protects_highlights():
    adapter, meta = resolve_grade_adapter(
        Path("IMG.DNG"), {"Make": "Apple", "Model": "iPhone 17 Pro Max"}
    )
    assert adapter.id == "iphone_17_promax"
    assert meta["brand"] == "apple"
    base = get_look("natural")
    adapted = apply_adapter(base, adapter)
    assert adapted["highlights"] < base["highlights"]
    assert adapted["noise_luma"] > base["noise_luma"]


def test_look_params_for_camera_meta():
    params, meta = look_params_for_camera(
        "sony-fl",
        path=Path("x.ARW"),
        exif={"Make": "SONY", "Model": "ILCE-7C"},
    )
    assert params["look"] == "sony-fl"
    assert meta["adapter_id"] == "sony_a7c_imx410"
    assert "tint" in params
