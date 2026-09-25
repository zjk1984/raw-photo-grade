#!/usr/bin/env python3
"""Camera-aware look brand + grade adapters (EXIF Make/Model → precise develop).

Flow:
  EXIF → look brand pool (sony/fuji/nikon/apple/canon)
       → body adapter deltas (α7C / iPhone 17 Pro Max / …)
       → look params ready for apply_grade
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from looks import SLIDER_KEYS, complete_params, get_look
from sensor_profiles import (
    camera_identity,
    detect_sensor_profile_with_reason,
)


@dataclass(frozen=True)
class GradeAdapter:
    id: str
    label: str
    brand: str
    # Additive slider deltas (exposure is float EV; others int-ish)
    deltas: dict[str, float]


# --- Body-specific grade adapters ---

SONY_A7C = GradeAdapter(
    id="sony_a7c_imx410",
    label="Sony α7C / IMX410",
    brand="sony",
    deltas={
        # Mild green cast → magenta; FF can take clarity/sharpen
        "tint": 3,
        "clarity": 2,
        "sharpen": 2,
        "noise_luma": -2,
        "highlights": -4,
        "hsl_skin_sat": -2,
    },
)

SONY_FF_24 = GradeAdapter(
    id="sony_ff_24mp",
    label="Sony IMX410-class 24MP FF",
    brand="sony",
    deltas={
        "tint": 2,
        "clarity": 2,
        "sharpen": 2,
        "noise_luma": -2,
        "highlights": -3,
    },
)

SONY_GENERIC = GradeAdapter(
    id="sony_generic",
    label="Sony body (generic)",
    brand="sony",
    deltas={"tint": 2, "noise_luma": -1},
)

FUJI_GENERIC = GradeAdapter(
    id="fuji_raf",
    label="Fujifilm RAF",
    brand="fuji",
    deltas={
        # Film-sim looks already carry character — ease sharpen/clarity pile-on
        "clarity": -1,
        "sharpen": -1,
        "vibrance": -2,
        "hsl_green_sat": 2,
    },
)

NIKON_GENERIC = GradeAdapter(
    id="nikon_nef",
    label="Nikon NEF",
    brand="nikon",
    deltas={
        "temperature": 2,  # NEF often reads slightly cool
        "clarity": 1,
        "noise_luma": -1,
    },
)

CANON_GENERIC = GradeAdapter(
    id="canon_cr",
    label="Canon CR2/CR3",
    brand="canon",
    deltas={
        "temperature": -1,
        "tint": -1,
        "vibrance": 1,
    },
)

IPHONE_17 = GradeAdapter(
    id="iphone_17_promax",
    label="iPhone 17 Pro / Pro Max ProRAW",
    brand="apple",
    deltas={
        # Computational DNG already tone-mapped — protect highs, tame punch, NR up
        "noise_luma": 10,
        "clarity": -5,
        "sharpen": -6,
        "highlights": -14,
        "shadows": -6,
        "vibrance": -5,
        "saturation": -4,
        "fade": -4,
        "lut_amount": -12,
        "hsl_skin_sat": -3,
    },
)

IPHONE_PRORAW = GradeAdapter(
    id="iphone_proraw",
    label="iPhone ProRAW (generic)",
    brand="apple",
    deltas={
        "noise_luma": 8,
        "clarity": -4,
        "sharpen": -4,
        "highlights": -12,
        "shadows": -4,
        "vibrance": -4,
        "saturation": -3,
        "fade": -3,
        "lut_amount": -10,
    },
)

PHONE_GENERIC = GradeAdapter(
    id="phone_generic",
    label="Phone DNG (generic)",
    brand="apple",
    deltas={
        "noise_luma": 6,
        "clarity": -3,
        "sharpen": -3,
        "highlights": -10,
        "vibrance": -3,
        "lut_amount": -8,
    },
)

CAMERA_GENERIC = GradeAdapter(
    id="camera_generic",
    label="Camera RAW (generic)",
    brand="sony",
    deltas={},
)

_PROFILE_TO_ADAPTER = {
    "sony_a7c_imx410": SONY_A7C,
    "sony_imx410_family": SONY_FF_24,
    "iphone_17_promax": IPHONE_17,
    "iphone_proraw": IPHONE_PRORAW,
    "generic_phone": PHONE_GENERIC,
    "generic_camera": CAMERA_GENERIC,
}

_INT_KEYS = {k for k in SLIDER_KEYS if k != "exposure"}


def detect_look_brand(exif: dict[str, Any] | None = None, *, path: Path | None = None) -> tuple[str, str]:
    """Return (brand, reason) for look-pool selection from EXIF Make/Model."""
    ident = camera_identity(exif)
    make = ident["make"]
    model = ident["model_raw"] or ident["model"]
    suffix = path.suffix.lower() if path else ""

    if "fujifilm" in make or "fuji" in make or suffix == ".raf":
        return "fuji", f"exif:Make={ident['make_raw'] or 'FUJIFILM'}"
    if "nikon" in make or suffix == ".nef":
        return "nikon", f"exif:Make={ident['make_raw'] or 'NIKON'}"
    if "canon" in make or suffix in {".cr2", ".cr3"}:
        return "canon", f"exif:Make={ident['make_raw'] or 'Canon'}"
    if (
        "apple" in make
        or "iphone" in ident["model"]
        or "iphone" in ident.get("host", "")
        or "proraw" in ident["software"]
    ):
        return "apple", f"exif:Make/Model={ident['make_raw']} {model}".strip()
    if "sony" in make or "ilce" in ident["model"] or suffix == ".arw":
        return "sony", f"exif:Make={ident['make_raw'] or 'SONY'} Model={model}".strip()
    if any(k in make for k in ("google", "samsung", "xiaomi", "oneplus", "huawei")):
        return "apple", f"exif:phone Make={ident['make_raw']}"
    if suffix == ".dng":
        return "apple", "exif:DNG → phone look pools"
    if suffix in {".arw", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2", ".pef", ".srw"}:
        return "sony", f"exif:camera RAW ({suffix})"
    return "sony", "fallback:default_sony"


def resolve_grade_adapter(
    path: Path | None = None,
    exif: dict[str, Any] | None = None,
) -> tuple[GradeAdapter, dict[str, str]]:
    """Pick body adapter + brand from EXIF / sensor profile."""
    if exif is None and path is not None:
        try:
            from raw_inspect import exiftool_tags

            exif = exiftool_tags(path)
        except Exception:
            exif = {}
    profile, profile_reason = detect_sensor_profile_with_reason(path, exif)
    brand, brand_reason = detect_look_brand(exif, path=path)
    adapter = _PROFILE_TO_ADAPTER.get(profile.id, CAMERA_GENERIC)

    # Prefer make-specific adapters when profile is generic
    if profile.id == "generic_camera":
        if brand == "fuji":
            adapter = FUJI_GENERIC
        elif brand == "nikon":
            adapter = NIKON_GENERIC
        elif brand == "canon":
            adapter = CANON_GENERIC
        elif brand == "sony":
            adapter = SONY_GENERIC
    elif profile.id == "generic_phone" and brand == "apple":
        adapter = PHONE_GENERIC

    ident = camera_identity(exif)
    meta = {
        "adapter_id": adapter.id,
        "adapter_label": adapter.label,
        "brand": brand,
        "brand_reason": brand_reason,
        "profile_id": profile.id,
        "profile_match": profile_reason,
        "camera_make": ident["make_raw"],
        "camera_model": ident["model_raw"],
    }
    # Look brand always from detect_look_brand (may differ from profile family)
    adapter = GradeAdapter(
        id=adapter.id,
        label=adapter.label,
        brand=brand,
        deltas=adapter.deltas,
    )
    return adapter, meta


def apply_adapter(params: dict[str, Any], adapter: GradeAdapter) -> dict[str, Any]:
    """Return a new params dict with body deltas applied and clamped."""
    out = complete_params(deepcopy(params))
    for key, delta in adapter.deltas.items():
        if key not in SLIDER_KEYS:
            continue
        cur = out.get(key, 0)
        if key == "exposure":
            out[key] = float(cur) + float(delta)
        else:
            out[key] = int(round(float(cur) + float(delta)))

    # Clamps (same spirit as look-recipes "too far")
    out["noise_luma"] = int(max(0, min(40, out["noise_luma"])))
    out["clarity"] = int(max(-20, min(40, out["clarity"])))
    out["sharpen"] = int(max(0, min(50, out["sharpen"])))
    out["fade"] = int(max(0, min(70, out["fade"])))
    out["lut_amount"] = int(max(0, min(100, out["lut_amount"])))
    out["highlights"] = int(max(-100, min(40, out["highlights"])))
    out["shadows"] = int(max(-40, min(60, out["shadows"])))
    out["vibrance"] = int(max(-40, min(40, out["vibrance"])))
    out["saturation"] = int(max(-40, min(40, out["saturation"])))
    out["tint"] = int(max(-50, min(50, out["tint"])))
    out["temperature"] = int(max(-50, min(50, out["temperature"])))
    for k in _INT_KEYS:
        if k in out and k != "exposure":
            out[k] = int(out[k])
    return out


def look_params_for_camera(
    look_name: str,
    path: Path | None = None,
    exif: dict[str, Any] | None = None,
    *,
    looks: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """get_look + body adapter. Returns (params, grade_meta)."""
    adapter, meta = resolve_grade_adapter(path, exif)
    base = get_look(look_name, looks) if looks is not None else get_look(look_name)
    params = apply_adapter(base, adapter)
    meta = dict(meta)
    meta["look"] = look_name
    return params, meta
