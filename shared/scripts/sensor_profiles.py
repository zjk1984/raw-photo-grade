#!/usr/bin/env python3
"""Camera / sensor profiles for RAW-aware quality evaluation.

Profiles are selected primarily from EXIF Make/Model (and related tags),
not from filename heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SensorProfile:
    id: str
    label: str
    family: str  # camera | phone
    format: str  # full_frame | phone
    megapixels: float
    base_iso: int
    sharp_ref: float
    soft_sharp: float
    blur_sharp: float
    base_dr_ev: float
    noise_ref_base: float
    iso_noise_exp: float
    raw_trust: float
    computational: bool = False


SONY_A7C_IMX410 = SensorProfile(
    id="sony_a7c_imx410",
    label="Sony α7C / IMX410 24.2MP FF",
    family="camera",
    format="full_frame",
    megapixels=24.2,
    base_iso=100,
    sharp_ref=0.22,
    soft_sharp=38.0,
    blur_sharp=24.0,
    base_dr_ev=14.5,
    noise_ref_base=0.008,
    iso_noise_exp=0.85,
    raw_trust=0.85,
    computational=False,
)

# Closely related Sony 24MP FF bodies sharing IMX410-class sensors
SONY_IMX410_FAMILY = SensorProfile(
    id="sony_imx410_family",
    label="Sony IMX410-class 24MP FF (a7 III / a7C II etc.)",
    family="camera",
    format="full_frame",
    megapixels=24.2,
    base_iso=100,
    sharp_ref=0.22,
    soft_sharp=38.0,
    blur_sharp=24.0,
    base_dr_ev=14.5,
    noise_ref_base=0.008,
    iso_noise_exp=0.85,
    raw_trust=0.85,
    computational=False,
)

IPHONE_17_PROMAX = SensorProfile(
    id="iphone_17_promax",
    label="iPhone 17 Pro / Pro Max ProRAW",
    family="phone",
    format="phone",
    megapixels=48.0,
    base_iso=32,
    sharp_ref=0.30,
    soft_sharp=42.0,
    blur_sharp=28.0,
    base_dr_ev=12.0,
    noise_ref_base=0.014,
    iso_noise_exp=1.15,
    raw_trust=0.55,
    computational=True,
)

IPHONE_PRORAW_GENERIC = SensorProfile(
    id="iphone_proraw",
    label="iPhone ProRAW (generic)",
    family="phone",
    format="phone",
    megapixels=48.0,
    base_iso=32,
    sharp_ref=0.28,
    soft_sharp=42.0,
    blur_sharp=28.0,
    base_dr_ev=11.5,
    noise_ref_base=0.016,
    iso_noise_exp=1.2,
    raw_trust=0.5,
    computational=True,
)

GENERIC_CAMERA = SensorProfile(
    id="generic_camera",
    label="Generic camera RAW",
    family="camera",
    format="full_frame",
    megapixels=24.0,
    base_iso=100,
    sharp_ref=0.20,
    soft_sharp=40.0,
    blur_sharp=26.0,
    base_dr_ev=13.0,
    noise_ref_base=0.010,
    iso_noise_exp=0.95,
    raw_trust=0.7,
    computational=False,
)

GENERIC_PHONE = SensorProfile(
    id="generic_phone",
    label="Generic phone DNG",
    family="phone",
    format="phone",
    megapixels=12.0,
    base_iso=50,
    sharp_ref=0.26,
    soft_sharp=42.0,
    blur_sharp=28.0,
    base_dr_ev=11.0,
    noise_ref_base=0.018,
    iso_noise_exp=1.2,
    raw_trust=0.45,
    computational=True,
)

# EXIF Model substrings → profile (checked in order; more specific first)
_SONY_A7C_MODELS = (
    "ilce-7c",
    "ilce7c",
    "a7c",
    "alpha 7c",
    "α7c",
    "ɑ7c",
)
# Other Sony bodies known to use IMX410 / same 24MP BSI gen
_SONY_IMX410_MODELS = (
    "ilce-7m3",
    "ilce-7m3",
    "a7 iii",
    "a7iii",
    "ilce-7c2",
    "a7c ii",
    "a7c2",
    "zv-e1",  # closely related; treat as IMX410-class for noise/DR
)

_IPHONE_17_PRO_MODELS = (
    "iphone 17 pro max",
    "iphone17,2",  # possible Apple hardware string form
    "iphone 17 pro",
    "iphone17,1",
)


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def _compact(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


def camera_identity(exif: dict[str, Any] | None) -> dict[str, str]:
    """Normalize Make / Model fields commonly written by cameras and phones."""
    exif = dict(exif or {})
    make = (
        exif.get("Make")
        or exif.get("make")
        or exif.get("Manufacturer")
        or ""
    )
    model = (
        exif.get("Model")
        or exif.get("model")
        or exif.get("UniqueCameraModel")
        or exif.get("CameraModelName")
        or exif.get("Camera Model Name")
        or ""
    )
    # Apple sometimes puts device name in LensModel / HostComputer for DNG
    lens = exif.get("LensModel") or exif.get("Lens") or ""
    soft = exif.get("Software") or ""
    unique = exif.get("UniqueCameraModel") or ""
    host = exif.get("HostComputer") or ""
    # Prefer Model; fall back to UniqueCameraModel / HostComputer (common on ProRAW)
    model_raw = (
        str(model).strip()
        or str(unique).strip()
        or str(host).strip()
    )
    return {
        "make": _norm(make),
        "model": _norm(model) or _norm(unique) or _norm(host),
        "unique_model": _norm(unique),
        "host": _norm(host),
        "lens": _norm(lens),
        "software": _norm(soft),
        "make_raw": str(make).strip(),
        "model_raw": model_raw,
    }


def _model_blob(ident: dict[str, str]) -> str:
    return " ".join(
        [
            ident["make"],
            ident["model"],
            ident["unique_model"],
            ident.get("host", ""),
            ident["lens"],
        ]
    )


def detect_sensor_profile(
    path: Path | None = None,
    exif: dict[str, Any] | None = None,
    *,
    prefer: str | None = None,
) -> SensorProfile:
    """Pick a sensor profile. Prefer EXIF Make/Model; `prefer` only if set and not 'auto'."""
    profile, _reason = detect_sensor_profile_with_reason(path, exif, prefer=prefer)
    return profile


def detect_sensor_profile_with_reason(
    path: Path | None = None,
    exif: dict[str, Any] | None = None,
    *,
    prefer: str | None = None,
) -> tuple[SensorProfile, str]:
    """Return (profile, reason) explaining which EXIF cue matched."""
    if prefer and prefer not in {"auto", ""}:
        forced = {
            "sony_a7c_imx410": SONY_A7C_IMX410,
            "sony_imx410_family": SONY_IMX410_FAMILY,
            "iphone_17_promax": IPHONE_17_PROMAX,
            "iphone17": IPHONE_17_PROMAX,
            "iphone_proraw": IPHONE_PRORAW_GENERIC,
        }.get(prefer)
        if forced:
            return forced, f"forced:{prefer}"

    ident = camera_identity(exif)
    blob = _model_blob(ident)
    compact = _compact(blob)
    suffix = path.suffix.lower() if path else ""

    # --- Apple iPhone (EXIF Make=Apple, Model=iPhone …) ---
    is_apple = "apple" in ident["make"] or "iphone" in blob or "iphone" in ident["software"]
    if is_apple:
        for token in _IPHONE_17_PRO_MODELS:
            if token in blob or _compact(token) in compact:
                which = "Pro Max" if "max" in token or "17,2" in token else "Pro"
                return (
                    IPHONE_17_PROMAX,
                    f"exif:Make/Model matched iPhone 17 {which} ({ident['model_raw'] or ident['model']})",
                )
        if "iphone" in blob or "proraw" in ident["software"] or "halide" in ident["software"]:
            return (
                IPHONE_PRORAW_GENERIC,
                f"exif:Apple/iPhone ProRAW ({ident['model_raw'] or 'iPhone'})",
            )
        return GENERIC_PHONE, f"exif:Apple device ({ident['model_raw']})"

    # --- Sony ILCE / α bodies ---
    is_sony = (
        "sony" in ident["make"]
        or "ilce" in blob
        or "ilce" in compact
        or ident["make"].startswith("sony")
    )
    if is_sony:
        model = ident["model"] or ident["unique_model"]
        model_c = _compact(model)
        # α7C II first (must not map to α7C)
        if any(
            k in model or k in model_c
            for k in ("ilce-7c2", "ilce7c2", "a7c ii", "a7cii", "a7c2")
        ):
            return (
                SONY_IMX410_FAMILY,
                f"exif:Model={ident['model_raw']} → IMX410-class (α7C II)",
            )
        # Exact α7C (ILCE-7C) — exclude *7c2 / *7cii
        is_a7c = (
            ("ilce-7c" in model and "ilce-7c2" not in model)
            or (model_c == "ilce7c")
            or model in _SONY_A7C_MODELS
            or (model_c in {"a7c", "alpha7c"} and "7c2" not in model_c)
        )
        if is_a7c:
            return SONY_A7C_IMX410, f"exif:Model={ident['model_raw']} → α7C/IMX410"

        for token in _SONY_IMX410_MODELS:
            if token in blob or _compact(token) in compact:
                return (
                    SONY_IMX410_FAMILY,
                    f"exif:Model={ident['model_raw']} → IMX410-class",
                )

        return (
            GENERIC_CAMERA,
            f"exif:Sony body ({ident['model_raw'] or 'unknown model'}) — generic camera profile",
        )

    # --- Other phones writing DNG ---
    if any(k in ident["make"] for k in ("google", "samsung", "xiaomi", "oneplus", "huawei")):
        return GENERIC_PHONE, f"exif:phone Make={ident['make_raw']}"

    if suffix == ".dng":
        if "apple" in ident["software"] or "proraw" in ident["software"]:
            return IPHONE_PRORAW_GENERIC, "exif:Software hints Apple ProRAW"
        return GENERIC_CAMERA, "exif:DNG without Apple/Sony identity"

    if suffix in {".arw", ".nef", ".cr2", ".cr3", ".raf", ".orf", ".rw2", ".pef", ".srw"}:
        brand = ident["make_raw"] or suffix.upper()
        return GENERIC_CAMERA, f"exif:camera RAW ({brand} {ident['model_raw']})".strip()

    if ident["make"] or ident["model"]:
        return GENERIC_CAMERA, f"exif:unclassified ({ident['make_raw']} {ident['model_raw']})"

    return GENERIC_CAMERA, "fallback:no_exif"


def expected_noise(profile: SensorProfile, iso: float) -> float:
    """Expected Laplacian MAD at this ISO for a 'normal' exposure."""
    iso = max(float(iso or profile.base_iso), 1.0)
    ratio = iso / float(profile.base_iso)
    return float(profile.noise_ref_base) * (ratio ** float(profile.iso_noise_exp))
