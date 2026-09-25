"""Tests for soft vs blurry, batch ranking, attention∩patch focus, verdict."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from batch_rank import apply_batch_relative_ranking, build_verdict_reason  # noqa: E402
from focus_sharpness import compute_plane_field_sharpness  # noqa: E402


@dataclass
class _FakeEv:
    overall_score: float
    tier: str
    sharpness: float = 50.0
    dynamic_range: float = 80.0
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    filename: str = "x.arw"


def test_attention_intersect_prefers_subject_over_corner():
    """Sharp leaf in corner + soft center attention → plane from attention∩ or attention."""
    h, w = 120, 120
    img = np.full((h, w, 3), 0.4, dtype=np.float32)
    # Soft center
    img[40:80, 40:80] = 0.45
    # Sharp high-contrast corner (should not dominate if attention is center)
    yy, xx = np.mgrid[0:20, 0:20]
    tile = ((xx // 2) + (yy // 2)) % 2
    img[0:20, 0:20] = np.where(tile[..., None] > 0, 0.95, 0.05)
    # Sharpish subject in center
    yy2, xx2 = np.mgrid[50:70, 50:70]
    tile2 = ((xx2 // 3) + (yy2 // 3)) % 2
    img[50:70, 50:70] = np.where(tile2[..., None] > 0, 0.9, 0.1)

    out = compute_plane_field_sharpness(img, subject_center=[0.5, 0.5])
    assert out["focus_pick_mode"] in {"attention_intersect", "attention_window", "top2_wide"}
    cx = (out["focus_patch"]["x0"] + out["focus_patch"]["x1"]) / 2
    cy = (out["focus_patch"]["y0"] + out["focus_patch"]["y1"]) / 2
    # Winning patch should not be the top-left corner leaf
    assert not (cx < 0.25 and cy < 0.25)


def test_batch_promotes_top_soft_with_latitude():
    soft_top = _FakeEv(
        overall_score=52.0,
        tier="C",
        sharpness=20.0,
        flags=["soft"],
        details={"edit_latitude": 86.0, "blur_cut": 24.0},
    )
    hard = _FakeEv(
        overall_score=40.0,
        tier="C",
        sharpness=8.0,
        flags=["blurry"],
        details={"edit_latitude": 90.0, "blur_cut": 24.0},
    )
    mid = _FakeEv(
        overall_score=48.0,
        tier="C",
        sharpness=30.0,
        flags=[],
        details={"edit_latitude": 70.0, "blur_cut": 24.0},
    )
    low = _FakeEv(
        overall_score=35.0,
        tier="C",
        sharpness=25.0,
        flags=["soft"],
        details={"edit_latitude": 55.0, "blur_cut": 24.0},
    )
    # Need ≥5 so soft_top is top quintile (pct >= 0.8)
    extras = [
        _FakeEv(overall_score=45.0 - i, tier="C", details={"edit_latitude": 72.0})
        for i in range(3)
    ]
    out = apply_batch_relative_ranking([soft_top, hard, mid, low, *extras])
    promoted = [r for r in out if "batch_promoted" in (r.flags or [])]
    assert any(r is soft_top or soft_top.tier == "B" for r in out)
    assert soft_top.tier == "B"
    assert "batch_promoted" in soft_top.flags
    assert hard.tier == "C"
    assert "batch_promoted" not in hard.flags
    assert all(isinstance(r.details.get("shoot_percentile"), float) for r in out)


def test_batch_phone_promotes_with_lower_latitude():
    """Phone computational ceiling → promote at lat≥65."""
    soft_top = _FakeEv(
        overall_score=50.0,
        tier="C",
        sharpness=18.0,
        flags=["soft"],
        details={
            "edit_latitude": 66.0,
            "blur_cut": 24.0,
            "sensor_profile": "iphone_17_promax",
            "family": "phone",
        },
    )
    extras = [
        _FakeEv(
            overall_score=44.0 - i,
            tier="C",
            details={"edit_latitude": 60.0, "sensor_profile": "iphone_proraw", "family": "phone"},
        )
        for i in range(4)
    ]
    apply_batch_relative_ranking([soft_top, *extras])
    assert soft_top.tier == "B"
    assert "batch_promoted" in soft_top.flags


def test_verdict_reason_mentions_soft_and_latitude():
    ev = _FakeEv(
        overall_score=58.0,
        tier="B",
        sharpness=20.0,
        flags=["soft", "batch_promoted"],
        details={
            "edit_latitude": 86.0,
            "blur_cut": 24.0,
            "shoot_percentile": 0.9,
            "batch_promotion": "C→B (top of shoot + edit latitude)",
        },
    )
    text = build_verdict_reason(ev)
    assert "偏软" in text or "soft" in text.lower()
    assert "86" in text or "后期" in text
    assert "卷内" in text
