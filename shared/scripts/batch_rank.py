#!/usr/bin/env python3
"""Batch / shoot-relative ranking (skill: cull like an editor within one roll)."""

from __future__ import annotations

from typing import Any


def _percentile_ranks(values: list[float]) -> list[float]:
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank, i in enumerate(order):
        ranks[i] = rank / (n - 1)
    return ranks


def apply_batch_relative_ranking(results: list[Any]) -> list[Any]:
    """Annotate shoot percentile / z-score; promote top soft keepers to B.

    Hard `blurry` stays C/A-banned. Soft focus + high latitude + top of roll → B.
    """
    if not results:
        return results

    scores = [float(r.overall_score) for r in results]
    mean = sum(scores) / len(scores)
    var = sum((s - mean) ** 2 for s in scores) / max(len(scores), 1)
    std = var ** 0.5
    pcts = _percentile_ranks(scores)

    for r, pct in zip(results, pcts):
        z = (float(r.overall_score) - mean) / std if std > 1e-6 else 0.0
        d = r.details if isinstance(r.details, dict) else {}
        d["shoot_percentile"] = round(float(pct), 3)
        d["shoot_z"] = round(float(z), 2)
        d["shoot_mean"] = round(mean, 1)
        d["shoot_n"] = len(results)
        r.details = d

        flags = list(r.flags or [])
        hard_blur = "blurry" in flags
        soft = "soft" in flags
        lat = float(d.get("edit_latitude") or r.dynamic_range or 0.0)
        is_phone = (
            str(d.get("sensor_profile") or "").startswith("iphone")
            or str(d.get("sensor_profile") or "") in {"generic_phone"}
            or str(d.get("family") or "").lower() == "phone"
        )
        # Phones: computational DR ceiling is lower — promote with slightly softer lat floor
        lat_floor = 65.0 if is_phone else 70.0
        score_floor = 45.0 if is_phone else 48.0

        # Top quintile of the roll: promote soft / near-miss C → B when latitude OK
        if (
            not hard_blur
            and pct >= 0.80
            and lat >= lat_floor
            and r.tier == "C"
            and (soft or float(r.overall_score) >= score_floor)
        ):
            r.tier = "B"
            if "batch_promoted" not in flags:
                flags.append("batch_promoted")
            d["batch_promotion"] = "C→B (top of shoot + edit latitude)"
            # Mild score lift so table reflects keep-worthiness (capped)
            r.overall_score = round(min(71.9, max(float(r.overall_score), 58.0)), 1)

        # Top decile, no soft/blurry, already B with strong absolute → nudge toward A floor
        lat_a = 70.0 if is_phone else 75.0
        if (
            not hard_blur
            and not soft
            and pct >= 0.90
            and lat >= lat_a
            and r.tier == "B"
            and float(r.overall_score) >= 68.0
        ):
            r.overall_score = round(min(84.9, float(r.overall_score) + 4.0), 1)
            if r.overall_score >= 72.0:
                r.tier = "A"
                d["batch_promotion"] = "B→A (top of shoot)"
                if "batch_promoted" not in flags:
                    flags.append("batch_promoted")

        r.flags = flags

    results.sort(key=lambda x: x.overall_score, reverse=True)
    return results


def build_verdict_reason(result: Any) -> str:
    """One-line editor-facing diagnosis (Chinese)."""
    d = result.details if isinstance(getattr(result, "details", None), dict) else {}
    flags = list(getattr(result, "flags", None) or [])
    parts: list[str] = []

    sharp = float(getattr(result, "sharpness", 0) or d.get("sharp_plane") or 0)
    cut = float(d.get("blur_cut") or 28)
    if "blurry" in flags:
        parts.append(f"焦平面不足（{sharp:.0f}<{cut:.0f}，硬否决）")
    elif "soft" in flags:
        parts.append(f"焦平面偏软（{sharp:.0f}<{cut:.0f}，可进B）")
    else:
        parts.append(f"焦平面尚可（{sharp:.0f}）")

    lat = d.get("edit_latitude")
    if lat is not None:
        lat_f = float(lat)
        if lat_f >= 80:
            parts.append(f"后期空间充足（{lat_f:.0f}）")
        elif lat_f >= 60:
            parts.append(f"后期空间一般（{lat_f:.0f}）")
        else:
            parts.append(f"后期空间偏紧（{lat_f:.0f}）")

    if "underexposed_as_shot" in flags:
        parts.append("直出欠曝可拉")
    if "raw_highlight_clip" in flags:
        parts.append("传感器高光难救")
    if "motion_risk" in flags:
        parts.append("快门偏慢")
    if "batch_promoted" in flags:
        parts.append(d.get("batch_promotion") or "卷内相对提升")

    pct = d.get("shoot_percentile")
    if pct is not None:
        parts.append(f"卷内P{int(float(pct) * 100)}")

    return "；".join(parts)
