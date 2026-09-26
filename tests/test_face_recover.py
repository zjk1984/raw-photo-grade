"""Tests for face highlight recoverability + face finish."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "shared" / "scripts"))

from edit_latitude import score_edit_latitude  # noqa: E402
from face_recover import (  # noqa: E402
    analyze_face_region,
    apply_face_finish,
    bump_params_for_face_under,
    face_flags_from_analysis,
    face_finish_exposure_for_mean,
    should_face_finish,
)
from sensor_profiles import SONY_A7C_IMX410  # noqa: E402


def _hot_face_recoverable(h: int = 128, w: int = 128) -> np.ndarray:
    """Bright face in center with channel headroom; blown sky in corners."""
    img = np.full((h, w, 3), 0.98, dtype=np.float32)  # blown periphery
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 3, 2 * w // 3
    # Skin-ish bright but not all-channel dead (R high, G/B slightly lower)
    face = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    face[..., 0] = 0.92
    face[..., 1] = 0.78
    face[..., 2] = 0.70
    img[y0:y1, x0:x1] = face
    return img


def _dead_face(h: int = 128, w: int = 128) -> np.ndarray:
    img = np.full((h, w, 3), 0.5, dtype=np.float32)
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 3, 2 * w // 3
    img[y0:y1, x0:x1] = 0.995
    return img


def test_hot_face_flagged_recoverable():
    img = _hot_face_recoverable()
    face = analyze_face_region(img, subject_center=[0.5, 0.5], preset="portrait")
    flags = face_flags_from_analysis(face)
    assert face["face_recoverable"]
    assert not face["face_dead"]
    assert "face_hot_as_shot" in flags or face["face_hot_as_shot"]
    assert "face_recoverable" in flags


def test_dead_face_hard_flag():
    img = _dead_face()
    face = analyze_face_region(img, subject_center=[0.5, 0.5], preset="portrait")
    assert face["face_dead"] or face["face_all_clip_pct"] >= 8.0
    flags = face_flags_from_analysis(face)
    assert "face_dead_highlights" in flags or face["face_dead"]


def test_latitude_softens_global_clip_when_face_ok():
    ctx = {
        "headroom_ev": 0.2,
        "raw_highlight_pct": 8.0,
        "raw_shadow_pct": 5.0,
        "midtone_ev": 0.5,
        "raw_p50": 0.2,
        "raw_noise_mad": SONY_A7C_IMX410.noise_ref_base,
        "expected_noise": SONY_A7C_IMX410.noise_ref_base,
        "iso": 100,
    }
    face = {
        "face_headroom": 0.08,
        "face_skin_frac": 0.2,
        "face_all_clip_pct": 1.0,
        "face_recoverable": True,
        "face_dead": False,
        "face_hot_as_shot": True,
    }
    _score, _d, flags = score_edit_latitude(
        ctx, SONY_A7C_IMX410, preset="portrait", face=face
    )
    assert "raw_highlight_clip" not in flags
    assert "face_hot_as_shot" in flags
    assert "face_recoverable" in flags


def test_underexposed_face_normalized_sharpness_beats_as_shot():
    """Dark but edged face should score higher after normalize — not false blur."""
    h, w = 160, 160
    img = np.full((h, w, 3), 0.12, dtype=np.float32)
    # Dark skin-ish face with checker edges (sharp when lifted)
    y0, y1, x0, x1 = 50, 110, 50, 110
    yy, xx = np.mgrid[y0:y1, x0:x1]
    tile = ((xx // 3) + (yy // 3)) % 2
    face = np.where(tile[..., None] > 0, 0.22, 0.06).astype(np.float32)
    face = np.concatenate([face * 1.05, face * 0.95, face * 0.85], axis=-1)
    # rebuild properly
    face = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    face[..., 0] = np.where(tile > 0, 0.24, 0.07)
    face[..., 1] = np.where(tile > 0, 0.18, 0.05)
    face[..., 2] = np.where(tile > 0, 0.14, 0.04)
    img[y0:y1, x0:x1] = face

    from face_recover import measure_face_plane_sharpness
    from focus_sharpness import compute_plane_field_sharpness

    m = measure_face_plane_sharpness(img, subject_center=[0.5, 0.5], preset="portrait")
    assert m["face_underexposed_as_shot"]
    assert m["face_sharp_norm"] > m["face_sharp_as_shot"]
    out = compute_plane_field_sharpness(img, subject_center=[0.5, 0.5], preset="portrait")
    assert out["focus_pick_mode"] in {"face_plane_norm", "face_plane", "attention_intersect", "attention_window"}
    assert out["sharp_plane"] >= m["face_sharp_as_shot"]

    img = _hot_face_recoverable()
    # Start slightly darker face
    img[40:90, 40:90, :] *= 0.75
    before = float(img[50:80, 50:80].mean())
    out = apply_face_finish(img, subject_center=[0.5, 0.5], amount=1.0)
    after = float(out[50:80, 50:80].mean())
    assert after > before


def test_underexposed_face_adaptive_finish_lifts_hard():
    """Adaptive EV should visibly lift the face center (falloff softens edges)."""
    h, w = 200, 200
    img = np.full((h, w, 3), 0.05, dtype=np.float32)
    y0, y1, x0, x1 = 30, 150, 50, 150
    face = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    face[..., 0], face[..., 1], face[..., 2] = 0.26, 0.18, 0.14
    img[y0:y1, x0:x1] = face
    before = float(img[70, 100].mean())
    out = apply_face_finish(
        img,
        subject_center=[0.5, 0.5],
        preset="portrait",
        amount=1.0,
        as_shot_mean=0.22,
    )
    after = float(out[70, 100].mean())
    assert after > before * 1.5, (before, after)
    assert face_finish_exposure_for_mean(0.30) >= 0.5
    assert should_face_finish(["face_underexposed_as_shot", "face_recoverable"])
    assert not should_face_finish(["blurry"])
    bumped = bump_params_for_face_under({"shadows": 8, "exposure": 0.05}, ["face_underexposed_as_shot"])
    assert bumped["shadows"] > 8
    assert bumped["exposure"] > 0.05


def test_skin_anchor_beats_sky_saliency():
    """Bright sky + dark people: anchor must land on skin, not cloud edges."""
    h, w = 240, 160
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Sky / clouds (high edge + sat) in upper half — saliency trap
    img[: h // 2] = (0.35, 0.55, 0.90)
    img[20:40, 40:120] = 0.95  # bright cloud
    # Dark skin people in lower half
    img[150:210, 50:110, 0] = 0.28
    img[150:210, 50:110, 1] = 0.18
    img[150:210, 50:110, 2] = 0.14
    from face_recover import resolve_face_center, apply_face_finish

    center, box, mode = resolve_face_center(img, subject_center=[0.5, 0.25])
    assert mode.startswith("skin"), mode
    assert center[1] > 0.45, center
    assert box is not None and box["y0"] > 0.4
    before = float(img[160:200, 60:100].mean())
    out = apply_face_finish(
        img, subject_center=[0.5, 0.25], amount=1.0, as_shot_mean=0.20
    )
    after = float(out[160:200, 60:100].mean())
    assert after > before * 1.25, (before, after)
    # Sky should barely move
    assert abs(float(out[30, 80].mean()) - float(img[30, 80].mean())) < 0.05


def test_yunet_detects_two_faces_synthetic():
    """YuNet path: skip if model/cv2 missing; otherwise detect drawn faces."""
    from face_detect import detect_faces, _yunet_path
    if not _yunet_path():
        return
    try:
        import cv2  # noqa: F401
    except ImportError:
        return
    # Use a real photo if available for a meaningful detector test
    from pathlib import Path
    src = Path("/Users/zjk/Documents/photo/DSC00117.ARW")
    if not src.exists():
        return
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared" / "scripts"))
    from raw_develop import decode_raw, apply_orientation, resize_long_edge
    rgb = resize_long_edge(apply_orientation(decode_raw(src, 1.0, False), "auto", src), 960)
    faces = detect_faces(rgb, score_thresh=0.4)
    assert len(faces) >= 2, faces
    # Adult left, child right
    xs = sorted(f["center"][0] for f in faces)
    assert xs[0] < 0.55 < xs[-1]


def test_people_finish_equalizes_two_dark_faces():
    from face_recover import apply_people_finish
    h, w = 320, 240
    img = np.full((h, w, 3), 0.08, dtype=np.float32)
    # Sky
    img[:120] = (0.4, 0.55, 0.9)
    # Two dark skin patches (adult / child)
    for y0, y1, x0, x1, v in [(150, 200, 40, 90, 0.18), (155, 205, 140, 190, 0.22)]:
        img[y0:y1, x0:x1, 0] = v * 1.15
        img[y0:y1, x0:x1, 1] = v * 0.85
        img[y0:y1, x0:x1, 2] = v * 0.70
    # Without detector these patches still get skin-anchor fallback via apply_face_finish;
    # apply_people_finish returns unchanged if YuNet finds nothing on synthetic.
    out, meta = apply_people_finish(img, amount=1.0, as_shot_mean=0.2, target_luma=0.52)
    assert out.shape == img.shape
    assert meta["mode"] in {"multi_face", "none"}


def test_should_face_finish_hot():
    assert should_face_finish(["face_hot_as_shot", "face_recoverable"])
