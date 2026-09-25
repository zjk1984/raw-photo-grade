"""Comprehensive unit & integration tests for photo evaluation and tiering."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "shared" / "scripts"))

from eval_photo import (
    DEFAULT_WEIGHTS,
    PRESET_WEIGHTS,
    ImageEvaluation,
    PhotoEvaluator,
    format_table,
    get_torch_device,
    organize_files,
    run_eval,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_images(temp_dir):
    """Generate various test images with predictable characteristics."""
    # 1. Sharp, colorful, well-composed image (Expected A or S tier)
    sharp_img = Image.new("RGB", (1000, 800), (45, 55, 75))
    draw = ImageDraw.Draw(sharp_img)
    # High frequency grid
    for x in range(50, 950, 30):
        draw.line([(x, 50), (x, 750)], fill=(220, 180, 50), width=3)
    for y in range(50, 750, 40):
        draw.line([(50, y), (950, y)], fill=(60, 190, 220), width=3)
    # Thirds subject
    draw.ellipse([300, 220, 420, 340], fill=(240, 50, 50), outline=(255, 255, 255), width=4)
    sharp_path = temp_dir / "sharp_keeper.jpg"
    sharp_img.save(sharp_path, quality=95)

    # 2. Blurry, low contrast image (Expected C tier, blurry flag)
    arr_blur = np.full((600, 800, 3), 120, dtype=np.uint8)
    blur_img = Image.fromarray(arr_blur)
    blur_path = temp_dir / "blurry_reject.jpg"
    blur_img.save(blur_path, quality=90)

    # 3. Blown out / clipped highlights image (Expected clipped_highlights flag)
    blown_img = Image.new("RGB", (600, 600), (255, 255, 255))
    draw_blown = ImageDraw.Draw(blown_img)
    draw_blown.rectangle([50, 50, 150, 150], fill=(200, 200, 200))
    blown_path = temp_dir / "blown_highlights.jpg"
    blown_img.save(blown_path, quality=90)

    # 4. Underexposed dark image (Expected underexposed flag)
    dark_img = Image.new("RGB", (600, 600), (8, 8, 12))
    dark_path = temp_dir / "underexposed.jpg"
    dark_img.save(dark_path, quality=90)

    return {
        "sharp": sharp_path,
        "blurry": blur_path,
        "blown": blown_path,
        "dark": dark_path,
    }


def test_device_selection():
    """Verify get_torch_device returns CPU when requested, or MPS/CUDA/CPU appropriately."""
    dev_cpu = get_torch_device("cpu")
    if dev_cpu is None:
        pytest.skip("PyTorch not installed in test environment")
    assert dev_cpu.type == "cpu"

    dev_auto = get_torch_device("auto")
    assert dev_auto is not None
    assert dev_auto.type in {"cpu", "mps", "cuda"}


def test_sharpness_discrimination(sample_images):
    """Verify in-focus sharp image scores dramatically higher than flat blurred image."""
    evaluator = PhotoEvaluator(device_name="auto")

    ev_sharp = evaluator.evaluate(sample_images["sharp"])
    ev_blurry = evaluator.evaluate(sample_images["blurry"])

    assert ev_sharp.sharpness > 70.0
    assert ev_blurry.sharpness < 25.0
    assert "blurry" in ev_blurry.flags
    assert "blurry" not in ev_sharp.flags


def test_dynamic_range_and_clipping(sample_images):
    """Verify blown highlight / underexposure as-shot soft flags (Track C)."""
    evaluator = PhotoEvaluator(device_name="auto")

    ev_blown = evaluator.evaluate(sample_images["blown"])
    assert (
        "preview_highlights" in ev_blown.flags
        or "overexposed_as_shot" in ev_blown.flags
        or "clipped_highlights" in ev_blown.flags
    )
    assert ev_blown.details.get("clipped_highlights_pct", 0) > 80.0 or ev_blown.details.get(
        "as_shot_score", 100
    ) < 40

    ev_dark = evaluator.evaluate(sample_images["dark"])
    assert (
        "underexposed_as_shot" in ev_dark.flags
        or "underexposed" in ev_dark.flags
        or "crushed_shadows" in ev_dark.flags
        or "preview_shadows" in ev_dark.flags
    )


def test_noise_estimation(temp_dir):
    """Verify noise estimator detects clean vs intentionally noisy image."""
    evaluator = PhotoEvaluator(device_name="auto")

    # Clean smooth gradient
    clean_arr = np.linspace(50, 200, 400 * 400, dtype=np.float32).reshape(400, 400)
    clean_rgb = np.stack([clean_arr, clean_arr, clean_arr], axis=-1).astype(np.uint8)
    clean_path = temp_dir / "clean.jpg"
    Image.fromarray(clean_rgb).save(clean_path)

    # Noisy image
    noisy_arr = np.clip(clean_arr + np.random.normal(0, 35, clean_arr.shape), 0, 255).astype(np.uint8)
    noisy_rgb = np.stack([noisy_arr, noisy_arr, noisy_arr], axis=-1)
    noisy_path = temp_dir / "noisy.jpg"
    Image.fromarray(noisy_rgb).save(noisy_path)

    ev_clean = evaluator.evaluate(clean_path)
    ev_noisy = evaluator.evaluate(noisy_path)

    assert ev_clean.noise_control > ev_noisy.noise_control
    assert ev_noisy.details["raw_noise"] > ev_clean.details["raw_noise"]


def test_color_harmony_monochrome_handling(temp_dir):
    """Verify pure black-and-white images do not trigger crash and receive sensible score."""
    evaluator = PhotoEvaluator(device_name="auto")
    gray_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    gray_arr[:, :] = [128, 128, 128]
    bw_path = temp_dir / "bw.jpg"
    Image.fromarray(gray_arr).save(bw_path)

    ev = evaluator.evaluate(bw_path)
    assert 50.0 <= ev.color_harmony <= 70.0
    assert ev.details["mean_saturation"] == 0.0


def test_horizon_tilt_detection(temp_dir):
    """Verify tilted horizon is detected by Hough accumulator."""
    evaluator = PhotoEvaluator(device_name="auto")

    # Image with prominent line rotated by ~5 degrees
    im = Image.new("RGB", (800, 600), (200, 200, 200))
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 300, 800, 600], fill=(50, 70, 90))
    tilted_im = im.rotate(5.0, resample=Image.BICUBIC)
    tilted_path = temp_dir / "tilted.jpg"
    tilted_im.save(tilted_path)

    ev = evaluator.evaluate(tilted_path)
    tilt = abs(ev.details["tilt_angle_deg"])
    assert tilt > 1.5


def test_tier_classification_and_veto(sample_images):
    """Verify S/A/B/C tiering rules and veto constraints."""
    evaluator = PhotoEvaluator(device_name="auto")

    ev_sharp = evaluator.evaluate(sample_images["sharp"])
    assert ev_sharp.tier in {"S", "A"}

    ev_blurry = evaluator.evaluate(sample_images["blurry"])
    assert ev_blurry.tier == "C"

    ev_blown = evaluator.evaluate(sample_images["blown"])
    assert ev_blown.tier in {"B", "C"}
    # Blown highlight should not be tier S
    assert ev_blown.tier != "S"


def test_file_organization(sample_images, temp_dir):
    """Verify organize_files sorts images into S, A, B, C subdirectories."""
    evaluator = PhotoEvaluator(device_name="auto")
    results = [
        evaluator.evaluate(sample_images["sharp"]),
        evaluator.evaluate(sample_images["blurry"]),
    ]

    target_dir = temp_dir / "curated"
    organize_files(results, target_dir, method="copy")

    # The sharp keeper should be in S/ or A/
    sharp_dest = target_dir / results[0].tier / sample_images["sharp"].name
    assert sharp_dest.exists()

    # The blurry reject should be in C/
    blurry_dest = target_dir / "C" / sample_images["blurry"].name
    assert blurry_dest.exists()


def test_table_formatting(sample_images):
    """Verify format_table renders without errors and contains tier counts."""
    evaluator = PhotoEvaluator(device_name="auto")
    results = [evaluator.evaluate(sample_images["sharp"])]
    table_str = format_table(results)
    assert "Tier" in table_str
    assert "Summary:" in table_str
    assert sample_images["sharp"].name in table_str


def test_run_eval_cli_json(sample_images, capsys):
    """Test CLI execution with JSON output."""
    status = run_eval(
        inputs=[str(sample_images["sharp"])],
        output_json=True,
    )
    assert status == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["total_count"] == 1
    assert data["results"][0]["filename"] == "sharp_keeper.jpg"
