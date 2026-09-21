"""Integration test for the photo evaluation pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import sys
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "photo-eval-grade" / "scripts"))
import pipeline


@pytest.fixture
def temp_workspace():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_pipeline_integration(temp_workspace, monkeypatch):
    """Test full automated pipeline execution on sample input images."""
    in_dir = temp_workspace / "input"
    in_dir.mkdir()
    out_dir = temp_workspace / "output"

    # Create a sharp photo
    img1 = Image.new("RGB", (800, 600), (60, 80, 100))
    d1 = ImageDraw.Draw(img1)
    for x in range(50, 750, 30):
        d1.line([(x, 50), (x, 550)], fill=(240, 200, 50), width=4)
    img1.save(in_dir / "keeper.jpg")

    # Create a blurry photo
    img2 = Image.new("RGB", (800, 600), (128, 128, 128))
    img2.save(in_dir / "blurry.jpg")

    # Mock CLI arguments
    test_args = [
        "pipeline.py",
        str(in_dir),
        "--tiers", "S,A",
        "--look", "natural",
        "--out-dir", str(out_dir),
        "--preview",
        "--straighten",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    exit_code = pipeline.main()
    assert exit_code == 0

    # Verify output directory contents
    manifest_file = out_dir / "pipeline_manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["total_evaluated"] == 2
    assert manifest["total_developed"] == 1
    assert manifest["developed"][0]["tier"] in {"S", "A"}
    assert Path(manifest["developed"][0]["output"]).exists()
