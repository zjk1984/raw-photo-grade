"""Verify DMG and App bundle generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_make_dmg_creates_installer(monkeypatch):
    """Test make_dmg.py runs and creates both PhotoGrade M4.app and PhotoGrade-M4-Installer.dmg."""
    script = _REPO_ROOT / "photo-eval-grade" / "ui" / "make_dmg.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0

    dist_dir = _REPO_ROOT / "dist"
    app_bundle = dist_dir / "PhotoGrade M4.app"
    dmg_file = dist_dir / "PhotoGrade-M4-Installer.dmg"

    assert app_bundle.exists()
    assert (app_bundle / "Contents" / "Info.plist").exists()
    assert (app_bundle / "Contents" / "MacOS" / "PhotoGradeM4").exists()

    assert dmg_file.exists()
    assert dmg_file.stat().st_size > 100000  # greater than 100KB
