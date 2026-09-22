#!/usr/bin/env python3
"""Build and package PhotoGrade M4.app & PhotoGrade-M4-Installer.dmg.

Can run on both macOS (using hdiutil or create-dmg) and Linux (using genisoimage/hfsplus).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIST_DIR = REPO_ROOT / "dist"
APP_DIR = DIST_DIR / "PhotoGrade M4.app"
FINAL_DMG = DIST_DIR / "PhotoGrade-M4-Installer.dmg"


def create_app_bundle():
    print(f"==> 1. 构建 macOS Application Bundle: {APP_DIR}")
    contents = APP_DIR / "Contents"
    macos = contents / "MacOS"
    resources = contents / "Resources"

    shutil.rmtree(APP_DIR, ignore_errors=True)
    macos.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    info_plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>PhotoGradeM4</string>
    <key>CFBundleIdentifier</key>
    <string>com.photograde.m4</string>
    <key>CFBundleName</key>
    <string>PhotoGrade M4</string>
    <key>CFBundleDisplayName</key>
    <string>PhotoGrade M4</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""
    (contents / "Info.plist").write_text(info_plist, encoding="utf-8")

    launcher_script = """#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
RESOURCES="$DIR/../Resources"
APP_ROOT="$(cd "$DIR/../../.." && pwd)"

# Prioritize virtual environments in project or system
if [ -f "$APP_ROOT/.venv/bin/python3" ]; then
    PY="$APP_ROOT/.venv/bin/python3"
elif [ -f "$HOME/.venv/bin/python3" ]; then
    PY="$HOME/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PY="$(command -v python3)"
else
    PY="/usr/bin/python3"
fi

export PYTHONPATH="$RESOURCES:$RESOURCES/shared/scripts:$RESOURCES/photo-eval-grade/scripts:$PYTHONPATH"
exec "$PY" "$RESOURCES/photo-eval-grade/ui/app.py" "$@"
"""
    bin_file = macos / "PhotoGradeM4"
    bin_file.write_text(launcher_script, encoding="utf-8")
    bin_file.chmod(0o755)

    # Copy scripts and resources
    shutil.copytree(REPO_ROOT / "shared", resources / "shared", dirs_exist_ok=True)
    shutil.copytree(REPO_ROOT / "photo-eval-grade", resources / "photo-eval-grade", dirs_exist_ok=True)
    shutil.copy2(REPO_ROOT / "photo-eval-grade" / "ui" / "index.html", resources / "index.html")

    print(f"✓ 成功生成 {APP_DIR.name}")


def create_dmg():
    print(f"==> 2. 生成 macOS DMG 安装文件: {FINAL_DMG}")
    staging_dir = DIST_DIR / "_dmg_staging"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. 复制 .app
    dest_app = staging_dir / "PhotoGrade M4.app"
    shutil.copytree(APP_DIR, dest_app, symlinks=True)

    # 2. 软链接 Applications
    app_link = staging_dir / "Applications"
    if app_link.exists() or app_link.is_symlink():
        app_link.unlink()
    try:
        app_link.symlink_to("/Applications")
    except Exception:
        pass

    FINAL_DMG.unlink(missing_ok=True)

    # 3. 优先检查 macOS 原生 hdiutil
    if shutil.which("hdiutil"):
        print("✓ 使用 macOS 原生 hdiutil 制作高压缩 DMG...")
        temp_dmg = DIST_DIR / "temp.dmg"
        temp_dmg.unlink(missing_ok=True)
        try:
            subprocess.run([
                "hdiutil", "create",
                "-srcfolder", str(staging_dir),
                "-volname", "PhotoGrade M4",
                "-format", "UDRW",
                str(temp_dmg)
            ], check=True)
            subprocess.run([
                "hdiutil", "convert", str(temp_dmg),
                "-format", "UDZO",
                "-imagekey", "zlib-level=9",
                "-o", str(FINAL_DMG)
            ], check=True)
            temp_dmg.unlink(missing_ok=True)
        except Exception as e:
            print(f"hdiutil UDRW/UDZO failed: {e}, falling back to direct create...")
            subprocess.run([
                "hdiutil", "create",
                "-srcfolder", str(staging_dir),
                "-volname", "PhotoGrade M4",
                "-format", "UDZO",
                "-ov",
                str(FINAL_DMG)
            ], check=True)
    elif shutil.which("genisoimage") or shutil.which("mkisofs"):
        tool = shutil.which("genisoimage") or shutil.which("mkisofs")
        print(f"✓ 使用 {tool} (Apple HFS/ISO 格式) 制作标准兼容 DMG...")
        subprocess.run([
            tool,
            "-V", "PhotoGrade M4",
            "-D", "-R", "-apple",
            "-no-pad",
            "-o", str(FINAL_DMG),
            str(staging_dir),
        ], check=True)
    else:
        sys.exit("错误: 未找到可用的 DMG 构建工具 (hdiutil, genisoimage 或 mkisofs)。")

    shutil.rmtree(staging_dir, ignore_errors=True)

    if FINAL_DMG.exists():
        size_mb = round(FINAL_DMG.stat().st_size / (1024 * 1024), 2)
        print("======================================================")
        print(f"✓ DMG 安装文件直接生成成功!")
        print(f"文件位置: {FINAL_DMG}")
        print(f"文件大小: {size_mb} MB")
        print("======================================================")
    else:
        sys.exit("DMG 生成失败。")


def main():
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    create_app_bundle()
    create_dmg()


if __name__ == "__main__":
    main()
