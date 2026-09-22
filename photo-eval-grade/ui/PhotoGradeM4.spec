# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build specification for PhotoGrade M4.app on macOS
Build command on Mac:
    pyinstaller photo-eval-grade/ui/PhotoGradeM4.spec
"""

import sys
from pathlib import Path

block_cipher = None

SPEC_ROOT = Path(SPECPATH).resolve()
REPO_ROOT = SPEC_ROOT.parent.parent

added_files = [
    (str(SPEC_ROOT / "index.html"), "ui"),
    (str(REPO_ROOT / "photo-eval-grade" / "references"), "photo-eval-grade/references"),
    (str(REPO_ROOT / "shared" / "scripts"), "shared/scripts"),
]

a = Analysis(
    ['app.py'],
    pathex=[
        str(SPEC_ROOT),
        str(REPO_ROOT / "shared" / "scripts"),
        str(REPO_ROOT / "photo-eval-grade" / "scripts"),
    ],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'torch',
        'rawpy',
        'PIL',
        'numpy',
        'webview',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhotoGradeM4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='PhotoGradeM4',
)

app = BUNDLE(
    coll,
    name='PhotoGrade M4.app',
    icon=None,
    bundle_identifier='com.photograde.m4',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'NSHighResolutionCapable': True,
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleDisplayName': 'PhotoGrade M4',
    },
)
