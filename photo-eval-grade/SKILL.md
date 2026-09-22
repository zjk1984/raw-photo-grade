---
name: photo-eval-grade
description: Evaluate RAW, DNG, and JPEG photo quality using Apple Silicon M4 / GPU (MPS) acceleration. Scores photos across sharpness, dynamic range, noise, color harmony, and composition, classifies into S/A/B/C tiers, and filters top keepers for RAW development and editing. Use when the user mentions photo culling, scoring, selecting best photos, grading photo quality, S/A/B/C classification, or running evaluation on Mac M4.
license: MIT
compatibility: Requires Python 3.10+ with torch, rawpy, numpy, pillow. Metal/MPS acceleration on Apple Silicon (M4/M3/M2/M1); falls back to CUDA or CPU.
metadata:
  version: "1.0"
  type: workflow
---

# Photo Quality Evaluation & S/A/B/C Tiering

Evaluate photo batches like a seasoned photo editor — assess sharpness, exposure headroom, noise control, color harmony, and composition using local M4 GPU (MPS) acceleration.

Scores every image from 0 to 100, flags technical defects (missed focus, blown sky, clipped shadows, horizon tilt), and classifies images into **S / A / B / C** tiers.

Scripts live next to this file, plus the shared engine in `shared/scripts/eval_photo.py`.

```bash
SKILL_DIR="${SKILL_DIR:-$(cd "$(dirname "$0")" && pwd)}"
python3 "$SKILL_DIR/scripts/eval.py" path/to/photos/ --device mps
python3 "$SKILL_DIR/scripts/eval.py" path/to/photos/ --filter S,A --organize ./selected
python3 "$SKILL_DIR/ui/app.py"  # 启动本地可视化桌面应用 (Mac App 模式)
```

## Hard rules

1. **Focus is King**: Photos with missed focus or camera shake (\(S_{\text{sharp}} < 28\)) receive a hard veto (`blurry` flag) and cannot rank into S or A tier, regardless of how nice the colors are.
2. **Never delete originals automatically**: C-tier rejects are identified and organized, but never permanently deleted without explicit user instruction.
3. **Hardware Acceleration**: Default to `--device auto` (automatically selects `mps` on Apple Silicon M4, `cuda` on NVIDIA, or `cpu`).
4. **Fast First Pass**: Uses half-size RAW decoding for initial scoring to process 40–80 photos/minute on M4, reserving full-resolution decoding for the actual development phase.
5. **Feed Top Keepers to Develop**: Once S and A tiers are selected, feed them directly into `camera-raw-grade` or `phone-dng-grade` for color grading and cropping.

---

## Workflow

### 1. Evaluate & Score Batch

Point at a directory containing RAW files (`.CR3`, `.NEF`, `.ARW`, `.DNG`, etc.) or JPEGs:

```bash
python3 "$SKILL_DIR/scripts/eval.py" ./DCIM/ --preset general
```

For specific genres, select a tuned evaluation preset:
- `--preset general`: Balanced across all technical and artistic metrics.
- `--preset landscape`: Prioritizes dynamic range headroom, horizon flatness, and edge detail.
- `--preset portrait`: Emphasizes subject eye sharpness and clean skin tones; softer noise penalty.
- `--preset street`: Prioritizes decisive composition and contrast; tolerates higher ISO noise.
- `--preset night`: Strict noise control and highlight clipping protection on light sources.

### 2. Inspect Tiering Output

Review the colorized terminal table:
```
----------------------------------------------------------------------------------
Tier    Score  Sharp  DynRng  Noise  Color  Comp   Flags                  Filename
----------------------------------------------------------------------------------
[ S ]   89.4   94.2   88.5    92.0   85.6   87.1   ok                     DSC_0128.NEF
[ A ]   79.1   84.0   76.2    89.3   78.0   82.4   ok                     DSC_0130.NEF
[ B ]   68.5   65.2   70.1    82.0   64.3   71.0   tilted_horizon         DSC_0135.NEF
[ C ]   38.2   12.0   68.4    78.0   55.0   60.2   blurry                 DSC_0142.NEF
----------------------------------------------------------------------------------
Summary: 4 photos | S: 1 (25.0%) | A: 1 (25.0%) | B: 1 (25.0%) | C: 1 (25.0%)
```

### 3. Filter & Curate

Organize files automatically into subdirectories (`selected/S/`, `selected/A/`, etc.) without moving originals:

```bash
# Non-destructive symlinks:
python3 "$SKILL_DIR/scripts/eval.py" ./DCIM/ --organize ./curated --organize-method symlink

# Or output clean JSON for scripts / automation:
python3 "$SKILL_DIR/scripts/eval.py" ./DCIM/ --json --filter S,A > keepers.json
```

### 3.1 Visual Desktop Mac App (GUI)

Launch the native macOS-styled interactive desktop interface to drag-and-drop photos, view real-time M4 GPU compute status, inspect S/A/B/C cards, and export keepers with one click:

```bash
# Launch GUI (Native window via pywebview or browser)
python3 "$SKILL_DIR/ui/app.py"
# Or run launcher script
"$SKILL_DIR/ui/launch_app.sh"
```

### 4. Direct Pipeline to Develop & Crop

Feed the curated S and A keepers into the sibling development skills:

```bash
# Develop camera RAW keepers using camera-raw-grade:
python3 camera-raw-grade/scripts/develop.py ./curated/S/ --out-dir ./edited --look natural --preview

# Straighten and crop hero shot:
python3 shared/scripts/crop.py ./edited/DSC_0128_preview.jpg -o ./edited/DSC_0128_crop.jpg --aspect 3:2 --straighten --horizon
```

---

## Tier Definitions & Action Guide

- **S-Tier (Score \(\ge 85\))**: Flawless focus, clean shadows, great dynamic range. Prioritize for full-res export and portfolio.
- **A-Tier (Score \(72 \sim 84.9\))**: High quality keepers. Great for client delivery after standard grading.
- **B-Tier (Score \(58 \sim 71.9\))**: Usable second choices. May need localized shadow recovery, noise reduction, or cropping.
- **C-Tier (Score \(< 58\) or Critical Defect)**: Out-of-focus, severe camera shake, heavily clipped highlights. Discard or archive.
