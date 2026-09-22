# raw-photo-grade

Three Claude Code skills that evaluate and develop RAW photos like a photographer would: evaluate technical & aesthetic quality, cull and tier into S/A/B/C using local GPU acceleration, inspect the file, make a preview, actually look at it, then grade, crop, and export. Not a one-shot filter.

- **[photo-eval-grade](photo-eval-grade/)** — Batch quality evaluation & S/A/B/C tiering. Accelerated by Apple Silicon M4 / Metal GPU (MPS). Multi-dimensional scoring across sharpness (Tenengrad), dynamic range & exposure, noise control, color harmony, and composition.
- **[phone-dng-grade](phone-dng-grade/)** — iPhone ProRAW, Google Pixel RAW+, Samsung Expert RAW DNG. Tuned for computational/linear DNG: baked-in tone-mapping, small-sensor noise, gain maps.
- **[camera-raw-grade](camera-raw-grade/)** — DSLR/mirrorless RAW: Nikon NEF, Canon CR2/CR3, Sony ARW, Fujifilm RAF, Olympus/OM ORF, Panasonic RW2, Pentax PEF, Leica/generic DNG. Tuned for real sensor headroom and brand color science.

All skills share one core RAW-processing and evaluation engine (`shared/`) so the math is written and tested once. Each skill's own `SKILL.md` and `references/` stay focused on what's actually different for that camera type or workflow — see [writing-skills SDO guidance](https://github.com/obra/superpowers) on why the docs don't repeat themselves.

## Install

Copy the skill folders you want **and** `shared/` as siblings into your skills directory. `shared/` is not a skill itself; it's imported by the skill scripts via a relative path, so it must sit next to them.

```bash
# personal skills (Claude Code)
cp -r photo-eval-grade phone-dng-grade camera-raw-grade shared ~/.claude/skills/

# or project-local
cp -r photo-eval-grade phone-dng-grade camera-raw-grade shared /path/to/project/.claude/skills/
```

Only need evaluation/culling plus camera RAW?
```bash
cp -r photo-eval-grade camera-raw-grade shared ~/.claude/skills/
```

Then install the Python dependencies:

```bash
# For evaluation and grading (with PyTorch MPS acceleration on Apple Silicon M4):
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # On macOS, pip3 install torch directly includes MPS
pip3 install -r photo-eval-grade/requirements.txt
```

Needs `rawpy` (LibRaw binding), `numpy`, `Pillow`, `torch`, and libraw itself on the system (`brew install libraw exiftool` / `apt install libraw-dev`). Canon CR3 needs a reasonably recent libraw.

## Usage

Once installed, mention what you're working with and Claude Code should pick up the right skill on its own — "evaluate this shoot on my M4 Mac and pick the S/A keepers", "grade this iPhone ProRAW", or "develop these NEF files from my D850 shoot." You can also invoke scripts directly:

```bash
# 1. Evaluate & rank batch into S/A/B/C tiers using M4 Metal/MPS GPU:
python3 photo-eval-grade/scripts/eval.py /path/to/shoot --device mps
python3 photo-eval-grade/scripts/eval.py /path/to/shoot --filter S,A --organize ./selected

# 2. Inspect individual RAW files:
python3 phone-dng-grade/scripts/inspect_dng.py photo.dng
python3 camera-raw-grade/scripts/inspect_raw.py photo.nef

# 3. Develop top keepers:
python3 camera-raw-grade/scripts/develop.py ./selected/S/ --out-dir ./edited --look natural --preview

# 4. Or run the unified automated pipeline (Eval -> Filter -> Develop -> Crop):
python3 photo-eval-grade/scripts/pipeline.py /path/to/shoot --tiers S,A --look natural --out-dir ./edited --straighten

# 5. Launch the Visual Mac Desktop Application (GUI):
python3 photo-eval-grade/ui/app.py
# or: ./photo-eval-grade/ui/launch_app.sh
```

See each skill's `SKILL.md` for the full workflow.

## Repo layout

```
photo-eval-grade/  skill: M4 GPU-accelerated quality evaluation & S/A/B/C tiering
phone-dng-grade/   skill: phone/computational DNG
camera-raw-grade/  skill: DSLR/mirrorless RAW
shared/             evaluation + RAW decode + grade engine, imported by all skills
tests/              unit and integration test suite
```

## License

MIT — see [LICENSE](LICENSE).
