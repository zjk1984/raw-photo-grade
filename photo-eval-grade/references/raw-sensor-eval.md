# Sensor-aware RAW evaluation

`eval_photo.py` can blend **sensor-level RAW metrics** with the RGB preview scores.

## Profiles

| Profile id | Body / sensor | Notes |
| --- | --- | --- |
| `sony_a7c_imx410` | Sony α7C · IMX410 24.2MP FF BSI | Matched when EXIF `Model=ILCE-7C`. |
| `sony_imx410_family` | a7 III / α7C II / ZV-E1 class | Same noise/DR curve; not exact α7C. |
| `iphone_17_promax` | iPhone 17 Pro / Pro Max ProRAW | EXIF `Model=iPhone 17 Pro Max` etc. |
| `iphone_proraw` | Other iPhone ProRAW | Generic Apple/iPhone DNG. |
| `generic_camera` / `generic_phone` | Fallback | Unknown Make/Model or other brands. |

## Extra signals (vs plain JPEG eval)

1. **Highlight headroom** from mosaic levels vs `black_level` / `white_level` (stops below clip).
2. **Focus** on a **16-bit linear demosaic center crop** (not half-size 8-bit preview).
3. **Noise** vs profile-expected MAD at the file’s ISO (FF vs phone curves differ).
4. Profile-specific **blur soft/hard cutoffs** before S/A veto.

## How the camera is chosen

Profile selection is **EXIF-first**:

1. Read `Make`, `Model`, `UniqueCameraModel`, `CameraModelName`, `LensModel`, `Software` (exiftool, else Pillow).
2. Match rules (examples):
   - `Make=SONY` + `Model=ILCE-7C` → `sony_a7c_imx410`
   - `Make=SONY` + `Model=ILCE-7C2` → `sony_imx410_family` (not α7C)
   - `Make=Apple` + `Model=iPhone 17 Pro Max` → `iphone_17_promax`
   - Other iPhone → `iphone_proraw`
3. JSON details include `camera_make`, `camera_model`, `profile_match` (human-readable reason).

Do **not** rely on filename (`DSC*.ARW`). Force only with `--sensor-profile`.

## CLI

```bash
# Auto-detect from EXIF (α7C ARW → sony_a7c_imx410, etc.)
python3 camera-raw-grade/scripts/eval_raw.py /path/to/ARWs --device auto

# iPhone ProRAW — profile from Make/Model
python3 phone-dng-grade/scripts/eval_dng.py ./DCIM --device auto

# Force a profile
python3 photo-eval-grade/scripts/eval.py ./RAWS --raw-aware --sensor-profile sony_a7c_imx410 --json
```

Use `--no-raw-aware` for preview-only scoring.
