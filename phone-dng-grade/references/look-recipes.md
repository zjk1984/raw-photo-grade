# Look recipes

Named looks are starting points. After the first preview, change 2–4 sliders, not all of them.

## Slider ranges (develop.py)

| slider | typical | too far |
| --- | --- | --- |
| exposure | −0.4 … +0.6 EV | +1 on a night phone file |
| contrast | −10 … +20 | +35 (crunchy, halo) |
| highlights | −40 … +10 | +25 on ProRAW skies |
| shadows | 0 … +25 | +40 (noise + HDR look) |
| whites | −15 … +10 | |
| blacks | −20 … +8 | |
| temperature | −20 … +20 | ±40 unless mixed neon |
| tint | −8 … +8 | |
| vibrance | 0 … +20 | |
| saturation | −10 … +10 | +20 on skin |
| clarity | 0 … +14 landscape, 0 … +6 portrait | +25 |
| vignette | 0 … −12 | −25 |
| sharpen | 8 … 22 | 40 |
| noise_luma | 0 … 12 daylight, 16–28 night | 40 (plastic) |
| fade / hsl_* / lut | see camera-raw-grade look-recipes | Sony Creative Look presets shared |

Phone defaults keep slightly higher `noise_luma` and softer `clarity`. At develop time,
EXIF body adapters add more (iPhone 17 Pro Max: stronger NR, highlight protect, lower LUT).
Pipeline `--look auto` picks **apple** look pools from Make/Model; force with `--brand apple`.

Eval → Look feedback (same as camera path): high `edit_latitude` + `underexposed_as_shot` biases NT/Flat/Eterna-class looks; `shallow_dof` + skin cues bias Astia/PT/portrait pools; **`face_hot_as_shot` / `face_underexposed_as_shot` + recoverable** biases highlight-protect or shadow-lift looks and pipeline auto face-finish (local adaptive lift after the look).

Units: exposure is EV. Everything else is roughly −100…+100 like Lightroom, but the implementation is simpler — treat numbers as taste, not as a Lightroom match.

## Develop order (Lightroom-aligned)

Pipeline `develop_lr_stack` follows LR edit order:

1. **Geometry** — straighten / crop (`--straighten`)
2. **Global Auto Basic** — whole-frame exposure, shadows, highlight protect, slight WB
3. **Look + Detail** — creative look / LUT / sharpen / NR (`apply_grade`)
4. **People refine** — MediaPipe multiclass + YuNet. Scene-matched: face ≈ non-sky mid + ~0.04 (cap ~0.42), face/body/clothes EV within ~0.1, gain ≤1.75×. Mild skin nudge only.
5. **Selective masks** — light sky/linear pull; subject fill only if crushed vs sky

Do **not** jump to hard local face lifts before global auto — that creates spotlight faces and mismatched clothes.

## When to pick which look

- **neutral** — diagnostic. Use this for the first preview if the scene is unknown.
- **natural** — default client-safe finish. Daylight streets, family, travel without a gimmick.
- **warm-golden** — late sun, interiors with tungsten, bakeries, summer skin. If shade goes orange, drop temperature.
- **cool-cinematic** — blue hour, concrete, rain, night shopfronts. Skin will go pale; add +temperature if a face is the subject.
- **portrait** — people close. Low clarity, gentle shadows, slight warm tint. Do not add food-level vibrance.
- **food** — plates, coffee, markets. Extra clarity and vibrance. Watch specular highlights on plates.
- **travel** — punch without looking like a postcard filter. Good default for mixed outdoor sets.
- **night** — signs, streets after dark. Pull highlights hard, modest shadow lift, more NR, less sharpen.
- **editorial-flat** — fashion / lookbook / further grading later. Do not add vignette.

## Scene recipes (overrides on top of a look)

Open shade portrait, green cast:

```
--look portrait --temperature 10 --tint 6 --shadows 16 --vibrance 4
```

Backlit hair / window:

```
--look natural --exposure 0.35 --highlights -35 --shadows 22 --whites -8
```

Sunset sky hero:

```
--look warm-golden --highlights -28 --vibrance 18 --saturation 4 --clarity 10
```

Office LED / mixed white:

```
--look natural --temperature -6 --tint 4 --saturation -4 --vibrance 8
```

Rain / asphalt:

```
--look cool-cinematic --contrast 12 --clarity 12 --vibrance 4 --vignette -10
```

## Batch discipline

One shoot, one grade. If half the set is sun and half is shade, split into two param files rather than averaging a look that fits neither.
