# Look recipes

Named looks are starting points. After the first preview, change 2–4 sliders, not all of them.

## Slider ranges (develop.py)

| slider | typical | too far |
| --- | --- | --- |
| exposure | −0.4 … +0.6 EV | +1 unless it's a deliberate night lift |
| contrast | −10 … +20 | +35 (crunchy, halo) |
| highlights | −45 … +10 | +30 on a bright sky |
| shadows | 0 … +28 | +42 (flat, HDR look) |
| whites | −15 … +10 | |
| blacks | −20 … +8 | |
| temperature | −20 … +20 | ±40 unless mixed neon |
| tint | −8 … +8 | |
| vibrance | 0 … +20 | |
| saturation | −10 … +10 | +20 on skin |
| clarity | 0 … +16 landscape/architecture, 0 … +6 portrait | +25 |
| vignette | 0 … −12 | −25 |
| sharpen | 10 … 24 | 40 |
| noise_luma | 0 … 8 base-to-moderate ISO, 10–16 high ISO | 30 (plastic) |

| fade | 0 … 45 (matte / Instant) | 70 (washed out) |
| sharpen_range | 1 … 3 (Sony Sharpness Range) | 5 |
| hsl_skin_hue / hsl_skin_sat | −15 … +10 | protect skin; negative sat on Film looks |
| hsl_sky_hue / hsl_sky_sat | hue −20…0, sat 0…20 | FL pulls sky cooler |
| hsl_green_hue / hsl_green_sat | hue 0…12, sat 0…15 | foliage punch |
| lut / lut_amount | path to `.cube`, amount 0–100 | ship `shared/luts/sony_fl_approx.cube` |

Units: exposure is EV. Everything else is roughly −100…+100 like Lightroom, but the implementation is simpler — treat numbers as taste, not as a Lightroom match. Note the shadow/highlight/noise ranges are wider (and default noise reduction lower) than the phone version — a real sensor at base ISO earns that headroom.

## Sony Creative Look presets

Inspired by in-camera Creative Look (JPEG grammar). RAW does not bake these in — we recreate the character with fade + HSL + optional 3D LUT.

| look | when |
| --- | --- |
| **sony-st** | Default Standard. Safe daylight / mixed sets. |
| **sony-pt** | Portraits / skin. Soft, low clarity, mild fade. |
| **sony-nt** | Flat source for further grading. |
| **sony-vv** / **sony-vv2** | Colorful subjects, high-key travel; VV2 uses a vivid LUT blend. |
| **sony-fl** / **sony-fl2** | Moody film: cool calm colors, strong contrast, sky/green HSL, FL LUT. |
| **sony-in** | Matte Instant — fade is the point. |
| **sony-sh** | Soft high-key, bright and airy. |

Pipeline / UI can use `--look auto` (optionally `--brand sony|fuji|nikon`) to pick from scene heuristics. Preferences live in `~/.photograde/look_prefs.json` (`brand`, `scene_map`, `default_look`).

## Fujifilm Film Simulation presets

Inspired by in-camera Film Simulations (JPEG grammar). Approximate character via fade + HSL + LUT — not a pixel-perfect match of Fuji color science.

| look | when |
| --- | --- |
| **fuji-provia** | Standard / PROVIA. Safe general default. |
| **fuji-astia** | Soft portraits, gentle skin. |
| **fuji-velvia** | Landscape punch — deep greens / blues (Velvia LUT). |
| **fuji-classic-chrome** | Documentary muted teal, restrained reds. |
| **fuji-classic-neg** | Urban Superia-ish muted colors, mild fade. |
| **fuji-nostalgic-neg** | Warm amber nostalgia (Nostalgic Neg LUT). |
| **fuji-eterna** | Cinema-flat, low contrast for further grade. |
| **fuji-acros** | B&W ACROS-style contrast. |

## Nikon Picture Control presets

Inspired by Nikon Picture Controls (Standard / Neutral / Vivid / Portrait / Landscape / Flat / …).

| look | when |
| --- | --- |
| **nikon-standard** | Balanced everyday finish. |
| **nikon-neutral** | Minimal processing, edit later. |
| **nikon-vivid** | Photoprint punch on primary colors. |
| **nikon-portrait** | Smooth complexions, soft clarity. |
| **nikon-rich-tone-portrait** | Richer skin detail, protected highlights. |
| **nikon-landscape** | Blues/greens emphasis (Landscape LUT). |
| **nikon-flat** | Wide tonal range for grading. |
| **nikon-monochrome** | Straight B&W. |

## When to pick which classic look

- **neutral** — diagnostic. Use this for the first preview if the scene is unknown.
- **natural** — default client-safe finish. Daylight, family, documentary, anything without a gimmick.
- **warm-golden** — golden hour, interiors with tungsten, autumn light, skin in warm sun.
- **cool-cinematic** — blue hour, concrete, rain, night streets. Skin will go pale; add +temperature if a face is the subject.
- **portrait** — people close. Low clarity, gentle shadows, slight warm tint. Do not add food-level vibrance.
- **food** — plates, product, markets. Extra clarity and vibrance. Watch specular highlights on glossy surfaces.
- **travel** — punch without looking like a postcard filter. Good default for mixed outdoor sets.
- **night** — city lights, astro, long exposures. Pull highlights hard on light sources, modest shadow lift — a real sensor rarely needs the noise reduction a phone night mode needs.
- **editorial-flat** — fashion / lookbook / further grading later. Do not add vignette.

## Scene recipes (overrides on top of a look)

Overcast landscape, flat sky:

```
--look natural --contrast 18 --clarity 14 --highlights -10 --vibrance 8
```

Backlit portrait / rim light:

```
--look portrait --exposure 0.3 --highlights -30 --shadows 24 --whites -6
```

Golden hour sky hero:

```
--look warm-golden --highlights -30 --vibrance 16 --saturation 4 --clarity 10
```

Studio / mixed indoor white:

```
--look natural --temperature -6 --tint 4 --saturation -4 --vibrance 8
```

Astro / Milky Way:

```
--look night --exposure 0.15 --shadows 6 --noise-luma 10 --clarity 8 --vibrance 6
```

## Batch discipline

One shoot, one grade. If half the set is sun and half is shade, split into two param files rather than averaging a look that fits neither.
