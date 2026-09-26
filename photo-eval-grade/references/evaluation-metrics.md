# Evaluation Metrics Reference

Technical specifications of the multi-dimensional photo assessment algorithms used by `eval_photo.py` and `photo-eval-grade`.

---

## 1. Metrics Breakdown

### 1.1 Sharpness & Subject Focus (`sharpness` = \(S_{\text{plane}}\))
- **Core Concept**: Photographic sharpness is determined by the **focal plane / in-focus subject** micro-contrast, not by the entire image (which would unfairly penalize shallow depth-of-field portraits or macro).
- **Algorithm**:
  1. Convert to luminance \( Y = 0.2126 R + 0.7152 G + 0.0722 B \).
  2. **Field metric** \(S_{\text{field}}\): **median** of 5×5 patch scores (bokeh-aware; not full-frame 95th).
  3. **Plane metric** \(S_{\text{plane}}\) (drives score + focus veto):
     - Score a **5×5 grid** of patches; prefer **attention ∩ top patches** (IoU ≥ 0.12), else max / top-2 when \(F \le 2.8\).
     - Also score an **attention window** from composition subject center (portrait preset biases upward toward eyes).
     - \(S_{\text{plane}}\) prefers subject-plane overlap over a sharp corner leaf.
  4. RAW-aware: re-measure edge95 on a **16-bit linear demosaic crop of the winning patch** (not a blind center crop); blend with `raw_trust`.
  5. Critical band (\(S_{\text{plane}}\) within ±8 of `blur_cut`): higher-res RAW focus crop (~2200px).
- **Scoring / flags**:
  - \(S_{\text{plane}} < \text{blur\_cut} - m\) → `blurry` (−22, ban S/A). Margin \(m=8\) camera / \(m=10\) phone.
  - \(\text{blur\_cut} - m \le S_{\text{plane}} < \text{blur\_cut}\) → `soft` (−14, ban S/A, **B allowed**).
  - \(\text{blur\_cut} \le S_{\text{plane}} < \text{soft\_cut}\) → `soft` (−10, no S/A ban).
  - Large \(S_{\text{plane}} - S_{\text{field}}\) → `shallow_dof` (no hard veto).
- **Phone / ProRAW**: lower `blur_sharp` defaults (26), OIS/multi-frame IBIS, missing F → assumed wide; computational latitude ceiling.
- **Batch ranking**: after a shoot is scored, annotate `shoot_percentile` / `shoot_z`; top-quintile soft keepers with strong `edit_latitude` may promote C→B (`batch_promoted`). Phone lat floor 65 vs camera 70.
- **Verdict**: `details.verdict_reason` — one-line Chinese diagnosis (focus + latitude + roll rank).

---

### 1.2 Dynamic Range → Edit Latitude (`dynamic_range` ≈ \(S_{\text{lat}}\))
- **Core Concept**: For RAW, score **recoverability** (headroom below white level, shadow floor, ISO-relative noise), not JPEG-like as-shot brightness.
- **Track B (drives DR metric on RAW)**: `edit_latitude` from sensor levels + profile DR ceiling; preset×ISO tolerance. **Face / attention window** headroom blends in for people so as-shot hot faces with channel room stay recoverable.
- **Track C (informational)**: `as_shot_score` from preview entropy/clipping — soft flags like `underexposed_as_shot` / `face_hot_as_shot` (light penalty, no S ban alone when recoverable).
- **Hard integrity**: `face_dead_highlights` / `no_latitude` / unrecovered `raw_highlight_clip` may block S. Global sky clip + recoverable face → soft `preview_highlights`, not a hard ban.
- **Develop (Lightroom order)**: Geometry → **Global Auto Basic** → Look/Detail → **People refine** (**MediaPipe selfie multiclass**: face-skin / body-skin / clothes / hair) → **Selective** (subject / sky / linear / radial). GrabCut only if the model is missing.

---

### 1.3 Noise & Signal-to-Noise Ratio (`noise_control`)
- **Core Concept**: High ISO grain or sensor read noise degrades fine textures.
- **Algorithm**:
  1. Convolve luminance with a discrete 3×3 Laplacian high-pass filter:
     \[
     L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}
     \]
  2. Compute the **Median Absolute Deviation (MAD)** of the residual:
     \[
     \text{Noise Level} = \text{median}(|L \ast Y|)
     \]
     Because the median is insensitive to sparse strong edges, it cleanly extracts Gaussian/Poisson noise from smooth surfaces (sky, skin, backgrounds).
- **Scoring**: Base ISO clean images score 90–100; noisy images drop below 50.

---

### 1.4 Color Harmony & Gamut Richness (`color_harmony`)
- **Core Concept**: Natural and aesthetically pleasing photos maintain healthy color separation without oversaturation or drab colorlessness.
- **Algorithm**:
  - Uses the **Hasler and Süsstrunk Colorfulness Metric**:
    \[
    \Delta_{rg} = R - G, \quad \Delta_{yb} = \frac{1}{2}(R + G) - B
    \]
    \[
    M_{\text{color}} = \sqrt{\sigma_{rg}^2 + \sigma_{yb}^2} + 0.3 \sqrt{\mu_{rg}^2 + \mu_{yb}^2}
    \]
  - Optimal values range between 0.15 and 0.35.
  - Near-zero saturation (black & white photography) is safely assigned a neutral artistic baseline (65) rather than zero.

---

### 1.5 Composition & Subject Prominence (`composition`)
- **Core Concept**: Evaluates visual flow, subject attention center, and horizon alignment.
- **Algorithm**:
  1. **Attention Center**: Combined edge density, local color saturation, and center-weighted distribution (matching the heuristic in `shared/scripts/crop.py`).
  2. **Rule-of-Thirds Distance**: Euclidean distance between the attention center and the nearest classic third intersections \((1/3, 1/3)\), \((2/3, 1/3)\), etc., or the optical center \((0.5, 0.5)\).
  3. **Horizon Tilt Estimation**: Uses Hough transform accumulator voting across near-horizontal edges \(\pm 10^\circ\). A tilt \(\ge 2^\circ\) reduces composition points; \(\ge 4^\circ\) flags `tilted_horizon`.

---

## 2. Overall Scoring & Veto Rules

The base score is computed via weighted sum of normalized indicators:
\[
\text{Score}_{\text{base}} = \sum_{k} w_k \cdot S_k
\]

### Penalties & Hard Vetoes:
| Condition | Penalty | Flag Added | Consequence |
| :--- | :--- | :--- | :--- |
| **Hard blur** (\(S_{\text{plane}} < \text{blur\_cut}' - 8\)) | -22 pts | `blurry` | Banned from S & A |
| **Soft / critical band** (\(\text{blur\_cut}'-8 \le S < \text{blur\_cut}'\)) | -14 pts | `soft` | Banned from S & A; **B allowed** |
| **Noticeable Softness** (\(S_{\text{plane}} < \text{soft\_cut}\)) | -10 pts | `soft` | Score reduced (no S/A ban if ≥ blur_cut) |
| **Cloud / white sky** (upper bright, low sat) | 0 pts | `cloud_sky` | **Not a defect** — natural clouds |
| **Sensor highlight clip** (no recoverable face, not cloud) | -12 pts | `raw_highlight_clip` | Blocks S |
| **Face underexposed (recoverable)** | −2 pts | `face_underexposed_as_shot` | Measure face after luma normalize; **not** hard `blurry` when face plane holds; develop auto face-finish + shadow bump |
| **Face dead highlights** | -14 pts | `face_dead_highlights` | Blocks S/A integrity |
| **Face hot as-shot (recoverable)** | −1…−2 pts | `face_hot_as_shot` + `face_recoverable` | Info; prefer highlight-protect look + `--face-finish` |
| **No edit latitude** | -10 pts | `no_latitude` | Blocks S |
| **As-shot underexposed (recoverable)** | -3 pts | `underexposed_as_shot` | Informational / light |
| **Shallow DOF** (large plane−field gap) | 0 pts | `shallow_dof` | Informational only |
| **Motion risk** (shutter ≫ safe \(1/f_{35}\)) | 0 pts | `motion_risk` | Informational; may slightly lower `blur_cut'` |
| **High ISO** (ISO ≫ body base) | 0 pts | `high_iso` | Informational; noise scored vs ISO expectation |
| **Exposure mismatch** (scene EV vs midtones) | −6 pts | `exposure_mismatch` | Light penalty only |
| **Clipped Highlights** (\(> 8\%\)) | -14 pts | `clipped_highlights` | Disqualified from S |
| **Crushed Shadows** (\(> 20\%\)) | -8 pts | `crushed_shadows` | Score reduced |
| **Tilted Horizon** (\(|\theta| \ge 4^\circ\)) | -6 pts | `tilted_horizon` | Score reduced |
| **Batch promote** (top quintile + latitude) | score floor | `batch_promoted` | Soft C→B within shoot |
| **Severe Underexposure** (\(\bar{Y} < 0.10\)) | -15 pts | `underexposed` | Disqualified from S |

---

## 3. Tier Definitions

| Tier | Overall Score | Criteria & Photographic Meaning | Recommended Action |
| :---: | :---: | :--- | :--- |
| **S** | **\(\ge 85.0\)** | **Masterpiece / Hero Shot**: Flawless focus, rich dynamic range, clean shadows, striking composition. Zero severe flags. | Priority full-resolution development, portfolio candidate. |
| **A** | **\(72.0 \sim 84.9\)** | **Keeper / Delivery Ready**: Excellent technical foundation, sharp subject, slight adjustments in color/crop needed. | Batch develop with `--look natural` or custom recipe. |
| **B** | **\(58.0 \sim 71.9\)** | **Backup / Marginal**: Usable scene, but contains minor flaws (e.g. slight noise, suboptimal framing, needs crop or shadow lift). | Keep in secondary archive; develop if client requests more frames. |
| **C** | **\(< 58.0\)** | **Reject / Cull**: Critical blur, missed focus, severe blown sky or ruined exposure. | Safe to delete or move to reject bin. |
