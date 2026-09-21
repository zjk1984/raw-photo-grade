# Evaluation Metrics Reference

Technical specifications of the multi-dimensional photo assessment algorithms used by `eval_photo.py` and `photo-eval-grade`.

---

## 1. Metrics Breakdown

### 1.1 Sharpness & Subject Focus (`sharpness`)
- **Core Concept**: Photographic sharpness is determined by the subject's in-focus micro-contrast and edge transitions, not by the entire image average (which would unfairly penalize shallow depth-of-field portraits or macro photography).
- **Algorithm**:
  1. Convert image to linear/sRGB luminance \( Y = 0.2126 R + 0.7152 G + 0.0722 B \).
  2. Apply Sobel 3×3 convolution filters (\( G_x, G_y \)) on GPU/MPS:
     \[
     G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
     G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}
     \]
  3. Calculate gradient magnitude: \( M(x, y) = \sqrt{G_x(x,y)^2 + G_y(x,y)^2} \).
  4. Compute the **95th percentile** of edge magnitudes. This specifically isolates the sharpest in-focus areas while ignoring smooth out-of-focus background bokeh.
- **Scoring**: Mapped to 0–100. Values below 28 trigger a `blurry` defect flag.

---

### 1.2 Dynamic Range & Exposure Balance (`dynamic_range`)
- **Core Concept**: A well-exposed photo retains detail across shadows, midtones, and highlights without unrecoverable clipping or flat muddy tones.
- **Algorithm**:
  1. **Histogram Shannon Entropy**:
     \[
     H(Y) = -\sum_{i=1}^{K} p_i \log_2(p_i)
     \]
     Higher entropy signifies a rich tonal distribution across the 64 luminance bins.
  2. **Highlight Clipping Ratio**: Percentage of pixels where \( Y > 0.985 \). More than 8% clipped highlights triggers `clipped_highlights` penalty.
  3. **Shadow Crushing Ratio**: Percentage of pixels where \( Y < 0.015 \). More than 20% crushed shadows triggers `crushed_shadows` penalty.
  4. **Midtone Luminance Drift**: Measures distance between mean scene luminance and the perceptual mid-point (~0.42).

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
| **Severe Blur / Focus Miss** (\(S_{\text{sharp}} < 28\)) | -22 pts | `blurry` | Automatically disqualified from S & A |
| **Noticeable Softness** (\(S_{\text{sharp}} < 42\)) | -10 pts | - | Score reduced |
| **Clipped Highlights** (\(> 8\%\)) | -14 pts | `clipped_highlights` | Disqualified from S |
| **Crushed Shadows** (\(> 20\%\)) | -8 pts | `crushed_shadows` | Score reduced |
| **Tilted Horizon** (\(|\theta| \ge 4^\circ\)) | -6 pts | `tilted_horizon` | Score reduced |
| **Severe Underexposure** (\(\bar{Y} < 0.10\)) | -15 pts | `underexposed` | Disqualified from S |

---

## 3. Tier Definitions

| Tier | Overall Score | Criteria & Photographic Meaning | Recommended Action |
| :---: | :---: | :--- | :--- |
| **S** | **\(\ge 85.0\)** | **Masterpiece / Hero Shot**: Flawless focus, rich dynamic range, clean shadows, striking composition. Zero severe flags. | Priority full-resolution development, portfolio candidate. |
| **A** | **\(72.0 \sim 84.9\)** | **Keeper / Delivery Ready**: Excellent technical foundation, sharp subject, slight adjustments in color/crop needed. | Batch develop with `--look natural` or custom recipe. |
| **B** | **\(58.0 \sim 71.9\)** | **Backup / Marginal**: Usable scene, but contains minor flaws (e.g. slight noise, suboptimal framing, needs crop or shadow lift). | Keep in secondary archive; develop if client requests more frames. |
| **C** | **\(< 58.0\)** | **Reject / Cull**: Critical blur, missed focus, severe blown sky or ruined exposure. | Safe to delete or move to reject bin. |
