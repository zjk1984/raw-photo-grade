#!/usr/bin/env python3
"""M4/Metal GPU-Accelerated Photo Quality Evaluation & Tiering Engine.

Assesses RAW, DNG, and standard image files across multiple dimensions:
1. Sharpness & Focus (Tenengrad edge energy on MPS/Metal)
2. Dynamic Range & Exposure (histogram entropy, clipping ratio)
3. Noise Control (Laplacian high-frequency residual MAD)
4. Color Harmony (Colorfulness, saturation distribution)
5. Composition (Subject saliency center & rule-of-thirds alignment)

Assigns an overall quality score (0-100) and classifies into S, A, B, C tiers.
Designed for Apple Silicon (M4 / M3 / M2 / M1) Unified Memory architecture via PyTorch MPS.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Suffixes supported
RAW_SUFFIXES = {
    ".nef", ".NEF",
    ".cr2", ".CR2", ".cr3", ".CR3",
    ".arw", ".ARW",
    ".raf", ".RAF",
    ".orf", ".ORF",
    ".rw2", ".RW2",
    ".pef", ".PEF",
    ".srw", ".SRW",
    ".dng", ".DNG",
}
STANDARD_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG", ".tif", ".tiff", ".TIF", ".TIFF", ".webp", ".WEBP"}
ALL_SUPPORTED_SUFFIXES = RAW_SUFFIXES | STANDARD_SUFFIXES

# Default weights for general photography
DEFAULT_WEIGHTS = {
    "sharpness": 0.35,
    "dynamic_range": 0.25,
    "noise_control": 0.15,
    "color_harmony": 0.15,
    "composition": 0.10,
}

# Scene-specific weight presets
PRESET_WEIGHTS = {
    "general": DEFAULT_WEIGHTS,
    "landscape": {
        "sharpness": 0.30,
        "dynamic_range": 0.35,
        "noise_control": 0.15,
        "color_harmony": 0.10,
        "composition": 0.10,
    },
    "portrait": {
        "sharpness": 0.35,
        "dynamic_range": 0.20,
        "noise_control": 0.20,
        "color_harmony": 0.15,
        "composition": 0.10,
    },
    "street": {
        "sharpness": 0.25,
        "dynamic_range": 0.20,
        "noise_control": 0.10,
        "color_harmony": 0.20,
        "composition": 0.25,
    },
    "night": {
        "sharpness": 0.25,
        "dynamic_range": 0.30,
        "noise_control": 0.30,
        "color_harmony": 0.10,
        "composition": 0.05,
    },
}


def get_torch_device(requested: str = "auto"):
    """Select the best available compute device, prioritizing Apple Silicon MPS."""
    try:
        import torch
        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            sys.stderr.write("Warning: MPS requested but not available. Falling back to CPU.\n")
            return torch.device("cpu")
        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            sys.stderr.write("Warning: CUDA requested but not available. Falling back to CPU.\n")
            return torch.device("cpu")
        if requested == "cpu":
            return torch.device("cpu")
        # auto
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None


@dataclass
class ImageEvaluation:
    path: str
    filename: str
    overall_score: float
    tier: str  # S, A, B, C
    sharpness: float
    dynamic_range: float
    noise_control: float
    color_harmony: float
    composition: float
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class PhotoEvaluator:
    def __init__(
        self,
        device_name: str = "auto",
        weights: dict[str, float] | None = None,
        max_edge: int = 1920,
        use_half_raw: bool = True,
    ):
        self.device = get_torch_device(device_name)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.max_edge = max_edge
        self.use_half_raw = use_half_raw

    def load_image(self, path: Path) -> tuple[np.ndarray, Any]:
        """Load an image (RAW or standard) and return normalized RGB np.ndarray [H,W,3] in 0..1,

        plus a PyTorch tensor on target device if torch is available.
        """
        ext = path.suffix
        np_rgb: np.ndarray

        if ext in RAW_SUFFIXES:
            try:
                import rawpy
                with rawpy.imread(str(path)) as raw:
                    # For fast evaluation, half_size=True reduces decoding time by ~4x
                    # while retaining full spatial dynamic range and edge information.
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        half_size=self.use_half_raw,
                        output_bps=8,
                        bright=1.0,
                    )
                    np_rgb = rgb.astype(np.float32) / 255.0
            except Exception as e:
                # Fallback to PIL if rawpy fails or file can be opened by PIL
                try:
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        np_rgb = np.asarray(im, dtype=np.float32) / 255.0
                except Exception:
                    raise RuntimeError(f"Failed to decode RAW file {path.name}: {e}")
        else:
            with Image.open(path) as im:
                im = im.convert("RGB")
                np_rgb = np.asarray(im, dtype=np.float32) / 255.0

        # Downsample to max_edge if larger
        h, w = np_rgb.shape[:2]
        m = max(h, w)
        if m > self.max_edge:
            scale = self.max_edge / m
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            pil_im = Image.fromarray((np.clip(np_rgb, 0, 1) * 255.0).astype(np.uint8), mode="RGB")
            pil_im = pil_im.resize((nw, nh), Image.Resampling.BILINEAR)
            np_rgb = np.asarray(pil_im, dtype=np.float32) / 255.0

        tensor_img = None
        if self.device is not None:
            try:
                import torch
                # Shape [1, 3, H, W]
                tensor_img = torch.from_numpy(np_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
            except Exception:
                tensor_img = None

        return np_rgb, tensor_img

    def compute_sharpness(self, np_rgb: np.ndarray, tensor_img: Any) -> tuple[float, float]:
        """Compute sharpness using Tenengrad gradient energy of luminance.

        Returns (score 0-100, raw_metric).
        Evaluates top 5% high frequency edge energy to detect in-focus subject edges.
        """
        if tensor_img is not None and self.device is not None:
            import torch
            import torch.nn.functional as F

            # Luminance Y = 0.2126 R + 0.7152 G + 0.0722 B
            gray = (
                0.2126 * tensor_img[:, 0:1]
                + 0.7152 * tensor_img[:, 1:2]
                + 0.0722 * tensor_img[:, 2:3]
            )

            # Sobel horizontal & vertical kernels
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 1, 3, 3)

            gx = F.conv2d(gray, sobel_x, padding=1)
            gy = F.conv2d(gray, sobel_y, padding=1)
            mag = torch.sqrt(gx**2 + gy**2 + 1e-6)

            # 95th percentile captures sharpest edges (subject in focus)
            # while avoiding bokeh/background blur dragging down the score
            raw_val = float(torch.quantile(mag, 0.95).item())
        else:
            # NumPy fallback
            gray = (
                0.2126 * np_rgb[..., 0]
                + 0.7152 * np_rgb[..., 1]
                + 0.0722 * np_rgb[..., 2]
            )
            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            mag = np.sqrt(gx**2 + gy**2)
            raw_val = float(np.quantile(mag, 0.95))

        # Map to 0-100 scale: typical sharp photos have 95th percentile ~0.15 - 0.35+
        score = float(np.clip((raw_val / 0.28) * 80.0, 0.0, 100.0))
        return round(score, 1), round(raw_val, 4)

    def compute_dynamic_range(
        self, np_rgb: np.ndarray, tensor_img: Any
    ) -> tuple[float, dict[str, float], list[str]]:
        """Evaluate exposure, highlight/shadow clipping, and histogram entropy.

        Returns (score 0-100, details, flags).
        """
        flags = []
        if tensor_img is not None and self.device is not None:
            import torch

            gray = (
                0.2126 * tensor_img[:, 0]
                + 0.7152 * tensor_img[:, 1]
                + 0.0722 * tensor_img[:, 2]
            )
            clipped_hi = float((gray > 0.985).float().mean().item())
            clipped_sh = float((gray < 0.015).float().mean().item())
            mean_luma = float(gray.mean().item())

            hist = torch.histc(gray, bins=64, min=0.0, max=1.0)
            prob = hist / (hist.sum() + 1e-6)
            mask = prob > 0
            entropy = float(-torch.sum(prob[mask] * torch.log2(prob[mask])).item())
        else:
            gray = (
                0.2126 * np_rgb[..., 0]
                + 0.7152 * np_rgb[..., 1]
                + 0.0722 * np_rgb[..., 2]
            )
            clipped_hi = float((gray > 0.985).mean())
            clipped_sh = float((gray < 0.015).mean())
            mean_luma = float(gray.mean())

            hist, _ = np.histogram(gray, bins=64, range=(0.0, 1.0))
            prob = hist / (hist.sum() + 1e-6)
            mask = prob > 0
            entropy = float(-np.sum(prob[mask] * np.log2(prob[mask])))

        # Defect flags
        if clipped_hi > 0.08:
            flags.append("clipped_highlights")
        if clipped_sh > 0.20:
            flags.append("crushed_shadows")
        if mean_luma < 0.10:
            flags.append("underexposed")
        elif mean_luma > 0.85:
            flags.append("overexposed")

        # Score calculation:
        # Ideal entropy is around 4.8 - 5.8 (rich tones throughout range)
        entropy_score = np.clip((entropy / 5.4) * 85.0, 20.0, 95.0)

        # Clipping penalties
        hi_penalty = np.clip(clipped_hi * 150.0, 0.0, 40.0)
        sh_penalty = np.clip(clipped_sh * 100.0, 0.0, 30.0)

        # Exposure center deviation penalty (mid-grey around 0.18 - 0.45 in linear/gamma)
        exp_dev = abs(mean_luma - 0.42)
        exp_penalty = np.clip(max(0.0, exp_dev - 0.15) * 60.0, 0.0, 30.0)

        dr_score = float(np.clip(entropy_score - hi_penalty - sh_penalty - exp_penalty + 10.0, 0.0, 100.0))

        details = {
            "entropy": round(entropy, 2),
            "clipped_highlights_pct": round(clipped_hi * 100.0, 2),
            "clipped_shadows_pct": round(clipped_sh * 100.0, 2),
            "mean_luma": round(mean_luma, 3),
        }
        return round(dr_score, 1), details, flags

    def compute_noise_control(self, np_rgb: np.ndarray, tensor_img: Any) -> tuple[float, float]:
        """Estimate noise level via high-frequency Laplacian residual in smooth areas.

        Returns (score 0-100, noise_level). Higher score = cleaner image.
        """
        if tensor_img is not None and self.device is not None:
            import torch
            import torch.nn.functional as F

            gray = (
                0.2126 * tensor_img[:, 0:1]
                + 0.7152 * tensor_img[:, 1:2]
                + 0.0722 * tensor_img[:, 2:3]
            )
            laplacian = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            res = torch.abs(F.conv2d(gray, laplacian, padding=1))
            # Median of residual gives a robust noise estimate (unaffected by strong edges)
            noise_val = float(torch.median(res).item())
        else:
            gray = (
                0.2126 * np_rgb[..., 0]
                + 0.7152 * np_rgb[..., 1]
                + 0.0722 * np_rgb[..., 2]
            )
            # 3x3 Laplacian
            padded = np.pad(gray, 1, mode="edge")
            res = np.abs(
                padded[0:-2, 1:-1]
                + padded[2:, 1:-1]
                + padded[1:-1, 0:-2]
                + padded[1:-1, 2:]
                - 4 * gray
            )
            noise_val = float(np.median(res))

        # Base ISO camera RAW typically has noise_val ~ 0.001 - 0.015
        # High ISO / noisy phone image ~ 0.035 - 0.08+
        noise_score = float(np.clip(100.0 - (noise_val / 0.045) * 50.0, 10.0, 100.0))
        return round(noise_score, 1), round(noise_val, 5)

    def compute_color_harmony(self, np_rgb: np.ndarray) -> tuple[float, dict[str, float]]:
        """Compute Hasler and Süsstrunk Colorfulness index & saturation richness.

        Returns (score 0-100, details).
        """
        r, g, b = np_rgb[..., 0], np_rgb[..., 1], np_rgb[..., 2]
        # Hasler & Süsstrunk metric components:
        # rg = R - G, yb = 0.5 * (R + G) - B
        rg = r - g
        yb = 0.5 * (r + g) - b

        std_rg = float(np.std(rg))
        std_yb = float(np.std(yb))
        mean_rg = float(np.mean(rg))
        mean_yb = float(np.mean(yb))

        std_rgyb = math.sqrt(std_rg**2 + std_yb**2)
        mean_rgyb = math.sqrt(mean_rg**2 + mean_yb**2)
        colorfulness = std_rgyb + 0.3 * mean_rgyb  # usually 0.05 (muted) to 0.40+ (vivid)

        # Saturation max-min
        sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        mean_sat = float(sat.mean())

        # If image is almost monochrome (B&W intentional), colorfulness is near 0
        if mean_sat < 0.03:
            # Monochrome picture can still be fine art, score neutral 65
            score = 65.0
        else:
            # Optimal colorfulness range: 0.12 - 0.35
            # Score peaks around 0.25
            score = 100.0 - abs(colorfulness - 0.25) * 160.0
            score = float(np.clip(score, 30.0, 100.0))

        details = {
            "colorfulness": round(colorfulness, 3),
            "mean_saturation": round(mean_sat, 3),
        }
        return round(score, 1), details

    def compute_composition(self, np_rgb: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Evaluate composition: subject attention center, rule-of-thirds, and horizon tilt.

        Returns (score 0-100, details).
        """
        h, w = np_rgb.shape[:2]
        step = max(1, min(h, w) // 240)
        small = np_rgb[::step, ::step]
        sh, sw = small.shape[:2]

        sat = small.max(axis=2) - small.min(axis=2)
        gray = 0.2126 * small[..., 0] + 0.7152 * small[..., 1] + 0.0722 * small[..., 2]
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        edge = gx + gy

        yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
        cy, cx = (sh - 1) / 2.0, (sw - 1) / 2.0
        dist = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
        center_bias = np.clip(1.0 - dist * 0.55, 0.15, 1.0)
        attention = (0.55 * edge + 0.45 * sat) * center_bias
        attention = attention - attention.min()
        total = float(attention.sum()) + 1e-8
        subj_y = float((attention * yy).sum() / total) / max(sh - 1, 1)
        subj_x = float((attention * xx).sum() / total) / max(sw - 1, 1)

        # Measure closeness to rule-of-thirds nodes: (1/3, 1/3), (1/3, 2/3), etc. or center (0.5, 0.5)
        third_nodes = [
            (1 / 3, 1 / 3), (1 / 3, 2 / 3),
            (2 / 3, 1 / 3), (2 / 3, 2 / 3),
            (0.5, 0.5), (0.5, 1 / 3), (0.5, 2 / 3),
        ]
        min_dist = min(math.hypot(subj_x - nx, subj_y - ny) for nx, ny in third_nodes)

        # Tilt angle estimation on horizontal edges
        tilt_angle = self._estimate_horizon_tilt(gray)

        # Baseline score: closer to thirds / center = higher score
        comp_score = 90.0 - min_dist * 75.0
        if abs(tilt_angle) > 2.0:
            comp_score -= min(15.0, abs(tilt_angle) * 2.5)

        comp_score = float(np.clip(comp_score, 40.0, 100.0))
        details = {
            "subject_center": [round(subj_x, 3), round(subj_y, 3)],
            "tilt_angle_deg": round(tilt_angle, 2),
        }
        return round(comp_score, 1), details

    def _estimate_horizon_tilt(self, gray: np.ndarray) -> float:
        """Estimate horizon tilt angle via gradient Hough vote in +/-10 degrees."""
        sh, sw = gray.shape
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        thr = float(np.quantile(mag, 0.96))
        mask = mag >= max(thr, 1e-4)
        ys, xs = np.nonzero(mask)
        if xs.size < 120:
            return 0.0

        xs = xs.astype(np.float32) - sw / 2.0
        ys = ys.astype(np.float32) - sh / 2.0
        thetas = np.arange(-10.0, 10.001, 0.5)
        rad = np.radians(90.0 + thetas)
        rho = xs[:, None] * np.cos(rad)[None, :] + ys[:, None] * np.sin(rad)[None, :]
        rho_bin = np.floor(rho * 1.5).astype(np.int32)
        scores = np.zeros(thetas.shape, dtype=np.float64)
        wts = mag[mask].astype(np.float64)
        for i in range(thetas.shape[0]):
            col = rho_bin[:, i] - rho_bin[:, i].min()
            scores[i] = float(np.bincount(col, weights=wts).max())
        a = float(thetas[int(np.argmax(scores))])
        return a if abs(a) <= 10.0 else 0.0

    def evaluate(self, path: Path) -> ImageEvaluation:
        """Run full multi-dimensional quality assessment on an image."""
        np_rgb, tensor_img = self.load_image(path)

        # 1. Metric calculations
        sharpness, raw_sharp = self.compute_sharpness(np_rgb, tensor_img)
        dr_score, dr_details, flags = self.compute_dynamic_range(np_rgb, tensor_img)
        noise_score, raw_noise = self.compute_noise_control(np_rgb, tensor_img)
        color_score, color_details = self.compute_color_harmony(np_rgb)
        comp_score, comp_details = self.compute_composition(np_rgb)

        # 2. Weighted overall score
        w = self.weights
        overall = (
            sharpness * w["sharpness"]
            + dr_score * w["dynamic_range"]
            + noise_score * w["noise_control"]
            + color_score * w["color_harmony"]
            + comp_score * w["composition"]
        )

        # 3. Rule-based vetoes and penalty adjustments
        # Critical blur veto:
        if sharpness < 28.0:
            if "blurry" not in flags:
                flags.append("blurry")
            overall -= 22.0
        elif sharpness < 42.0:
            overall -= 10.0

        # Severe clipping vetoes:
        if "clipped_highlights" in flags:
            overall -= 14.0
        if "crushed_shadows" in flags:
            overall -= 8.0
        if abs(comp_details.get("tilt_angle_deg", 0.0)) >= 4.0:
            flags.append("tilted_horizon")

        overall = round(float(np.clip(overall, 0.0, 100.0)), 1)

        # 4. Tier classification: S, A, B, C
        # S-Tier: Outstanding technical and aesthetic score, no major flags
        if overall >= 85.0 and not any(f in flags for f in ["blurry", "clipped_highlights", "underexposed"]):
            tier = "S"
        # A-Tier: Strong candidate, slight tuning needed
        elif overall >= 72.0 and "blurry" not in flags:
            tier = "A"
        # B-Tier: Acceptable backup or requires heavier post-processing
        elif overall >= 58.0:
            tier = "B"
        # C-Tier: Reject / Discard
        else:
            tier = "C"

        details = {
            "raw_sharpness": raw_sharp,
            "raw_noise": raw_noise,
            **dr_details,
            **color_details,
            **comp_details,
            "compute_device": str(self.device) if self.device is not None else "cpu",
        }

        return ImageEvaluation(
            path=str(path.resolve()),
            filename=path.name,
            overall_score=overall,
            tier=tier,
            sharpness=sharpness,
            dynamic_range=dr_score,
            noise_control=noise_score,
            color_harmony=color_score,
            composition=comp_score,
            flags=flags,
            details=details,
        )


def format_table(results: list[ImageEvaluation]) -> str:
    """Format evaluations as a clean, colorized terminal table."""
    lines = []
    # Tier colors if ANSI supported
    tier_colors = {
        "S": "\033[1;35m[ S ]\033[0m",  # Bold Magenta
        "A": "\033[1;32m[ A ]\033[0m",  # Bold Green
        "B": "\033[1;36m[ B ]\033[0m",  # Bold Cyan
        "C": "\033[1;31m[ C ]\033[0m",  # Bold Red
    }
    header = f"{'Tier':<7} {'Score':<6} {'Sharp':<6} {'DynRng':<7} {'Noise':<6} {'Color':<6} {'Comp':<6} {'Flags':<22} {'Filename'}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for r in results:
        t_str = tier_colors.get(r.tier, f"[{r.tier}]")
        flags_str = ",".join(r.flags) if r.flags else "ok"
        if len(flags_str) > 20:
            flags_str = flags_str[:18] + ".."
        line = (
            f"{t_str:<16} {r.overall_score:<6.1f} {r.sharpness:<6.1f} {r.dynamic_range:<7.1f} "
            f"{r.noise_control:<6.1f} {r.color_harmony:<6.1f} {r.composition:<6.1f} "
            f"{flags_str:<22} {r.filename}"
        )
        lines.append(line)
    lines.append(sep)

    # Summary counts
    counts = {"S": 0, "A": 0, "B": 0, "C": 0}
    for r in results:
        counts[r.tier] = counts.get(r.tier, 0) + 1
    total = len(results)
    lines.append(
        f"Summary: {total} photos | S: {counts['S']} ({counts['S']/total*100:.1f}%) | "
        f"A: {counts['A']} ({counts['A']/total*100:.1f}%) | "
        f"B: {counts['B']} ({counts['B']/total*100:.1f}%) | "
        f"C: {counts['C']} ({counts['C']/total*100:.1f}%)"
    )
    return "\n".join(lines)


def organize_files(results: list[ImageEvaluation], target_dir: Path, method: str = "copy") -> None:
    """Organize assessed photos into target_dir/S, /A, /B, /C folders."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        src = Path(r.path)
        dest_folder = target_dir / r.tier
        dest_folder.mkdir(exist_ok=True)
        dest = dest_folder / src.name
        if method == "symlink":
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src.resolve())
        elif method == "move":
            shutil.move(str(src), str(dest))
        else:  # copy
            shutil.copy2(str(src), str(dest))


def run_eval(
    inputs: list[str],
    device: str = "auto",
    preset: str = "general",
    weights: dict[str, float] | None = None,
    output_json: bool = False,
    filter_tiers: list[str] | None = None,
    organize_dir: str | None = None,
    organize_method: str = "copy",
    max_edge: int = 1920,
) -> int:
    """Core CLI execution function."""
    selected_weights = weights or PRESET_WEIGHTS.get(preset, DEFAULT_WEIGHTS)
    evaluator = PhotoEvaluator(
        device_name=device,
        weights=selected_weights,
        max_edge=max_edge,
    )

    # Collect files
    files: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser()
        if not p.exists():
            sys.stderr.write(f"Not found: {p}\n")
            continue
        if p.is_dir():
            found = sorted(
                q for q in p.rglob("*")
                if q.is_file() and q.suffix in ALL_SUPPORTED_SUFFIXES and not q.name.startswith(".")
            )
            files.extend(found)
        else:
            if p.suffix in ALL_SUPPORTED_SUFFIXES:
                files.append(p)
            else:
                sys.stderr.write(f"Unsupported extension for {p.name}\n")

    if not files:
        sys.stderr.write("No valid photo files found to evaluate.\n")
        return 1

    # Evaluate
    results: list[ImageEvaluation] = []
    for f in files:
        try:
            ev = evaluator.evaluate(f)
            results.append(ev)
        except Exception as e:
            sys.stderr.write(f"Error evaluating {f.name}: {e}\n")

    if not results:
        return 1

    # Sort results by overall score descending
    results.sort(key=lambda x: x.overall_score, reverse=True)

    # Optional filtering (e.g. S,A)
    if filter_tiers:
        allowed = {t.upper() for t in filter_tiers}
        results = [r for r in results if r.tier in allowed]

    # Organize if requested
    if organize_dir:
        organize_files(results, Path(organize_dir).expanduser(), method=organize_method)
        sys.stderr.write(f"Organized {len(results)} files into {organize_dir}/(S,A,B,C)\n")

    # Output
    if output_json:
        out_data = {
            "device": str(evaluator.device) if evaluator.device is not None else "cpu",
            "weights": selected_weights,
            "total_count": len(results),
            "results": [asdict(r) for r in results],
        }
        json.dump(out_data, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_table(results))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M4 Mac Local Photo Quality Evaluator & S/A/B/C Classifier"
    )
    parser.add_argument("inputs", nargs="+", help="Image/RAW file(s) or directory")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device (default: auto; uses MPS on Apple Silicon M4)",
    )
    parser.add_argument(
        "--preset",
        default="general",
        choices=sorted(PRESET_WEIGHTS.keys()),
        help="Evaluation weighting preset",
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument(
        "--filter",
        dest="filter_tiers",
        help="Comma-separated tiers to output (e.g. 'S,A' or 'S')",
    )
    parser.add_argument(
        "--organize",
        dest="organize_dir",
        help="Copy/link photos into S/A/B/C subfolders in target directory",
    )
    parser.add_argument(
        "--organize-method",
        default="copy",
        choices=["copy", "symlink", "move"],
        help="How to place files when --organize is set",
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=1920,
        help="Longest edge downsample size for evaluation (default: 1920)",
    )
    args = parser.parse_args()

    filter_list = [x.strip() for x in args.filter_tiers.split(",")] if args.filter_tiers else None

    return run_eval(
        inputs=args.inputs,
        device=args.device,
        preset=args.preset,
        output_json=args.json,
        filter_tiers=filter_list,
        organize_dir=args.organize_dir,
        organize_method=args.organize_method,
        max_edge=args.max_edge,
    )


if __name__ == "__main__":
    raise SystemExit(main())
