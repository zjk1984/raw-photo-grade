#!/usr/bin/env python3
"""Minimal .cube 3D LUT loader and trilinear applicator (sRGB domain).

.cube data order follows IRIDAS: R varies fastest, then G, then B.
Reshaped table is indexed as table[b, g, r].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_cube(path: Path | str) -> tuple[np.ndarray, int]:
    """Parse a .cube LUT. Returns (table[size,size,size,3], size) indexed [b,g,r]."""
    path = Path(path)
    size = None
    values: list[list[float]] = []
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper.startswith("TITLE") or upper.startswith("DOMAIN_"):
                continue
            if upper.startswith("LUT_3D_SIZE"):
                size = int(line.split()[-1])
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    values.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
    if size is None:
        raise ValueError(f"LUT_3D_SIZE missing in {path}")
    expected = size * size * size
    if len(values) < expected:
        raise ValueError(f"LUT {path} has {len(values)} entries, need {expected}")
    # R fastest → reshape to [B, G, R, 3]
    table = np.asarray(values[:expected], dtype=np.float32).reshape(size, size, size, 3)
    return table, size


def apply_lut(rgb: np.ndarray, table: np.ndarray, size: int, amount: float = 1.0) -> np.ndarray:
    """Apply a 3D LUT with trilinear interpolation. amount in [0, 1] blends with identity."""
    if amount <= 0:
        return rgb
    img = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    scaled = img * (size - 1)
    i0 = np.floor(scaled).astype(np.int32)
    i1 = np.minimum(i0 + 1, size - 1)
    f = scaled - i0

    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    # table indexed [b, g, r]
    c000 = table[b0, g0, r0]
    c001 = table[b1, g0, r0]
    c010 = table[b0, g1, r0]
    c011 = table[b1, g1, r0]
    c100 = table[b0, g0, r1]
    c101 = table[b1, g0, r1]
    c110 = table[b0, g1, r1]
    c111 = table[b1, g1, r1]

    c00 = c000 * (1 - fb) + c001 * fb
    c01 = c010 * (1 - fb) + c011 * fb
    c10 = c100 * (1 - fb) + c101 * fb
    c11 = c110 * (1 - fb) + c111 * fb
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    mapped = c0 * (1 - fr) + c1 * fr

    if amount >= 1.0:
        return np.clip(mapped, 0.0, 1.0)
    return np.clip(img * (1.0 - amount) + mapped * amount, 0.0, 1.0)


def write_cube(path: Path | str, table: np.ndarray, title: str = "Generated") -> None:
    """Write LUT. Input table indexed [b, g, r] (IRIDAS order)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = table.shape[0]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f'TITLE "{title}"\n')
        fh.write(f"LUT_3D_SIZE {size}\n")
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    rr, gg, bb = table[b, g, r]
                    fh.write(f"{rr:.6f} {gg:.6f} {bb:.6f}\n")


def _grid_rgb(size: int) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, size, dtype=np.float32)
    bb, gg, rr = np.meshgrid(xs, xs, xs, indexing="ij")
    return np.stack([rr, gg, bb], axis=-1)


def make_film_approx_lut(size: int = 33) -> np.ndarray:
    """Procedural FL-ish look: cooler mids, lifted blacks, punchy greens/blues."""
    rgb = _grid_rgb(size)
    x = np.clip((rgb - 0.5) * 1.18 + 0.5, 0, 1)
    x = 0.06 + x * 0.90
    luma = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    cool = np.stack([x[..., 0] * 0.96, x[..., 1] * 1.00, x[..., 2] * 1.06], axis=-1)
    warm = np.stack([x[..., 0] * 1.04, x[..., 1] * 1.01, x[..., 2] * 0.96], axis=-1)
    w = np.clip(luma[..., None], 0, 1)
    x = cool * (1 - w) + warm * w
    out = x.copy()
    out[..., 1] = np.clip(x[..., 1] * 1.04 + x[..., 2] * 0.02, 0, 1)
    out[..., 2] = np.clip(x[..., 2] * 1.05 - x[..., 0] * 0.03, 0, 1)
    out[..., 0] = np.clip(x[..., 0] * 0.97 + x[..., 1] * 0.01, 0, 1)
    return out.astype(np.float32)


def make_classic_chrome_lut(size: int = 33) -> np.ndarray:
    """Fuji Classic Chrome-ish: muted mids, cool documentary teal, restrained reds."""
    x = _grid_rgb(size)
    x = np.clip((x - 0.5) * 1.08 + 0.5, 0, 1)
    x = 0.04 + x * 0.93
    gray = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = gray + (x - gray) * 0.82
    out = x.copy()
    out[..., 0] = np.clip(x[..., 0] * 0.94 + 0.02, 0, 1)
    out[..., 1] = np.clip(x[..., 1] * 0.98 + x[..., 2] * 0.02, 0, 1)
    out[..., 2] = np.clip(x[..., 2] * 1.06 - x[..., 0] * 0.02, 0, 1)
    return out.astype(np.float32)


def make_velvia_lut(size: int = 33) -> np.ndarray:
    """Fuji Velvia-ish: deep greens, electric blues, saturated primary punch."""
    x = _grid_rgb(size)
    x = np.clip((x - 0.5) * 1.22 + 0.5, 0, 1)
    gray = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = gray + (x - gray) * 1.18
    out = x.copy()
    out[..., 1] = np.clip(x[..., 1] * 1.10 + x[..., 2] * 0.02, 0, 1)
    out[..., 2] = np.clip(x[..., 2] * 1.12 - x[..., 0] * 0.04, 0, 1)
    out[..., 0] = np.clip(x[..., 0] * 1.06, 0, 1)
    return np.clip(out, 0, 1).astype(np.float32)


def make_nostalgic_neg_lut(size: int = 33) -> np.ndarray:
    """Fuji Nostalgic Neg-ish: warm amber lift, soft contrast, creamy highlights."""
    x = _grid_rgb(size)
    x = np.clip((x - 0.5) * 1.05 + 0.5, 0, 1)
    x = 0.07 + x * 0.88
    out = x.copy()
    out[..., 0] = np.clip(x[..., 0] * 1.08 + 0.03, 0, 1)
    out[..., 1] = np.clip(x[..., 1] * 1.02 + 0.01, 0, 1)
    out[..., 2] = np.clip(x[..., 2] * 0.92, 0, 1)
    return out.astype(np.float32)


def make_nikon_landscape_lut(size: int = 33) -> np.ndarray:
    """Nikon Landscape-ish: crisp blues/greens, clean mid contrast."""
    x = _grid_rgb(size)
    x = np.clip((x - 0.5) * 1.14 + 0.5, 0, 1)
    gray = (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2])[..., None]
    x = gray + (x - gray) * 1.08
    out = x.copy()
    out[..., 1] = np.clip(x[..., 1] * 1.06, 0, 1)
    out[..., 2] = np.clip(x[..., 2] * 1.08, 0, 1)
    out[..., 0] = np.clip(x[..., 0] * 0.98, 0, 1)
    return out.astype(np.float32)


def ensure_builtin_luts(lut_dir: Path | None = None) -> Path:
    """Ensure shared/luts has approximation cubes; return lut_dir."""
    if lut_dir is None:
        lut_dir = Path(__file__).resolve().parent.parent / "luts"
    lut_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("sony_fl_approx.cube", make_film_approx_lut, "Sony FL approx"),
        ("fuji_classic_chrome_approx.cube", make_classic_chrome_lut, "Fuji Classic Chrome approx"),
        ("fuji_velvia_approx.cube", make_velvia_lut, "Fuji Velvia approx"),
        ("fuji_nostalgic_neg_approx.cube", make_nostalgic_neg_lut, "Fuji Nostalgic Neg approx"),
        ("nikon_landscape_approx.cube", make_nikon_landscape_lut, "Nikon Landscape approx"),
    ]
    for name, factory, title in specs:
        path = lut_dir / name
        if not path.exists():
            write_cube(path, factory(33), title=title)
    vv = lut_dir / "sony_vv2_approx.cube"
    if not vv.exists():
        table = make_film_approx_lut(33)
        mid = 0.5
        table = np.clip((table - mid) * 1.12 + mid, 0, 1).astype(np.float32)
        write_cube(vv, table, title="Sony VV2 approx")
    return lut_dir
