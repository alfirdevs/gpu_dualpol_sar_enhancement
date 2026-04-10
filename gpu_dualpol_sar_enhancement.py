#!/usr/bin/env python3
"""
GPU-Accelerated Dual-Polarization SAR Enhancement and Saliency Mapping

This script reads dual-polarization SAR images (VV and VH), performs GPU-accelerated
preprocessing and feature fusion, builds a pseudo-RGB composite, computes a saliency
map, and saves printable outputs including a PDF summary.

Primary target:
- Ubuntu machine with NVIDIA GPU
- CuPy-based GPU acceleration

Outputs:
- vv.png
- vh.png
- rgb_composite.png
- sum_map.png
- diff_map.png
- norm_diff_map.png
- saliency_map.png
- saliency_heatmap.png
- overlay.png
- report.pdf
- timing_report.txt

Usage examples:
    python3 gpu_dualpol_sar_enhancement.py \
        --vv /path/to/VV.tif \
        --vh /path/to/VH.tif \
        --outdir results

    python3 gpu_dualpol_sar_enhancement.py \
        --vv VV.tif --vh VH.tif --outdir results --benchmark-cpu

Recommended installation on Ubuntu:
    pip install numpy matplotlib opencv-python tifffile pillow scipy
    pip install cupy-cuda12x

Optional for GeoTIFF support:
    pip install rasterio
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import tifffile
except Exception as exc:
    print(f"ERROR: tifffile is required: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import rasterio  # type: ignore
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False
    cp = None
    cnd = None

try:
    from scipy.ndimage import gaussian_filter as cpu_gaussian_filter
    from scipy.ndimage import uniform_filter as cpu_uniform_filter
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False
    cpu_gaussian_filter = None
    cpu_uniform_filter = None


@dataclass
class Timing:
    read_s: float = 0.0
    gpu_preprocess_s: float = 0.0
    gpu_features_s: float = 0.0
    gpu_saliency_s: float = 0.0
    gpu_post_s: float = 0.0
    cpu_total_s: Optional[float] = None
    gpu_total_s: float = 0.0


@dataclass
class Config:
    vv_path: str
    vh_path: str
    outdir: str
    gaussian_sigma: float = 1.2
    local_window: int = 15
    saliency_percentile_clip_low: float = 1.0
    saliency_percentile_clip_high: float = 99.0
    overlay_alpha: float = 0.38
    eps: float = 1e-6
    benchmark_cpu: bool = False


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="GPU dual-pol SAR enhancement")
    parser.add_argument("--vv", required=True, help="Path to VV TIFF image")
    parser.add_argument("--vh", required=True, help="Path to VH TIFF image")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--gaussian-sigma", type=float, default=1.2)
    parser.add_argument("--local-window", type=int, default=15)
    parser.add_argument("--benchmark-cpu", action="store_true",
                        help="Run a CPU benchmark path for timing comparison")
    args = parser.parse_args()

    if args.local_window < 3 or args.local_window % 2 == 0:
        parser.error("--local-window must be an odd integer >= 3")

    return Config(
        vv_path=args.vv,
        vh_path=args.vh,
        outdir=args.outdir,
        gaussian_sigma=args.gaussian_sigma,
        local_window=args.local_window,
        benchmark_cpu=args.benchmark_cpu,
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_tiff(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    arr = None
    last_err = None

    if HAS_RASTERIO:
        try:
            with rasterio.open(path) as src:
                arr = src.read(1)
        except Exception as exc:
            last_err = exc

    if arr is None:
        try:
            arr = tifffile.imread(path)
        except Exception as exc:
            last_err = exc

    if arr is None:
        raise RuntimeError(f"Unable to read TIFF: {path}. Last error: {last_err}")

    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr[0]

    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.float32, copy=False)

    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0
    return arr


def sync_gpu() -> None:
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


def percentile_normalize_cpu(img: np.ndarray, low: float, high: float, eps: float) -> np.ndarray:
    lo = np.percentile(img, low)
    hi = np.percentile(img, high)
    if hi - lo < eps:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def percentile_normalize_gpu(img: "cp.ndarray", low: float, high: float, eps: float) -> "cp.ndarray":
    lo = cp.percentile(img, low)
    hi = cp.percentile(img, high)
    denom = cp.maximum(hi - lo, eps)
    out = (img - lo) / denom
    return cp.clip(out, 0.0, 1.0).astype(cp.float32)


def to_uint8_cpu(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def gpu_preprocess(vv: np.ndarray, vh: np.ndarray, cfg: Config) -> Tuple[Dict[str, np.ndarray], Timing]:
    if not HAS_CUPY:
        raise RuntimeError("CuPy is not available. Install cupy-cuda12x or matching version for your CUDA stack.")

    timing = Timing()
    t0 = time.perf_counter()

    vv_gpu = cp.asarray(vv, dtype=cp.float32)
    vh_gpu = cp.asarray(vh, dtype=cp.float32)
    sync_gpu()
    timing.read_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    vv_log = cp.log1p(vv_gpu)
    vh_log = cp.log1p(vh_gpu)

    vv_norm = percentile_normalize_gpu(vv_log, 1.0, 99.0, cfg.eps)
    vh_norm = percentile_normalize_gpu(vh_log, 1.0, 99.0, cfg.eps)

    vv_smooth = cnd.gaussian_filter(vv_norm, sigma=cfg.gaussian_sigma)
    vh_smooth = cnd.gaussian_filter(vh_norm, sigma=cfg.gaussian_sigma)
    sync_gpu()
    timing.gpu_preprocess_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    sum_map = vv_smooth + vh_smooth
    diff_map = cp.abs(vv_smooth - vh_smooth)
    norm_diff = (vv_smooth - vh_smooth) / (vv_smooth + vh_smooth + cfg.eps)
    norm_diff_shift = 0.5 * (norm_diff + 1.0)

    local_mean = cnd.uniform_filter(sum_map, size=cfg.local_window)
    local_sq_mean = cnd.uniform_filter(sum_map * sum_map, size=cfg.local_window)
    local_var = cp.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    local_std = cp.sqrt(local_var + cfg.eps)
    local_contrast = (sum_map - local_mean) / (local_std + cfg.eps)
    local_contrast = cp.maximum(local_contrast, 0.0)

    sum_n = percentile_normalize_gpu(sum_map, 1.0, 99.0, cfg.eps)
    diff_n = percentile_normalize_gpu(diff_map, 1.0, 99.0, cfg.eps)
    nd_n = percentile_normalize_gpu(norm_diff_shift, 1.0, 99.0, cfg.eps)
    lc_n = percentile_normalize_gpu(local_contrast, 1.0, 99.0, cfg.eps)
    sync_gpu()
    timing.gpu_features_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    saliency = 0.34 * sum_n + 0.26 * diff_n + 0.18 * (1.0 - nd_n) + 0.22 * lc_n
    saliency = cnd.gaussian_filter(saliency, sigma=1.0)
    saliency = percentile_normalize_gpu(
        saliency,
        cfg.saliency_percentile_clip_low,
        cfg.saliency_percentile_clip_high,
        cfg.eps,
    )
    sync_gpu()
    timing.gpu_saliency_s = time.perf_counter() - t3

    t4 = time.perf_counter()
    rgb = cp.stack([
        percentile_normalize_gpu(vv_smooth, 1.0, 99.0, cfg.eps),
        percentile_normalize_gpu(vh_smooth, 1.0, 99.0, cfg.eps),
        percentile_normalize_gpu(diff_map, 1.0, 99.0, cfg.eps),
    ], axis=-1)
    rgb = cp.clip(rgb, 0.0, 1.0)

    outputs = {
        "vv": cp.asnumpy(vv_smooth),
        "vh": cp.asnumpy(vh_smooth),
        "sum_map": cp.asnumpy(sum_n),
        "diff_map": cp.asnumpy(diff_n),
        "norm_diff_map": cp.asnumpy(nd_n),
        "local_contrast": cp.asnumpy(lc_n),
        "saliency": cp.asnumpy(saliency),
        "rgb": cp.asnumpy(rgb),
    }
    sync_gpu()
    timing.gpu_post_s = time.perf_counter() - t4
    timing.gpu_total_s = (
        timing.read_s + timing.gpu_preprocess_s + timing.gpu_features_s +
        timing.gpu_saliency_s + timing.gpu_post_s
    )

    return outputs, timing


def cpu_preprocess(vv: np.ndarray, vh: np.ndarray, cfg: Config) -> Dict[str, np.ndarray]:
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required for CPU benchmark path. Install scipy.")

    vv_log = np.log1p(vv)
    vh_log = np.log1p(vh)

    vv_norm = percentile_normalize_cpu(vv_log, 1.0, 99.0, cfg.eps)
    vh_norm = percentile_normalize_cpu(vh_log, 1.0, 99.0, cfg.eps)

    vv_smooth = cpu_gaussian_filter(vv_norm, sigma=cfg.gaussian_sigma)
    vh_smooth = cpu_gaussian_filter(vh_norm, sigma=cfg.gaussian_sigma)

    sum_map = vv_smooth + vh_smooth
    diff_map = np.abs(vv_smooth - vh_smooth)
    norm_diff = (vv_smooth - vh_smooth) / (vv_smooth + vh_smooth + cfg.eps)
    norm_diff_shift = 0.5 * (norm_diff + 1.0)

    local_mean = cpu_uniform_filter(sum_map, size=cfg.local_window)
    local_sq_mean = cpu_uniform_filter(sum_map * sum_map, size=cfg.local_window)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    local_std = np.sqrt(local_var + cfg.eps)
    local_contrast = np.maximum((sum_map - local_mean) / (local_std + cfg.eps), 0.0)

    sum_n = percentile_normalize_cpu(sum_map, 1.0, 99.0, cfg.eps)
    diff_n = percentile_normalize_cpu(diff_map, 1.0, 99.0, cfg.eps)
    nd_n = percentile_normalize_cpu(norm_diff_shift, 1.0, 99.0, cfg.eps)
    lc_n = percentile_normalize_cpu(local_contrast, 1.0, 99.0, cfg.eps)

    saliency = 0.34 * sum_n + 0.26 * diff_n + 0.18 * (1.0 - nd_n) + 0.22 * lc_n
    saliency = cpu_gaussian_filter(saliency, sigma=1.0)
    saliency = percentile_normalize_cpu(
        saliency,
        cfg.saliency_percentile_clip_low,
        cfg.saliency_percentile_clip_high,
        cfg.eps,
    )

    rgb = np.stack([
        percentile_normalize_cpu(vv_smooth, 1.0, 99.0, cfg.eps),
        percentile_normalize_cpu(vh_smooth, 1.0, 99.0, cfg.eps),
        percentile_normalize_cpu(diff_map, 1.0, 99.0, cfg.eps),
    ], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)

    return {
        "vv": vv_smooth.astype(np.float32),
        "vh": vh_smooth.astype(np.float32),
        "sum_map": sum_n.astype(np.float32),
        "diff_map": diff_n.astype(np.float32),
        "norm_diff_map": nd_n.astype(np.float32),
        "local_contrast": lc_n.astype(np.float32),
        "saliency": saliency.astype(np.float32),
        "rgb": rgb.astype(np.float32),
    }


def make_heatmap(img01: np.ndarray) -> np.ndarray:
    img_u8 = to_uint8_cpu(img01)
    heatmap_bgr = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def make_overlay(base_gray01: np.ndarray, heatmap_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base_u8 = to_uint8_cpu(base_gray01)
    base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)


def save_direct_png(path: str, img: np.ndarray) -> None:
    if img.ndim == 2:
        cv2.imwrite(path, img)
    elif img.ndim == 3:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError(f"Unsupported shape: {img.shape}")


def write_timing_report(path: str, timing: Timing, shape: Tuple[int, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("GPU Dual-Pol SAR Enhancement Timing Report\n")
        f.write("=" * 48 + "\n")
        f.write(f"Image shape: {shape[0]} x {shape[1]}\n")
        f.write(f"Read time: {timing.read_s:.4f} s\n")
        f.write(f"GPU preprocess time: {timing.gpu_preprocess_s:.4f} s\n")
        f.write(f"GPU features time: {timing.gpu_features_s:.4f} s\n")
        f.write(f"GPU saliency time: {timing.gpu_saliency_s:.4f} s\n")
        f.write(f"GPU post/save-prep time: {timing.gpu_post_s:.4f} s\n")
        f.write(f"GPU total time: {timing.gpu_total_s:.4f} s\n")
        if timing.cpu_total_s is not None and timing.gpu_total_s > 0:
            speedup = timing.cpu_total_s / timing.gpu_total_s
            f.write(f"CPU total time: {timing.cpu_total_s:.4f} s\n")
            f.write(f"Speedup (CPU/GPU): {speedup:.2f}x\n")


def build_pdf_report(path: str, outputs: Dict[str, np.ndarray], overlay: np.ndarray,
                     heatmap: np.ndarray, timing: Timing) -> None:
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        title = "GPU-Accelerated Dual-Pol SAR Enhancement Report"
        subtitle = (
            f"GPU total: {timing.gpu_total_s:.3f} s"
            + (f" | CPU total: {timing.cpu_total_s:.3f} s" if timing.cpu_total_s is not None else "")
        )
        plt.text(0.5, 0.85, title, ha="center", va="center", fontsize=20, fontweight="bold")
        plt.text(0.5, 0.78, subtitle, ha="center", va="center", fontsize=12)
        plt.text(
            0.08,
            0.58,
            "This report summarizes dual-polarization SAR enhancement using VV and VH inputs.\n"
            "RGB = [VV, VH, |VV-VH|], and saliency is built from intensity, polarimetric contrast,\n"
            "normalized difference, and local contrast. This is an enhancement and visualization\n"
            "pipeline, not a final validated ship detector.",
            fontsize=11,
            va="top",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        pages = [
            [(outputs["vv"], "VV (processed)", "gray"),
             (outputs["vh"], "VH (processed)", "gray"),
             (outputs["rgb"], "RGB composite [VV, VH, |VV-VH|]", None)],
            [(outputs["sum_map"], "Sum map", "gray"),
             (outputs["diff_map"], "|VV-VH|", "gray"),
             (outputs["norm_diff_map"], "Normalized difference", "gray")],
            [(outputs["local_contrast"], "Local contrast", "gray"),
             (outputs["saliency"], "Saliency map", "inferno"),
             (heatmap, "Saliency heatmap", None)],
            [(overlay, "Heatmap overlay on VV", None),
             (outputs["rgb"], "RGB composite", None),
             (outputs["saliency"], "Final saliency", "inferno")],
        ]

        for row in pages:
            fig, axs = plt.subplots(1, 3, figsize=(14, 5))
            for ax, (img, title, cmap) in zip(axs, row):
                if img.ndim == 2:
                    ax.imshow(img, cmap=cmap if cmap else "gray")
                else:
                    ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> int:
    cfg = parse_args()
    ensure_dir(cfg.outdir)

    read_t0 = time.perf_counter()
    vv = read_tiff(cfg.vv_path)
    vh = read_tiff(cfg.vh_path)
    read_dt = time.perf_counter() - read_t0

    if vv.shape != vh.shape:
        raise ValueError(f"VV and VH shapes do not match: {vv.shape} vs {vh.shape}")
    if vv.size == 0:
        raise ValueError("Input image is empty")

    outputs, timing = gpu_preprocess(vv, vh, cfg)
    timing.read_s = read_dt

    if cfg.benchmark_cpu:
        cpu_t0 = time.perf_counter()
        _ = cpu_preprocess(vv, vh, cfg)
        timing.cpu_total_s = time.perf_counter() - cpu_t0

    heatmap = make_heatmap(outputs["saliency"])
    overlay = make_overlay(outputs["vv"], heatmap, cfg.overlay_alpha)

    save_direct_png(os.path.join(cfg.outdir, "vv.png"), to_uint8_cpu(outputs["vv"]))
    save_direct_png(os.path.join(cfg.outdir, "vh.png"), to_uint8_cpu(outputs["vh"]))
    save_direct_png(os.path.join(cfg.outdir, "rgb_composite.png"), to_uint8_cpu(outputs["rgb"]))
    save_direct_png(os.path.join(cfg.outdir, "sum_map.png"), to_uint8_cpu(outputs["sum_map"]))
    save_direct_png(os.path.join(cfg.outdir, "diff_map.png"), to_uint8_cpu(outputs["diff_map"]))
    save_direct_png(os.path.join(cfg.outdir, "norm_diff_map.png"), to_uint8_cpu(outputs["norm_diff_map"]))
    save_direct_png(os.path.join(cfg.outdir, "saliency_map.png"), to_uint8_cpu(outputs["saliency"]))
    save_direct_png(os.path.join(cfg.outdir, "saliency_heatmap.png"), heatmap)
    save_direct_png(os.path.join(cfg.outdir, "overlay.png"), overlay)

    build_pdf_report(
        os.path.join(cfg.outdir, "report.pdf"),
        outputs,
        overlay,
        heatmap,
        timing,
    )
    write_timing_report(
        os.path.join(cfg.outdir, "timing_report.txt"),
        timing,
        outputs["vv"].shape,
    )

    print("Done.")
    print(f"Results written to: {cfg.outdir}")
    print(f"GPU total time: {timing.gpu_total_s:.4f} s")
    if timing.cpu_total_s is not None and timing.gpu_total_s > 0:
        print(f"CPU total time: {timing.cpu_total_s:.4f} s")
        print(f"Speedup: {timing.cpu_total_s / timing.gpu_total_s:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
