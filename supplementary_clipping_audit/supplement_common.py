"""Shared locked metric and serialization helpers for the clipping supplement."""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
HERE = ROOT / "experiments/supplementary_bosr_additive_clipping_20260823"
OUTPUTS = HERE / "outputs"
METRIC_ROOT = ROOT / "experiments/innovation_search/ecf_wtnet/paper_final_protocol_20260810/metric_statistics"
if str(METRIC_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIC_ROOT))
from metrics_core import calculate_full_reference_metrics, calculate_psnr, calculate_ssim  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Mirror the frozen metric formulas but permit finite predictions outside [0,1]."""
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"invalid RGB pair: {pred.shape}, {gt.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("non-finite prediction or target")
    if gt.min() < 0 or gt.max() > 1:
        raise ValueError("target outside [0,1]")
    mse = float(np.mean(np.square(pred - gt), dtype=np.float64))
    # The frozen BasicSR implementation infers its dynamic range from
    # ``prediction.max()``.  That heuristic is invalid for the deliberately
    # out-of-range pre-clipping output: a value such as 1.01 would be treated
    # as an image in [0,255].  Scaling both arrays by 255 forces BasicSR onto
    # its 255 branch while remaining exactly equivalent to data_range=1.
    pred_metric = pred * 255.0
    gt_metric = gt * 255.0
    return {
        "mse": mse,
        "psnr_db": float(calculate_psnr(pred_metric, gt_metric, crop_border=0, input_order="HWC", test_y_channel=False)),
        "ssim": float(calculate_ssim(pred_metric, gt_metric, crop_border=0, input_order="HWC", test_y_channel=False)),
        "delta_e00": float(np.mean(deltaE_ciede2000(rgb2lab(pred), rgb2lab(gt)), dtype=np.float64)),
    }


def final_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return calculate_full_reference_metrics(
        np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    )


def clipping_stats(raw: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(raw, dtype=np.float64)
    below = array < 0
    above = array > 1
    mask = below | above
    clipped = np.clip(array, 0, 1)
    magnitude = np.abs(array - clipped)
    active = magnitude[mask]
    return {
        "element_count": int(array.size),
        "below_zero_count": int(below.sum()),
        "above_one_count": int(above.sum()),
        "total_oor_count": int(mask.sum()),
        "below_zero_fraction": float(below.mean()),
        "above_one_fraction": float(above.mean()),
        "total_oor_fraction": float(mask.mean()),
        "mean_clip_magnitude_active": float(active.mean()) if active.size else 0.0,
        "max_clip_magnitude": float(active.max()) if active.size else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def save_float_prediction(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, prediction=np.asarray(array, dtype=np.float32))
    os.replace(temp, path)
