"""Correct pre-clipping PSNR/SSIM from frozen float caches without inference."""
from __future__ import annotations

import json
import csv
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from supplement_common import OUTPUTS, raw_metrics, sha256

ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
SOURCE = ROOT / "experiments/supplementary_bosr_mechanism_oof_20260817"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from mechanism_common import fold_rows, read_rgb  # noqa: E402


def main() -> None:
    metric_path = OUTPUTS / "metrics/oof_additive_raw_clipped_per_image.csv"
    backup = OUTPUTS / "audit/oof_metrics_before_dynamic_range_correction.csv"
    if backup.exists():
        raise RuntimeError("correction already applied; second run forbidden")
    before_hash = sha256(metric_path)
    shutil.copy2(metric_path, backup)
    with metric_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    lookup = {}
    for fold in range(5):
        for item in fold_rows(fold, "holdout"):
            lookup[(fold, item["filename"])] = item["gt_path"]
    selected = [row for row in rows if row["method"] == "RGB_ADD_FULL_B0P1" and row["stage"] == "pre-clipping"]
    if len(selected) != 360:
        raise RuntimeError(f"expected 360 pre-clipping rows, got {len(selected)}")
    for number, row in enumerate(selected, 1):
        cache = OUTPUTS / "additive_raw/oof_seed_20260813" / f"{Path(row['image_id']).stem}.npz"
        with np.load(cache, allow_pickle=False) as data:
            prediction = data["prediction"].astype(np.float64)
        target = read_rgb(lookup[(int(row["fold"]), row["image_id"])]).astype(np.float64)
        metrics = raw_metrics(prediction, target)
        for name, value in metrics.items():
            row[name] = repr(value)
        if number % 20 == 0:
            print(f"recomputed={number}/360", flush=True)
    temp = metric_path.with_suffix(".corrected.tmp.csv")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, metric_path)
    record = {
        "status": "CORRECTED_FROM_FROZEN_FLOAT_CACHE_NO_MODEL_INFERENCE",
        "reason": "BasicSR dynamic-range heuristic misclassified pre-clipping values above 1 as [0,255]",
        "fix": "scale prediction and GT by 255 for PSNR/SSIM, algebraically enforcing data_range=1",
        "rows_corrected": 360,
        "before_csv_sha256": before_hash,
        "backup_sha256": sha256(backup),
        "corrected_csv_sha256": sha256(metric_path),
        "training": 0,
        "model_inference": 0,
    }
    path = OUTPUTS / "audit/oof_raw_metric_dynamic_range_correction.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
