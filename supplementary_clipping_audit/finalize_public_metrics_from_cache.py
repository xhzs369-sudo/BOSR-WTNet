"""Finalize the public-test table after the write-only serialization incident.

No model is loaded. Frozen final-output metrics are copied from the already
audited public-test CSV; only pre-clipping GenericHead metrics are calculated
from the newly saved float32 caches.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from supplement_common import OUTPUTS, clipping_stats, final_metrics, raw_metrics, sha256, write_csv

ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
SOURCE_CSV = ROOT / "experiments/innovation_search/ecf_wtnet/bosr_final_public_test_20260814/results/PUBLIC_TEST_ALL_GROUPS_PER_IMAGE.csv"
GT_DIR = ROOT / "datasets/CEC/test/gt"
SEEDS = (20260801, 20260802, 20260803)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_prediction(seed: int, image_id: str, name: str) -> np.ndarray:
    path = OUTPUTS / "public_float_predictions" / f"seed_{seed}" / Path(image_id).stem / name
    with np.load(path, allow_pickle=False) as data:
        array = data["prediction"].astype(np.float64)
    if array.shape != (420, 420, 3) or not np.isfinite(array).all():
        raise RuntimeError(f"invalid cache: {path} {array.shape}")
    return array


def read_gt(image_id: str) -> np.ndarray:
    import cv2

    path = GT_DIR / image_id
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"unreadable GT: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def metric_fields(source: dict) -> dict:
    return {name: float(source[name]) for name in ("mse", "psnr_db", "ssim", "delta_e00")}


def main() -> None:
    target = OUTPUTS / "metrics/test100_all_stages_per_image.csv"
    if target.exists():
        raise RuntimeError("public metric table already exists; overwrite forbidden")
    source_rows = read_rows(SOURCE_CSV)
    lookup = {(int(row["seed"]), row["filename"], row["group"]): row for row in source_rows}
    expected = 3 * 100 * 4
    cache_files = list((OUTPUTS / "public_float_predictions").rglob("*.npz"))
    if len(cache_files) != expected:
        raise RuntimeError(f"expected {expected} caches, got {len(cache_files)}")
    rows: list[dict] = []
    sample_checks = []
    for seed in SEEDS:
        for index in range(100):
            image_id = f"{index}.png"
            gt = read_gt(image_id)
            raw = load_prediction(seed, image_id, "additive_raw.npz")
            clipped = load_prediction(seed, image_id, "additive_clipped.npz")
            bosr = load_prediction(seed, image_id, "bosr_final.npz")
            parent = load_prediction(seed, image_id, "wtnet50_final.npz")
            if float(np.max(np.abs(clipped - np.clip(raw, 0, 1)))) > 1e-7:
                raise RuntimeError(f"clipping identity mismatch: {seed} {image_id}")
            stats = clipping_stats(raw)
            base = {"seed": seed, "image_id": image_id}
            rows.append({**base, "method": "WTNet-50", "stage": "final", **metric_fields(lookup[(seed, image_id, "WTNet-50")]), "total_oor_fraction": 0.0})
            rows.append({**base, "method": "GenericHead", "stage": "pre-clipping", **raw_metrics(raw, gt), **stats})
            rows.append({**base, "method": "GenericHead", "stage": "post-clipping", **metric_fields(lookup[(seed, image_id, "GenericHead")]), **stats})
            rows.append({**base, "method": "BOSR-WTNet", "stage": "final", **metric_fields(lookup[(seed, image_id, "BOSR-WTNet")]), "total_oor_fraction": 0.0})
            if index == 0:
                for method, prediction, group in (("WTNet-50", parent, "WTNet-50"), ("GenericHead", clipped, "GenericHead"), ("BOSR-WTNet", bosr, "BOSR-WTNet")):
                    recomputed = final_metrics(prediction, gt)
                    frozen = metric_fields(lookup[(seed, image_id, group)])
                    diffs = {key: abs(recomputed[key] - frozen[key]) for key in recomputed}
                    # Float caches are stored as float32, so PSNR can differ
                    # by a few micro-dB from the original in-memory result.
                    if max(diffs.values()) > 1e-5:
                        raise RuntimeError(f"frozen metric reproduction mismatch: {seed} {method} {diffs}")
                    sample_checks.append({"seed": seed, "image_id": image_id, "method": method, "max_abs_metric_difference": max(diffs.values())})
            if (index + 1) % 20 == 0:
                print(f"seed={seed} finalized={index + 1}/100", flush=True)
    if len(rows) != 1200:
        raise RuntimeError(f"expected 1200 rows, got {len(rows)}")
    write_csv(target, rows)
    incident = {
        "status": "RECOVERED_FROM_COMPLETE_FLOAT_CACHES_WITHOUT_REINFERENCE",
        "incident": "initial CSV serialization used the first row's fieldnames and rejected later clipping-stat fields",
        "inference_completed_before_incident": "300/300 images",
        "cache_count": len(cache_files),
        "source_frozen_public_csv": str(SOURCE_CSV),
        "source_frozen_public_csv_sha256": sha256(SOURCE_CSV),
        "raw_metrics_recomputed_from_cache": 300,
        "final_metrics_copied_from_frozen_public_csv": 900,
        "sample_reproduction_checks": sample_checks,
        "result_rows": len(rows),
        "result_sha256": sha256(target),
        "training": 0,
        "model_reinference": 0,
    }
    path = OUTPUTS / "audit/public_csv_write_incident_and_recovery.json"
    path.write_text(json.dumps(incident, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = {"status": "COMPLETE_AFTER_SERIALIZATION_RECOVERY", "rows": 1200, "images": 100, "seeds": list(SEEDS), "training": 0}
    (OUTPUTS / "audit/public_inference_complete.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
