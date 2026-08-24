"""Read-only inventory for the advisor-requested BOSR clipping supplement."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
HERE = ROOT / "experiments/supplementary_bosr_additive_clipping_20260823"
OUT = HERE / "outputs/audit"
PUBLIC = ROOT / "experiments/innovation_search/ecf_wtnet/bosr_final_public_test_20260814"
PUBLIC_LOCK = PUBLIC / "PUBLIC_TEST_PRESTART_LOCK.json"
PUBLIC_RESULTS = PUBLIC / "results/PUBLIC_TEST_ALL_GROUPS_PER_IMAGE.csv"
OOF = ROOT / "experiments/supplementary_bosr_mechanism_oof_20260817"
OOF_SOURCE = ROOT / "experiments/innovation_search/ecf_wtnet/cr_lrst_gate0_oof_20260813"
TRAIN_MANIFEST = ROOT / "datasets/CEC/internal_split_seed20260731/train_manifest.csv"
TEST_INPUT = ROOT / "datasets/CEC/test/input_under"
TEST_GT = ROOT / "datasets/CEC/test/gt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.is_dir() else 0


def record(item: str, paths: list[Path], directly_usable: bool, needs_inference: bool, notes: str) -> dict:
    existing = [path for path in paths if path.is_file() or path.is_dir()]
    return {
        "item": item,
        "exists": len(existing) == len(paths),
        "path": "; ".join(str(path) for path in paths),
        "count": len(existing),
        "directly_usable": directly_usable and len(existing) == len(paths),
        "needs_inference": needs_inference,
        "notes": notes,
    }


def main() -> None:
    lock = json.loads(PUBLIC_LOCK.read_text(encoding="utf-8"))
    weights = lock["weights"]

    def weight_paths(group: str) -> list[Path]:
        return [Path(item["path"]) for item in weights if item["group"] == group]

    oof_generic = [OOF / f"runs/fold_{fold}/rgb_add_full_b0p1/epoch_50_last.pt" for fold in range(5)]
    oof_bosr = [OOF / f"runs/fold_{fold}/bosr_full_b1/epoch_50_last.pt" for fold in range(5)]
    oof_caches = [OOF_SOURCE / f"runs/fold_{fold}/parent/epoch50_all360_cache" for fold in range(5)]
    cache_count = sum(len(list(path.glob("*.npz"))) for path in oof_caches if path.is_dir())

    inventory = [
        record("WTNet-50 three-seed checkpoints", weight_paths("WTNET-50"), False, True, "Three frozen checkpoints; inference required for uncached float output."),
        record("BOSR three-seed checkpoints", weight_paths("BOSR-ONLY"), False, True, "Three frozen checkpoints; existing clipped public metrics are directly available."),
        record("GenericHead three-seed checkpoints", weight_paths("WTNET-GENERIC"), False, True, "Three frozen checkpoints; inference required to recover raw pre-clipping tensors."),
        record("OOF RGB additive checkpoints", oof_generic, False, True, "Five frozen fold-specific checkpoints; no retraining authorized."),
        record("OOF BOSR checkpoints", oof_bosr, False, True, "Five frozen fold-specific checkpoints."),
        record("train360 OOF parent caches", oof_caches, cache_count == 1800, True, f"{cache_count} NPZ files across five folds (360 per fold); held-out subset selected by frozen fold manifest."),
        record("CEC test100 existing metric predictions", [PUBLIC_RESULTS], True, False, "Contains 1,800 rows for 3 seeds x 6 groups x 100 images; metrics are computed from final outputs."),
        record("CEC test100 input images", [TEST_INPUT], png_count(TEST_INPUT) == 100, False, f"{png_count(TEST_INPUT)} PNG files."),
        record("CEC test100 GT/reference images", [TEST_GT], png_count(TEST_GT) == 100, False, f"{png_count(TEST_GT)} PNG files."),
        record("train360 manifest", [TRAIN_MANIFEST], True, False, "Frozen train360 image list."),
        record("OOF fold assignment", [OOF_SOURCE / "FOLD_ASSIGNMENTS.csv"], True, False, "Frozen five-fold assignment."),
        record("Frozen metric implementation", [ROOT / "experiments/innovation_search/ecf_wtnet/paper_final_protocol_20260810/metric_statistics/metrics_core.py"], True, False, "BasicSR PSNR/SSIM and scikit-image 0.24.0 CIEDE2000."),
        record("Existing OOF per-image metrics", [OOF / "OOF_ALL_GROUPS_PER_IMAGE.csv"], True, False, "Table-5 source rows, including preclamp and final OOR fields."),
        record("Existing bootstrap/statistical script", [OOF / "analyze_mechanism_oof.py"], True, False, "Existing image-level 10,000-resample implementation."),
    ]

    missing = [item["item"] for item in inventory if not item["exists"]]
    payload = {
        "status": "PASS_FILE_AUDIT" if not missing else "STOP_MISSING_REQUIRED_OBJECTS",
        "missing": missing,
        "public_lock_status": lock.get("status"),
        "inventory": inventory,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "source_hashes": {
            "public_lock": sha256(PUBLIC_LOCK),
            "public_results": sha256(PUBLIC_RESULTS),
            "oof_results": sha256(OOF / "OOF_ALL_GROUPS_PER_IMAGE.csv"),
            "train_manifest": sha256(TRAIN_MANIFEST),
            "fold_manifest": sha256(OOF_SOURCE / "FOLD_ASSIGNMENTS.csv"),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "file_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    columns = ["item", "exists", "path", "count", "directly_usable", "needs_inference", "notes"]
    lines = ["# File inventory", "", f"Status: **{payload['status']}**", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in inventory:
        values = [str(row[key]).replace("|", "\\|") for key in columns]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Environment", "", "```json", json.dumps(payload["environment"], ensure_ascii=False, indent=2), "```", ""])
    (OUT / "file_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "items": len(inventory), "missing": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
