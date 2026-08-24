"""Re-infer frozen train360 OOF additive raw/clipped and BOSR outputs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from supplement_common import OUTPUTS, clipping_stats, final_metrics, raw_metrics, save_float_prediction, write_csv


ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
SOURCE = ROOT / "experiments/supplementary_bosr_mechanism_oof_20260817"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from mechanism_common import fold_rows, group_paths, read_rgb, source_cache  # noqa: E402
from mechanism_heads import build_head  # noqa: E402


def load_head(fold: int, group: str, device: torch.device):
    model = build_head(group).to(device).eval()
    payload = torch.load(group_paths(fold, group)["final"], map_location="cpu", weights_only=False)
    model.load_state_dict(payload["params"], strict=True)
    return model


def metric_row(fold: int, filename: str, method: str, stage: str, metrics: dict, extra: dict | None = None) -> dict:
    row = {"fold": fold, "seed": 20260813, "image_id": filename, "method": method, "stage": stage, **metrics}
    if extra:
        row.update(extra)
    return row


def main() -> None:
    metric_path = OUTPUTS / "metrics/oof_additive_raw_clipped_per_image.csv"
    if metric_path.exists():
        raise RuntimeError("OOF supplementary result already exists; overwrite forbidden")
    raw_dir = OUTPUTS / "additive_raw/oof_seed_20260813"
    clipped_dir = OUTPUTS / "additive_clipped/oof_seed_20260813"
    bosr_dir = OUTPUTS / "bosr_final/oof_seed_20260813"
    rows: list[dict] = []
    device = torch.device("cuda")
    with torch.inference_mode():
        for fold in range(5):
            additive = load_head(fold, "RGB_ADD_FULL_B0P1", device)
            bosr = load_head(fold, "BOSR_FULL_B1", device)
            for index, item in enumerate(fold_rows(fold, "holdout"), 1):
                low = read_rgb(item["input_path"])
                gt = read_rgb(item["gt_path"]).astype(np.float64)
                cache_path = source_cache(fold) / f"{Path(item['filename']).stem}.npz"
                with np.load(cache_path, allow_pickle=False) as data:
                    ic_np = data["Ic_raw"].astype(np.float32)
                    i1_np = data["I1_raw"].astype(np.float32)
                i0 = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).to(device)
                ic = torch.from_numpy(ic_np).permute(2, 0, 1).unsqueeze(0).to(device)
                i1 = torch.from_numpy(i1_np).permute(2, 0, 1).unsqueeze(0).to(device)
                add_clipped_t, diag = additive(i0, ic, i1)
                bosr_t, _ = bosr(i0, ic, i1)
                add_raw = diag["preclamp"][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                add_clipped = add_clipped_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                bosr_final = bosr_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                if float(np.max(np.abs(add_clipped - np.clip(add_raw, 0, 1)))) > 1e-7:
                    raise RuntimeError(f"clipping identity mismatch: fold={fold} {item['filename']}")
                stats = clipping_stats(add_raw)
                rows.append(metric_row(fold, item["filename"], "RGB_ADD_FULL_B0P1", "pre-clipping", raw_metrics(add_raw, gt), stats))
                rows.append(metric_row(fold, item["filename"], "RGB_ADD_FULL_B0P1", "post-clipping", final_metrics(add_clipped, gt), stats))
                rows.append(metric_row(fold, item["filename"], "BOSR_FULL_B1", "final", final_metrics(bosr_final, gt), {"total_oor_fraction": 0.0}))
                save_float_prediction(raw_dir / f"{Path(item['filename']).stem}.npz", add_raw)
                save_float_prediction(clipped_dir / f"{Path(item['filename']).stem}.npz", add_clipped)
                save_float_prediction(bosr_dir / f"{Path(item['filename']).stem}.npz", bosr_final)
                print(f"fold={fold} image={index}/72 {item['filename']}", flush=True)
            del additive, bosr
            torch.cuda.empty_cache()
    if len(rows) != 1080 or len({(row["image_id"], row["method"], row["stage"]) for row in rows}) != 1080:
        raise RuntimeError("OOF completeness failure")
    write_csv(metric_path, rows)
    status = {"status": "COMPLETE", "rows": len(rows), "images": 360, "training": 0, "seed": 20260813}
    (OUTPUTS / "audit/oof_inference_complete.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
