"""Re-infer frozen three-seed CEC test100 GenericHead raw/clipped and BOSR output."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from supplement_common import OUTPUTS, clipping_stats, final_metrics, raw_metrics, save_float_prediction, sha256, write_csv


ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
PUBLIC = ROOT / "experiments/innovation_search/ecf_wtnet/bosr_final_public_test_20260814"
LOCK_PATH = PUBLIC / "PUBLIC_TEST_PRESTART_LOCK.json"
INPUT = ROOT / "datasets/CEC/test/input_under"
GT = ROOT / "datasets/CEC/test/gt"
SEEDS = (20260801, 20260802, 20260803)
sys.path[:0] = [
    str(ROOT / "experiments/innovation_search/ecf_wtnet/formal_bosr_three_seed_20260814"),
    str(ROOT / "experiments/innovation_search/ecf_wtnet/paper_final_protocol_20260810/implementation"),
    str(ROOT / "experiments/innovation_search/ecf_wtnet/advisor_confirmation_single_seed"),
]
from formal_bosr_head import FormalBOSRHead  # noqa: E402
from formal_heads import build_head  # noqa: E402
from ecf_common import CaptureCalibration, load_wtnet  # noqa: E402


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"unreadable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def infer(model, image: torch.Tensor) -> torch.Tensor:
    height, width = image.shape[-2:]
    pad_h, pad_w = (-height) % 8, (-width) % 8
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else image
    return model(padded)[..., :height, :width]


def get_weight(lock: dict, seed: int, group: str) -> Path:
    matches = [Path(item["path"]) for item in lock["weights"] if item["seed"] == seed and item["group"] == group]
    if len(matches) != 1:
        raise RuntimeError(f"weight identity failure: {seed} {group}")
    return matches[0]


def metric_row(seed: int, image_id: str, method: str, stage: str, metrics: dict, extra: dict | None = None) -> dict:
    row = {"seed": seed, "image_id": image_id, "method": method, "stage": stage, **metrics}
    if extra:
        row.update(extra)
    return row


def main() -> None:
    metric_path = OUTPUTS / "metrics/test100_all_stages_per_image.csv"
    if metric_path.exists():
        raise RuntimeError("public supplementary result already exists; overwrite forbidden")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("status") != "AUTHORIZED_FOR_ONE_PASS_CEC_PUBLIC_TEST":
        raise RuntimeError("unexpected public-test lock")
    for item in lock["locked_files"] + lock["weights"]:
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen file changed: {path}")
    rows: list[dict] = []
    device = torch.device("cuda")
    with torch.inference_mode():
        for seed in SEEDS:
            parent = load_wtnet(device, get_weight(lock, seed, "WTNET-50")).eval()
            capture = CaptureCalibration(parent)
            generic = build_head("WTNET-GENERIC").to(device).eval()
            generic.load_state_dict(torch.load(get_weight(lock, seed, "WTNET-GENERIC"), map_location="cpu", weights_only=False)["params"], strict=True)
            bosr = FormalBOSRHead("BOSR-ONLY").to(device).eval()
            bosr.load_state_dict(torch.load(get_weight(lock, seed, "BOSR-ONLY"), map_location="cpu", weights_only=False)["params"], strict=True)
            for index in range(100):
                name = f"{index}.png"
                low, gt = read_rgb(INPUT / name), read_rgb(GT / name).astype(np.float64)
                image = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).to(device)
                parent_raw_t = infer(parent, image)
                ic = capture.value
                if ic is None:
                    raise RuntimeError("ECM capture failed")
                ic = ic[..., :image.shape[-2], :image.shape[-1]]
                generic_clipped_t, diag = generic(image, ic, parent_raw_t)
                generic_raw_t = parent_raw_t + diag["residual"]
                bosr_t, _ = bosr(image, ic, parent_raw_t)
                parent_final = torch.clamp(parent_raw_t, 0, 1)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                add_raw = generic_raw_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                add_clipped = generic_clipped_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                bosr_final = bosr_t[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                if float(np.max(np.abs(add_clipped - np.clip(add_raw, 0, 1)))) > 1e-7:
                    raise RuntimeError(f"public clipping identity mismatch: {seed} {name}")
                stats = clipping_stats(add_raw)
                rows.append(metric_row(seed, name, "WTNet-50", "final", final_metrics(parent_final, gt), {"total_oor_fraction": 0.0}))
                rows.append(metric_row(seed, name, "GenericHead", "pre-clipping", raw_metrics(add_raw, gt), stats))
                rows.append(metric_row(seed, name, "GenericHead", "post-clipping", final_metrics(add_clipped, gt), stats))
                rows.append(metric_row(seed, name, "BOSR-WTNet", "final", final_metrics(bosr_final, gt), {"total_oor_fraction": 0.0}))
                base = OUTPUTS / "public_float_predictions" / f"seed_{seed}" / Path(name).stem
                save_float_prediction(base / "wtnet50_final.npz", parent_final)
                save_float_prediction(base / "additive_raw.npz", add_raw)
                save_float_prediction(base / "additive_clipped.npz", add_clipped)
                save_float_prediction(base / "bosr_final.npz", bosr_final)
                print(f"seed={seed} image={index + 1}/100", flush=True)
            capture.close()
            del parent, generic, bosr
            torch.cuda.empty_cache()
    if len(rows) != 1200 or len({(row["seed"], row["image_id"], row["method"], row["stage"]) for row in rows}) != 1200:
        raise RuntimeError("public completeness failure")
    write_csv(metric_path, rows)
    status = {"status": "COMPLETE", "rows": len(rows), "images": 100, "seeds": list(SEEDS), "training": 0}
    (OUTPUTS / "audit/public_inference_complete.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
