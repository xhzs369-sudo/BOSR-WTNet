"""Hash-locked, one-pass evaluation of frozen BOSR heads on development-used CEC dev40."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BOSR_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
LOCK = HERE / "EVALUATION_PRESTART_LOCK.json"
OUTDIR = HERE / "evaluation"
FINAL = OUTDIR / "DEV40_PAPER_MATRIX_PER_IMAGE.csv"
JOURNAL = OUTDIR / "DEV40_EVALUATION_IN_PROGRESS.ndjson"
ACCESS = OUTDIR / "DEV40_EVALUATION_ACCESS.json"
MANIFEST = ROOT / "datasets/CEC/internal_split_seed20260731/val_manifest.csv"
BASELINE = ROOT / "experiments/innovation_search/ecf_wtnet/paper_final_protocol_20260810/formal_evaluation/DEV40_ALL_GROUPS_PER_IMAGE.csv"
METRICS_DIR = ROOT / "experiments/innovation_search/ecf_wtnet/paper_final_protocol_20260810/metric_statistics"
COMMON_DIR = ROOT / "experiments/innovation_search/ecf_wtnet/paper_final_protocol_20260810/implementation"
ECF_DIR = ROOT / "experiments/innovation_search/ecf_wtnet/advisor_confirmation_single_seed"
SEEDS = (20260801, 20260802, 20260803)
GROUPS = (("BOSR-ONLY", "bosr_only"), ("ECF+BOSR", "ecf_plus_bosr"))

sys.path[:0] = [str(HERE), str(METRICS_DIR), str(COMMON_DIR), str(ECF_DIR)]
from formal_bosr_head import FormalBOSRHead  # noqa: E402
from formal_common import existing_parent_and_cache  # noqa: E402
from ecf_common import CaptureCalibration, load_wtnet  # noqa: E402
from metrics_core import calculate_full_reference_metrics, calculate_region_luminance_metrics  # noqa: E402

FIELDS = [
    "seed", "group", "filename", "mse", "psnr_db", "ssim", "delta_e00",
    "dark_pixel_count", "dark_computable", "dark_luminance_mae",
    "normal_pixel_count", "normal_computable", "normal_luminance_mae",
    "highlight_pixel_count", "highlight_computable", "highlight_luminance_mae",
    "alpha_abs_mean", "beta_abs_mean", "u_abs_mean",
    "preclip_out_of_range_fraction", "final_out_of_range_fraction",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"unreadable image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def infer_parent(model, image: torch.Tensor) -> torch.Tensor:
    h, w = image.shape[-2:]
    ph, pw = (-h) % 8, (-w) % 8
    x = F.pad(image, (0, pw, 0, ph), mode="reflect") if ph or pw else image
    return model(x)[..., :h, :w]


def load_items() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    forbidden = ("/test/", "\\test\\", "endo4ie", "rle")
    if len(rows) != 40 or len({r["filename"] for r in rows}) != 40 or any(r.get("split") != "val" for r in rows):
        raise RuntimeError("dev40 manifest structure mismatch")
    if any(any(t in (r["input_path"] + r["gt_path"]).lower() for t in forbidden) for r in rows):
        raise RuntimeError("sealed dataset path detected")
    return rows


def record(seed: int, group: str, name: str, output: torch.Tensor, gt: np.ndarray, diag=None) -> dict:
    pred = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float64), 0.0, 1.0)
    row = {"seed": seed, "group": group, "filename": name}
    row.update(calculate_full_reference_metrics(pred, gt))
    regions = calculate_region_luminance_metrics(pred, gt)
    for region in ("dark", "normal", "highlight"):
        for key in ("pixel_count", "computable", "luminance_mae"):
            row[f"{region}_{key}"] = regions[region][key]
    for key in ("alpha_abs_mean", "beta_abs_mean", "u_abs_mean", "preclip_out_of_range_fraction", "final_out_of_range_fraction"):
        row[key] = float("nan")
    if diag is not None:
        row["alpha_abs_mean"] = float(diag["alpha"].abs().mean().cpu())
        row["beta_abs_mean"] = float(diag["beta"].abs().mean().cpu())
        row["u_abs_mean"] = float(diag["u"].abs().mean().cpu())
        pre = diag["stage_preclip"]
        row["preclip_out_of_range_fraction"] = float(((pre < 0) | (pre > 1)).float().mean().cpu())
        row["final_out_of_range_fraction"] = float(((output < 0) | (output > 1)).float().mean().cpu())
    return row


def main() -> None:
    if FINAL.exists() or ACCESS.exists() or JOURNAL.exists():
        raise RuntimeError("evaluation is one-pass; output already exists")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock.get("status") != "AUTHORIZED_FOR_ONE_READONLY_DEV40_EVALUATION":
        raise RuntimeError("invalid prestart lock")
    for item in lock["files"]:
        path = Path(item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"locked file mismatch: {path}")
    items = load_items()
    with BASELINE.open("r", encoding="utf-8-sig", newline="") as f:
        old = list(csv.DictReader(f))
    selected = []
    rename = {"ECF-FULL": "ECF"}
    for r in old:
        if r["group"] in ("WTNET-50", "WTNET-100", "ECF-FULL"):
            selected.append({key: r.get(key, "nan") for key in FIELDS} | {"group": rename.get(r["group"], r["group"])})
    if len(selected) != 360:
        raise RuntimeError("frozen baseline extraction mismatch")
    baseline_index = {(int(r["seed"]), r["filename"]): r for r in selected if r["group"] == "WTNET-50"}
    records = list(selected)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    with torch.inference_mode(), JOURNAL.open("x", encoding="utf-8") as journal:
        for seed in SEEDS:
            parent_path, _ = existing_parent_and_cache(seed)
            parent = load_wtnet(device, parent_path).eval()
            capture = CaptureCalibration(parent)
            heads = {}
            for group, folder in GROUPS:
                model = FormalBOSRHead(group).to(device).eval()
                payload = torch.load(HERE / f"runs/seed_{seed}/{folder}/epoch_50_last.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(payload["params"], strict=True)
                heads[group] = model
            for idx, item in enumerate(items, 1):
                low, gt = read_rgb(item["input_path"]), read_rgb(item["gt_path"])
                if low.shape != gt.shape:
                    raise RuntimeError(f"shape mismatch: {item['filename']}")
                image = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).to(device)
                out1 = infer_parent(parent, image)
                ic = capture.value
                if ic is None:
                    raise RuntimeError("ECM capture failed")
                ic = ic[..., :image.shape[-2], :image.shape[-1]]
                check = record(seed, "WTNET-50", item["filename"], torch.clamp(out1, 0, 1), gt)
                ref = baseline_index[(seed, item["filename"])]
                for metric in ("mse", "psnr_db", "ssim", "delta_e00"):
                    if abs(float(check[metric]) - float(ref[metric])) > 1e-8:
                        raise RuntimeError(f"parent reproducibility mismatch: {seed} {item['filename']} {metric}")
                for group, head in heads.items():
                    output, diag = head(image, ic, out1)
                    row = record(seed, group, item["filename"], output, gt, diag)
                    records.append(row)
                    journal.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                journal.flush()
                print(f"seed={seed} image={idx}/40", flush=True)
            capture.close()
            del parent, heads
            torch.cuda.empty_cache()
    if len(records) != 600 or len({(int(r["seed"]), r["group"], r["filename"]) for r in records}) != 600:
        raise RuntimeError("paper matrix completeness failure")
    tmp = FINAL.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(records)
    os.replace(tmp, FINAL)
    JOURNAL.unlink()
    ACCESS.write_text(json.dumps({
        "status": "COMPLETE_ONE_READONLY_DEV40_EVALUATION", "unique_images": 40,
        "paper_matrix_rows": 600, "new_inference_rows": 240, "input_reads": 120, "gt_reads": 120,
        "model_parameter_updates": 0, "result_sha256": sha(FINAL),
        "CEC_public_test_access": 0, "Endo4IE_access": 0, "RLE_access": 0,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "rows": 600, "sha256": sha(FINAL)}))


if __name__ == "__main__":
    main()
