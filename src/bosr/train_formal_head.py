"""Train one frozen full-train360 formal BOSR group with exact resume support."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
FORMAL_IMPL = HERE.parent / "ecf"
sys.path.insert(0, str(FORMAL_IMPL))
from formal_common import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
    capture_rng_state,
    existing_parent_and_cache,
    read_rgb,
    restore_rng_state,
    seed_all,
    sha256_file,
    train_rows,
)
from formal_bosr_head import FormalBOSRHead, parameter_count  # noqa: E402

PROTOCOL_ID = "ECF_BOSR_ADVISOR_AUTHORIZED_THREE_SEED_20260814_V1"
SEEDS = (20260801, 20260802, 20260803)
EPOCHS, CROP_SIZE, BATCH_SIZE = 50, 128, 8
LEARNING_RATE, MIN_LEARNING_RATE = 1e-3, 1e-5


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedTrainingDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], cache: Path) -> None:
        self.rows, self.cache = rows, cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.rows[index]
        i0, target = read_rgb(row["input_path"]), read_rgb(row["gt_path"])
        with np.load(self.cache / f"{Path(row['filename']).stem}.npz", allow_pickle=False) as stored:
            ic = stored["Ic_raw"].astype(np.float32)
            i1 = stored["I1_raw"].astype(np.float32)
        height, width = i0.shape[:2]
        top, left = random.randint(0, height - CROP_SIZE), random.randint(0, width - CROP_SIZE)
        arrays = [x[top : top + CROP_SIZE, left : left + CROP_SIZE] for x in (i0, ic, i1, target)]
        if random.random() < 0.5:
            arrays = [x[:, ::-1] for x in arrays]
        if random.random() < 0.5:
            arrays = [x[::-1] for x in arrays]
        return tuple(torch.from_numpy(x.copy()).permute(2, 0, 1) for x in arrays)


def ssim_index(x: torch.Tensor, y: torch.Tensor, window: int = 11) -> torch.Tensor:
    pad = window // 2
    mx, my = F.avg_pool2d(x, window, 1, pad), F.avg_pool2d(y, window, 1, pad)
    vx = F.avg_pool2d(x * x, window, 1, pad) - mx.square()
    vy = F.avg_pool2d(y * y, window, 1, pad) - my.square()
    vxy = F.avg_pool2d(x * y, window, 1, pad) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * vxy + c2)) /
            ((mx.square() + my.square() + c1) * (vx + vy + c2))).mean()


def identity(group: str, seed: int) -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "group": group,
        "seed": seed,
        "preregistration_sha256": sha256_text(HERE / "PREREGISTRATION.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=FormalBOSRHead.GROUPS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()

    output = HERE / "runs" / f"seed_{args.seed}" / args.group.lower().replace("+", "_plus_").replace("-", "_")
    final = output / "epoch_50_last.pt"
    resume = output / "latest_resume.pt"
    complete = output / "TRAINING_COMPLETE.json"
    if final.exists() or complete.exists():
        raise RuntimeError("formal final output already exists; retraining is forbidden")

    parent_checkpoint, cache = existing_parent_and_cache(args.seed)
    rows = train_rows()
    seed_all(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        CachedTrainingDataset(rows, cache), batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, generator=loader_generator, drop_last=True,
    )
    device = torch.device("cuda")
    model = FormalBOSRHead(args.group).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(loader), eta_min=MIN_LEARNING_RATE
    )
    start_epoch, history = 0, []
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state.get("identity") != identity(args.group, args.seed):
            raise RuntimeError("resume identity mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng_state(state["rng"], loader_generator)
        start_epoch, history = int(state["epoch"]), state["history"]

    started = time.time()
    model.train()
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        losses, gradients, magnitudes = [], [], []
        for i0, ic, i1, target in loader:
            i0, ic, i1, target = [x.to(device) for x in (i0, ic, i1, target)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction, diagnostics = model(i0, ic, i1)
                loss = F.mse_loss(prediction, target) + 0.05 * (1.0 - ssim_index(prediction, target))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(loss) or not torch.isfinite(gradient):
                raise RuntimeError("non-finite loss or gradient")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            gradients.append(float(gradient.detach().cpu()))
            magnitudes.append(float(diagnostics["u"].abs().mean().detach().cpu()))
        record = {
            "protocol_id": PROTOCOL_ID, "group": args.group, "seed": args.seed,
            "epoch": epoch, "loss": float(np.mean(losses)),
            "mean_abs_u": float(np.mean(magnitudes)),
            "max_preclip_grad_norm": max(gradients),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_this_invocation_seconds": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        atomic_torch_save({
            "identity": identity(args.group, args.seed), "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(loader_generator), "history": history,
            "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        }, resume)

    atomic_torch_save({
        "identity": identity(args.group, args.seed), "epoch": EPOCHS,
        "params": model.state_dict(), "history": history,
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
    }, final)
    atomic_json_save({
        "status": "FORMAL_TRAIN360_COMPLETE", "protocol_id": PROTOCOL_ID,
        "group": args.group, "seed": args.seed, "parameters": parameter_count(model),
        "fit_images": len(rows), "epochs": EPOCHS,
        "decision_checkpoint": str(final), "decision_checkpoint_sha256": sha256_file(final),
        "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
        "sealed_access": {"CEC_dev40": 0, "RLE": 0, "CEC_public_test": 0, "Endo4IE": 0},
    }, complete)


if __name__ == "__main__":
    main()
