"""Reproduce the manuscript Table 5 source values before new analysis."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
HERE = ROOT / "experiments/supplementary_bosr_additive_clipping_20260823"
SOURCE = ROOT / "experiments/supplementary_bosr_mechanism_oof_20260817"
CSV_PATH = SOURCE / "OOF_ALL_GROUPS_PER_IMAGE.csv"
OUT = HERE / "outputs/audit"

EXPECTED = {
    "BOSR_FULL_B1": {"mse": 0.0008225, "psnr_db": 32.5608, "ssim": 0.973526, "delta_e00": 2.5364, "preclamp_out_of_range_fraction": 0.0},
    "RGB_ADD_FULL_B0P1": {"mse": 0.0008079, "psnr_db": 32.6938, "ssim": 0.981728, "delta_e00": 2.4954, "preclamp_out_of_range_fraction": 0.04505},
}
TOLERANCE = {"mse": 5e-8, "psnr_db": 5e-5, "ssim": 5e-7, "delta_e00": 5e-5, "preclamp_out_of_range_fraction": 5e-6}


def main() -> None:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    output = {}
    failures = []
    for group, expected in EXPECTED.items():
        subset = [row for row in rows if row["group"] == group]
        if len(subset) != 360:
            failures.append(f"{group}: expected 360 rows, got {len(subset)}")
            continue
        actual = {key: float(np.mean([float(row[key]) for row in subset], dtype=np.float64)) for key in expected}
        deltas = {key: actual[key] - expected[key] for key in expected}
        passed = {key: abs(deltas[key]) <= TOLERANCE[key] for key in expected}
        if not all(passed.values()):
            failures.append(f"{group}: mismatch {passed}")
        output[group] = {"n": len(subset), "expected": expected, "actual": actual, "delta": deltas, "passed": passed}

    result = {
        "status": "PASS_EXISTING_TABLE5_REPRODUCED" if not failures else "STOP_TABLE5_NOT_REPRODUCED",
        "failures": failures,
        "groups": output,
        "implementation_finding": {
            "metrics_stage": "post-clipping final output",
            "basis": "mechanism_heads.py constructs output=torch.clamp(preclamp,0,1); evaluate_mechanism_oof.py computes all four metrics from output and records preclamp only for OOR.",
            "consequence": "The numerical Table 5 additive values already describe Additive+clip, even if surrounding manuscript wording labels them pre-clipping.",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table5_reproduction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": failures}, ensure_ascii=False))


if __name__ == "__main__":
    main()
