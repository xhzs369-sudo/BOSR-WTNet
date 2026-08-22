"""Frozen image-level analysis and decision for ECF+BOSR on development-used dev40."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE / "evaluation"
SOURCE, ACCESS = EVAL / "DEV40_PAPER_MATRIX_PER_IMAGE.csv", EVAL / "DEV40_EVALUATION_ACCESS.json"
ANALYSIS, DECISION, REPORT = EVAL / "DEV40_ANALYSIS.json", EVAL / "DEV40_DECISION.json", EVAL / "DEV40_RESULT_REPORT.md"
SEEDS = (20260801, 20260802, 20260803)
GROUPS = ("WTNET-50", "WTNET-100", "ECF", "BOSR-ONLY", "ECF+BOSR")
METRICS = ("mse", "psnr_db", "ssim", "delta_e00")
HIGHER = {"mse": False, "psnr_db": True, "ssim": True, "delta_e00": False}
sys.path.insert(0, str(HERE.parent / "metrics"))
from statistics_core import paired_percentile_bootstrap  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_ci(candidate: np.ndarray, control: np.ndarray) -> dict:
    candidate_by_image, control_by_image = candidate.mean(axis=0), control.mean(axis=0)
    effect = (control_by_image.mean() - candidate_by_image.mean()) / control_by_image.mean() * 100.0
    rng, samples = np.random.default_rng(20260810), np.empty(10000)
    for start in range(0, 10000, 1000):
        ix = rng.integers(0, 40, size=(1000, 40))
        b, c = control_by_image[ix].mean(axis=1), candidate_by_image[ix].mean(axis=1)
        samples[start:start + 1000] = (b - c) / b * 100.0
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"relative_improvement_percent": float(effect), "ci95_low": float(low), "ci95_high": float(high), "n": 40, "resamples": 10000, "seed": 20260810}


def main() -> None:
    if any(path.exists() for path in (ANALYSIS, DECISION, REPORT)):
        raise RuntimeError("analysis is one-pass; output already exists")
    access = json.loads(ACCESS.read_text(encoding="utf-8"))
    if access.get("status") != "COMPLETE_ONE_READONLY_DEV40_EVALUATION" or access.get("result_sha256") != sha(SOURCE):
        raise RuntimeError("evaluation result lock mismatch")
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    table = {(int(row["seed"]), row["group"], row["filename"]): row for row in rows}
    names = sorted({row["filename"] for row in rows})
    if len(rows) != 600 or len(table) != 600 or len(names) != 40:
        raise RuntimeError("paper matrix completeness failure")

    def matrix(group: str, metric: str) -> np.ndarray:
        return np.asarray([[float(table[(seed, group, name)][metric]) for name in names] for seed in SEEDS])

    summaries = {}
    for group in GROUPS:
        summaries[group] = {}
        for metric in METRICS:
            means = matrix(group, metric).mean(axis=1)
            summaries[group][metric] = {"seed_means": means.tolist(), "three_seed_mean": float(means.mean()), "between_seed_sample_sd": float(means.std(ddof=1))}

    comparisons = {}
    pairs = (("ECF+BOSR", "ECF"), ("ECF+BOSR", "WTNET-50"), ("BOSR-ONLY", "WTNET-50"), ("ECF+BOSR", "BOSR-ONLY"), ("WTNET-100", "WTNET-50"))
    for candidate, control in pairs:
        result = comparisons.setdefault(f"{candidate}_vs_{control}", {})
        for metric in METRICS:
            cand, ctrl = matrix(candidate, metric), matrix(control, metric)
            difference = cand - ctrl
            entry = {
                "signed_definition": f"{candidate} minus {control}",
                "seed_mean_differences": difference.mean(axis=1).tolist(),
                "aggregate_image_bootstrap": paired_percentile_bootstrap(difference.mean(axis=0)),
                "favorable_images": int(np.sum(difference.mean(axis=0) > 0 if HIGHER[metric] else difference.mean(axis=0) < 0)),
            }
            if metric in ("mse", "delta_e00"):
                entry["relative_improvement"] = relative_ci(cand, ctrl)
            result[metric] = entry

    primary = comparisons["ECF+BOSR_vs_ECF"]
    checks = {
        "mse_relative_improvement_at_least_1pct": primary["mse"]["relative_improvement"]["relative_improvement_percent"] >= 1.0,
        "mse_relative_ci_lower_above_zero": primary["mse"]["relative_improvement"]["ci95_low"] > 0.0,
        "mse_favorable_at_least_24_of_40": primary["mse"]["favorable_images"] >= 24,
        "psnr_positive_all_seeds": all(value > 0 for value in primary["psnr_db"]["seed_mean_differences"]),
        "ssim_each_seed_noninferior": all(value >= -0.001 for value in primary["ssim"]["seed_mean_differences"]),
        "ssim_ci_lower_noninferior": primary["ssim"]["aggregate_image_bootstrap"]["ci95_low"] >= -0.001,
        "delta_e_mean_non_degrading": primary["delta_e00"]["relative_improvement"]["relative_improvement_percent"] >= 0.0,
        "delta_e_relative_ci_lower_at_least_minus_1pct": primary["delta_e00"]["relative_improvement"]["ci95_low"] >= -1.0,
    }
    safety = ("ssim_each_seed_noninferior", "ssim_ci_lower_noninferior", "delta_e_mean_non_degrading", "delta_e_relative_ci_lower_at_least_minus_1pct")
    if all(checks.values()):
        classification = "STRONG_DEV40_SUPPORT"
    elif primary["mse"]["relative_improvement"]["relative_improvement_percent"] > 0 and np.mean(primary["psnr_db"]["seed_mean_differences"]) > 0 and all(checks[key] for key in safety):
        classification = "DIRECTIONAL_DEV40_SUPPORT_ONLY"
    else:
        classification = "NOT_SUPPORTED_ON_DEV40"

    regions = {}
    for region in ("dark", "normal", "highlight"):
        seed_values = []
        for seed in SEEDS:
            values = []
            for name in names:
                candidate, control = table[(seed, "ECF+BOSR", name)], table[(seed, "ECF", name)]
                if candidate[f"{region}_computable"].lower() == "true" and control[f"{region}_computable"].lower() == "true":
                    values.append(float(candidate[f"{region}_luminance_mae"]) - float(control[f"{region}_luminance_mae"]))
            seed_values.append(values)
        regions[region] = {"computable_images_per_seed": [len(value) for value in seed_values]}
        if len({len(value) for value in seed_values}) == 1 and seed_values[0]:
            regions[region]["aggregate_difference_bootstrap"] = paired_percentile_bootstrap(np.asarray(seed_values).mean(axis=0))

    diagnostics = {}
    for group in ("BOSR-ONLY", "ECF+BOSR"):
        selected = [row for row in rows if row["group"] == group]
        diagnostics[group] = {key: float(np.mean([float(row[key]) for row in selected])) for key in ("alpha_abs_mean", "beta_abs_mean", "u_abs_mean", "preclip_out_of_range_fraction", "final_out_of_range_fraction")}

    sealed = {"CEC_public_test": 0, "Endo4IE": 0, "RLE": 0}
    payload = {"status": "DEV40_FROZEN_ANALYSIS_COMPLETE", "scope_warning": "CEC dev40 participated in development; this is not independent final-test evidence.", "group_summary": summaries, "comparisons": comparisons, "regional_ecf_bosr_vs_ecf": regions, "diagnostics": diagnostics, "sealed_access": sealed}
    ANALYSIS.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    decision = {"status": classification, "passed_strong_support": classification == "STRONG_DEV40_SUPPORT", "checks": checks, "primary_comparison": primary, "post_result_policy": "No further tuning on dev40. Public test remains sealed pending a separate final-model lock and advisor authorization.", "sealed_access": sealed}
    DECISION.write_text(json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    mse, psnr = primary["mse"]["relative_improvement"], primary["psnr_db"]["aggregate_image_bootstrap"]
    ssim, delta_e = primary["ssim"]["aggregate_image_bootstrap"], primary["delta_e00"]["relative_improvement"]
    lines = ["# Frozen ECF+BOSR three-seed evaluation", "", "## Decision", "", f"**{classification}**", "", "CEC dev40 participated in development and is not an independent final test set.", "", "## ECF+BOSR versus ECF", "", f"- Relative MSE improvement: {mse['relative_improvement_percent']:.4f}% (95% CI {mse['ci95_low']:.4f}% to {mse['ci95_high']:.4f}%); favorable images: {primary['mse']['favorable_images']}/40.", f"- PSNR difference: {psnr['mean']:+.6f} dB (95% CI {psnr['ci95_low']:+.6f} to {psnr['ci95_high']:+.6f}).", f"- SSIM difference: {ssim['mean']:+.8f} (95% CI {ssim['ci95_low']:+.8f} to {ssim['ci95_high']:+.8f}).", f"- Relative DeltaE00 improvement: {delta_e['relative_improvement_percent']:.4f}% (95% CI {delta_e['ci95_low']:.4f}% to {delta_e['ci95_high']:.4f}%).", "", "## Frozen checks", ""]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: {key}" for key, value in checks.items())
    lines.extend(["", "## Data boundary", "", "CEC public test, Endo4IE and RLE access remained zero. No further dev40-driven tuning is authorized."])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": classification, "passed": decision["passed_strong_support"], "checks": checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
