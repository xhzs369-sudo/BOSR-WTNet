"""Generate frozen supplementary statistics, figures, final tables, and report."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(os.environ["BOSR_PROJECT_ROOT"]).expanduser().resolve()
HERE = ROOT / "experiments/supplementary_bosr_additive_clipping_20260823"
OUT = HERE / "outputs"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
FINAL = OUT / "final_tables"
AUDIT = OUT / "audit"
INPUT_DIR = ROOT / "datasets/CEC/test/input_under"
GT_DIR = ROOT / "datasets/CEC/test/gt"
SEEDS = (20260801, 20260802, 20260803)
BOOTSTRAP_SEED = 20260823
N_BOOT = 10_000
LOW_THR = 0.06220269948244095
HIGH_THR = 0.0814473032951355


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def metric_effect(candidate: np.ndarray, control: np.ndarray, metric: str) -> np.ndarray:
    if metric in ("mse", "delta_e00"):
        return 100.0 * (control - candidate) / control
    return candidate - control


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(N_BOOT, dtype=np.float64)
    n = len(values)
    for start in range(0, N_BOOT, 500):
        count = min(500, N_BOOT - start)
        indices = rng.integers(0, n, size=(count, n))
        draws[start:start + count] = values[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def bootstrap_effect(candidate: np.ndarray, control: np.ndarray, metric: str,
                     rng: np.random.Generator) -> tuple[float, float, float]:
    """Match the frozen public protocol for relative and additive effects."""
    candidate = np.asarray(candidate, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if candidate.shape != control.shape or candidate.ndim != 1:
        raise ValueError("paired one-dimensional arrays required")
    if metric in ("mse", "delta_e00"):
        point = 100.0 * (control.mean() - candidate.mean()) / control.mean()
    else:
        point = (candidate - control).mean()
    draws = np.empty(N_BOOT, dtype=np.float64)
    n = len(candidate)
    for start in range(0, N_BOOT, 500):
        count = min(500, N_BOOT - start)
        indices = rng.integers(0, n, size=(count, n))
        cand_mean = candidate[indices].mean(axis=1)
        ctrl_mean = control[indices].mean(axis=1)
        draws[start:start + count] = (100.0 * (ctrl_mean - cand_mean) / ctrl_mean
                                      if metric in ("mse", "delta_e00") else cand_mean - ctrl_mean)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(point), float(low), float(high)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def prediction(seed: int, image_id: str, filename: str) -> np.ndarray:
    path = OUT / "public_float_predictions" / f"seed_{seed}" / Path(image_id).stem / filename
    with np.load(path, allow_pickle=False) as data:
        return data["prediction"].astype(np.float64)


def summarize_oof(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (oof.groupby(["method", "stage"], sort=False)
               .agg(n_images=("image_id", "nunique"), mse=("mse", "mean"), psnr_db=("psnr_db", "mean"),
                    ssim=("ssim", "mean"), delta_e00=("delta_e00", "mean"),
                    total_oor_fraction=("total_oor_fraction", "mean"))
               .reset_index())
    save_csv(summary, METRICS / "oof_additive_clipping_summary.csv")
    wide = oof.pivot(index="image_id", columns=["method", "stage"], values=["mse", "psnr_db", "ssim", "delta_e00"])
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows = []
    for comparison, candidate_key, control_key in (
        ("Additive post-clipping vs pre-clipping", ("RGB_ADD_FULL_B0P1", "post-clipping"), ("RGB_ADD_FULL_B0P1", "pre-clipping")),
        ("BOSR final vs Additive pre-clipping", ("BOSR_FULL_B1", "final"), ("RGB_ADD_FULL_B0P1", "pre-clipping")),
        ("BOSR final vs Additive post-clipping", ("BOSR_FULL_B1", "final"), ("RGB_ADD_FULL_B0P1", "post-clipping")),
    ):
        for metric in ("mse", "psnr_db", "ssim", "delta_e00"):
            cand = wide[(metric, *candidate_key)].to_numpy()
            ctl = wide[(metric, *control_key)].to_numpy()
            effects = metric_effect(cand, ctl, metric)
            mean, low, high = bootstrap_effect(cand, ctl, metric, rng)
            rows.append({"comparison": comparison, "metric": metric, "favorable_effect": mean,
                         "ci95_low": low, "ci95_high": high, "favorable_images": int((effects > 0).sum()), "n_images": len(effects),
                         "effect_definition": "relative reduction (%)" if metric in ("mse", "delta_e00") else "candidate - control"})
    effects = pd.DataFrame(rows)
    save_csv(effects, METRICS / "oof_paired_effects_bootstrap.csv")
    return summary, effects


def clipping_distribution(oof: pd.DataFrame, public: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = []
    magnitudes = {"OOF train360": [], "CEC test100 three-seed": []}
    channel_rows = []
    for label, paths in (
        ("OOF train360", sorted((OUT / "additive_raw/oof_seed_20260813").glob("*.npz"), key=lambda p: int(p.stem))),
        ("CEC test100 three-seed", sorted((OUT / "public_float_predictions").rglob("additive_raw.npz"), key=lambda p: str(p))),
    ):
        below = np.zeros(3, dtype=np.int64)
        above = np.zeros(3, dtype=np.int64)
        count = np.zeros(3, dtype=np.int64)
        active_sum = np.zeros(3, dtype=np.float64)
        active_count = np.zeros(3, dtype=np.int64)
        all_active = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                raw = data["prediction"].astype(np.float64)
            low = raw < 0
            high = raw > 1
            mag = np.abs(raw - np.clip(raw, 0, 1))
            active = low | high
            below += low.sum(axis=(0, 1))
            above += high.sum(axis=(0, 1))
            count += np.array([raw.shape[0] * raw.shape[1]] * 3)
            active_sum += (mag * active).sum(axis=(0, 1))
            active_count += active.sum(axis=(0, 1))
            if active.any():
                all_active.append(mag[active].astype(np.float32))
        merged = np.concatenate(all_active) if all_active else np.zeros(1, dtype=np.float32)
        magnitudes[label] = merged
        datasets.append({"dataset": label, "n_outputs": len(paths), "element_count": int(count.sum()),
                         "below_zero_fraction": float(below.sum() / count.sum()),
                         "above_one_fraction": float(above.sum() / count.sum()),
                         "total_oor_fraction": float((below.sum() + above.sum()) / count.sum()),
                         "active_clipping_count": int(len(merged))})
        for c, name in enumerate(("R", "G", "B")):
            channel_rows.append({"dataset": label, "channel": name, "element_count": int(count[c]),
                                 "below_zero_fraction": float(below[c] / count[c]), "above_one_fraction": float(above[c] / count[c]),
                                 "total_oor_fraction": float((below[c] + above[c]) / count[c]),
                                 "mean_active_clip_magnitude": float(active_sum[c] / active_count[c])})
    summary = pd.DataFrame(datasets)
    quantile_rows = []
    for label, values in magnitudes.items():
        quantiles = np.quantile(values, [0, .25, .5, .75, .9, .95, .99, 1])
        quantile_rows.append({"dataset": label, "active_values": len(values), "mean": float(values.mean()),
                              "q00": quantiles[0], "q25": quantiles[1], "q50": quantiles[2], "q75": quantiles[3],
                              "q90": quantiles[4], "q95": quantiles[5], "q99": quantiles[6], "q100": quantiles[7]})
    quantiles = pd.DataFrame(quantile_rows)
    channels = pd.DataFrame(channel_rows)
    save_csv(summary, METRICS / "clipping_oor_summary.csv")
    save_csv(quantiles, METRICS / "clipping_magnitude_summary.csv")
    save_csv(channels, METRICS / "clipping_by_rgb_channel.csv")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for label, values in magnitudes.items():
        limit = max(float(np.quantile(values, .995)), 1e-6)
        axes[0].hist(np.clip(values, 0, limit), bins=70, density=True, alpha=.55, label=label)
    axes[0].set_xlabel("Absolute clipping magnitude (values above 99.5th percentile pooled at edge)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Active out-of-range values")
    axes[0].legend(frameon=False, fontsize=8)
    x = np.arange(3)
    width = .35
    for j, label in enumerate(("OOF train360", "CEC test100 three-seed")):
        subset = channels[channels.dataset == label]
        axes[1].bar(x + (j - .5) * width, 100 * subset.total_oor_fraction.to_numpy(), width, label=label)
    axes[1].set_xticks(x, ["R", "G", "B"])
    axes[1].set_ylabel("Out-of-range elements (%)")
    axes[1].set_title("Channel-wise out-of-range rate")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(FIGURES / "FIG_S1_CLIPPING_DISTRIBUTION.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary, quantiles, channels


def public_statistics(public: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_means = (public.groupby(["seed", "method", "stage"], sort=False)
                  .agg(n_images=("image_id", "nunique"), mse=("mse", "mean"), psnr_db=("psnr_db", "mean"),
                       ssim=("ssim", "mean"), delta_e00=("delta_e00", "mean"), total_oor_fraction=("total_oor_fraction", "mean"))
                  .reset_index())
    save_csv(seed_means, METRICS / "public_test_seed_means.csv")
    group = (seed_means.groupby(["method", "stage"], sort=False)
             .agg(mse_mean=("mse", "mean"), mse_sd=("mse", "std"), psnr_db_mean=("psnr_db", "mean"), psnr_db_sd=("psnr_db", "std"),
                  ssim_mean=("ssim", "mean"), ssim_sd=("ssim", "std"), delta_e00_mean=("delta_e00", "mean"), delta_e00_sd=("delta_e00", "std"),
                  total_oor_fraction_mean=("total_oor_fraction", "mean"))
             .reset_index())
    save_csv(group, METRICS / "public_test_three_seed_mean_sd.csv")

    final = public[((public.method == "GenericHead") & (public.stage == "post-clipping")) | ((public.method == "BOSR-WTNet") & (public.stage == "final"))]
    pivot = final.pivot(index=["seed", "image_id"], columns="method", values=["mse", "psnr_db", "ssim", "delta_e00"])
    effect_rows = []
    for (seed, image_id), row in pivot.iterrows():
        item = {"seed": seed, "image_id": image_id}
        for metric in ("mse", "psnr_db", "ssim", "delta_e00"):
            item[f"bosr_{metric}"] = row[(metric, "BOSR-WTNet")]
            item[f"additive_clipped_{metric}"] = row[(metric, "GenericHead")]
            item[f"favorable_effect_{metric}"] = metric_effect(np.array([row[(metric, "BOSR-WTNet")]]), np.array([row[(metric, "GenericHead")]]), metric)[0]
        effect_rows.append(item)
    per_seed_image = pd.DataFrame(effect_rows)
    save_csv(per_seed_image, METRICS / "bosr_vs_additive_clipped_per_seed_image.csv")
    per_image = per_seed_image.groupby("image_id", sort=False).mean(numeric_only=True).reset_index()
    save_csv(per_image, METRICS / "bosr_vs_additive_clipped_per_image_three_seed_average.csv")

    rng = np.random.default_rng(BOOTSTRAP_SEED + 2)
    bootstrap_rows = []
    for metric in ("mse", "psnr_db", "ssim", "delta_e00"):
        candidate = per_image[f"bosr_{metric}"].to_numpy()
        control = per_image[f"additive_clipped_{metric}"].to_numpy()
        values = metric_effect(candidate, control, metric)
        mean, low, high = bootstrap_effect(candidate, control, metric, rng)
        bootstrap_rows.append({"comparison": "BOSR-WTNet vs GenericHead post-clipping", "metric": metric,
                               "favorable_effect": mean, "ci95_low": low, "ci95_high": high,
                               "favorable_images": int((values > 0).sum()), "n_images": len(values),
                               "effect_definition": "relative reduction (%)" if metric in ("mse", "delta_e00") else "BOSR - Additive clipped"})
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, METRICS / "public_bosr_vs_additive_clipped_bootstrap.csv")

    by_seed_rows = []
    for seed in SEEDS:
        subset = per_seed_image[per_seed_image.seed == seed]
        for metric in ("mse", "psnr_db", "ssim", "delta_e00"):
            candidate = subset[f"bosr_{metric}"].to_numpy()
            control = subset[f"additive_clipped_{metric}"].to_numpy()
            values = metric_effect(candidate, control, metric)
            mean, low, high = bootstrap_effect(candidate, control, metric, rng)
            by_seed_rows.append({"seed": seed, "metric": metric, "favorable_effect": mean, "ci95_low": low, "ci95_high": high,
                                 "favorable_images": int((values > 0).sum()), "n_images": len(values)})
    by_seed = pd.DataFrame(by_seed_rows)
    save_csv(by_seed, METRICS / "public_bosr_vs_additive_clipped_by_seed.csv")
    return seed_means, group, bootstrap, by_seed


def luminance_analysis(public: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index in range(100):
        image_id = f"{index}.png"
        image = read_rgb(INPUT_DIR / image_id)
        h, w = image.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - (w - 1) / 2) ** 2 + (yy - (h - 1) / 2) ** 2 <= (0.45 * min(h, w)) ** 2
        luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        value = float(luminance[mask].mean())
        group = "low" if value <= LOW_THR else ("middle" if value <= HIGH_THR else "high")
        rows.append({"image_id": image_id, "fixed_circle_mean_rec709_luminance": value, "luminance_tertile": group})
    membership = pd.DataFrame(rows)
    save_csv(membership, METRICS / "fixed_luminance_tertile_membership.csv")
    definition = (
        "Fixed existing luminance stratification (not re-estimated in this supplement):\n"
        "- Image: CEC test100 underexposed input, decoded RGB in [0,1].\n"
        "- Luminance: Y = 0.2126 R + 0.7152 G + 0.0722 B (Rec.709).\n"
        "- Spatial region: centered circle with radius 0.45 * min(height,width).\n"
        f"- Low: mean Y <= {LOW_THR:.17g}.\n"
        f"- Middle: {LOW_THR:.17g} < mean Y <= {HIGH_THR:.17g}.\n"
        f"- High: mean Y > {HIGH_THR:.17g}.\n"
        "- Expected counts: 34 low, 33 middle, 33 high.\n"
        "- Purpose: exploratory heterogeneity description only; no model selection or threshold tuning.\n"
    )
    (METRICS / "fixed_luminance_tertile_definition.txt").write_text(definition, encoding="utf-8")
    effects = pd.read_csv(METRICS / "bosr_vs_additive_clipped_per_image_three_seed_average.csv")
    merged = effects.merge(membership, on="image_id", how="left")
    rng = np.random.default_rng(BOOTSTRAP_SEED + 3)
    result = []
    for group in ("low", "middle", "high"):
        sub = merged[merged.luminance_tertile == group]
        for metric in ("mse", "psnr_db", "ssim", "delta_e00"):
            candidate = sub[f"bosr_{metric}"].to_numpy()
            control = sub[f"additive_clipped_{metric}"].to_numpy()
            values = metric_effect(candidate, control, metric)
            mean, low, high = bootstrap_effect(candidate, control, metric, rng)
            result.append({"luminance_tertile": group, "metric": metric, "favorable_effect": mean, "ci95_low": low,
                           "ci95_high": high, "favorable_images": int((values > 0).sum()), "n_images": len(values)})
    frame = pd.DataFrame(result)
    save_csv(frame, METRICS / "public_bosr_vs_additive_clipped_by_fixed_luminance_tertile.csv")
    return frame


def create_forest() -> None:
    labels = ["MSE reduction (%)", "PSNR difference (dB)", "SSIM difference", "DeltaE00 reduction (%)"]
    estimates = np.array([2.9965, 0.3788, 0.001690, 5.5556])
    lows = np.array([-7.2670, -0.0644, 0.001247, 0.3996])
    highs = np.array([12.5610, 0.8122, 0.002138, 10.4894])
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8), constrained_layout=True)
    for ax, label, est, lo, hi in zip(axes.flat, labels, estimates, lows, highs):
        ax.axvline(0, color="#666666", lw=1, ls="--")
        ax.errorbar(est, 0, xerr=[[est - lo], [hi - est]], fmt="o", color="#1f5a94", capsize=4, lw=2)
        ax.set_yticks([])
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(f"{est:.6g} [{lo:.6g}, {hi:.6g}]")
        span = max(abs(lo), abs(hi), 1e-6)
        ax.set_xlim(min(lo - .12 * span, -0.08 * span), max(hi + .12 * span, .08 * span))
        ax.grid(axis="x", alpha=.18)
    fig.suptitle("BOSR-WTNet vs WTNet-50 on CEC test100 (existing frozen estimates)", fontsize=12)
    fig.savefig(FIGURES / "FIG_2_PRIMARY_EFFECT_FOREST.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_visualization(public: pd.DataFrame) -> pd.DataFrame:
    raw = public[(public.method == "GenericHead") & (public.stage == "pre-clipping")]
    ranks = raw.groupby("image_id").total_oor_fraction.mean().sort_values()
    positive = ranks[ranks > 0]
    selected = [positive.index[-1], positive.index[len(positive) // 2], positive.index[0]]
    roles = ["Highest OOR", "Median nonzero OOR", "Lowest nonzero OOR"]
    seed = 20260801
    selection = pd.DataFrame({"selection_rule": roles, "image_id": selected, "three_seed_mean_oor_fraction": [ranks[x] for x in selected], "display_seed": seed})
    save_csv(selection, METRICS / "fixed_qualitative_case_selection.csv")
    fig, axes = plt.subplots(3, 6, figsize=(15.2, 8.0), constrained_layout=True)
    titles = ["Input", "WTNet-50", "Additive raw*", "Additive clipped", "BOSR", "GT / OOR mask inset"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=10)
    for row, (image_id, role) in enumerate(zip(selected, roles)):
        inp = read_rgb(INPUT_DIR / image_id)
        gt = read_rgb(GT_DIR / image_id)
        parent = prediction(seed, image_id, "wtnet50_final.npz")
        add_raw = prediction(seed, image_id, "additive_raw.npz")
        add_clip = prediction(seed, image_id, "additive_clipped.npz")
        bosr = prediction(seed, image_id, "bosr_final.npz")
        oor = ((add_raw < 0) | (add_raw > 1)).any(axis=2)
        shown = [inp, parent, np.clip(add_raw, 0, 1), add_clip, bosr, gt]
        for col, image in enumerate(shown):
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"{role}\n{image_id}\nOOR={100*ranks[image_id]:.3f}%", fontsize=9)
        inset = axes[row, 5].inset_axes([.67, .02, .31, .31])
        inset.imshow(oor, cmap="magma", vmin=0, vmax=1)
        inset.axis("off")
        inset.set_title("OOR", fontsize=7, color="white", pad=1)
    fig.savefig(FIGURES / "FIG_S2_FIXED_OOR_CASES.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return selection


def final_tables(oof_summary: pd.DataFrame, oof_effects: pd.DataFrame, public_group: pd.DataFrame,
                 public_boot: pd.DataFrame, public_seed: pd.DataFrame, luminance: pd.DataFrame,
                 clipping_summary: pd.DataFrame, clipping_quantiles: pd.DataFrame) -> None:
    table1 = oof_summary.copy()
    table1["evidence_scope"] = "5-fold OOF train360; prespecified seed 20260813"
    save_csv(table1, FINAL / "FINAL_TABLE_1_OOF_RAW_CLIPPED_BOSR.csv")
    table2 = clipping_summary.merge(clipping_quantiles, on="dataset", how="left")
    save_csv(table2, FINAL / "FINAL_TABLE_2_CLIPPING_DISTRIBUTION.csv")
    table3 = public_group.merge(public_boot[["metric", "favorable_effect", "ci95_low", "ci95_high", "favorable_images", "n_images"]].assign(key=1), how="cross")
    table3 = table3[(table3.method == "BOSR-WTNet") & (table3.stage == "final")].drop(columns="key", errors="ignore")
    save_csv(table3, FINAL / "FINAL_TABLE_3_PUBLIC_BOSR_VS_ADDITIVE_CLIPPED.csv")
    table4 = public_seed.merge(public_boot[["metric", "favorable_effect", "ci95_low", "ci95_high"]], on="metric", suffixes=("_seed", "_pooled"))
    save_csv(table4, FINAL / "FINAL_TABLE_4_SEED_AND_LUMINANCE_STABILITY.csv")
    save_csv(luminance, FINAL / "FINAL_TABLE_5_FIXED_LUMINANCE_TERTILES.csv")


def build_report(oof_summary: pd.DataFrame, oof_effects: pd.DataFrame, clipping_summary: pd.DataFrame,
                 clipping_quantiles: pd.DataFrame, public_group: pd.DataFrame, public_boot: pd.DataFrame,
                 public_seed: pd.DataFrame, luminance: pd.DataFrame, selection: pd.DataFrame) -> None:
    def row(frame: pd.DataFrame, **kwargs):
        sub = frame.copy()
        for key, value in kwargs.items():
            sub = sub[sub[key] == value]
        if len(sub) != 1:
            raise RuntimeError(f"row selection failed: {kwargs}, n={len(sub)}")
        return sub.iloc[0]

    oof_pre = row(oof_summary, method="RGB_ADD_FULL_B0P1", stage="pre-clipping")
    oof_post = row(oof_summary, method="RGB_ADD_FULL_B0P1", stage="post-clipping")
    oof_bosr = row(oof_summary, method="BOSR_FULL_B1", stage="final")
    pub_pre = row(public_group, method="GenericHead", stage="pre-clipping")
    pub_post = row(public_group, method="GenericHead", stage="post-clipping")
    pub_bosr = row(public_group, method="BOSR-WTNet", stage="final")
    test_oor = row(clipping_summary, dataset="CEC test100 three-seed")
    test_mag = row(clipping_quantiles, dataset="CEC test100 three-seed")
    effects = {r.metric: r for _, r in public_boot.iterrows()}
    seed_direction = public_seed.groupby("metric").apply(lambda d: int((d.favorable_effect > 0).sum()), include_groups=False).to_dict()
    report = f"""# BOSR-WTNet补充实验报告：Additive裁剪依赖性审计

## 1. 执行结论

本轮严格执行老师任务书规定的补充实验，没有新建或训练任何模型，没有修改既有权重、数据划分、训练轮数、种子或公共测试结果。新增工作只包括：读取冻结权重做只读推理、保存Additive裁剪前float32输出、将裁剪前/后结果拆分统计，以及生成图表。

**总判定：C（重建能力—范围约束权衡得到直接证据）。** Additive裁剪前在OOF和CEC公共测试上四项指标均弱于BOSR；经过`clip([0,1])`后，Additive四项指标均超过BOSR。由此可见，Additive的最终重建优势明显依赖裁剪操作。BOSR的合理定位不是“全面优于Additive”，而是以一定重建自由度为代价，获得更新后解析地位于[0,1]、无需后裁剪的性质。

## 2. 关键实现核对

- 既有OOF Table 5中的`RGB_ADD_FULL_B0P1`四项指标来自**裁剪后输出**，不是裁剪前输出；代码中`output=torch.clamp(preclamp,0,1)`，指标读取`output`，而`preclamp`只用于越界率。
- 本轮因此新增独立的`pre-clipping`行，不篡改旧Table 5。
- 初次计算裁剪前PSNR/SSIM时发现BasicSR会把最大值略大于1的数组误判为[0,255]量纲。已保留错误CSV及事故记录，并以等价缩放方式固定`data_range=1`后从float缓存重算；没有重新推理。
- CEC推理300/300完成后，首次CSV写入因字段集合不一致中断。全部1200个float缓存完整保留，随后从缓存和冻结公共结果CSV恢复1200行表；没有重新推理。

## 3. OOF结果（train360，五折，每图均为折外预测）

| 方法阶段 | MSE↓ | PSNR↑ | SSIM↑ | ΔE00↓ | 越界率 |
|---|---:|---:|---:|---:|---:|
| Additive裁剪前 | {oof_pre.mse:.9f} | {oof_pre.psnr_db:.6f} | {oof_pre.ssim:.6f} | {oof_pre.delta_e00:.6f} | {100*oof_pre.total_oor_fraction:.4f}% |
| Additive裁剪后 | {oof_post.mse:.9f} | {oof_post.psnr_db:.6f} | {oof_post.ssim:.6f} | {oof_post.delta_e00:.6f} | 0（输出范围） |
| BOSR最终输出 | {oof_bosr.mse:.9f} | {oof_bosr.psnr_db:.6f} | {oof_bosr.ssim:.6f} | {oof_bosr.delta_e00:.6f} | 0 |

解释：裁剪使Additive从“四项均弱于BOSR”变为“四项均优于BOSR”，排名发生反转。这是“clip参与性能形成”的直接证据，而不是仅根据越界率进行推测。

## 4. 裁剪分布

CEC test100三种子Additive裁剪前共有{100*test_oor.total_oor_fraction:.4f}%的RGB元素越界，其中低于0为{100*test_oor.below_zero_fraction:.4f}%，高于1为{100*test_oor.above_one_fraction:.4f}%。对实际被裁剪元素，绝对裁剪幅度中位数为{test_mag.q50:.6f}，95百分位为{test_mag.q95:.6f}，最大值为{test_mag.q100:.6f}。通道分解见`clipping_by_rgb_channel.csv`，分布图见`FIG_S1_CLIPPING_DISTRIBUTION.png`。

## 5. CEC公共测试三种子结果

| 方法阶段 | MSE（种子均值）↓ | PSNR↑ | SSIM↑ | ΔE00↓ | 裁剪前越界率 |
|---|---:|---:|---:|---:|---:|
| Additive裁剪前 | {pub_pre.mse_mean:.9f} | {pub_pre.psnr_db_mean:.6f} | {pub_pre.ssim_mean:.6f} | {pub_pre.delta_e00_mean:.6f} | {100*pub_pre.total_oor_fraction_mean:.4f}% |
| Additive裁剪后 | {pub_post.mse_mean:.9f} | {pub_post.psnr_db_mean:.6f} | {pub_post.ssim_mean:.6f} | {pub_post.delta_e00_mean:.6f} | 同上 |
| BOSR最终输出 | {pub_bosr.mse_mean:.9f} | {pub_bosr.psnr_db_mean:.6f} | {pub_bosr.ssim_mean:.6f} | {pub_bosr.delta_e00_mean:.6f} | 0 |

以“正值有利于BOSR”统一方向，BOSR相对Additive裁剪后的配对图像级效应为：

- MSE相对降低：{effects['mse'].favorable_effect:.4f}%（95% CI {effects['mse'].ci95_low:.4f}%至{effects['mse'].ci95_high:.4f}%；{int(effects['mse'].favorable_images)}/100张有利）。
- PSNR差：{effects['psnr_db'].favorable_effect:+.6f} dB（95% CI {effects['psnr_db'].ci95_low:+.6f}至{effects['psnr_db'].ci95_high:+.6f}；{int(effects['psnr_db'].favorable_images)}/100张有利）。
- SSIM差：{effects['ssim'].favorable_effect:+.6f}（95% CI {effects['ssim'].ci95_low:+.6f}至{effects['ssim'].ci95_high:+.6f}；{int(effects['ssim'].favorable_images)}/100张有利）。
- ΔE00相对降低：{effects['delta_e00'].favorable_effect:.4f}%（95% CI {effects['delta_e00'].ci95_low:.4f}%至{effects['delta_e00'].ci95_high:.4f}%；{int(effects['delta_e00'].favorable_images)}/100张有利）。

本节的负值表示Additive裁剪后更优。SSIM和ΔE00的区间完全位于不利于BOSR的一侧；MSE和PSNR区间跨0，表示均值不利但仍有不确定性。四项指标中，BOSR有利方向分别出现在{seed_direction.get('mse',0)}/3、{seed_direction.get('psnr_db',0)}/3、{seed_direction.get('ssim',0)}/3和{seed_direction.get('delta_e00',0)}/3个种子。

## 6. 亮度分层与固定案例

亮度分层完全复用既有定义：输入RGB在[0,1]，Rec.709亮度，图像中心半径0.45×短边的固定圆；阈值为{LOW_THR:.8f}和{HIGH_THR:.8f}，组数34/33/33。它只用于探索性异质性描述，不用于调参或模型选择。结果见`FINAL_TABLE_5_FIXED_LUMINANCE_TERTILES.csv`。

可视化案例按三种子平均越界率自动锁定，不经肉眼挑选：
{selection.to_markdown(index=False)}

## 7. 对论文叙事的建议

可写：

> The capacity-matched additive head achieved stronger post-clipping reconstruction scores, but its pre-clipping output violated the valid range. Importantly, the additive head was weaker than BOSR before clipping and became stronger after clipping, demonstrating that clipping materially contributed to its observed performance. BOSR therefore represents a deliberate quality–constraint trade-off rather than an unconstrained reconstruction optimum.

不可写：

- “BOSR在准确率上优于Additive+clip”；数据不支持。
- “越界必然造成临床危害”；本轮没有临床端点。
- “clip是作弊”；clip是合法工程操作，但其贡献必须被透明分离。
- “三种子相当于300个独立样本”；配对bootstrap单位是100张图像，先在每图内平均三种子。

## 8. 统计与推断边界（11类谬误扫描）

- Simpson悖论：已核对总体、分种子和固定亮度分层方向；存在异质性，但未见总体方向与所有子组一致反转。
- 生态谬误：不作患者、视频或临床个体推断。
- Berkson/选择偏倚：CEC是固定公开配对测试集，结论限定于该数据。
- Collider偏倚：未进行协变量调整，不适用。
- 基础率忽视：没有诊断敏感度/特异度结论，不适用。
- 回归均值：没有按极端表现重新训练或选模型；固定案例仅展示。
- 生存者偏倚：100/100图像、3/3种子均纳入。
- 多重寻找效应：四项指标全部报告，未只挑有利指标；分层为探索性。
- 分叉路径：阈值、种子、比较组和案例规则固定并留档；本轮不调参。
- 相关不等于因果：这里比较的是同图、同父模型、固定权重下的确定性算法输出，可归因于输出映射差异，但不能外推临床因果。
- 反向因果：不适用于算法配对输出比较。

**扫描覆盖：11/11。总体置信等级：CAUTION。** 原因是公共测试仅100张图像、无患者/视频标识，且同时报告四项指标而未做多重比较校正；本报告以效应量和完整95% CI为主，不用单一显著性标签包装结论。

## 9. 完整性与可追溯性

- OOF结果：360张×3阶段=1080行；公共测试：100张×3种子×4阶段=1200行。
- 训练次数：0；模型参数改动：0；权重覆盖：0。
- 事故、修正前CSV、修正公式、缓存恢复和SHA256均保存在`outputs/audit/`。
- 最终可提交表位于`outputs/final_tables/`，图位于`outputs/figures/`。
"""
    (OUT / "EXPERIMENT_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    for path in (METRICS, FIGURES, FINAL, AUDIT):
        path.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(METRICS / "oof_additive_raw_clipped_per_image.csv")
    public = pd.read_csv(METRICS / "test100_all_stages_per_image.csv")
    if len(oof) != 1080 or len(public) != 1200:
        raise RuntimeError("input completeness failure")
    oof_summary, oof_effects = summarize_oof(oof)
    clipping_summary, clipping_quantiles, _ = clipping_distribution(oof, public)
    public_seed_means, public_group, public_boot, public_by_seed = public_statistics(public)
    luminance = luminance_analysis(public)
    counts = pd.read_csv(METRICS / "fixed_luminance_tertile_membership.csv").luminance_tertile.value_counts().to_dict()
    if counts != {"low": 34, "middle": 33, "high": 33}:
        raise RuntimeError(f"fixed luminance group mismatch: {counts}")
    create_forest()
    selection = create_visualization(public)
    final_tables(oof_summary, oof_effects, public_group, public_boot, public_by_seed, luminance, clipping_summary, clipping_quantiles)
    build_report(oof_summary, oof_effects, clipping_summary, clipping_quantiles, public_group, public_boot, public_by_seed, luminance, selection)
    artifacts = [p for p in OUT.rglob("*") if p.is_file()]
    manifest = {"status": "COMPLETE", "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": N_BOOT,
                "training": 0, "files": [{"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(artifacts)]}
    (AUDIT / "FINAL_OUTPUT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "artifacts": len(artifacts), "report": str(OUT / 'EXPERIMENT_REPORT.md')}))


if __name__ == "__main__":
    main()
