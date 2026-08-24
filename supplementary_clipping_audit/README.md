# Additive clipping audit

This directory contains the code and machine-readable aggregate/per-image outputs for the post-submission BOSR-WTNet clipping audit. The audit compares:

1. the capacity-matched additive RGB head before clipping;
2. the same additive output after the fixed `clip([0,1])` operation;
3. the intrinsically range-preserving BOSR output.

No model was trained or fine-tuned in this audit. Frozen checkpoints and cached float predictions were reused. Raw images, checkpoints, and float prediction caches are intentionally not redistributed.

## Scientific finding

The additive head was weaker than BOSR before clipping and stronger after clipping. The final interpretation is therefore a reconstruction-quality versus intrinsic-range-validity trade-off. The audit does not support a claim that BOSR universally outperforms an additive head followed by clipping.

## Path configuration

The original scripts used a workstation-specific project root. The public copies change only that path declaration and read it from an environment variable:

```powershell
$env:BOSR_PROJECT_ROOT = 'D:\path\to\your\project_root'
```

```bash
export BOSR_PROJECT_ROOT=/path/to/your/project_root
```

The expected historical experiment tree is documented by the scripts and `../DATASETS.md`. These scripts are an audit snapshot rather than a one-command training package.

## Script order

Run only on data and checkpoints for which you have access and authorization.

```text
audit_inventory.py
reproduce_table5.py
run_oof_additive_clipping.py
recompute_oof_raw_metrics_from_cache.py
run_public_additive_clipping.py
finalize_public_metrics_from_cache.py  # recovery path used after a CSV write incident
analyze_supplement.py
```

The public-test script should not be rerun merely to regenerate already frozen results. The released CSV files in `results/` allow inspection of the reported analysis without accessing the raw medical images.

## Released results

- `oof_additive_raw_clipped_per_image.csv` contains 360-image OOF raw/clipped/BOSR metrics.
- `test100_all_stages_per_image.csv` contains the three-seed, four-stage CEC test metrics.
- `bosr_vs_additive_clipped_per_seed_image.csv` contains the matched per-seed, per-image comparison.
- the smaller CSV files contain aggregate clipping, bootstrap, seed, and luminance-stratum summaries.

Positive `favorable_effect` values denote an effect favorable to BOSR. For MSE and CIEDE2000, the effect is relative; PSNR and SSIM use additive differences.

## Original frozen source identities

The hashes below identify the local scripts before workstation-path sanitization. No scientific formula, threshold, seed, statistic, or result was changed in the public copy.

| File | Original SHA-256 |
|---|---|
| `supplement_common.py` | `3cc12b70dc0800fc3b750959d02e23922f0261c819d871b5e6ad13226ebda5f3` |
| `audit_inventory.py` | `f7e1135814a3c5e43501f926f5dcd2ece26e381fcc84ed44875875963dc36284` |
| `reproduce_table5.py` | `8f7cdd2ca7199dcc43cd07dbfc4178ade7c9c35b57b79bf6db5b390bd467777f` |
| `run_oof_additive_clipping.py` | `2f26b6261c104aa656cb363adb90628e006ed034efb5f9ee6ba90537df5a5b49` |
| `recompute_oof_raw_metrics_from_cache.py` | `8f6903fef9b3b019e9fc05d3552208dbfe9c894bbbb3fabfaa4724fc04c5c2da` |
| `run_public_additive_clipping.py` | `f69c8084b318625912ea0b8bde66531f7ff319ee0c8f86fa7d8db983884caaa2` |
| `finalize_public_metrics_from_cache.py` | `7d3fad78734ee99cda611287b111fd624102f0b1b692b1487e71154b6b43605b` |
| `analyze_supplement.py` | `455fcfdd09dcdf9aa768110c8ccc16b987f65ac5ca93f6a44a71bdcecfd673a1` |

## Reproducibility limits

- CEC lacks patient/video identifiers in the released protocol, so statistical statements are image-level.
- KCL is a post hoc synthetic-pair evaluation.
- Dataset licenses and third-party WTNet checkpoint terms remain controlled by their original providers.
