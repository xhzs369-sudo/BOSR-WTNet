# BOSR-WTNet

Official manuscript-facing code snapshot for **BOSR-WTNet: bounded odds-space refinement of WTNet for underexposed capsule-endoscopy image correction**.

## What is included

- the BOSR refinement head and its ECF ablation;
- the WTNet architecture used by the frozen parent, redistributed under the upstream MIT license;
- frozen training, development-evaluation and public-test scripts;
- metric, bootstrap-statistics and complexity definitions;
- experiment protocols, split manifests and environment records;
- efficiency and post hoc LPIPS analysis code and numerical outputs.
- post-submission additive-clipping audit code and released numerical outputs;
- official access links and provenance notes for manuscript-related datasets.

## What is not included

CEC/KCL image data, model checkpoints, cached tensors and generated images are not redistributed. Dataset access remains subject to the original providers' terms. Checkpoint release also remains subject to the third-party WTNet licensing and manuscript-review plan.

## Repository status

This is a sanitized release layer derived from the frozen local submission snapshot. Local absolute paths and workstation identifiers were replaced for safe publication; the scientific implementation and recorded numerical results were not changed. `SOURCE_HASHES.csv` records the identity of the original frozen source files.

The public-test decision remains `DIRECTIONAL_INDEPENDENT_TEST_SUPPORT_ONLY`. The repository does not establish clinical, patient-level, video-level or state-of-the-art superiority.

## Project layout

```text
src/bosr/         BOSR head, training and locked development evaluation
src/ecf/          ECF controls and shared experiment utilities
src/wtnet/        WTNet parent architecture
src/metrics/      image-quality, bootstrap and complexity utilities
src/public_test/  frozen public-test evaluation and analysis
protocols/        preregistration and model-selection records
manifests/        CEC train360/dev40 file identities without raw images
environment/      recorded software environment
efficiency/       measured efficiency protocol and outputs
posthoc_lpips/    explicitly post hoc LPIPS analysis
supplementary_clipping_audit/  additive raw/clipped/BOSR audit and released CSV outputs
DATASETS.md       official dataset links, roles, provenance and redistribution limits
```

## Environment

The recorded experiments used Python 3.9 and PyTorch 2.8.0 with CUDA 12.6. See `environment/` for the full frozen environment records. The WTNet parent is based on the official [WTNet repository](https://github.com/charonf/WTNet) at commit `e7197e09be0dd844317dda84b74275585d1a8d39`.

## Data layout

The sanitized manifests use repository-independent relative locations such as:

```text
CEC/train/input_under/0.png
CEC/train/gt/0.png
CEC/test/input_under/0.png
CEC/test/gt/0.png
```

Set the `BOSR_PROJECT_ROOT` environment variable when adapting the frozen scripts to another workstation. The scripts retain their historical experiment-directory assumptions, so this snapshot should be treated as manuscript audit code rather than a turnkey training package.

Official access links and the exact roles of CEC, KCL, Kvasir-Capsule and RLE are recorded in [`DATASETS.md`](DATASETS.md). Raw third-party medical images are not redistributed.

## Minimal model check

```python
import torch
from src.bosr.formal_bosr_head import FormalBOSRHead

model = FormalBOSRHead("BOSR-ONLY")
i0 = torch.rand(1, 3, 128, 128)
ic = torch.rand(1, 3, 128, 128)
i1 = torch.rand(1, 3, 128, 128)
output, diagnostics = model(i0, ic, i1)
assert output.shape == i1.shape
assert output.min() >= 0 and output.max() <= 1
```

## Citation

The manuscript is under preparation. Citation metadata and the final archival DOI will be added after submission.

## Licensing and third-party code

The copied WTNet architecture remains covered by the upstream MIT license in `third_party/WTNet_LICENSE`. No CEC or KCL image is included. A repository-wide license for the new BOSR code will be finalized by the authors before archival publication.
