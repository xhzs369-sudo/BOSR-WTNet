# Dataset access and provenance

This repository does not redistribute third-party medical images. Download each dataset from its original provider and comply with the provider's current terms, citation requirements, and ethics conditions.

## Datasets used in the manuscript

### Capsule-endoscopy Exposure Correction (CEC)

- **Role:** primary paired dataset for BOSR-WTNet training, development analysis, and the frozen 100-pair public test.
- **Official project:** [EndoUIC](https://github.com/longbai1006/EndoUIC)
- **Official CEC download:** [Google Drive](https://drive.google.com/file/d/1h1mqugWx7PfmU_H7uGK-AIMcMhf1C2n6/view?usp=sharing)
- **Local protocol used here:** the official 400-pair underexposure training split was divided into `train360` and `dev40`; the official 100-pair underexposure test was retained as `test100`. Filename-only manifests are provided in `manifests/`.
- **Redistribution note:** the EndoUIC code repository is MIT licensed, but the dataset download page does not provide a separate dataset-specific license. Raw CEC images are therefore not copied into this repository.

### KCL eval15 paired set

- **Role:** post hoc, zero-finetuning, image-level cross-dataset evaluation on 400 synthetic low/high pairs.
- **Package used:** `dataset_IGLFM.zip`, prefix `dataset_IGLFM/KCL/eval15/`, distributed with the official Dual-Generalization project.
- **Official project:** [Dual-Gen-Frame](https://github.com/superwsc/Dual-Gen-Frame)
- **Official download page:** [Baidu Netdisk, extraction code 4240](https://pan.baidu.com/s/1XPxP-p05YdbbVRgy1RR-AA?pwd=4240)
- **Source-domain dataset:** [Kvasir-Capsule official repository](https://github.com/simula/kvasir-capsule), [official dataset page](https://datasets.simula.no/kvasir-capsule/), and [OSF data record](https://osf.io/dv2ag/).
- **Evidence boundary:** this synthetic paired set was selected after the primary CEC analysis. It is not an untouched natural-exposure, patient-level, video-level, or clinical validation endpoint.

## Dataset used during earlier development history

### Red Lesion Endoscopy (RLE)

- **Role:** earlier ECF development and external-confirmation history. It is not presented as an untouched final BOSR external test.
- **Original dataset page:** [INESC TEC Research Data](https://rdm.inesctec.pt/dataset/nis-2018-003)
- **Low/high paired package used in the project history:** the official [Dual-Gen-Frame](https://github.com/superwsc/Dual-Gen-Frame) download above.

## Expected local layout

The released scripts assume the following repository-independent paths below the directory referenced by `BOSR_PROJECT_ROOT`:

```text
datasets/
  CEC/
    train/input_under/
    train/gt/
    test/input_under/
    test/gt/
    internal_split_seed20260731/
  KCL_IGLFM_external_v1/
    low/
    high/
```

The KCL and RLE data are not needed to reproduce the CEC clipping audit in `supplementary_clipping_audit/`; their links are included to document every manuscript-related dataset source.
