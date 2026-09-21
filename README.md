# Compact Gabor and color features with scarce labels — code and results

This repository holds the code, the registered protocol and the aggregated results of the study *How Far Can Compact Gabor and Color Features Go with Scarce Labels? A Controlled Study in Colorectal Histology*. The study asks how far a 22-dimensional texture-and-color descriptor, paired with a classical classifier, can go on colorectal histology patches when labels are scarce, and compares it against ResNet-18 and YOLO11n-cls under both random and ImageNet initialization on identical splits, seeds and perturbations.

**Reviewers and anyone trying to replicate a number in the paper should start at [`reproducibility/`](reproducibility/README.md).** It maps every table and figure of the manuscript to the file that contains its numbers and to the command that produced that file, and it explains what ships in this repository and what has to be recomputed.

## What is here

| Component | Role |
| --- | --- |
| [`polygarbor/`](polygarbor/README.md) | The descriptor (a bank of eight quadrature Gabor filters plus CIELAB statistics) and the polynomial Mahalanobis distance (PMD) classifier, with a CLI for the dataset, training, evaluation and single-image prediction flows. |
| [`cnn/`](cnn/README.md) | The reference CNNs — ResNet-18 and YOLO11n-cls, each with random and ImageNet initialization — behind a CLI with the same commands as `polygarbor`. |
| [`comparison/`](comparison/README.md) | The experiment harness: fixed splits, scenario runner, per-run monitoring, metrics, statistics, figures and reports. `comparison/gabor_svm.py` holds the RBF-SVM on the same descriptor, which is the paper's primary classifier. |
| [`comparison/PROTOCOL.md`](comparison/PROTOCOL.md) | The protocol, registered before the experiments, and every extension, each registered before it was run. |
| [`comparison/results/`](comparison/results/) | The aggregated results: CSV tables, Markdown reports and figures, one directory per campaign. |
| [`reproducibility/`](reproducibility/README.md) | Setup, the ordered command pipeline, the paper-to-artifact map and the verification procedure. |

Two exploratory notebooks, `cnn_test.ipynb` and `garbor-downsampling.ipynb`, are kept as the historical prototypes from which the two packages were derived. They were run on a remote GPU server, which is why their outputs show paths from that machine; no reported result comes from them.

## Quick start

The projects are managed with [uv](https://docs.astral.sh/uv/) and target Python 3.11. From the repository root:

```bash
uv sync --project polygarbor --extra dataset          # descriptor + PMD, with the dataset flow
polygarbor/.venv/bin/polygarbor dataset --data-dir polygarbor/data --preview 9 --export-per-class 1
polygarbor/.venv/bin/polygarbor train --data-dir polygarbor/data --model-dir polygarbor/artifacts/model --figures
polygarbor/.venv/bin/polygarbor predict -i polygarbor/artifacts/dataset/samples/00_tumor_1.png --model-dir polygarbor/artifacts/model
```

The first command downloads TFDS `colorectal_histology` 2.0.0 (5,000 patches, about 750 MB on disk) once; every later run only reads from that directory. Training the default model takes about 20 s on a CPU. [`reproducibility/SETUP.md`](reproducibility/SETUP.md) covers the CNN and quantization environments, the GPU libraries and the second dataset used in the replication.

## Reproducing the paper

The results come from six campaigns, each with its own directory under `comparison/results/`:

| Campaign | Contents |
| --- | --- |
| `2026-09-12` | The registered main campaign: 278 audited runs of the PMD variants and the four CNN configurations over full training, seven scarce-label budgets, label noise, imbalance, ten leave-one-source-out folds and eight test perturbations, plus the edge benchmark, post-training quantization, mosaic localization and the microcontroller preflight. |
| `2026-09-14_posthoc` | Post-hoc analyses added during the paper review: PMD distance rescaling under imbalance, PolyGabor memory footprint, and CNN training repeated on the CPU for a like-for-like cost comparison. |
| `2026-09-15_descriptor` | The registered classifier ablation: PMD (L = 3 and L = 1), nearest centroid and two RBF-SVMs fitted to exactly the same descriptor vectors. |
| `2026-09-15_svm` | The registered campaign that runs the fixed RBF-SVM through every scenario of the main campaign. Most of the paper's headline numbers live here. |
| `2026-09-15_nct` | The registered independent-cohort replication, training on NCT-CRC-HE-100K and testing on all 7,180 patches of CRC-VAL-HE-7K. |
| `2026-09-15_posthoc_standardized_pmd` | A post-hoc check of whether feature standardization explains the PMD's weakness. |

[`reproducibility/PIPELINE.md`](reproducibility/PIPELINE.md) gives the commands for each campaign in order, with the wall-clock time they took on the reference machine; [`reproducibility/PAPER_MAP.md`](reproducibility/PAPER_MAP.md) resolves each table, figure and quoted number to its source file.

## What ships and what does not

Versioned here: all source code, the protocol, the aggregated CSV tables, the Markdown reports, the figures and the per-campaign checksum manifests (`FROZEN.sha256`). Not versioned, and regenerated by the commands above: the TFDS cache (`polygarbor/data/`), the shared array cache (`comparison/cache/`), the per-run directories with models, per-image predictions and explanation maps (`comparison/runs/`), and the downloaded YOLO weights. The per-run directories total tens of gigabytes, which is why the repository keeps the aggregated tables that every reported number is computed from, and the code that rebuilds those tables from the runs.

## Datasets

The main dataset is the CRC texture collection of Kather et al. (2016), used through TensorFlow Datasets as `colorectal_histology` 2.0.0: 5,000 patches of 150×150 pixels from ten H&E slides of a single institution, evenly split over eight tissue classes. The replication uses NCT-CRC-HE-100K and CRC-VAL-HE-7K (Zenodo record 1214456, color-normalized version, nine classes), whose test patients do not overlap with its training slides. Neither collection is redistributed here; both are fetched by the scripts described in [`reproducibility/SETUP.md`](reproducibility/SETUP.md).

## Tests

```bash
cnn/.venv/bin/python -m pytest
```

`pytest.ini` at the root collects the three suites (`cnn/tests`, `comparison/test_scenarios.py`, `polygarbor/tests`), which cover the descriptor, training, persistence, the ImageNet weight port, probability ordering, metrics and the fixed scenario definitions on synthetic data, without needing either dataset.
