# Colorectal histology classification

Code, registered protocol and aggregated results for the study *How Far Can Compact Gabor and Color Features Go with Scarce Labels? A Controlled Study in Colorectal Histology*. The question is how far a 22-dimensional texture and color descriptor with a classical classifier can go on colorectal histology patches when labels are scarce, against ResNet-18 and YOLO11n-cls under random and ImageNet initialization, on identical splits, seeds and perturbations.

To replicate a number from the paper, start at [`reproducibility/`](reproducibility/README.md). It maps every table and figure to the file holding its values and the command that wrote it.

## Components

| Path | Role |
| --- | --- |
| [`polygarbor/`](polygarbor/README.md) | The descriptor (eight quadrature Gabor filters plus CIELAB statistics) and the polynomial Mahalanobis distance classifier, with a CLI for dataset, training, evaluation and prediction. |
| [`cnn/`](cnn/README.md) | ResNet-18 and YOLO11n-cls, each with random and ImageNet initialization, behind the same CLI commands. |
| [`comparison/`](comparison/README.md) | The experiment harness: fixed splits, scenario runner, per-run monitoring, metrics, statistics, figures, reports. `gabor_svm.py` holds the RBF-SVM on the same descriptor, which is the paper's primary classifier. |
| [`comparison/PROTOCOL.md`](comparison/PROTOCOL.md) | The protocol registered before the experiments, and every extension, each registered before it ran. |
| [`comparison/results/`](comparison/results/) | CSV tables, Markdown reports and figures, one directory per campaign. |
| [`reproducibility/`](reproducibility/README.md) | Setup, command pipeline, paper map, verification. |

`cnn_test.ipynb` and `garbor-downsampling.ipynb` are the prototypes the two packages were derived from. They ran on a remote GPU server, which is why their outputs show paths from that machine. No reported result comes from them.

## Quick start

The projects use [uv](https://docs.astral.sh/uv/) and target Python 3.11. From the repository root:

```bash
uv sync --project polygarbor --extra dataset
polygarbor/.venv/bin/polygarbor dataset --data-dir polygarbor/data --preview 9 --export-per-class 1
polygarbor/.venv/bin/polygarbor train --data-dir polygarbor/data --model-dir polygarbor/artifacts/model --figures
polygarbor/.venv/bin/polygarbor predict -i polygarbor/artifacts/dataset/samples/00_tumor_1.png --model-dir polygarbor/artifacts/model
```

The first command downloads TFDS `colorectal_histology` 2.0.0 once, about 750 MB on disk; later runs only read from that directory. Training the default model takes about 20 s on a CPU. [`reproducibility/SETUP.md`](reproducibility/SETUP.md) covers the CNN and quantization environments, the GPU libraries and the replication dataset.

## Campaigns

| Directory under `comparison/results/` | Contents |
| --- | --- |
| `2026-09-12` | Registered main campaign: 278 audited runs of the PMD variants and four CNN configurations over full training, seven scarce budgets, label noise, imbalance, ten leave-one-source-out folds and eight test perturbations, plus the edge benchmark, quantization, mosaic localization and the microcontroller preflight. |
| `2026-09-14_posthoc` | Post-hoc analyses from the paper review: PMD distance rescaling under imbalance, PolyGabor memory footprint, CNNs retrained on the CPU for a like-for-like cost comparison. |
| `2026-09-15_descriptor` | Registered classifier ablation: PMD at L = 3 and L = 1, nearest centroid and two RBF-SVMs on exactly the same descriptor vectors. |
| `2026-09-15_svm` | Registered campaign running the fixed RBF-SVM through every scenario of the main campaign. Most headline numbers live here. |
| `2026-09-15_nct` | Registered replication: training on NCT-CRC-HE-100K, testing on all 7,180 patches of CRC-VAL-HE-7K. |
| `2026-09-15_posthoc_standardized_pmd` | Post-hoc check of whether feature standardization explains the PMD's weakness. |

[`reproducibility/PIPELINE.md`](reproducibility/PIPELINE.md) gives the commands for each campaign in order, with measured wall-clock times.

## What is versioned

Source code, the protocol, the aggregated CSV tables, the Markdown reports, the figures and the per-campaign checksum manifests (`FROZEN.sha256`).

Not versioned, and rebuilt by the commands above: the TFDS cache (`polygarbor/data/`), the shared array cache (`comparison/cache/`), the per-run directories with models, per-image predictions and explanation maps (`comparison/runs/`), and the downloaded YOLO weights. The per-run directories run to tens of gigabytes, so the repository keeps the aggregated tables every reported number is computed from, plus the code that rebuilds them.

## Datasets

The main dataset is the CRC texture collection of Kather et al. (2016), used through TensorFlow Datasets as `colorectal_histology` 2.0.0: 5,000 patches of 150x150 pixels from ten H&E slides of a single institution, evenly split over eight tissue classes. The replication uses NCT-CRC-HE-100K and CRC-VAL-HE-7K (Zenodo record 1214456, color-normalized, nine classes), whose test patients do not overlap its training slides. Neither collection is redistributed here; the scripts in [`reproducibility/SETUP.md`](reproducibility/SETUP.md) fetch both.

## Tests

```bash
cnn/.venv/bin/python -m pytest
```

`pytest.ini` collects `cnn/tests`, `comparison/test_scenarios.py` and `polygarbor/tests`, covering the descriptor, training, persistence, the ImageNet weight port, probability ordering, metrics and the fixed scenario definitions on synthetic data. Neither dataset is needed.
