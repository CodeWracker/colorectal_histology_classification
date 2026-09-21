# Pipeline: every campaign, in order

All commands are run from the repository root, with the environments of [SETUP.md](SETUP.md) already created and the main dataset already prepared. The campaigns are ordered by date, and later ones read artifacts produced by earlier ones, so this is also the order in which they must be run. Times are the wall-clock times measured on the reference machine.

A run is never overwritten: `suite.py` skips any run directory that already has a `status.json`, including failed ones. That makes every stage restartable, and it means a botched run has to be repeated under a new campaign name rather than silently replaced.

## 0. Preparation

```bash
cnn/.venv/bin/python comparison/prepare.py          # ~5 s
cnn/.venv/bin/python comparison/prepare_groups.py   # recovers the ten slide sources
cnn/.venv/bin/python comparison/loso.py             # fixes the ten LOSO folds in loso_manifest.json
```

`loso.py` writes `loso_manifest.json` and then verifies it against the committed copy on later runs. Each fold tests one entire slide source; validation is 10% of each class drawn from the other nine sources with partition seed 20260913; training is the rest. The indices are identical for every method and seed.

## 1. Main campaign — `2026-09-12`

Nine methods (`polygarbor`, `polygarbor_aug`, `polygarbor_p3`, `polygarbor_p3_aug`, `polygarbor_p5`, `resnet18`, `resnet18_imagenet`, `yolo11n`, `yolo11n_random`) over full training, the seven scarce-label budgets, imbalance and label noise. `suite.py` runs seeds 42 and 43 for the size scenarios and only the first seed for imbalance and label noise, which is the registered design.

```bash
cnn/.venv/bin/python comparison/suite.py --campaign 2026-09-12 \
  --methods polygarbor polygarbor_aug polygarbor_p3 polygarbor_p3_aug polygarbor_p5 \
            resnet18 resnet18_imagenet yolo11n yolo11n_random

cnn/.venv/bin/python comparison/suite.py --campaign 2026-09-12 \
  --methods polygarbor resnet18 resnet18_imagenet yolo11n yolo11n_random \
  --scenarios loso_01 loso_02 loso_03 loso_04 loso_05 loso_06 loso_07 loso_08 loso_09 loso_10

cnn/.venv/bin/python comparison/finish.py --campaign 2026-09-12
cnn/.venv/bin/python comparison/loso_report.py --campaign 2026-09-12
```

The random-split part came to about 4.5 hours of training across 178 runs; the LOSO part added 100 runs at roughly 3 h 15 min per seed. One process runs at a time, on purpose, so that the recorded time and memory of a run are not contaminated by a neighbour.

`finish.py` is the closing stage and does several things in sequence: it measures inference in fresh processes (one CPU thread with restricted affinity, four CPU threads, and GPU) on the full seed-42 models, runs the mosaic localization test, runs the test suite, audits every run, validates the campaign and rebuilds every table and figure. It propagates a non-zero exit from the tests or the benchmarks instead of swallowing it. `report.py`, `loso_report.py` and `audit.py` only read saved artifacts, so they can be re-run at any time without touching a model.

Quantization and the microcontroller preflight are separate closing steps:

```bash
comparison/quantization/.venv/bin/python comparison/quantization/quantize.py --campaign 2026-09-12
comparison/quantization/.venv/bin/python comparison/quantization/bench.py    --campaign 2026-09-12
comparison/quantization/.venv/bin/python comparison/quantization/report.py   --campaign 2026-09-12
cnn/.venv/bin/python comparison/embedded_feasibility.py \
  --model comparison/runs/2026-09-12/full__polygarbor__seed42/model \
  --output comparison/results/2026-09-12/embedded
```

`quantize.py` exports each full CNN to LiteRT and produces float16, dynamic int8 and static int8 variants, evaluating each on the clean test and the eight perturbations; it runs one model and seed per process, because keeping all of them in one process exhausted memory during the study. Do not run `bench.py` while anything else is training: it measures latency and peak RSS.

## 2. Post-hoc analyses — `2026-09-14_posthoc`

These were added during the paper review, with the authors' authorization, and are not part of the registered protocol. They must be described as exploratory wherever they are used.

```bash
polygarbor/.venv/bin/python comparison/posthoc_imbalance_normalization.py   # PMD decision-rule rescaling
polygarbor/.venv/bin/python comparison/posthoc_footprint_scaling.py         # PolyGabor memory footprint
cnn/.venv/bin/python comparison/cpu_training.py                             # CNN training repeated on the CPU
```

The first two retrain nothing: they load the saved models of the main campaign and change only the decision rule or subsample the retained vectors. `cpu_training.py` reruns `run.py` with `--device cpu`, seed 42 and four threads, at 10 images per class and on the full training set, into `runs/2026-09-14_cpu`; that is where the CPU training times of the paper's cost tables come from. It is the slowest step here, dominated by the 2,437 s ImageNet ResNet-18 run.

## 3. Classifier ablation — `2026-09-15_descriptor`

```bash
cnn/.venv/bin/python comparison/descriptor_baselines.py
```

One script, registered before running. For every scarce-label and full-training run of the main campaign, with and without augmentation, it takes exactly the descriptor vectors the PMD retained and refits five decision rules on them: the registered PMD with L = 3, the PMD with L = 1, a nearest centroid, an RBF-SVM with C = 1 and gamma = 'scale', and the same SVM with C and gamma selected on the 500 validation images. The descriptor, the training images and the retained vectors never change, so the only difference between rows is the decision function.

The registered stopping condition is a reproduction check: the refitted PMD with L = 3 must reproduce the saved test predictions of the main campaign on every image. It did, on all 30 runs that exist. The two that do not exist are `few1__polygarbor__seed42` and `seed43`, which fail identically here and in the campaign (see [README.md](README.md#known-failures-and-their-causes)).

## 4. RBF-SVM campaign — `2026-09-15_svm`

```bash
cnn/.venv/bin/python comparison/svm_campaign.py   # ~1,400 s, CPU only, 38 runs
cnn/.venv/bin/python comparison/svm_report.py
```

`svm_campaign.py` drives `suite.py`/`run.py` with the `gabor_svm` method, in this order: full plus the seven scarce budgets on seeds 42 and 43; then the registered consistency check that these test predictions equal `svm_default` of the previous campaign, aborting everything if they do not; then label noise and imbalance on seed 42; then the ten LOSO folds on both seeds; then the cpu1 and cpu4 inference benchmarks on the full seed-42 model. The check passed and is recorded in `runs/2026-09-15_svm/consistency.json`.

`svm_report.py` builds the comparison tables. It reads three campaigns at once — `2026-09-15_svm` for the SVM, `2026-09-12` for the PMD and the CNNs, and `2026-09-14_cpu` for the CNN CPU training cost — and reuses the source-bootstrap functions of `loso_report.py` with the same 5,000 draws and seed 123. Most of the paper's headline numbers come out of this step.

## 5. Independent-cohort replication — `2026-09-15_nct`

```bash
cnn/.venv/bin/python comparison/nct_prepare.py        # see SETUP.md; network-bound
cnn/.venv/bin/python comparison/nct_descriptor.py     # gabor_svm, svm_tuned, nearest_centroid, pmd_l3
cnn/.venv/bin/python comparison/suite.py --campaign 2026-09-15_nct \
  --methods yolo11n_random yolo11n \
  --scenarios nct_few1 nct_few2 nct_few5 nct_few10 nct_few20 --seeds 42 43 44 45 46
cnn/.venv/bin/python comparison/nct_report.py
```

Five seeds instead of two, five budgets from 1 to 20 images per class, and the full 7,180-patch test set of CRC-VAL-HE-7K, whose patients do not overlap with the training slides. The descriptor methods extract features once from the cached crops and are fitted in one process; the two YOLO configurations are trained through the normal runner, on the GPU, with the campaign's policy (up to 100 epochs, patience 8, early stopping on the 450 validation patches). All 50 YOLO runs and every descriptor fit completed.

## 6. Standardized-PMD check — `2026-09-15_posthoc_standardized_pmd`

```bash
cnn/.venv/bin/python comparison/posthoc_standardized_pmd.py
```

Exploratory, run after the two registered campaigns above had been frozen with their `FROZEN.sha256` manifests. It repeats the classifier ablation exactly, except that each vector is standardized with the mean and standard deviation of its run's retained fitting vectors before the PMD is fitted. No validation or test label enters the transformation. It exists to answer one question raised by the ablation's outcome — whether feature scale explains the PMD's weakness — and its answer is "partly, but not enough to change the conclusion".

## Figures of the manuscript

The five figures are drawn by `scripts/make_figures.py`, which lives with the LaTeX manuscript rather than in this repository, and which trains nothing. Two of them (the pipeline walk-through and the class signatures) load the full seed-42 SVM and PMD models from `comparison/runs/` and run inference on test crops; the other three read only committed CSV files plus the saved confusion matrices of the two full `gabor_svm` runs. [PAPER_MAP.md](PAPER_MAP.md) lists the exact input of each figure.
