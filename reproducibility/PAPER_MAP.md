# Paper map

Each entry gives the file that holds a published value, the command that wrote it, and how to get from the rows to the number. Files under `comparison/results/` are versioned here, so most entries can be checked without running anything. Files under `comparison/runs/` are per-run outputs and are not versioned; the entries that need them say so.

Values are means over seeds 42 and 43 on the `test_clean` evaluation unless an entry says otherwise. Bracketed ranges are the minimum and maximum over those seeds.

## Tables

### Table 1, classifiers on the same descriptor vectors

`comparison/results/2026-09-15_descriptor/metrics.csv`, with the means already computed in `summary.md`. Written by `descriptor_baselines.py`.

Filter `vectors == polygarbor` and average `macro_f1` over the two seeds for each `method` and `scenario`. The methods are `pmd_l3`, `pmd_l1`, `nearest_centroid`, `svm_default` and `svm_tuned`. The `vectors == polygarbor_aug` rows are the augmented variant discussed in the text, and the `C`, `gamma` and `selection_val_macro_f1` columns record what `svm_tuned` selected. The standardized PMD value quoted in the same subsection (0.592 at n = 10) is in `results/2026-09-15_posthoc_standardized_pmd/metrics.csv`.

### Table 2, scarce labels with size and cost

Macro-F1 columns: `results/2026-09-15_svm/scarce_macro_f1.csv`, one row per scenario and one column per method, already formatted as the mean with the seed range in brackets. The G-mean twin is `scarce_multiclass_gmean.csv`.

Size, CPU fit time and RAM: `results/2026-09-15_svm/cost.csv`, columns `model_mb_few2`, `train_s_few10` and `rss_mb_few10`. The CNN rows there come from the CPU-only runs of `runs/2026-09-14_cpu`, which is what makes the comparison like for like; the `training_runs` column names the campaign each row was measured in.

The GPU fit column is the only one outside `cost.csv`. It is `train_seconds` of the `few10`, seed 42 rows of `results/2026-09-12/metrics.csv`, where the CNNs trained on the GPU.

Both files come from `svm_report.py`.

### Table 3, replication on CRC-VAL-HE-7K

`results/2026-09-15_nct/metrics.csv`, with means and ranges in `summary.md`. Written by `nct_descriptor.py` for the descriptor methods and `nct_report.py` for the tables; the YOLO rows come from `runs/2026-09-15_nct`.

One row per method, scenario (`few1` to `few20`) and seed (42 to 46), evaluated on all 7,180 test patches over nine classes. Published cells are means over the five seeds. Per-class recall is in `recall.csv`, paired differences in `paired.csv`. The `pmd_l3` cells at one image per class are absent by construction.

### Table 4, full training set and cost

Macro-F1 and G-mean for all six trained rows: `results/2026-09-15_svm/runs.csv`, filtered to `scenario == full` and `evaluation == test_clean`, averaged over the two seeds. That file carries every method of both campaigns, and the `campaign` column says where each row was measured. Its `path` column records the absolute run directory on the machine that produced it and is metadata, not a path to resolve.

Cost columns: `results/2026-09-15_svm/cost.csv`, columns `train_s_full`, `model_mb_full`, `cpu1_median_ms` and `cpu1_rss_mb`.

The two LiteRT int8 rows come from the quantization extension: accuracy and file size from `results/2026-09-12/quantization/summary.csv`, latency and peak RSS from `speed.csv` in the same directory, mode `cpu1`, both written by `quantization/report.py`. LiteRT peak RSS is `ru_maxrss` while the non-quantized rows are sampled, a difference the quantization README documents. The `original` rows of those files reproduce the corresponding rows of `cost.csv`.

### Table 5, LOSO

`results/2026-09-15_svm/loso_pooled.csv` for macro-F1, macro-F1 without background and the drop against the random split (columns `macro_f1`, `macro_f1_without_empty`, `random_split_macro_f1`). The three recall columns are in `loso_recall_per_class.csv`. Both from `svm_report.py`. The drop is `random_split_macro_f1 - macro_f1`, averaged over the two seeds.

The source bootstrap quoted in the text, with its intervals, is `loso_paired_svm_vs_others.csv` (5,000 draws, seed 123). Per-fold metrics and the fold definitions for the registered campaign alone are in `results/2026-09-12/loso/`, from `loso_report.py`.

### Table 6, class imbalance

The imbalanced SVM row and the registered PMD row: `results/2026-09-15_svm/imbalance.csv`, which carries macro-F1, G-mean and per-class recall for all seven methods under the seed 42 imbalance scenario.

The two post-hoc PMD rules and the balanced rows for them: `results/2026-09-14_posthoc/imbalance_normalization.csv`, from `posthoc_imbalance_normalization.py`, filtered to `split == test`. That file also holds the `insample` rule and the validation splits, which the paper leaves out. Nothing was retrained for those rows; the saved models are loaded and only the decision rule changes.

### Table 7, spatial support and augmentation

All five rows come from `results/2026-09-12/`.

Macro-F1 and G-mean: `metrics.csv`, rows with `scenario == full`, `evaluation == test_clean` and `method` in `polygarbor`, `polygarbor_aug`, `polygarbor_p3`, `polygarbor_p3_aug`, `polygarbor_p5`, averaged over the two seeds. The noisy-label column is the same file with `scenario == label_noise`, seed 42 only.

The CPU latency column is `edge.csv`, column `median_ms` with `mode == cpu1`. It is not the `batch1_median_ms` column of `metrics.csv`, which is measured inside the training process and runs about 4 ms higher for every variant.

### Table 8, image perturbations

`results/2026-09-15_svm/robustness.csv`, one row per evaluation (`test_clean` plus the eight perturbations) and one column per method, already averaged over the seeds. From `svm_report.py`. The main campaign alone is in `results/2026-09-12/robustness.csv`.

The perturbations are defined once, in `scenarios.py::perturb`, and applied to the same pixels for every method: 9x9 Gaussian blur with sigma 2, additive Gaussian noise with sigma 20, brightness 1.35, channel scaling (1.2, 0.8, 1.1), JPEG quality 20, downsampling to 32x32 and back, a white occlusion of the central third, and a 90 degree rotation.

## Figures

`scripts/make_figures.py` draws all five. It lives with the LaTeX manuscript and trains nothing.

| Figure | Inputs |
| --- | --- |
| 1, descriptor pipeline on a tumor patch | full seed 42 `gabor_svm` model and the cached test arrays, under `runs/` and `cache/` (not versioned) |
| 2, class-conditional signatures | the 2,800 fitting vectors of `runs/2026-09-12/full__polygarbor__seed42/model` (not versioned) |
| 3, macro-F1 and G-mean vs. labels per class | `results/2026-09-12/learning_curve.csv` and `results/2026-09-15_svm/runs.csv` |
| 4, effect of 20% label noise | `results/2026-09-12/metrics.csv` and `results/2026-09-15_svm/runs.csv`; the same values are tabulated in `results/2026-09-15_svm/label_noise.csv` |
| 5, confusion matrix of the fixed SVM | `runs/2026-09-15_svm/full__gabor_svm__seed{42,43}/eval/test_clean/confusion_matrix.csv` (not versioned), averaged over the two seeds |

Figures 3 and 4 redraw from this repository alone. Figures 1, 2 and 5 need the per-run artifacts, so they require rerunning the main campaign and the SVM campaign.

## Numbers quoted only in the text

| Claim | File |
| --- | --- |
| The refitted PMD reproduces the campaign's test predictions on every image | `results/2026-09-15_descriptor/reproduction.json` |
| The SVM campaign's predictions equal those of `svm_default` | `runs/2026-09-15_svm/consistency.json` (not versioned); the outcome is in `results/2026-09-15_svm/README.md` |
| Cross-fitted median distances of 24 to 78 for the reduced classes against 0.11 to 0.99 for the others | `results/2026-09-14_posthoc/imbalance_normalization.json` |
| int8 quantization degrading only the ImageNet ResNet-18 | `results/2026-09-12/quantization/metrics.csv` and `summary.csv` |
| Campaign totals, run counts and audit outcome | `comparison/integrity_audit.json` and `results/2026-09-12/validation_summary.json` |
| When each experimental decision was registered | `comparison/PROTOCOL.md` |
| What was run, what failed and what was fixed | `comparison/WORK_LOG.md` |

## Results the paper does not use

The mosaic localization test (`results/2026-09-12/localization/`), the microcontroller memory preflight (`embedded/`), the calibration and aggregation tables, the error and corruption galleries, and the historical `source_a` and `source_b` splits are complete and reproducible but were left out of the manuscript. `results/2026-09-12/DISCUSSION.md` and `REPORT.md` cover them. The LOSO design replaced `source_a` and `source_b` for the reasons given in `PROTOCOL.md`, so those two should not be read as results about method ranking.
