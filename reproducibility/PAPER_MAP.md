# Paper map: from a published number to the file and the command behind it

Each entry names the artifact that holds the published values, the command from [PIPELINE.md](PIPELINE.md) that wrote it, and how the number is derived from the rows. Every artifact listed as a `.csv` or `.md` file is versioned in this repository, so these entries can be checked without rerunning anything. Artifacts under `comparison/runs/` are per-run outputs and are not versioned; the entries say so explicitly.

Unless an entry says otherwise, published values are means over seeds 42 and 43 of the `test_clean` evaluation, and a bracketed range is the minimum and maximum over those two seeds.

## Tables

### Table 1 — Test macro-F1 of classifiers fitted to the same descriptor vectors

`comparison/results/2026-09-15_descriptor/metrics.csv`, with the means already computed in `summary.md` of the same directory. Written by `comparison/descriptor_baselines.py`.

Filter to `vectors == polygarbor` for the published table and average `macro_f1` over the two seeds for each `method` and `scenario`. The five methods are `pmd_l3`, `pmd_l1`, `nearest_centroid`, `svm_default` and `svm_tuned`; the `full` scenario is the last column. The `vectors == polygarbor_aug` rows are the augmented-vector variant discussed in the text. The selected hyperparameters of `svm_tuned` are in the `C`, `gamma` and `selection_val_macro_f1` columns. The standardized-PMD figure quoted in the same subsection (0.592 with n = 10) is in `comparison/results/2026-09-15_posthoc_standardized_pmd/metrics.csv`, from `posthoc_standardized_pmd.py`.

### Table 2 — Macro-F1, model size and training cost with scarce labels

Two files, both written by `comparison/svm_report.py`:

- macro-F1 columns: `comparison/results/2026-09-15_svm/scarce_macro_f1.csv`, one row per scenario and one column per method, already formatted as `mean [min–max]`. The G-mean twin is `scarce_multiclass_gmean.csv`.
- the "Size at 2", "Fit at 10 (CPU)" and "RAM at 10" columns: `comparison/results/2026-09-15_svm/cost.csv`, columns `model_mb_few2`, `train_s_few10` and `rss_mb_few10`. The CNN rows of that file come from the CPU-only runs of `runs/2026-09-14_cpu`, which is what makes the comparison like-for-like; the `training_runs` column names the campaign each row was measured in.

The "Fit at 10 (GPU)" column is the only one not in `cost.csv`: it is `train_seconds` of the `few10`, seed 42 rows of `comparison/results/2026-09-12/metrics.csv`, where the CNNs were trained on the GPU.

### Table 3 — Replication on CRC-VAL-HE-7K

`comparison/results/2026-09-15_nct/metrics.csv`, with means and ranges in `summary.md`. Written by `comparison/nct_descriptor.py` (descriptor methods) and `comparison/nct_report.py` (tables), with the YOLO rows coming from `runs/2026-09-15_nct`.

One row per `method`, `scenario` (`few1` to `few20`) and `seed` (42 to 46), evaluated on all 7,180 test patches over nine classes; the published cells are means over the five seeds. Per-class recall is in `recall.csv` and the paired differences between methods in `paired.csv`. The `pmd_l3` cells at one image per class are absent by construction, not by omission.

### Table 4 — Test results and cost with the full training set

Macro-F1 and G-mean of all six trained rows: `comparison/results/2026-09-15_svm/runs.csv`, filtered to `scenario == full` and `evaluation == test_clean` and averaged over the two seeds. That file carries every method of both campaigns, with the `campaign` column saying where each row was measured; its `path` column records the absolute directory of the run on the machine that produced it and is metadata, not something to resolve. The cost columns (`Train CPU`, `Size`, `Lat.`, `RSS`) are the `train_s_full`, `model_mb_full`, `cpu1_median_ms` and `cpu1_rss_mb` columns of `comparison/results/2026-09-15_svm/cost.csv`.

The two LiteRT int8 rows come from the quantization extension: accuracy and file size from `comparison/results/2026-09-12/quantization/summary.csv` and latency and peak RSS from `speed.csv` in the same directory, mode `cpu1`, both written by `comparison/quantization/report.py`. Note that the LiteRT peak RSS is `ru_maxrss` while the non-quantized rows are sampled, a difference documented in the quantization README, and that the `original` rows of those files reproduce the corresponding rows of `cost.csv`.

### Table 5 — LOSO results

`comparison/results/2026-09-15_svm/loso_pooled.csv` for macro-F1, the macro-F1 without background and the drop relative to the random split (`macro_f1`, `macro_f1_without_empty`, `random_split_macro_f1`), and `loso_recall_per_class.csv` for the three recall columns. Both from `comparison/svm_report.py`. The "Drop" column is `random_split_macro_f1 - macro_f1`, averaged over the two seeds.

The paired bootstrap over the ten slide sources, quoted in the text with its intervals, is `loso_paired_svm_vs_others.csv` (5,000 draws, seed 123). The equivalent tables for the registered campaign alone, including per-fold metrics and the fold definitions, are in `comparison/results/2026-09-12/loso/`, from `comparison/loso_report.py`.

### Table 6 — Class imbalance and post-hoc rescaling

The imbalanced SVM row and the registered PMD row: `comparison/results/2026-09-15_svm/imbalance.csv`, which carries macro-F1, G-mean and per-class recall for all seven methods under the seed-42 imbalance scenario. The two post-hoc PMD rules (`N_c D_c` and `D_c / r_c`) and the balanced-training rows for those rules: `comparison/results/2026-09-14_posthoc/imbalance_normalization.csv`, from `comparison/posthoc_imbalance_normalization.py`, filtered to `split == test`.

That file also contains the `insample` rule and the validation splits, which the paper does not report. Nothing was retrained for those rows: the saved models are loaded and only the decision rule changes.

### Table 7 — Effect of spatial support and augmentation

All five rows come from the main campaign, `comparison/results/2026-09-12/`:

- macro-F1 and G-mean: `metrics.csv`, rows with `scenario == full`, `evaluation == test_clean` and `method` in `polygarbor`, `polygarbor_aug`, `polygarbor_p3`, `polygarbor_p3_aug`, `polygarbor_p5`, averaged over the two seeds.
- the noisy-label column: the same file with `scenario == label_noise`, seed 42 only, as registered.
- the CPU latency column: `edge.csv`, column `median_ms` with `mode == cpu1`. This is the fresh-process benchmark with affinity restricted to one logical processor, not the `batch1_median_ms` column of `metrics.csv`, which is measured inside the training process and is systematically higher.

### Table 8 — Test macro-F1 under image perturbations

`comparison/results/2026-09-15_svm/robustness.csv`, one row per evaluation (`test_clean` plus the eight `test_<perturbation>` rows) and one column per method, already averaged over the seeds. From `comparison/svm_report.py`. The equivalent table for the main campaign alone is `comparison/results/2026-09-12/robustness.csv`.

The perturbations themselves are defined once, in `comparison/scenarios.py::perturb`, and applied to the same pixels for every method: 9×9 Gaussian blur with sigma 2, additive Gaussian noise with sigma 20, brightness ×1.35, channel scaling (1.2, 0.8, 1.1), JPEG quality 20, downsampling to 32×32 and back, a white occlusion of the central third, and a 90° rotation.

## Figures

The figures are drawn by `scripts/make_figures.py`, which is kept with the LaTeX manuscript. It trains nothing.

| Figure | Inputs |
| --- | --- |
| 1 — descriptor pipeline on a tumor patch | the full seed-42 `gabor_svm` model and the cached test arrays, both under `comparison/runs/` and `comparison/cache/` (not versioned) |
| 2 — class-conditional texture and color signatures | the 2,800 fitting vectors of the full seed-42 PolyGabor model, `comparison/runs/2026-09-12/full__polygarbor__seed42/model` (not versioned) |
| 3 — macro-F1 and G-mean vs. labels per class | `comparison/results/2026-09-12/learning_curve.csv` and `comparison/results/2026-09-15_svm/runs.csv` (both versioned) |
| 4 — effect of 20% label noise | `comparison/results/2026-09-12/metrics.csv` and `comparison/results/2026-09-15_svm/runs.csv` (both versioned); the same values are tabulated in `comparison/results/2026-09-15_svm/label_noise.csv` |
| 5 — confusion matrix of the descriptor with the fixed SVM | `comparison/runs/2026-09-15_svm/full__gabor_svm__seed{42,43}/eval/test_clean/confusion_matrix.csv` (not versioned), averaged over the two seeds |

Figures 3 and 4 can therefore be redrawn from this repository alone. Figures 1, 2 and 5 need the per-run artifacts, which means rerunning the main campaign and the SVM campaign, or obtaining the run directories separately.

## Numbers quoted only in the running text

| Claim | Where it lives |
| --- | --- |
| Reproduction check: the refitted PMD reproduces the campaign's test predictions on every image | `comparison/results/2026-09-15_descriptor/reproduction.json` |
| Consistency check: the SVM campaign's predictions equal those of `svm_default` | `runs/2026-09-15_svm/consistency.json` (not versioned); the outcome is recorded in `comparison/results/2026-09-15_svm/README.md` |
| Cross-fitted median distances of 24–78 for the reduced classes against 0.11–0.99 for the others | `comparison/results/2026-09-14_posthoc/imbalance_normalization.json` |
| int8 quantization degrading only the ImageNet ResNet-18 | `comparison/results/2026-09-12/quantization/metrics.csv` and `summary.csv` |
| Campaign totals, run counts and audit outcome | `comparison/integrity_audit.json` and `comparison/results/2026-09-12/validation_summary.json` |
| Every experimental decision and the date it was registered | `comparison/PROTOCOL.md` |
| Narrative of what was run, what failed and what was fixed | `comparison/WORK_LOG.md` |

## Results in the repository that the paper does not use

The mosaic localization test (`comparison/results/2026-09-12/localization/`), the microcontroller memory preflight (`.../embedded/`), the calibration and aggregation tables, the error and corruption galleries, and the historical `source_a` and `source_b` splits are all complete, documented and reproducible, but they were left out of the manuscript. `comparison/results/2026-09-12/DISCUSSION.md` and `REPORT.md` cover them. The `source_a` and `source_b` scenarios were replaced by the LOSO design for the reasons stated in `PROTOCOL.md`, and should not be read as results about method ranking.
