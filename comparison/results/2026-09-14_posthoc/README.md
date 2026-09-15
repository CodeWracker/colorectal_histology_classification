# Post-hoc analyses from 2026-09-14

These analyses were done after the `2026-09-12` campaign, during the review of the `ichi-polygarbor` paper, with explicit authorization from the authors. They are not part of the protocol registered in `../../PROTOCOL.md` and must be described in the paper as exploratory.

## Distance normalization under class imbalance

Script: `comparison/posthoc_imbalance_normalization.py`. Outputs: `imbalance_normalization.csv` and `imbalance_normalization.json`.

No model was retrained. The saved models of `imbalance__polygarbor__seed42`, `full__polygarbor__seed42` and `full__polygarbor__seed43` were loaded and only the decision rule changed:

- `raw`: argmin D_c, the registered rule. It reproduces the test macro-F1 of 0.408 of the imbalance scenario.
- `insample`: argmin D_c / r_c, where r_c is the median of D_c over the class's own fitting vectors, evaluated by the model built from them.
- `crossfit`: the same, but each vector is evaluated by a model fitted without it (5 folds within the class).
- `count`: argmin N_c · D_c. Dividing the scatter matrix by N_c scales all singular values and s_min = eps · s_max by 1/N_c, without changing bases, projections or relative thresholds, so every level of D_c is multiplied by N_c. This is the PMD with the scatter normalized by the number of vectors. With balanced classes, the predictions are identical to those of `raw`.

No validation or test label was used to define the rules. Test results:

| model | rule | macro-F1 | G-mean |
| --- | --- | --- | --- |
| imbalance seed 42 | raw | 0.408 | 0.000 |
| imbalance seed 42 | insample | 0.434 | 0.000 |
| imbalance seed 42 | count | 0.552 | 0.379 |
| imbalance seed 42 | crossfit | 0.653 | 0.625 |
| full, mean of seeds 42/43 | raw = count | 0.793 | 0.777 |
| full, mean of seeds 42/43 | insample | 0.780 | 0.767 |
| full, mean of seeds 42/43 | crossfit | 0.775 | 0.761 |

The rules were chosen after observing the failure and were evaluated with one seed and one imbalance ratio.

## PolyGabor memory footprint

Script: `comparison/posthoc_footprint_scaling.py`. Output: `footprint_scaling.csv`.

The descriptors retained by the `full__polygarbor__seed42` model (350 per class) were subsampled, with seed 42, to 2, 5, 10, 20, 50, 100 and 350 vectors per class with 8 classes, and to 2 and 4 classes with 350 vectors. No descriptor was extracted again and nothing was evaluated on the test set. `saved_mb` is the size written by `save()`, `model_array_mb` is the arrays of the PMD models plus the retained samples, `fit_peak_mb` and `eval_peak_mb` are `tracemalloc` peaks, and `distance_median_ms` is the median of 30 distance computations for one vector (4 threads). The measurement ran before the CNN training on CPU started.

## CNN training on CPU

Script: `comparison/cpu_training.py`. Runs: `runs/2026-09-14_cpu`. It uses the same `run.py`, seed 42, 4 threads and `--device cpu`, with 10 images per class and with full training. During the runs, the machine was also running the browser and, for a few seconds, the scripts above and the generation of the paper's figures, so the timings contain some noise.
