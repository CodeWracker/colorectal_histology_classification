# Descriptor-controlled classifiers (2026-09-15)

Extension registered in `../../PROTOCOL.md` ("Descriptor-controlled classifier extension") before running, with the authors' authorization. Script: `comparison/descriptor_baselines.py`, run with the `cnn` venv (scikit-learn 1.9.1, 4 threads). Outputs: `metrics.csv` (one row per vector set, method, scenario and seed), `summary.md` (mean and range of test macro-F1), `reproduction.json` and `environment.json`. Descriptors and per-image predictions are in `runs/2026-09-15_descriptor`, which is ignored by Git.

## Reproduction check

The re-fitted `pmd_l3` produced exactly the saved test predictions of the 30 campaign runs of `polygarbor` and `polygarbor_aug` that exist (agreement 1.0 in all of them). The two missing runs are `few1__polygarbor__seed42` and `seed43`, which failed in the campaign and fail here with the same `IndexError`: `polymahalanobis` reads the fitting vectors from a text file, and a file with a single row is loaded as a one-dimensional array, so the error is raised while loading, before any scatter matrix is computed. `pmd_l1` fails in the same cells for the same reason.

## Test macro-F1

Mean of seeds 42 and 43, fitting vectors of `polygarbor` (original images only). Ranges between seeds and the `polygarbor_aug` vectors are in `summary.md`.

| images/class | PMD, L = 3 | PMD, L = 1 | nearest centroid | RBF-SVM, C = 1 | RBF-SVM, tuned |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -- | -- | 0.459 | 0.459 | 0.459 |
| 2 | 0.214 | 0.214 | 0.485 | 0.432 | 0.526 |
| 5 | 0.418 | 0.361 | 0.523 | 0.573 | 0.620 |
| 10 | 0.501 | 0.465 | 0.607 | 0.689 | 0.735 |
| 20 | 0.569 | 0.539 | 0.620 | 0.738 | 0.775 |
| 50 | 0.673 | 0.706 | 0.612 | 0.773 | 0.823 |
| 100 | 0.709 | 0.772 | 0.644 | 0.813 | 0.862 |
| full | 0.793 | 0.816 | 0.655 | 0.878 | 0.929 |

With the augmented vectors, the PMD with L = 3 reached 0.371, 0.416, 0.482, 0.543 and 0.621 with 1, 2, 5, 10 and 20 images per class, and the tuned SVM 0.482, 0.514, 0.618, 0.738 and 0.782. With the full training set, augmentation lowered the tuned SVM from 0.929 to 0.915.

## Observations

- Both SVMs had a higher mean test macro-F1 than the PMD with L = 3 at every budget and with both vector sets, and so did the nearest centroid from 1 to 20 images per class. The G-mean led to the same ordering: the PMD with L = 3 had a G-mean of zero with 2 and 5 images per class in both seeds and both vector sets, whereas the minimum over seeds of the tuned SVM was 0.380–0.554 at those budgets.
- The first PMD level alone (L = 1) was worse than L = 3 up to 20 images per class and better from 50 images per class on, including the full training set (0.816 against 0.793).
- Only `svm_tuned` used the validation labels. With 20 or fewer images per class, the 500 validation images give it more labeled data than its training set. `svm_default` uses no validation label and still exceeded the PMD with L = 3 at every budget with both vector sets.
- The selected hyperparameters are in `metrics.csv` (`C`, `gamma`, `selection_val_macro_f1`); C = 10 was selected in 10 of the 16 seed-42 cells; validation and test macro-F1 of the selected models were close (0.928 and 0.929 with the full training set, seed 42).
- The nearest centroid and the SVMs use standardized vectors, whereas the PMD receives the raw descriptor, as registered for PolyGabor. Whether the PMD changes with standardized inputs was not tested.
- Model size, latency and memory of the SVMs were not measured. Fit times are in `metrics.csv` (at most 4.1 s for the tuning grid of 30 fits with the full training set).
