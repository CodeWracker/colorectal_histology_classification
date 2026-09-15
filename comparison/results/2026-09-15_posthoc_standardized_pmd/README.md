# Post-hoc: PMD fitted to standardized descriptors (2026-09-15)

Exploratory analysis, not part of the registered protocol. It was authorized by the authors on 2026-09-15 and motivated by results already observed: in the registered descriptor-controlled extension (`../2026-09-15_descriptor`), the RBF-SVM and the nearest centroid, which receive standardized vectors, exceeded the PMD, which receives the raw descriptor. The question was whether part of that gap comes from the interaction between feature scale, the relative truncation threshold and the Tikhonov regularization of the PMD. The registered descriptor-controlled results and the registered SVM campaign were frozen (`FROZEN.sha256` in each folder) before this analysis was run. These results do not replace the registered PMD results, which remain the reference.

Script: `comparison/posthoc_standardized_pmd.py`. Outputs: `metrics.csv` and `summary.md`; per-image predictions in `runs/2026-09-15_posthoc_standardized_pmd`. Everything is identical to the descriptor-controlled extension (cached descriptors, scenarios, seeds 42 and 43, both vector sets, retained vectors), except that each vector is standardized with the mean and standard deviation of the retained fitting vectors of the run before the PMD is fitted and evaluated. No validation or test label defines the transformation. The PMD still fails with one vector per class without augmentation, with the same `IndexError`.

## Test macro-F1, vectors of `polygarbor`

Means of two seeds. The raw PMD, SVM and nearest-centroid columns are the registered values of `../2026-09-15_descriptor`.

| images/class | PMD L = 3, raw | PMD L = 3, standardized | PMD L = 1, standardized | nearest centroid | RBF-SVM, C = 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.214 | 0.358 | 0.358 | 0.485 | 0.432 |
| 5 | 0.418 | 0.435 | 0.363 | 0.523 | 0.573 |
| 10 | 0.501 | 0.592 | 0.546 | 0.607 | 0.689 |
| 20 | 0.569 | 0.625 | 0.514 | 0.620 | 0.738 |
| 50 | 0.673 | 0.705 | 0.722 | 0.612 | 0.773 |
| 100 | 0.709 | 0.768 | 0.795 | 0.644 | 0.813 |
| full | 0.793 | 0.800 | 0.824 | 0.655 | 0.878 |

## Observations

- Without augmentation, standardization raised the macro-F1 of the PMD with L = 3 at every budget, by 0.007 (full) to 0.144 (2 images per class). The variation between seeds remained large with few images (0.262–0.455 with 2 and 0.544–0.640 with 10 images per class).
- The standardized PMD remained below the RBF-SVM with fixed hyperparameters at every budget, and below the nearest centroid with 2, 5 and 10 images per class. With 10 images per class, standardization closed about half of the gap to the SVM (0.501 → 0.592, against 0.689). The G-mean of the standardized PMD with L = 3 was still zero with 2 images per class in both seeds and with 5 images per class in seed 42.
- As with raw descriptors, L = 1 was better than L = 3 from 50 images per class on, including the full training set (0.824 against 0.800).
- With the augmented vectors, the effect was mixed: standardization raised the PMD with L = 3 with 1, 2, 10 and 100 images per class and with the full set, and lowered it with 5, 20 and 50 (for example 0.482 → 0.444 with 5). It remained below the SVM at every budget (`summary.md`).
- Feature scale therefore explains part of the weakness of the PMD on this descriptor, but not the difference to the SVM, and it does not change the conclusion of the registered extension. Because the test was chosen after the results were known and was evaluated with two seeds, it should be reported as exploratory.
