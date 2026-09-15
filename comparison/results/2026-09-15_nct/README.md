# Independent-cohort replication: NCT-CRC-HE-100K to CRC-VAL-HE-7K (2026-09-15)

Extension registered in `../../PROTOCOL.md` ("Independent-cohort replication") before any image was selected. Scripts: `comparison/nct_prepare.py` (data and splits), `comparison/nct_descriptor.py` (descriptor methods), `run.py` through `suite.py` (YOLO11n-cls, campaign `runs/2026-09-15_nct`) and `comparison/nct_report.py` (tables). Outputs: `summary.md` (all tables), `metrics.csv`, `descriptor_metrics.csv`, `paired.csv`, `recall.csv`. `FROZEN.sha256` records the checksums of the code, manifest, arrays, predictions and tables.

## Data preparation

The training pool and validation set come from the color-normalized NCT-CRC-HE-100K archive, read with HTTP range requests; the test set is the whole color-normalized CRC-VAL-HE-7K archive. The first preparation run, with 16 unthrottled threads, was stopped by HTTP 429 from Zenodo before anything was saved. The second run, throttled to about one request per second, fetched and cached all 1,347 needed members of the 100K archive and then stopped with "No space left on device" while downloading the 7K archive; the disk had 21 GB free before and after, so the condition was transient and was not caused by the project files. The third run reused the cached members, whose selection is deterministic, and completed. The union of the selected training images has 897 patches (98 to 100 per class), the validation set 450 (50 per class) and the test set 7,180 (339 to 1,338 per class). There are no exact pixel duplicates between the three sets.

All 50 YOLO runs and all descriptor fits completed; the PMD is undefined with one image per class in all five seeds, as expected.

## Test macro-F1 (mean of seeds 42–46)

| images/class | RBF-SVM, C = 1 | tuned RBF-SVM | nearest centroid | PMD, L = 3 | YOLO11n-cls, scratch | YOLO11n-cls, ImageNet |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.300 | 0.300 | 0.300 | -- | 0.019 | 0.415 |
| 2 | 0.310 | 0.358 | 0.294 | 0.174 | 0.020 | 0.503 |
| 5 | 0.410 | 0.448 | 0.392 | 0.175 | 0.023 | 0.593 |
| 10 | 0.459 | 0.538 | 0.449 | 0.199 | 0.199 | 0.708 |
| 20 | 0.481 | 0.596 | 0.436 | 0.216 | 0.406 | 0.752 |

Ranges between seeds, G-mean, balanced accuracy, paired differences and per-class recall are in `summary.md`.

## Observations

- The classifier ablation replicated: the fixed RBF-SVM was more accurate than the PMD in every seed and budget, by 0.14 to 0.27 on average. The PMD recognized background but almost no adipose tissue (recall 0.04 with 10 images per class) and stayed below 0.22.
- Against YOLO11n-cls from scratch, the fixed SVM was more accurate in all five seeds with 1, 2 and 5 images per class (0.300–0.410 against 0.019–0.023), in three of five seeds with 10 (mean difference +0.260; the macro-F1 of YOLO11n-cls ranged from 0.021 to 0.502) and in four of five seeds with 20 (mean difference +0.075). With 1, 2 and 5 images per class, YOLO11n-cls from scratch stopped after 9 epochs in every seed.
- The ImageNet-initialized YOLO11n-cls was more accurate than the fixed SVM in every seed and budget, by 0.12 to 0.27 on average.
- The absolute accuracy of the descriptor was much lower than on the eight-class dataset (0.459 and 0.481 with 10 and 20 images per class, against 0.689 and 0.738), and its G-mean was 0.09 to 0.41. With 10 images per class, the SVM recognized adipose tissue (recall 0.87) and background (0.99), but not debris (0.21), smooth muscle (0.21), stroma (0.18) or tumor (0.27).
- The nearest centroid was close to the fixed SVM (mean difference +0.010 to +0.046 in favor of the SVM, and 0.000 with one image per class).
- The tuned SVM was more accurate than the fixed SVM in every seed from 5 images per class on (0.538 and 0.596 with 10 and 20), a larger gain than on the eight-class dataset, so the default hyperparameters suit this cohort less well. It uses the 450 validation labels, which with 20 or fewer training images per class exceed its training set.

## Limits

The collections are color-normalized, so the replication does not test the descriptor without stain normalization. The center crop keeps 45% of the pixels of each 224×224 patch. The taxonomy differs from that of the eight-class dataset (MUC denotes mucus here), so the replication tests whether the scarce-label behavior repeats, not whether the eight-class models transfer. Only YOLO11n-cls was trained as deep baseline, and only the scarce-label budgets were evaluated. According to the Zenodo record, the test patients do not overlap with the training patients; this was not verified independently.
