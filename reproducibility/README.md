# Reproducibility

Four documents, so that a reviewer can get from a number in the paper to the file and command behind it without reading the harness source first.

| Document | Answers |
| --- | --- |
| [SETUP.md](SETUP.md) | Which environments exist, what hardware and disk the experiments need, how to get the two datasets. |
| [PIPELINE.md](PIPELINE.md) | Which commands produce each campaign, in which order, and how long they took. |
| [PAPER_MAP.md](PAPER_MAP.md) | Which artifact holds each table, figure and quoted number, and how to derive the published value from its rows. |
| This file | What can be checked without computing anything, what will differ on a rerun, and which runs are known to fail. |

## How much you want to reproduce

Verifying the reported values needs no computation. Every number in the paper's tables comes from a CSV file versioned in this repository, and [PAPER_MAP.md](PAPER_MAP.md) names the file, the rows and the aggregation for each one. No environment, no dataset, no GPU. The checksums below confirm those files are the ones the analysis scripts wrote.

Rebuilding the tables needs the per-run directories but no training. `report.py`, `loso_report.py`, `svm_report.py`, `nct_report.py` and `audit.py` read only saved artifacts and recompute every table, statistic and figure, which is the check that nothing was edited by hand.

Rerunning the campaigns means following [PIPELINE.md](PIPELINE.md) in order. Budget about a day of compute on a comparable machine, most of it in the main campaign and the LOSO folds, plus the network time of the replication download. The descriptor campaigns are much cheaper: the whole RBF-SVM campaign, 38 runs covering full training, seven scarce budgets, label noise, imbalance and ten LOSO folds, took 1,400 s on the CPU.

## Checksums

Three campaigns were frozen with a SHA-256 manifest before any analysis that came after them, which makes the registered-then-post-hoc ordering checkable instead of merely asserted. The manifests list paths relative to `comparison/` and cover code, tables, and the per-run predictions and models that are not versioned here.

To check the subset that ships in this repository:

```bash
cd comparison
for c in 2026-09-15_descriptor 2026-09-15_svm 2026-09-15_nct; do
  echo "== $c"
  while read -r h p; do [ -f "$p" ] && printf '%s  %s\n' "$h" "$p"; done < "results/$c/FROZEN.sha256" | sha256sum -c -
done
```

Every table, report and analysis script verifies. Two entries do not, both because the file was extended afterwards, and both explained by the dates in `PROTOCOL.md`. `PROTOCOL.md` fails against the `2026-09-15_descriptor` and `2026-09-15_svm` manifests because the replication was registered in it later. `run.py` fails against the `2026-09-15_svm` manifest because the `nct_` scenario branch was added for that replication. Both verify against the `2026-09-15_nct` manifest, which is the most recent of the three. So the later manifest confirms the current state of both files, and the earlier ones confirm that no result was produced after its inputs had been recorded.

The main campaign is covered instead by `comparison/integrity_audit.json` and `comparison/results/2026-09-12/validation_summary.json`, which record 278 of 278 runs passing the audit together with the test, CLI and benchmark checks of `validate.py`.

## Determinism

The descriptor, the PMD, the nearest centroid and the RBF-SVM are fully deterministic. The dataset reading order is fixed, the subset draws and the 350-vectors-per-class cap use explicit seeds, and repeating a command rebuilds the same model bit for bit. That is why the classifier ablation could use exact agreement of test predictions as a stopping condition, and why the SVM campaign could require its predictions to equal the earlier ablation's.

The CNNs are not bitwise reproducible on a GPU. During validation one prediction in a hundred from the ImageNet ResNet-18 differed between devices, on an image where the top two classes were 0.0017 apart; the one-thread and four-thread CPU runs agreed completely. `validate.py` therefore records agreement at the 99% level instead of demanding bitwise equality. Expect small differences in CNN metrics on a rerun, especially with few training images, where early stopping can pick a different epoch.

Timings, latencies and memory peaks belong to the reference machine in [SETUP.md](SETUP.md), not to the methods. The paper's claim is the relative ordering, a descriptor model that trains in seconds on a CPU against CNNs that take minutes to tens of minutes. The absolute seconds are not portable.

## Known failures

Recorded rather than hidden, because a reviewer will meet them.

PolyGabor with one image per class (`few1__polygarbor__seed42` and `seed43`) cannot build the PMD from a single fitting vector per class. `polymahalanobis` reads the fitting vectors from a text file, and a file with one row loads as a one-dimensional array, so an `IndexError` is raised while loading, before any scatter matrix is computed. The runs are preserved as observed failures with their traceback, and the cells are reported as undefined rather than as a macro-F1 of zero. `pmd_l1` fails in the same cells for the same reason. With augmentation the cell is defined, because three extra views per image give the class more than one vector.

Two YOLO runs were interrupted when the repository was moved and moved back during the campaign, which broke absolute paths for the `label_noise` and `source_b` runs in flight. They were archived under `runs/interrupted`, repeated, and only the repeats are in the results. This is also why an external interruption can leave a `running` status behind: it is not a success, and its log has to be read.

Quantization was killed for lack of memory on the first attempts, which kept nine preprocessed test sets and all runtimes in one process. The export now runs in a subprocess separate from evaluation, and each model and seed runs on its own. Keep that structure on a machine with less RAM.

The replication download was throttled by Zenodo, which returned HTTP 429 to the first unthrottled attempt. `nct_prepare.py` now issues about one request per second and caches every fetched member, so an interrupted preparation restarts and reuses what it has.

## Registered and post-hoc

`comparison/PROTOCOL.md` is the registered protocol, written before the main experiments, with every extension added as its own dated section before that extension ran. Two directories hold analyses motivated by results already observed, which the paper labels exploratory: `results/2026-09-14_posthoc` (PMD distance rescaling under imbalance, memory footprint, CPU retraining of the CNNs) and `results/2026-09-15_posthoc_standardized_pmd` (the PMD fitted to standardized descriptors). Each ran only after the registered results it comments on had been frozen, which the manifests above make verifiable.

Results are reported in whichever direction they came out. The classifier ablation was registered with the explicit condition that its outcome would be reported even if an off-the-shelf SVM matched or beat the PMD, which is what happened, and the paper's framing follows the outcome rather than the original hypothesis.

## Limits

The main dataset is 5,000 patches from ten slides of a single institution, so the LOSO experiment measures generalization across slides of one collection, not external validation. The slide codes were recovered from the original file names by pixel hashing and identify a slide, not a confirmed patient. The replication uses color-normalized collections, so it does not test the descriptor without stain normalization, and its nine classes are not the eight classes of the main dataset. The edge benchmark restricts an x86 CPU to one logical processor; it is not a measurement on ARM or on a microcontroller, and the embedded analysis is a static memory and operation preflight, not an emulation. None of the explanation maps are segmentation masks.
