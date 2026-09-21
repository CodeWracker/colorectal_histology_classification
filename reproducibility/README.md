# Reproducibility guide

This guide exists so that a reviewer can go from a number printed in the paper to the file that contains it and the command that produced it, without reading the whole codebase first. It has four parts:

| Document | Answers |
| --- | --- |
| [SETUP.md](SETUP.md) | Which environments exist, what hardware and disk the experiments need, and how the two datasets are obtained. |
| [PIPELINE.md](PIPELINE.md) | Which commands produce each campaign, in which order, and how long they took. |
| [PAPER_MAP.md](PAPER_MAP.md) | Which artifact holds each table, figure and quoted number, and how the published value is derived from its rows. |
| This file | What can be checked without computing anything, what to expect to differ on a rerun, and which runs are known to fail and why. |

## Three levels of reproduction

**Level 1 — verify the reported values, no computation.** Every number in the paper's tables is computed from a CSV file that is versioned in this repository. [PAPER_MAP.md](PAPER_MAP.md) names the file, the rows and the aggregation for each one. This takes minutes and needs no environment, no dataset and no GPU. The checksum manifests below let you confirm that those files are the ones the analysis scripts wrote.

**Level 2 — rebuild the tables from the saved runs.** `report.py`, `loso_report.py`, `svm_report.py`, `nct_report.py` and `audit.py` read only saved artifacts and recompute every table, statistic and figure. If you have the per-run directories, these reproduce the published tables exactly, because no model is loaded or re-evaluated. They are also the honest check that the tables were not edited by hand.

**Level 3 — rerun the campaigns.** Follow [PIPELINE.md](PIPELINE.md) in order. Budget roughly a day of compute on a comparable machine for everything, most of it in the main campaign and the LOSO folds, plus the network time of the replication download. The descriptor-based campaigns alone are far cheaper: the whole RBF-SVM campaign — 38 runs covering full training, seven scarce budgets, label noise, imbalance and ten LOSO folds — took 1,400 s on the CPU.

## Verifying the frozen artifacts

Three campaigns were frozen with a SHA-256 manifest before any analysis that came after them, which is what makes the "registered, then post-hoc" ordering checkable rather than merely asserted. The manifests list paths relative to `comparison/`, and they cover code, tables, and the per-run predictions and models that are not versioned here.

To check the subset that ships in this repository:

```bash
cd comparison
for c in 2026-09-15_descriptor 2026-09-15_svm 2026-09-15_nct; do
  echo "== $c"
  while read -r h p; do [ -f "$p" ] && printf '%s  %s\n' "$h" "$p"; done < "results/$c/FROZEN.sha256" | sha256sum -c -
done
```

Every table, report and analysis script verifies. Two entries do not, both because a file was legitimately extended after that manifest was written, and both explained by the dates in `PROTOCOL.md`:

- `PROTOCOL.md` fails against the `2026-09-15_descriptor` and `2026-09-15_svm` manifests, because the independent-cohort replication was registered in it afterwards. It verifies against the `2026-09-15_nct` manifest, which is the most recent of the three.
- `run.py` fails against the `2026-09-15_svm` manifest, because the `nct_` scenario branch was added to it for the replication. It too verifies against the `2026-09-15_nct` manifest.

In other words, the later manifest confirms the current state of both files, and the earlier ones confirm that no result was produced after its inputs had already been recorded. The main campaign is covered instead by `comparison/integrity_audit.json` and `comparison/results/2026-09-12/validation_summary.json`, which record that 278 of 278 runs passed the audit, together with the test, CLI and benchmark checks of `validate.py`.

## What is deterministic and what is not

The descriptor, the PMD, the nearest centroid and the RBF-SVM are fully deterministic: the reading order of the dataset is fixed, the subset draws and the 350-vectors-per-class cap use explicit seeds, and repeating a command rebuilds the same model bit for bit. That is why the classifier ablation could use exact agreement of test predictions as its stopping condition, and why the SVM campaign could require its predictions to equal those of the earlier ablation.

The CNNs are not bitwise reproducible on a GPU. During validation, one prediction in a hundred from the ImageNet ResNet-18 differed between devices, on an image where the two top classes were separated by 0.0017; the single-thread and four-thread CPU runs agreed completely. `validate.py` therefore records agreement at the 99% level rather than demanding bitwise equality. Expect small differences in CNN metrics on a rerun, especially with few training images, where early stopping can select a different epoch.

Timings, latencies and memory peaks are properties of the reference machine described in [SETUP.md](SETUP.md), not of the methods. The relative ordering — a descriptor model that trains in seconds on a CPU against CNNs that take minutes to tens of minutes — is what the paper claims; the absolute seconds are not portable.

## Known failures and their causes

These are recorded rather than hidden, because a reviewer will meet them.

**PolyGabor with one image per class (`few1__polygarbor__seed42` and `seed43`).** The PMD cannot be built from a single fitting vector per class: `polymahalanobis` reads the fitting vectors from a text file, and a file with one row loads as a one-dimensional array, so an `IndexError` is raised while loading, before any scatter matrix is computed. The runs are preserved as observed failures with their traceback, and the corresponding cells are reported as undefined rather than as a macro-F1 of zero. `pmd_l1` fails in the same cells for the same reason. With augmentation the cell is defined, because three extra views per image give the class more than one vector.

**Two YOLO runs interrupted by a directory move.** The repository was moved and moved back during the campaign, which broke absolute paths for the `label_noise` and `source_b` YOLO runs in flight. They were archived under `runs/interrupted`, repeated, and only the repeats are in the results. This is also why the runner records a `running` status that an external interruption can leave behind: a `running` status is not a success, and its log must be read.

**Quantization killed for lack of memory.** The first attempts kept nine preprocessed test sets and all runtimes in one process. The export now runs in a subprocess separate from evaluation, and each model and seed runs on its own. If you rerun it on a machine with less RAM, keep that structure.

**Replication download throttled.** Zenodo returned HTTP 429 to the first, unthrottled preparation attempt. `nct_prepare.py` now issues about one request per second and caches every fetched member, so an interrupted preparation can be restarted and will reuse what it already has.

## Registered results and post-hoc analyses

`comparison/PROTOCOL.md` is the registered protocol, written before the main experiments, with every extension added as its own dated section before that extension was run. Three directories hold analyses that were instead motivated by results already observed, and the paper labels them as exploratory: `results/2026-09-14_posthoc` (PMD distance rescaling under imbalance, memory footprint, CPU retraining of the CNNs), and `results/2026-09-15_posthoc_standardized_pmd` (the PMD fitted to standardized descriptors). Each of these was run only after the registered results it comments on had been frozen, which the manifests above make verifiable.

Results are reported in whichever direction they came out. The classifier ablation was registered with the explicit condition that its outcome would be reported even if an off-the-shelf SVM matched or beat the PMD, which is what happened, and the paper's framing follows that outcome rather than the original hypothesis.

## Scope and limits of what can be reproduced

The main dataset is 5,000 patches from ten slides of a single institution, so the LOSO experiment measures generalization across slides of one collection, not external validation. The slide codes were recovered from the original file names by pixel hashing and identify a slide, not a confirmed patient. The independent-cohort replication uses color-normalized collections, so it does not test the descriptor without stain normalization, and its nine classes are not the eight classes of the main dataset. The edge benchmark restricts an x86 CPU to one logical processor; it is not a measurement on ARM or on a microcontroller, and the embedded analysis is a static memory and operation preflight, not an emulation. None of the explanation maps are segmentation masks.
