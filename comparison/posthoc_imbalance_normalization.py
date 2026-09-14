"""Post-hoc exploratory ablation: per-class distance normalization for PolyGabor.

Not part of the registered protocol. Added on 2026-09-14 after the paper
review pointed out that the PMD scatter matrix is not normalized by the
number of fitting vectors, so class models fitted to fewer vectors produce
distances on a different scale and bias argmin_c D_c.

No model is retrained. The saved models of the 2026-09-12 campaign are
loaded, and the decision rule is changed from argmin_c D_c to
argmin_c D_c / r_c, where r_c is a reference scale of class c computed only
from the fitting vectors of that class (no validation or test labels):

- insample: median of D_c over the fitting vectors themselves;
- crossfit: median of D_c over the fitting vectors, each evaluated by a model
  fitted without it (5-fold cross-fitting inside the class);
- count: 1 / N_c, i.e. argmin_c N_c * D_c. Dividing the scatter matrix by N_c
  scales every singular value and s_min = eps * s_max by 1 / N_c while the
  bases, projections and relative thresholds stay the same, so every level
  of D_c is multiplied by N_c. This rule is therefore the PMD with a scatter
  matrix normalized by the number of fitting vectors.

The balanced full-training models are evaluated as a control, to check that
the normalization does not hurt when every class has 350 vectors.

Run from colorectal_histology_classification/ with the polygarbor venv:
    polygarbor/.venv/bin/python comparison/posthoc_imbalance_normalization.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))
from polygarbor import PolyGaborClassifier  # noqa: E402
from polygarbor.classifier import _make_space  # noqa: E402

RUNS = ROOT / "runs/2026-09-12"
CACHE = ROOT / "cache"
OUT = ROOT / "results/2026-09-14_posthoc"
FOLDS = 5
MODELS = (("imbalance", 42), ("full", 42), ("full", 43))


def summarize(y, pred, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    precision = np.diag(cm) / np.maximum(cm.sum(0), 1)
    f1 = np.where(recall + precision > 0, 2 * recall * precision / np.maximum(recall + precision, 1e-12), 0)
    return dict(accuracy=float(np.mean(y == pred)), macro_f1=float(f1.mean()),
                multiclass_gmean=float(np.prod(recall) ** (1 / n)),
                recall=recall.round(4).tolist(), precision=precision.round(4).tolist())


def reference_scales(clf):
    rng = np.random.default_rng(clf.random_state)
    insample, crossfit = [], []
    for c in range(clf.n_classes):
        samples = clf.train_samples[c]
        insample.append(float(np.median(clf.models[c].evaluate(samples)[:, -1])))
        folds = np.array_split(rng.permutation(len(samples)), FOLDS)
        held_out = []
        for fold in folds:
            keep = np.setdiff1d(np.arange(len(samples)), fold)
            model = _make_space(samples[keep], clf.num_levels)
            held_out.append(model.evaluate(samples[fold])[:, -1])
        crossfit.append(float(np.median(np.concatenate(held_out))))
    return np.asarray(insample), np.asarray(crossfit)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arrays = {s: (np.load(CACHE / f"{s}_images.npy", mmap_mode="r"), np.load(CACHE / f"{s}_labels.npy"))
              for s in ("val", "test")}
    rows, details = [], {}
    features = {}
    for scenario, seed in MODELS:
        run = f"{scenario}__polygarbor__seed{seed}"
        clf = PolyGaborClassifier.load(RUNS / run / "model")
        n = clf.n_classes
        for split, (images, _) in arrays.items():
            if split not in features:
                features[split] = np.vstack([clf.extract(image) for image in images])
        insample, crossfit = reference_scales(clf)
        details[run] = dict(fitted_vectors_per_class=[len(clf.train_samples[c]) for c in range(n)],
                            reference_insample_median=insample.tolist(),
                            reference_crossfit_median=crossfit.tolist())
        for split, (_, labels) in arrays.items():
            dist = clf.distances(features[split])
            counts = np.asarray([len(clf.train_samples[c]) for c in range(n)], dtype=float)
            for rule, scale in (("raw", np.ones(n)), ("insample", insample), ("crossfit", crossfit),
                                ("count", 1 / counts)):
                pred = np.argmin(dist / np.maximum(scale, 1e-12), axis=1)
                summary = summarize(labels, pred, n)
                rows.append(dict(run=run, scenario=scenario, seed=seed, split=split, rule=rule,
                                 accuracy=summary["accuracy"], macro_f1=summary["macro_f1"],
                                 multiclass_gmean=summary["multiclass_gmean"],
                                 **{f"recall_{clf.class_names[c]}": summary["recall"][c] for c in range(n)}))
                details[run][f"{split}_{rule}"] = summary
                print(f"{run:32s} {split:4s} {rule:8s} macroF1={summary['macro_f1']:.4f} "
                      f"gmean={summary['multiclass_gmean']:.4f} recall={summary['recall']}", flush=True)

    import csv
    with open(OUT / "imbalance_normalization.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "imbalance_normalization.json").write_text(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
