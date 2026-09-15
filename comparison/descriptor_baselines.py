"""Descriptor-controlled classifiers for the scarce-label scenarios.

Registered in PROTOCOL.md ("Descriptor-controlled classifier extension") on
2026-09-15, before running. The question is whether the scarce-label
behavior of PolyGabor comes from the polynomial Mahalanobis distance or from
the 22-dimensional Gabor + CIELAB descriptor alone, so the descriptor, the
training images and the fitting vectors are kept fixed and only the decision
function changes:

- pmd_l3: the registered PolyGabor classifier, fitted again. Its test
  predictions must match those saved in runs/2026-09-12, otherwise the script
  stops before writing any metric;
- pmd_l1: the same code with num_levels=1;
- nearest_centroid: Euclidean distance to the standardized class means;
- svm_default: RBF SVM with C=1 and gamma='scale';
- svm_tuned: RBF SVM with C and gamma selected by validation macro-F1.

Every classifier receives the vectors retained by the PMD (at most 350 per
class, same random draw as the campaign). Standardization uses those vectors
only. Validation and test use the original images.

Run from colorectal_histology_classification/ with the cnn venv:
    cnn/.venv/bin/python comparison/descriptor_baselines.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
from polygarbor import PolyGaborClassifier  # noqa: E402
from cnn.evaluation import multiclass_gmean  # noqa: E402
from scenarios import SIZES, select_training  # noqa: E402
from variants import VARIANTS, training_views  # noqa: E402

CACHE = ROOT / "cache"
REFERENCE = ROOT / "runs/2026-09-12"
RUNS = ROOT / "runs/2026-09-15_descriptor"
OUT = ROOT / "results/2026-09-15_descriptor"
SCENARIOS = tuple(f"few{n}" for n in SIZES) + ("full",)
SEEDS = (42, 43)
VECTOR_SETS = ("polygarbor", "polygarbor_aug")
METHODS = ("pmd_l3", "pmd_l1", "nearest_centroid", "svm_default", "svm_tuned")
C_GRID = (0.1, 1, 10, 100, 1000)
GAMMA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1, "scale")


def extract_all(names, arrays):
    """Descriptors of every original image and of the augmented views of each seed."""
    path = RUNS / "features.npz"
    if path.exists():
        return dict(np.load(path))
    clf = PolyGaborClassifier(names)
    train_images = arrays["train"][0]
    _, copies = VARIANTS["polygarbor_aug"]
    features = {split: np.vstack([clf.extract(x) for x in arrays[split][0]]) for split in ("train", "val", "test")}
    for seed in SEEDS:
        print(json.dumps(dict(stage="extract_views", seed=seed)), flush=True)
        views = [np.vstack([clf.extract(v) for v in list(training_views(image, copies, seed, index))[1:]])
                 for index, image in enumerate(train_images)]
        features[f"views_{seed}"] = np.stack(views)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)
    return features


def fitting_vectors(features, vector_set, indices, labels, seed, n_classes, cap):
    """Vectors in the order of run.py, capped per class with the draw of fit_features."""
    _, copies = VARIANTS[vector_set]
    if copies:
        X = np.concatenate([features["train"][indices, None], features[f"views_{seed}"][indices]], axis=1)
        X = X.reshape(-1, X.shape[-1])
    else:
        X = features["train"][indices]
    X = X.astype(np.float32)
    y = np.repeat(labels, copies + 1).astype(np.int32)
    rng = np.random.default_rng(seed)
    parts, part_labels = [], []
    for c in range(n_classes):
        samples = X[y == c]
        if len(samples) > cap:
            samples = samples[rng.choice(len(samples), cap, replace=False)]
        parts.append(samples)
        part_labels.append(np.full(len(samples), c, dtype=np.int32))
    return np.vstack(parts), np.concatenate(part_labels)


def fit_pmd(names, X, y, seed, levels):
    clf = PolyGaborClassifier(names, num_levels=levels, random_state=seed).fit_features(X, y)

    def predict(Z):
        distances = clf.distances(Z)
        if not np.isfinite(distances).all():
            raise FloatingPointError("non-finite PMD distances")
        return distances.argmin(1)
    return predict, {}


def fit_centroid(names, X, y, seed):
    scaler = StandardScaler().fit(X)
    model = NearestCentroid().fit(scaler.transform(X), y)
    return lambda Z: model.predict(scaler.transform(Z)), {}


def fit_svm(names, X, y, seed, C=1.0, gamma="scale"):
    scaler = StandardScaler().fit(X)
    model = SVC(C=C, gamma=gamma, kernel="rbf").fit(scaler.transform(X), y)
    return lambda Z: model.predict(scaler.transform(Z)), dict(C=C, gamma=gamma)


def fit_svm_tuned(names, X, y, seed, X_val, y_val):
    best = None
    for C in C_GRID:
        for gamma in GAMMA_GRID:
            predict, _ = fit_svm(names, X, y, seed, C, gamma)
            score = macro_f1(y_val, predict(X_val), len(names))
            if best is None or score > best[0]:
                best = (score, C, gamma)
    predict, info = fit_svm(names, X, y, seed, best[1], best[2])
    return predict, dict(info, selection_val_macro_f1=best[0])


def macro_f1(y, pred, n_classes):
    return float(metrics.f1_score(y, pred, labels=range(n_classes), average="macro", zero_division=0))


def summarize(y, pred, names):
    n = len(names)
    recall = metrics.recall_score(y, pred, labels=range(n), average=None, zero_division=0)
    return dict(macro_f1=macro_f1(y, pred, n), multiclass_gmean=multiclass_gmean(y, pred, n),
                balanced_accuracy=float(metrics.balanced_accuracy_score(y, pred)),
                accuracy=float(np.mean(y == pred)),
                **{f"recall_{name}": float(r) for name, r in zip(names, recall)})


def main():
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    names = manifest["class_names"]
    arrays = {s: (np.load(CACHE / f"{s}_images.npy", mmap_mode="r"), np.load(CACHE / f"{s}_labels.npy"))
              for s in ("train", "val", "test")}
    features = extract_all(names, arrays)
    y_val, y_test = arrays["val"][1], arrays["test"][1]
    cap = PolyGaborClassifier(names).max_samples_per_class
    rows, reproduction = [], []

    for vector_set in VECTOR_SETS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                indices, labels = select_training(arrays["train"][1], scenario, seed)
                X, y = fitting_vectors(features, vector_set, indices, labels, seed, len(names), cap)
                cell = RUNS / f"{scenario}__{vector_set}__seed{seed}"
                cell.mkdir(parents=True, exist_ok=True)
                predictions = {}
                fits = dict(
                    pmd_l3=lambda: fit_pmd(names, X, y, seed, 3),
                    pmd_l1=lambda: fit_pmd(names, X, y, seed, 1),
                    nearest_centroid=lambda: fit_centroid(names, X, y, seed),
                    svm_default=lambda: fit_svm(names, X, y, seed),
                    svm_tuned=lambda: fit_svm_tuned(names, X, y, seed, features["val"], y_val),
                )
                for method in METHODS:
                    row = dict(vectors=vector_set, method=method, scenario=scenario, seed=seed,
                               images_per_class=len(indices) // len(names), fitted_vectors=len(X))
                    try:
                        start = time.perf_counter()
                        predict, info = fits[method]()
                        row["fit_seconds"] = time.perf_counter() - start
                        pred_val, pred_test = predict(features["val"]), predict(features["test"])
                    except Exception as error:  # the PMD cannot be built from one vector per class
                        rows.append(dict(row, status="undefined", error=f"{type(error).__name__}: {error}"))
                        continue
                    predictions[f"{method}_val"], predictions[f"{method}_test"] = pred_val, pred_test
                    rows.append(dict(row, status="complete", **info, val_macro_f1=macro_f1(y_val, pred_val, len(names)),
                                     **summarize(y_test, pred_test, names)))

                    if method == "pmd_l3":
                        saved = REFERENCE / f"{scenario}__{vector_set}__seed{seed}/eval/test_clean/predictions.npz"
                        if saved.exists():
                            agreement = float(np.mean(np.load(saved)["y_pred"] == pred_test))
                            reproduction.append(dict(run=saved.parent.parent.parent.name, agreement=agreement))
                            if agreement < 1:
                                OUT.mkdir(parents=True, exist_ok=True)
                                (OUT / "reproduction.json").write_text(json.dumps(reproduction, indent=2))
                                raise SystemExit(f"pmd_l3 does not reproduce {saved} (agreement {agreement:.4f})")
                np.savez_compressed(cell / "predictions.npz", y_val=y_val, y_test=y_test, **predictions)
                done = [r for r in rows[-len(METHODS):] if r["status"] == "complete"]
                print(json.dumps(dict(cell=cell.name, **{r["method"]: round(r["macro_f1"], 4) for r in done})), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reproduction.json").write_text(json.dumps(reproduction, indent=2))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics.csv", index=False)

    order = {s: i for i, s in enumerate(SCENARIOS)}
    lines = ["# Descriptor-controlled classifiers: test macro-F1", "",
             "Mean of seeds 42 and 43, with the range between seeds in brackets. Generated by `comparison/descriptor_baselines.py`.", ""]
    for vector_set in VECTOR_SETS:
        part = table[(table.vectors == vector_set) & (table.status == "complete")]
        stats = part.groupby(["scenario", "method"]).macro_f1.agg(["mean", "min", "max", "count"])
        lines += [f"## Fitting vectors of `{vector_set}`", "", "| images/class | " + " | ".join(METHODS) + " |",
                  "| ---: | " + " | ".join("---:" for _ in METHODS) + " |"]
        for scenario in sorted(part.scenario.unique(), key=order.get):
            cells = []
            for method in METHODS:
                if (scenario, method) not in stats.index:
                    cells.append("--")
                    continue
                s = stats.loc[(scenario, method)]
                cells.append(f"{s['mean']:.3f} [{s['min']:.3f}–{s['max']:.3f}]" + ("" if s["count"] == len(SEEDS) else f" (n={int(s['count'])})"))
            lines.append(f"| {scenario.removeprefix('few')} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))
    import polymahalanobis, sklearn
    (OUT / "environment.json").write_text(json.dumps(dict(
        numpy=np.__version__, sklearn=sklearn.__version__,
        polymahalanobis=getattr(polymahalanobis, "__version__", "unknown"),
        threads=os.environ["OMP_NUM_THREADS"]), indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
