"""Post-hoc exploratory analysis: PMD fitted to standardized descriptors.

Not part of the registered protocol. Authorized by the authors on 2026-09-15
and motivated by the results already observed in the registered
descriptor-controlled extension (results/2026-09-15_descriptor, frozen in
FROZEN.sha256 before this script was written): the RBF-SVM and the nearest
centroid, which receive standardized vectors, exceeded the PMD, which receives
the raw descriptor. The question is whether part of that gap comes from the
interaction between feature scale, the relative truncation threshold and the
Tikhonov regularization of the PMD. This analysis does not replace the
registered PMD results, which remain the reference.

Everything else is identical to descriptor_baselines.py: the same cached
descriptors, scenarios, seeds, vector sets and retained vectors. The only
change is that each vector is standardized with the mean and standard
deviation of the retained fitting vectors of the run (as for the SVM) before
fitting and evaluating the PMD with L = 3 and L = 1. No validation or test
label is used to define the transformation.

Run from colorectal_histology_classification/ with the cnn venv, after
descriptor_baselines.py:
    cnn/.venv/bin/python comparison/posthoc_standardized_pmd.py
"""
from __future__ import annotations

import json
import os
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from descriptor_baselines import (CACHE, ROOT, RUNS as DESCRIPTOR_RUNS, SCENARIOS, SEEDS, VECTOR_SETS,
                                  fitting_vectors, macro_f1, select_training, summarize)
from polygarbor import PolyGaborClassifier

RUNS = ROOT / "runs/2026-09-15_posthoc_standardized_pmd"
OUT = ROOT / "results/2026-09-15_posthoc_standardized_pmd"
LEVELS = {"pmd_l3_standardized": 3, "pmd_l1_standardized": 1}


def main():
    names = json.loads((ROOT / "dataset_manifest.json").read_text())["class_names"]
    features_path = DESCRIPTOR_RUNS / "features.npz"
    if not features_path.exists():
        raise FileNotFoundError(f"{features_path} is missing; run descriptor_baselines.py first")
    features = dict(np.load(features_path))
    train_labels = np.load(CACHE / "train_labels.npy")
    y_val, y_test = np.load(CACHE / "val_labels.npy"), np.load(CACHE / "test_labels.npy")
    cap = PolyGaborClassifier(names).max_samples_per_class
    rows = []
    for vector_set in VECTOR_SETS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                indices, labels = select_training(train_labels, scenario, seed)
                X, y = fitting_vectors(features, vector_set, indices, labels, seed, len(names), cap)
                scaler = StandardScaler().fit(X)
                Z, Z_val, Z_test = (scaler.transform(a).astype(np.float32) for a in (X, features["val"], features["test"]))
                predictions = {}
                for method, levels in LEVELS.items():
                    row = dict(vectors=vector_set, method=method, scenario=scenario, seed=seed,
                               images_per_class=len(indices) // len(names), fitted_vectors=len(X))
                    try:
                        start = time.perf_counter()
                        clf = PolyGaborClassifier(names, num_levels=levels, random_state=seed).fit_features(Z, y)
                        row["fit_seconds"] = time.perf_counter() - start
                        distances_val, distances_test = clf.distances(Z_val), clf.distances(Z_test)
                        if not (np.isfinite(distances_val).all() and np.isfinite(distances_test).all()):
                            raise FloatingPointError("non-finite PMD distances")
                    except Exception as error:
                        rows.append(dict(row, status="undefined", error=f"{type(error).__name__}: {error}"))
                        continue
                    pred_val, pred_test = distances_val.argmin(1), distances_test.argmin(1)
                    predictions[f"{method}_val"], predictions[f"{method}_test"] = pred_val, pred_test
                    rows.append(dict(row, status="complete", val_macro_f1=macro_f1(y_val, pred_val, len(names)),
                                     **summarize(y_test, pred_test, names)))
                cell = RUNS / f"{scenario}__{vector_set}__seed{seed}"
                cell.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cell / "predictions.npz", y_val=y_val, y_test=y_test, **predictions)
                done = [r for r in rows[-len(LEVELS):] if r["status"] == "complete"]
                print(json.dumps(dict(cell=cell.name, **{r["method"]: round(r["macro_f1"], 4) for r in done})), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "metrics.csv", index=False)
    order = {s: i for i, s in enumerate(SCENARIOS)}
    lines = ["# Post-hoc: PMD on standardized descriptors, test macro-F1", "",
             "Exploratory, not registered. Mean of seeds 42 and 43, range in brackets. Generated by `comparison/posthoc_standardized_pmd.py`.", ""]
    for vector_set in VECTOR_SETS:
        part = table[(table.vectors == vector_set) & (table.status == "complete")]
        stats = part.groupby(["scenario", "method"]).macro_f1.agg(["mean", "min", "max", "count"])
        lines += [f"## Fitting vectors of `{vector_set}`", "", "| images/class | " + " | ".join(LEVELS) + " |",
                  "| ---: | " + " | ".join("---:" for _ in LEVELS) + " |"]
        for scenario in sorted(part.scenario.unique(), key=order.get):
            cells = []
            for method in LEVELS:
                if (scenario, method) not in stats.index:
                    cells.append("--")
                    continue
                s = stats.loc[(scenario, method)]
                cells.append(f"{s['mean']:.3f} [{s['min']:.3f}–{s['max']:.3f}]" + ("" if s["count"] == len(SEEDS) else f" (n={int(s['count'])})"))
            lines.append(f"| {scenario.removeprefix('few')} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
