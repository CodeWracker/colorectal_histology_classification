"""Descriptor methods of the NCT-CRC-HE-100K -> CRC-VAL-HE-7K replication.

Registered in PROTOCOL.md ("Independent-cohort replication") on 2026-09-15.
The 22-dimensional descriptor is extracted once from the 150x150 crops of
cache/nct (nct_prepare.py), and for every seed and budget of the manifest the
same training images are classified by:

- gabor_svm: RBF SVM, C=1, gamma='scale', standardized vectors (primary);
- svm_tuned: C and gamma selected by macro-F1 on the 450 validation patches;
- nearest_centroid: Euclidean distance to the standardized class means;
- pmd_l3: the registered PMD with L=3 on the raw descriptor.

The fitting functions are those of descriptor_baselines.py. With at most 20
images per class the cap of 350 vectors per class never applies.

Run from colorectal_histology_classification/ with the cnn venv, after nct_prepare.py:
    cnn/.venv/bin/python comparison/nct_descriptor.py
"""
from __future__ import annotations

import json
import os
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import numpy as np
import pandas as pd

from descriptor_baselines import ROOT, fit_centroid, fit_pmd, fit_svm, fit_svm_tuned, macro_f1, summarize
from polygarbor import PolyGaborClassifier

NCT = ROOT / "cache/nct"
RUNS = ROOT / "runs/2026-09-15_nct_descriptor"
OUT = ROOT / "results/2026-09-15_nct"
METHODS = ("gabor_svm", "svm_tuned", "nearest_centroid", "pmd_l3")


def main():
    manifest = json.loads((NCT / "manifest.json").read_text())
    names = manifest["class_names"]
    labels = {s: np.load(NCT / f"{s}_labels.npy") for s in ("train", "val", "test")}
    path = RUNS / "features.npz"
    if path.exists():
        features = dict(np.load(path))
    else:
        extractor = PolyGaborClassifier(names)
        features = {s: np.vstack([extractor.extract(x) for x in np.load(NCT / f"{s}_images.npy", mmap_mode="r")])
                    for s in ("train", "val", "test")}
        RUNS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **features)
    cap = PolyGaborClassifier(names).max_samples_per_class
    rows = []
    for seed in manifest["seeds"]:
        for scenario, indices in manifest["selections"][str(seed)].items():
            indices = np.asarray(indices)
            X, y = features["train"][indices].astype(np.float32), labels["train"][indices].astype(np.int32)
            assert np.bincount(y, minlength=len(names)).max() <= cap
            fits = dict(
                gabor_svm=lambda: fit_svm(names, X, y, seed),
                svm_tuned=lambda: fit_svm_tuned(names, X, y, seed, features["val"], labels["val"]),
                nearest_centroid=lambda: fit_centroid(names, X, y, seed),
                pmd_l3=lambda: fit_pmd(names, X, y, seed, 3),
            )
            predictions = {}
            for method in METHODS:
                row = dict(method=method, scenario=scenario, seed=seed, images_per_class=len(indices) // len(names))
                try:
                    start = time.perf_counter()
                    predict, info = fits[method]()
                    row["fit_seconds"] = time.perf_counter() - start
                    pred_val, pred_test = predict(features["val"]), predict(features["test"])
                except Exception as error:  # the PMD cannot be built from one vector per class
                    rows.append(dict(row, status="undefined", error=f"{type(error).__name__}: {error}"))
                    continue
                predictions[f"{method}_val"], predictions[f"{method}_test"] = pred_val, pred_test
                rows.append(dict(row, status="complete", **info, val_macro_f1=macro_f1(labels["val"], pred_val, len(names)),
                                 **summarize(labels["test"], pred_test, names)))
            cell = RUNS / f"{scenario}__seed{seed}"
            cell.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cell / "predictions.npz", y_val=labels["val"], y_test=labels["test"], **predictions)
            done = [r for r in rows[-len(METHODS):] if r["status"] == "complete"]
            print(json.dumps(dict(cell=cell.name, **{r["method"]: round(r["macro_f1"], 4) for r in done})), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "descriptor_metrics.csv", index=False)


if __name__ == "__main__":
    main()
