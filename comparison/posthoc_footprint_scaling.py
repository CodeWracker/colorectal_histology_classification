"""Post-hoc measurement: memory footprint of PolyGabor models versus data size.

Added on 2026-09-14 after the paper review. Not part of the registered
protocol. The descriptors retained by the full-training seed-42 model (350
per class) are reused, so no feature is extracted and nothing is evaluated
on the test subset. For each configuration a PolyGaborClassifier is fitted
with the project code and the following are recorded:

- saved_mb: size of the directory written by PolyGaborClassifier.save();
- model_array_mb: bytes held by the arrays of the fitted PMD models
  (centers, bases, weights, index vectors) plus the retained samples;
- fit_peak_mb: peak Python/NumPy allocation during fitting (tracemalloc);
- eval_peak_mb: peak allocation while computing the distances of one vector;
- level_dims: input dimension of each PMD level of the largest class model.

Run from colorectal_histology_classification/ with the polygarbor venv:
    polygarbor/.venv/bin/python comparison/posthoc_footprint_scaling.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import tracemalloc

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "4")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "polygarbor/src"))
from polygarbor import PolyGaborClassifier  # noqa: E402

SOURCE = ROOT / "runs/2026-09-12/full__polygarbor__seed42/model"
OUT = ROOT / "results/2026-09-14_posthoc"
VECTORS = (2, 5, 10, 20, 50, 100, 350)
CLASS_COUNTS = (2, 4, 8)


def array_bytes(model):
    total = sum(a.nbytes for a in (model.samples, model.center) if a is not None)
    for level in model.levels:
        total += sum(a.nbytes for a in (level.A_basis, level.dms, level.ind_use) if a is not None)
    return total


def measure(pool, names, classes, per_class, rng):
    X = np.vstack([pool[c][rng.choice(len(pool[c]), per_class, replace=False)] for c in classes])
    y = np.repeat(np.arange(len(classes)), per_class)
    clf = PolyGaborClassifier([names[c] for c in classes], random_state=42)

    tracemalloc.start()
    start = time.perf_counter()
    clf.fit_features(X, y)
    fit_seconds = time.perf_counter() - start
    fit_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    tracemalloc.start()
    clf.distances(X[:1])
    eval_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    latency = []
    for _ in range(30):
        start = time.perf_counter()
        clf.distances(X[:1])
        latency.append((time.perf_counter() - start) * 1000)

    folder = Path(tempfile.mkdtemp())
    try:
        clf.save(folder / "model")
        saved = sum(f.stat().st_size for f in (folder / "model").rglob("*") if f.is_file())
    finally:
        shutil.rmtree(folder)

    per_model = [array_bytes(m) for m in clf.models.values()]
    largest = clf.models[int(np.argmax(per_model))]
    return dict(n_classes=len(classes), vectors_per_class=per_class,
                saved_mb=saved / 1e6,
                model_array_mb=(sum(per_model) + sum(s.nbytes for s in clf.train_samples.values())) / 1e6,
                largest_class_model_mb=max(per_model) / 1e6,
                fit_peak_mb=fit_peak / 1e6, eval_peak_mb=eval_peak / 1e6,
                fit_seconds=fit_seconds, distance_median_ms=float(np.median(latency)),
                levels=len(largest.levels),
                level_dims="/".join(str(l.A_basis.shape[0]) for l in largest.levels),
                level_components="/".join(str(l.A_basis.shape[1]) for l in largest.levels))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = PolyGaborClassifier.load(SOURCE)
    names = source.class_names
    pool = [source.train_samples[c] for c in range(len(names))]
    configs = [(tuple(range(8)), v) for v in VECTORS] + [(tuple(range(k)), 350) for k in CLASS_COUNTS if k != 8]
    rows = []
    for classes, per_class in configs:
        row = measure(pool, names, classes, per_class, np.random.default_rng(42))
        row["classes"] = "+".join(names[c] for c in classes)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    with open(OUT / "footprint_scaling.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
