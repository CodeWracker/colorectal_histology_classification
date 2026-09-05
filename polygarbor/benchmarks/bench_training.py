"""Measure how long training the model takes, stage by stage.

Usage:
    uv run python benchmarks/bench_training.py --data-dir data

Training is one workflow, so instead of timing an isolated callable this script
runs the whole thing ``--runs`` times and records each stage within every run:
loading the dataset, extracting the descriptors from the training split, fitting
the per-class subspaces and writing the model to disk. The total is the sum of
the stages of a single run, which is what a person waits for.

Needs the dataset extra (``uv sync --extra dataset``) and a dataset already
materialized by ``polygarbor dataset``.

Writes a markdown report and a JSON with the same numbers to
``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

from _bench_utils import (
    Stopwatch,
    Timing,
    collect_specs,
    default_base_name,
    format_duration,
    specs_section,
    timings_table,
    write_report,
)

LOAD_DATASET = "Load the dataset (TFDS, already materialized)"
EXTRACT = "Extract descriptors from the training split"
FIT = "Fit the polynomial subspaces (one per class)"
SAVE = "Save the model to disk"
TOTAL = "Total training time"


def run_benchmark(
    data_dir: Path, runs: int, limit: int | None, patches_per_row: int, max_samples: int
) -> dict:
    import numpy as np

    from polygarbor import PolyGaborClassifier, dataset as ds_mod

    watch = Stopwatch()
    totals: list[float] = []
    info: dict = {}

    for run in range(runs):
        print(f"  run {run + 1}/{runs}...", flush=True)
        run_start = time.perf_counter()

        bundle = watch.time(LOAD_DATASET, lambda: ds_mod.load(data_dir))

        clf = PolyGaborClassifier(
            class_names=bundle.class_names,
            patches_per_row=patches_per_row,
            max_samples_per_class=max_samples,
        )

        def extract():
            X_parts, y_parts = [], []
            for image, label in ds_mod.iter_numpy(bundle["train"], limit=limit):
                feats = clf.extract(image)
                if len(feats):
                    X_parts.append(feats)
                    y_parts.append(np.full(len(feats), label, dtype=np.int32))
            return np.vstack(X_parts), np.concatenate(y_parts)

        X, y = watch.time(EXTRACT, extract)
        watch.time(FIT, lambda: clf.fit_features(X, y))

        with tempfile.TemporaryDirectory() as tmp:
            watch.time(SAVE, lambda: clf.save(Path(tmp) / "model"))

        totals.append((time.perf_counter() - run_start) * 1000.0)
        info = {
            "images": limit or bundle.num_examples["train"],
            "classes": len(bundle.class_names),
            "vectors_extracted": int(len(X)),
            "vectors_per_class_fitted": int(max_samples),
            "features_per_vector": int(X.shape[1]),
            "patches_per_image": int(len(X) / (limit or bundle.num_examples["train"])),
            "polynomial_levels": clf.num_levels,
        }

    stages = watch.timings()
    total = Timing(TOTAL, totals)

    per_image = statistics.fmean(
        t / info["images"] for t in next(s for s in stages if s.name == EXTRACT).samples
    )

    return {
        "benchmark": "training",
        "specs": collect_specs(extra_packages=("tensorflow-cpu", "tensorflow-datasets")),
        "setup": {
            "data_dir": str(data_dir),
            "runs": runs,
            "patches_per_row": patches_per_row,
            "max_samples_per_class": max_samples,
            **info,
        },
        "total": total.to_dict(),
        "stages": [s.to_dict() for s in stages],
        "derived": {
            "extraction_ms_per_image": round(per_image, 3),
            "extraction_images_per_second": round(1000.0 / per_image, 1),
        },
    }


def render_markdown(result: dict) -> str:
    specs = result["specs"]
    cfg = result["setup"]
    total = result["total"]
    by_name = {s["name"]: s for s in result["stages"]}
    derived = result["derived"]

    lines = [
        "# Training benchmark — polygarbor",
        "",
        f"Measured at {specs['timestamp_utc']} (UTC).",
        "",
    ]
    lines += specs_section(specs)
    lines += [
        "## What was measured",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| Dataset | `{cfg['data_dir']}` — {cfg['images']} training images,"
        f" {cfg['classes']} classes |",
        f"| Patch grid | {cfg['patches_per_row']}x{cfg['patches_per_row']}"
        f" ({cfg['patches_per_image']} vector(s) per image) |",
        f"| Vectors extracted | {cfg['vectors_extracted']} of"
        f" {cfg['features_per_vector']}D |",
        f"| Vectors used per subspace | up to {cfg['vectors_per_class_fitted']} |",
        f"| Polynomial levels | {cfg['polynomial_levels']} |",
        f"| Repetitions | {cfg['runs']} full training runs |",
        "",
        "## Total training time",
        "",
    ]
    lines += timings_table([total], header="total")
    lines += ["", "## Stage breakdown", ""]
    lines += timings_table(result["stages"])

    extract = by_name[EXTRACT]
    fit = by_name[FIT]
    load = by_name[LOAD_DATASET]

    lines += [
        "",
        "## Reading the numbers",
        "",
        f"- **Training the whole model takes {format_duration(total['mean_ms'])}**"
        f" on {cfg['images']} images.",
        f"- Feature extraction is {extract['mean_ms'] / total['mean_ms'] * 100:.0f}% of"
        f" that ({format_duration(extract['mean_ms'])}):"
        f" {derived['extraction_ms_per_image']:.1f} ms per image, about"
        f" {derived['extraction_images_per_second']:.0f} images/s. It is the only stage"
        " that grows with the dataset, and it is embarrassingly parallel — nothing here"
        " is parallelized yet.",
        f"- Fitting the {cfg['classes']} subspaces takes only"
        f" {format_duration(fit['mean_ms'])}, because each one sees at most"
        f" {cfg['vectors_per_class_fitted']} vectors of"
        f" {cfg['features_per_vector']} dimensions. Raising `--max-samples` is cheap;"
        " raising the number of images is not.",
        f"- Opening the already-materialized dataset costs"
        f" {format_duration(load['max_ms'])} on the first run and"
        f" {format_duration(load['median_ms'])} afterwards: the expensive part is"
        " importing TensorFlow once per process, not reading the data.",
        "- These numbers cover training only. The `train` command also evaluates the"
        " validation split afterwards, which the inference benchmark covers"
        " separately.",
        "",
        "## How to reproduce",
        "",
        "```bash",
        "cd polygarbor",
        "uv sync --extra dataset",
        f"uv run polygarbor dataset --data-dir {cfg['data_dir']}   # once",
        f"uv run python benchmarks/bench_training.py --data-dir {cfg['data_dir']}",
        "```",
        "",
        "The script writes this markdown plus a JSON with the same numbers to",
        "`benchmarks/results/`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure how long training the model takes, stage by stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data", help="permanent TFDS folder")
    parser.add_argument("--runs", type=int, default=3, help="full training runs to time")
    parser.add_argument("--limit", type=int, default=None, help="use only N training images")
    parser.add_argument("--patches-per-row", type=int, default=1, help="patch grid")
    parser.add_argument("--max-samples", type=int, default=350, help="vectors per subspace")
    parser.add_argument(
        "--out-dir", default="benchmarks/results", help="where to write the report"
    )
    parser.add_argument(
        "--name", default=None, help="base name of the files (default: hostname)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(
            f"error: dataset folder not found: {data_dir}\n"
            "run `uv run polygarbor dataset` first.",
            file=sys.stderr,
        )
        return 1

    print(f"Timing {args.runs} full training run(s)...")
    result = run_benchmark(
        data_dir, args.runs, args.limit, args.patches_per_row, args.max_samples
    )

    md_path, json_path = write_report(
        Path(args.out_dir), default_base_name(args.name), "training",
        render_markdown(result), result,
    )

    print()
    for m in result["stages"]:
        print(f"  {m['name']:<52} {format_duration(m['mean_ms']):>10}")
    print(f"  {'-' * 52} {'-' * 10}")
    print(f"  {result['total']['name']:<52} {format_duration(result['total']['mean_ms']):>10}")
    print(f"\nReport: {md_path}\nJSON:   {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
