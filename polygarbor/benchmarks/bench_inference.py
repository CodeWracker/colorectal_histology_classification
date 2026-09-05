"""Measure single-image inference time and record the machine specs.

Usage:
    uv run python benchmarks/bench_inference.py \
        --model-dir artifacts/model \
        --image artifacts/dataset/samples/00_tumor_1.png

Reports two kinds of total. The **cold** total spawns a fresh interpreter and
measures everything a one-shot command pays for: Python startup, imports, model
load and the prediction itself. The **warm** total is what a long-lived service
pays per image once the model is already in memory. The stage breakdown below
them exists so a future regression can be traced to a stage.

Writes a markdown report and a JSON with the same numbers to
``benchmarks/results/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

from _bench_utils import (
    Timing,
    collect_specs,
    default_base_name,
    format_duration,
    measure,
    specs_section,
    timings_table,
    write_report,
)

COLD_TOTAL = "Cold total: new process, import, load model, classify"
CLI_TOTAL = "Cold total: `polygarbor predict --no-viz` command"
CLI_FIGURES = "Cold total: `polygarbor predict` with all figures"
WARM_TOTAL = "Warm total: read file and classify (model in memory)"
WARM_ARRAY = "Warm total: classify an array already in memory"
LOAD_MODEL = "Load model from disk (rebuilds the subspaces)"
DENSE_TOTAL = "Warm total, dense mode (similarity maps)"


def _cold_run(model_dir: Path, image_path: Path) -> Timing:
    """Time a fresh interpreter doing import + load + predict, from the outside."""
    script = textwrap.dedent(
        f"""
        from polygarbor import PolyGaborClassifier
        clf = PolyGaborClassifier.load({str(model_dir)!r})
        clf.predict({str(image_path)!r})
        """
    )
    return measure(
        COLD_TOTAL,
        lambda: subprocess.run(
            [sys.executable, "-c", script], check=True, capture_output=True
        ),
        runs=5,
        warmup=1,
    )


def _cli_run(name: str, args: list[str], runs: int) -> Timing:
    """Time the real CLI command, argparse and console output included."""
    return measure(
        name,
        lambda: subprocess.run(
            [sys.executable, "-m", "polygarbor.cli", *args],
            check=True, capture_output=True,
        ),
        runs=runs,
        warmup=1,
    )


def run_benchmark(
    model_dir: Path, image_path: Path, runs: int, warmup: int, skip_cli: bool
) -> dict:
    from polygarbor import PolyGaborClassifier, load_image

    load_timing = measure(
        LOAD_MODEL,
        lambda: PolyGaborClassifier.load(model_dir),
        runs=max(3, runs // 10),
        warmup=1,
    )

    clf = PolyGaborClassifier.load(model_dir)
    image = load_image(image_path)
    feats = clf.extract(image)
    dense_feats, _ = clf.extract_dense(image)

    totals = [_cold_run(model_dir, image_path)]
    if not skip_cli:
        import tempfile

        # The figures are written to a throwaway folder: savefig is a real part
        # of the cost, so skipping the write would understate the command.
        with tempfile.TemporaryDirectory() as tmp:
            common = [
                "predict", "--model-dir", str(model_dir), "-i", str(image_path),
                "--no-color", "--out-dir", tmp,
            ]
            totals.append(_cli_run(CLI_TOTAL, [*common, "--no-viz"], 3))
            totals.append(_cli_run(CLI_FIGURES, common, 3))
    totals += [
        measure(WARM_TOTAL, lambda: clf.predict(image_path), runs, warmup),
        measure(WARM_ARRAY, lambda: clf.predict(image), runs, warmup),
        measure(DENSE_TOTAL, lambda: clf.predict(image, method="dense"), runs, warmup),
    ]

    stages = [
        load_timing,
        measure("Read and decode the image", lambda: load_image(image_path), runs, warmup),
        measure("Extract descriptor (patch)", lambda: clf.extract(image), runs, warmup),
        measure(
            "Mahalanobis distances (patch)", lambda: clf.distances(feats), runs, warmup
        ),
        measure(
            "Extract descriptor (dense 75x75)",
            lambda: clf.extract_dense(image), runs, warmup,
        ),
        measure(
            "Mahalanobis distances (dense)",
            lambda: clf.distances(dense_feats), runs, warmup,
        ),
    ]

    pred = clf.predict(image)
    return {
        "benchmark": "inference",
        "specs": collect_specs(),
        "setup": {
            "image": str(image_path),
            "dimensions": f"{image.shape[1]}x{image.shape[0]}",
            "model": str(model_dir),
            "classes": clf.n_classes,
            "features_per_vector": clf.n_features,
            "patches_per_image": int(len(feats)),
            "dense_vectors_per_image": int(len(dense_feats)),
            "polynomial_levels": clf.num_levels,
            "runs": runs,
            "warmup": warmup,
            "prediction": pred.class_name,
        },
        "totals": [t.to_dict() for t in totals],
        "stages": [t.to_dict() for t in stages],
    }


def render_markdown(result: dict) -> str:
    specs = result["specs"]
    cfg = result["setup"]
    by_name = {m["name"]: m for m in result["totals"] + result["stages"]}

    lines = [
        "# Inference benchmark — polygarbor",
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
        f"| Image | `{cfg['image']}` ({cfg['dimensions']} px) |",
        f"| Model | `{cfg['model']}` — {cfg['classes']} classes,"
        f" {cfg['polynomial_levels']} polynomial levels |",
        f"| Descriptor | {cfg['features_per_vector']}D |",
        f"| Vectors per image | {cfg['patches_per_image']} (patch) /"
        f" {cfg['dense_vectors_per_image']} (dense) |",
        f"| Repetitions | {cfg['runs']} measured after {cfg['warmup']} warm-up runs |",
        f"| Predicted class | {cfg['prediction']} |",
        "",
        "## Total time per image",
        "",
        "Cold totals include Python startup and loading the model; warm totals are"
        " what you pay per image once the model is in memory.",
        "",
    ]
    lines += timings_table(result["totals"], header="total")
    lines += [
        "",
        "## Stage breakdown",
        "",
    ]
    lines += timings_table(result["stages"])

    warm = by_name[WARM_TOTAL]
    warm_array = by_name[WARM_ARRAY]
    cold = by_name[COLD_TOTAL]
    dense = by_name[DENSE_TOTAL]
    model_load = by_name[LOAD_MODEL]

    lines += [
        "",
        "## Reading the numbers",
        "",
        f"- **Warm: {format_duration(warm['mean_ms'])} per image**"
        f" (~{warm['per_second']:.0f} images/s) reading the file from disk,"
        f" {format_duration(warm_array['mean_ms'])} if the array is already in memory."
        " This is the number that matters for a service.",
        f"- **Cold: {format_duration(cold['mean_ms'])} end to end** for a fresh process."
        f" Almost all of it is fixed overhead — {format_duration(model_load['mean_ms'])}"
        " to rebuild the subspaces plus interpreter startup and imports — so classifying"
        " one image per process wastes"
        f" {(1 - warm['mean_ms'] / cold['mean_ms']) * 100:.0f}% of the time on setup.",
        f"- `dense` mode costs {format_duration(dense['mean_ms'])},"
        f" ~{dense['mean_ms'] / warm_array['mean_ms']:.0f}× the patch mode: it evaluates"
        f" {cfg['dense_vectors_per_image']} vectors instead of"
        f" {cfg['patches_per_image']}. That is the price of the similarity maps, not of"
        " the decision.",
    ]
    if CLI_FIGURES in by_name:
        lines.append(
            f"- The full `predict` command with every figure takes"
            f" {format_duration(by_name[CLI_FIGURES]['mean_ms'])}; rendering the five"
            " matplotlib figures dominates it, not the classification."
        )
    lines += [
        "",
        "## How to reproduce",
        "",
        "```bash",
        "cd polygarbor",
        "uv run python benchmarks/bench_inference.py \\",
        f"  --model-dir {cfg['model']} \\",
        f"  --image {cfg['image']}",
        "```",
        "",
        "The script writes this markdown plus a JSON with the same numbers to",
        "`benchmarks/results/`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure single-image inference time and record the machine specs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default="artifacts/model", help="saved model")
    parser.add_argument(
        "--image", default="artifacts/dataset/samples/00_tumor_1.png",
        help="image used for the measurement",
    )
    parser.add_argument("--runs", type=int, default=200, help="timed repetitions")
    parser.add_argument("--warmup", type=int, default=20, help="warm-up repetitions")
    parser.add_argument(
        "--skip-cli", action="store_true",
        help="skip the end-to-end CLI totals (they spawn subprocesses)",
    )
    parser.add_argument(
        "--out-dir", default="benchmarks/results", help="where to write the report"
    )
    parser.add_argument(
        "--name", default=None, help="base name of the files (default: hostname)"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    image_path = Path(args.image)
    for path, what in ((model_dir, "model"), (image_path, "image")):
        if not path.exists():
            print(f"error: {what} not found: {path}", file=sys.stderr)
            return 1

    print(f"Measuring {args.runs} repetitions (after {args.warmup} warm-up runs)...")
    result = run_benchmark(
        model_dir, image_path, args.runs, args.warmup, args.skip_cli
    )

    md_path, json_path = write_report(
        Path(args.out_dir), default_base_name(args.name), "inference",
        render_markdown(result), result,
    )

    for section in ("totals", "stages"):
        print(f"\n{section}:")
        for m in result[section]:
            print(f"  {m['name']:<56} {format_duration(m['mean_ms']):>10}")
    print(f"\nReport: {md_path}\nJSON:   {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
