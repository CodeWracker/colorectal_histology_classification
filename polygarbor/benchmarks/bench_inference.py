"""Measure single-image inference time and record the machine specs.

Usage:
    uv run python benchmarks/bench_inference.py \
        --model-dir artifacts/model \
        --image artifacts/dataset/samples/00_tumor_1.png

Writes two files under ``benchmarks/results/``: a readable markdown report and a
JSON with the same numbers, for comparing across machines.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------- timing


@dataclass
class Timing:
    """Statistics of one timed step, in milliseconds."""

    name: str
    samples: list[float]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def best(self) -> float:
        return min(self.samples)

    @property
    def worst(self) -> float:
        return max(self.samples)

    @property
    def per_second(self) -> float:
        return 1000.0 / self.mean if self.mean else float("inf")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "runs": len(self.samples),
            "mean_ms": round(self.mean, 3),
            "median_ms": round(self.median, 3),
            "stdev_ms": round(self.stdev, 3),
            "min_ms": round(self.best, 3),
            "max_ms": round(self.worst, 3),
            "per_second": round(self.per_second, 1),
        }


def measure(name: str, fn, runs: int, warmup: int) -> Timing:
    """Run ``fn`` ``warmup + runs`` times and keep the timings of the last ones."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return Timing(name, samples)


# -------------------------------------------------------------------- hardware


def _run(cmd: list[str]) -> str:
    """Run a helper command, forcing the C locale so the labels stay parseable."""
    import os

    if not shutil.which(cmd[0]):
        return ""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _cpu_topology() -> dict:
    out = _run(["lscpu"])
    fields = {}
    for key, label in (
        ("Core(s) per socket", "cores_per_socket"),
        ("Socket(s)", "sockets"),
        ("Thread(s) per core", "threads_per_core"),
        ("CPU max MHz", "max_mhz"),
        ("Architecture", "arch"),
    ):
        match = re.search(rf"^{re.escape(key)}:\s*(.+)$", out, re.MULTILINE)
        if match:
            fields[label] = match.group(1).strip()
    return fields


def _memory_gb() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    match = re.search(r"^MemTotal:\s+(\d+) kB", meminfo.read_text(), re.MULTILINE)
    return round(int(match.group(1)) / 1024 / 1024, 1) if match else None


def _gpus() -> list[str]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    versions = {}
    for name in ("numpy", "opencv-python-headless", "polymahalanobis", "scikit-learn"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def collect_specs() -> dict:
    """Gather what describes the machine the measurement ran on."""
    import os

    import cv2

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "os_detail": platform.version(),
        "architecture": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_topology": _cpu_topology(),
        "logical_cpus": os.cpu_count(),
        "total_memory_gb": _memory_gb(),
        "gpus": _gpus(),
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "opencv_threads": cv2.getNumThreads(),
        "thread_env": {
            var: os.environ[var]
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if var in os.environ
        },
    }


# ------------------------------------------------------------------ benchmark

LOAD_MODEL = "Load model from disk (once per process)"
PATCH_MEMORY = "Patch inference, in-memory array"
PATCH_FILE = "Patch inference, from file"
DENSE_MEMORY = "Dense inference, in-memory array"


def run_benchmark(model_dir: Path, image_path: Path, runs: int, warmup: int) -> dict:
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

    timings = [
        load_timing,
        measure("Read and decode the image", lambda: load_image(image_path), runs, warmup),
        measure("Extract descriptor (patch)", lambda: clf.extract(image), runs, warmup),
        measure("Mahalanobis distances (patch)", lambda: clf.distances(feats), runs, warmup),
        measure(PATCH_MEMORY, lambda: clf.predict(image), runs, warmup),
        measure(PATCH_FILE, lambda: clf.predict(image_path), runs, warmup),
        measure(
            "Extract descriptor (dense 75x75)",
            lambda: clf.extract_dense(image), runs, warmup,
        ),
        measure(
            "Mahalanobis distances (dense)",
            lambda: clf.distances(dense_feats), runs, warmup,
        ),
        measure(DENSE_MEMORY, lambda: clf.predict(image, method="dense"), runs, warmup),
    ]

    pred = clf.predict(image)
    return {
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
        "measurements": [t.to_dict() for t in timings],
    }


# ------------------------------------------------------------------- reporting


def render_markdown(result: dict) -> str:
    specs = result["specs"]
    cfg = result["setup"]
    topo = specs["cpu_topology"]

    lines = [
        "# Inference benchmark — polygarbor",
        "",
        f"Measured at {specs['timestamp_utc']} (UTC).",
        "",
        "## Machine",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| CPU | {specs['cpu']} |",
        f"| Cores / threads | {topo.get('cores_per_socket', '?')} cores"
        f" × {topo.get('threads_per_core', '?')} thread(s) ="
        f" {specs['logical_cpus']} logical CPUs |",
        f"| Max frequency | {topo.get('max_mhz', 'n/a')} MHz |",
        f"| Memory | {specs['total_memory_gb']} GB |",
        f"| GPU | {', '.join(specs['gpus']) or 'none detected'} |",
        f"| System | {specs['os']} ({specs['architecture']}) |",
        f"| Python | {specs['python']} |",
        f"| OpenCV threads | {specs['opencv_threads']} |",
        "",
        "> The GPU is listed only to describe the machine: the pipeline is",
        "> CPU-only (OpenCV + NumPy) and does not use GPU acceleration.",
        "",
        "### Versions",
        "",
        "| package | version |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {ver} |" for name, ver in specs["packages"].items()]

    lines += [
        "",
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
        "## Results",
        "",
        "| step | mean (ms) | median (ms) | stdev (ms) | best (ms) |"
        " worst (ms) | per second |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in result["measurements"]:
        lines.append(
            f"| {m['name']} | {m['mean_ms']:.2f} | {m['median_ms']:.2f} |"
            f" {m['stdev_ms']:.2f} | {m['min_ms']:.2f} | {m['max_ms']:.2f} |"
            f" {m['per_second']:.1f} |"
        )

    by_name = {m["name"]: m for m in result["measurements"]}
    patch = by_name[PATCH_MEMORY]
    from_file = by_name[PATCH_FILE]
    dense = by_name[DENSE_MEMORY]
    model_load = by_name[LOAD_MODEL]

    lines += [
        "",
        "## Reading the numbers",
        "",
        f"- **Classifying one image costs {patch['mean_ms']:.1f} ms**"
        f" (~{patch['per_second']:.0f} images/s) in `patch` mode, which is the"
        f" default. Including reading the file from disk, {from_file['mean_ms']:.1f} ms.",
        f"- `dense` mode costs {dense['mean_ms']:.1f} ms,"
        f" ~{dense['mean_ms'] / patch['mean_ms']:.0f}× more: it evaluates"
        f" {cfg['dense_vectors_per_image']} vectors instead of"
        f" {cfg['patches_per_image']}. That is the price of the similarity maps.",
        f"- Loading the model takes {model_load['mean_ms']:.0f} ms, but it is a"
        " one-off cost per process: the subspaces are rebuilt from the samples"
        " inside `load()`. A service classifying in batch should load once and"
        " reuse the instance.",
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
    result = run_benchmark(model_dir, image_path, args.runs, args.warmup)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or platform.node() or "benchmark"
    md_path = out_dir / f"inference-{base}.md"
    json_path = out_dir / f"inference-{base}.json"

    md_path.write_text(render_markdown(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    for m in result["measurements"]:
        print(f"  {m['name']:<42} {m['mean_ms']:8.2f} ms  ({m['per_second']:.1f}/s)")
    print(f"\nReport: {md_path}\nJSON:   {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
