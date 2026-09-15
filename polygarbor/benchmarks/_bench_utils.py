"""Shared helpers for the benchmark scripts: timing, machine specs, reporting."""

from __future__ import annotations

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
            "per_second": round(self.per_second, 3),
        }


def measure(name: str, fn, runs: int, warmup: int = 0) -> Timing:
    """Run ``fn`` ``warmup + runs`` times and keep the timings of the last ones."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return Timing(name, samples)


class Stopwatch:
    """Collects several named stages across repeated runs of a larger workflow."""

    def __init__(self) -> None:
        self._stages: dict[str, list[float]] = {}

    def record(self, name: str, elapsed_seconds: float) -> None:
        self._stages.setdefault(name, []).append(elapsed_seconds * 1000.0)

    def time(self, name: str, fn):
        """Time ``fn``, record it under ``name`` and return its result."""
        start = time.perf_counter()
        result = fn()
        self.record(name, time.perf_counter() - start)
        return result

    def timings(self) -> list[Timing]:
        return [Timing(name, samples) for name, samples in self._stages.items()]


# -------------------------------------------------------------------- hardware


def run_command(cmd: list[str]) -> str:
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
    out = run_command(["lscpu"])
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
    out = run_command(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def _package_versions(extra: tuple[str, ...] = ()) -> dict[str, str]:
    import importlib.metadata as md

    names = ("numpy", "opencv-python-headless", "polymahalanobis", "scikit-learn") + extra
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def collect_specs(extra_packages: tuple[str, ...] = ()) -> dict:
    """Gather what describes the machine the measurement ran on."""
    import os

    import cv2

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()}",
        "os_detail": platform.version(),
        "architecture": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_topology": _cpu_topology(),
        "logical_cpus": os.cpu_count(),
        "total_memory_gb": _memory_gb(),
        "gpus": _gpus(),
        "python": sys.version.split()[0],
        "packages": _package_versions(extra_packages),
        "opencv_threads": cv2.getNumThreads(),
        "thread_env": {
            var: os.environ[var]
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if var in os.environ
        },
    }


# ------------------------------------------------------------------- reporting


def format_duration(ms: float) -> str:
    """Human-readable duration: ms below a second, seconds below a minute."""
    if ms < 1000:
        return f"{ms:.1f} ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f} s"
    minutes, seconds = divmod(ms / 1000, 60)
    return f"{int(minutes)} min {seconds:.1f} s"


def specs_section(specs: dict, gpu_note: bool = True) -> list[str]:
    topo = specs["cpu_topology"]
    lines = [
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
    ]
    if gpu_note:
        lines += [
            "> The GPU is listed only to describe the machine: the pipeline is",
            "> CPU-only (OpenCV + NumPy) and does not use GPU acceleration.",
            "",
        ]
    lines += ["### Versions", "", "| package | version |", "| --- | --- |"]
    lines += [f"| {name} | {ver} |" for name, ver in specs["packages"].items()]
    lines.append("")
    return lines


def timings_table(measurements: list[dict], header: str = "step") -> list[str]:
    lines = [
        f"| {header} | mean | median | stdev | best | worst | runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in measurements:
        lines.append(
            f"| {m['name']} | {format_duration(m['mean_ms'])} |"
            f" {format_duration(m['median_ms'])} | {format_duration(m['stdev_ms'])} |"
            f" {format_duration(m['min_ms'])} | {format_duration(m['max_ms'])} |"
            f" {m['runs']} |"
        )
    return lines


def write_report(out_dir: Path, base: str, prefix: str, markdown: str, result: dict):
    """Write the markdown report and its JSON twin; returns both paths."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{prefix}-{base}.md"
    json_path = out_dir / f"{prefix}-{base}.json"
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return md_path, json_path


def default_base_name(explicit: str | None) -> str:
    return explicit or "benchmark"
