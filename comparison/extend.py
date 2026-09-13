"""Recover interrupted runs, execute requested ablations, then finalize artifacts."""
from pathlib import Path
import subprocess
import sys
from suite import ROOT, environment

env = environment()
campaign = "2026-09-12"
base = [sys.executable, str(ROOT / "suite.py"), "--campaign", campaign]
variants = ["polygarbor_p3", "polygarbor_p5", "polygarbor_aug", "polygarbor_p3_aug"]
commands = [
    base + ["--methods", "yolo11n", "--scenarios", "label_noise", "source_b", "--seeds", "42"],
    base + ["--methods", *variants],
    base + ["--methods", *variants, "--scenarios", "source_a", "source_b", "--seeds", "42"],
    [sys.executable, str(ROOT / "backfill_metrics.py")],
    [sys.executable, str(ROOT / "finish.py"), "--campaign", campaign],
    [sys.executable, str(ROOT / "audit.py")],
]
for command in commands:
    print("COMMAND", command, flush=True)
    subprocess.run(command, env=env, check=True)
