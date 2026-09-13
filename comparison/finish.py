"""Run edge benchmarks and validation sequentially after the training campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from suite import ROOT, environment
from localization import PROTOCOL_VERSION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", default="2026-09-12")
    args = p.parse_args()
    root = ROOT / "runs" / args.campaign
    env = environment()
    subprocess.run([sys.executable, str(ROOT / "export_inference.py"), "--campaign", args.campaign], env=env, check=True)
    for method in ("polygarbor", "polygarbor_p3", "polygarbor_p5", "polygarbor_aug", "polygarbor_p3_aug", "resnet18", "resnet18_imagenet", "yolo11n", "yolo11n_random"):
        model = root / f"full__{method}__seed42" / "model"
        if not model.exists():
            continue
        for mode in (("cpu1", "cpu4") if method.startswith("polygarbor") else ("cpu1", "cpu4", "gpu")):
            out = root / "edge" / f"{method}__{mode}"
            status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
            cold_status = json.loads((out / "cold_total.json").read_text()) if (out / "cold_total.json").exists() else {}
            warm_complete = (out / "benchmark.json").exists() and status.get("returncode") == 0
            if warm_complete and cold_status.get("returncode") == 0:
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "edge.py"), "--model-dir", str(model),
                   "--method", method, "--mode", mode, "--out", str(out)]
            print("EDGE", method, mode, flush=True)
            if not warm_complete:
                start = time.perf_counter()
                with (out / "run.log").open("w") as log:
                    result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                (out / "status.json").write_text(json.dumps(dict(returncode=result.returncode,
                    process_wall_seconds=time.perf_counter() - start, command=cmd), indent=2))
                result.check_returncode()
            cold_start = time.perf_counter()
            with (out / "cold.log").open("w") as log:
                cold = subprocess.run(cmd[:-1] + [str(out / "cold"), "--cold-only"], env=env, stdout=log, stderr=subprocess.STDOUT)
            (out / "cold_total.json").write_text(json.dumps(dict(seconds=time.perf_counter() - cold_start, returncode=cold.returncode), indent=2))
            cold.check_returncode()
    localization = root / "localization"
    status = json.loads((localization / "status.json").read_text()) if (localization / "status.json").exists() else {}
    if status.get("status") != "complete" or status.get("protocol_version") != PROTOCOL_VERSION:
        subprocess.run([sys.executable, str(ROOT / "localization.py"), "--campaign", args.campaign], env=env, check=True)
    # Avoid test-module name collisions with the existing sibling test suite.
    with (root / "tests.log").open("w") as log:
        result = subprocess.run([sys.executable, "-m", "pytest", "--import-mode=importlib",
            str(ROOT.parent / "cnn/tests"), str(ROOT / "test_scenarios.py"),
            str(ROOT.parent / "polygarbor/tests"), "-q"], env=env, stdout=log, stderr=subprocess.STDOUT)
    print("TESTS exit", result.returncode, flush=True)
    result.check_returncode()
    subprocess.run([sys.executable, str(ROOT / "audit.py"), "--campaign", args.campaign], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / "validate.py"), "--campaign", args.campaign], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / "report.py"), "--campaign", args.campaign], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / "visual_compare.py"), "--campaign", args.campaign], env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / "refresh_explanations.py"), "--campaign", args.campaign], env=env, check=True)


if __name__ == "__main__":
    main()
