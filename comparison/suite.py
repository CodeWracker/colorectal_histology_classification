"""Sequential subprocess orchestration; completed/failed runs are never overwritten."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def environment():
    env = os.environ.copy()
    site = next((ROOT.parent / "cnn/.venv/lib").glob("python*/site-packages"))
    env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in (site / "nvidia").glob("*/lib")) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = str(ROOT.parent / "cnn/src") + ":" + str(ROOT.parent / "polygarbor/src")
    env["KERAS_HOME"] = str(ROOT.parent / "cnn/.keras")
    env["YOLO_CONFIG_DIR"] = str(ROOT.parent / "cnn/.config")
    env["MPLBACKEND"] = "Agg"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", default="2026-09-12")
    p.add_argument("--methods", nargs="+", default=["polygarbor", "resnet18", "yolo11n"])
    p.add_argument("--scenarios", nargs="+", default=["full", "few1", "few2", "few5", "few10", "few20", "few50", "few100", "imbalance", "label_noise"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    campaign = ROOT / "runs" / args.campaign
    campaign.mkdir(parents=True, exist_ok=True)
    env = environment()
    for scenario in args.scenarios:
        for seed in args.seeds:
            if scenario in ("imbalance", "label_noise") and seed != args.seeds[0]:
                continue
            for method in args.methods:
                out = campaign / f"{scenario}__{method}__seed{seed}"
                if (out / "status.json").exists():
                    print(f"SKIP existing {out.name}", flush=True)
                    continue
                out.mkdir(exist_ok=True)
                cmd = [sys.executable, str(ROOT / "run.py"), "--method", method, "--scenario", scenario,
                       "--seed", str(seed), "--epochs", str(args.epochs), "--out", str(out)]
                if args.pilot:
                    cmd += ["--pilot"]
                print(f"START {out.name}", flush=True)
                start = time.perf_counter()
                with (out / "run.log").open("w") as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                elapsed = time.perf_counter() - start
                with (campaign / "events.jsonl").open("a") as f:
                    f.write(json.dumps(dict(run=out.name, returncode=result.returncode, wall_seconds=elapsed, command=cmd)) + "\n")
                if result.returncode:
                    status_path = out / "status.json"
                    status = json.loads(status_path.read_text()) if status_path.exists() else {}
                    if status.get("status") in (None, "running"):
                        status_path.write_text(json.dumps(dict(status="failed", returncode=result.returncode,
                            error="Process exited without final status; inspect run.log")))
                print(f"END {out.name}: code={result.returncode}, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
