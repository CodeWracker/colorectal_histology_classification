"""CPU-only training of the reference CNNs, to compare training cost on the same hardware.

Added on 2026-09-14 after the paper review, with explicit authorization from
the authors. Not part of the registered 2026-09-12 campaign. The runs use the
same run.py, splits, seed and thread count as the campaign, but with
--device cpu, so the CNN training time can be compared with the CPU-only
PolyGabor training time. Runs go to runs/2026-09-14_cpu and existing runs are
never overwritten.

Order: the scarce-label budget of 10 images per class first (cheap), then the
full training set, ImageNet-initialized models before scratch models.

Run from colorectal_histology_classification/ with the cnn venv:
    cnn/.venv/bin/python comparison/cpu_training.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from suite import environment  # noqa: E402

CAMPAIGN = ROOT / "runs/2026-09-14_cpu"
QUEUE = [(scenario, method) for scenario in ("few10", "full")
         for method in ("resnet18_imagenet", "yolo11n", "resnet18", "yolo11n_random")]
SEED, THREADS = 42, 4


def main():
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    env = environment()
    for scenario, method in QUEUE:
        out = CAMPAIGN / f"{scenario}__{method}__seed{SEED}"
        if (out / "status.json").exists():
            print(f"SKIP existing {out.name}", flush=True)
            continue
        out.mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT / "run.py"), "--method", method, "--scenario", scenario,
               "--seed", str(SEED), "--device", "cpu", "--threads", str(THREADS), "--out", str(out)]
        print(f"START {out.name} {time.strftime('%H:%M:%S')}", flush=True)
        start = time.perf_counter()
        with (out / "run.log").open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        elapsed = time.perf_counter() - start
        with (CAMPAIGN / "events.jsonl").open("a") as f:
            f.write(json.dumps(dict(run=out.name, returncode=result.returncode, wall_seconds=elapsed, command=cmd)) + "\n")
        print(f"END {out.name}: code={result.returncode}, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
