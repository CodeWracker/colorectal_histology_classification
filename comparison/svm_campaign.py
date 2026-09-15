"""Campaign of the RBF-SVM on the descriptor (PROTOCOL.md, "RBF-SVM on the descriptor in the registered scenarios").

Runs, in order and in fresh processes through suite.py/run.py:

1. full and few1-few100 on seeds 42 and 43;
2. the registered consistency check: the test predictions of those runs must
   equal svm_default of runs/2026-09-15_descriptor, otherwise the campaign stops;
3. label_noise and imbalance on seed 42;
4. loso_01-loso_10 on seeds 42 and 43;
5. the cpu1/cpu4 inference benchmark of the full seed-42 model, warm and cold,
   as in finish.py.

Existing runs are never overwritten. Nothing else should run on the machine
meanwhile, because the runs record time and memory.

Run from colorectal_histology_classification/ with the cnn venv:
    cnn/.venv/bin/python comparison/svm_campaign.py
"""
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from suite import environment  # noqa: E402

CAMPAIGN = "2026-09-15_svm"
RUNS = ROOT / "runs" / CAMPAIGN
DESCRIPTOR = ROOT / "runs/2026-09-15_descriptor"
METHOD = "gabor_svm"
SCARCE = ["full"] + [f"few{n}" for n in (1, 2, 5, 10, 20, 50, 100)]
LOSO = [f"loso_{i:02d}" for i in range(1, 11)]


def suite(scenarios, seeds, env):
    subprocess.run([sys.executable, str(ROOT / "suite.py"), "--campaign", CAMPAIGN, "--methods", METHOD,
                    "--scenarios", *scenarios, "--seeds", *map(str, seeds)], env=env, check=True)


def check_consistency():
    rows = []
    for scenario in SCARCE:
        for seed in (42, 43):
            run = RUNS / f"{scenario}__{METHOD}__seed{seed}"
            status = json.loads((run / "status.json").read_text())["status"]
            if status != "complete":
                raise SystemExit(f"{run.name}: status {status}")
            saved = np.load(run / "eval/test_clean/predictions.npz")["y_pred"]
            reference = np.load(DESCRIPTOR / f"{scenario}__polygarbor__seed{seed}/predictions.npz")["svm_default_test"]
            rows.append(dict(run=run.name, agreement=float(np.mean(saved == reference))))
    (RUNS / "consistency.json").write_text(json.dumps(rows, indent=2))
    failed = [row for row in rows if row["agreement"] < 1]
    if failed:
        raise SystemExit(f"Consistency check failed, campaign stopped: {failed}")
    print(f"CONSISTENCY ok: {len(rows)} runs", flush=True)


def edge(env):
    model = RUNS / f"full__{METHOD}__seed42/model"
    for mode in ("cpu1", "cpu4"):
        out = RUNS / "edge" / f"{METHOD}__{mode}"
        if (out / "cold_total.json").exists():
            print(f"SKIP existing {out.name}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "edge.py"), "--model-dir", str(model), "--method", METHOD,
               "--mode", mode, "--out", str(out)]
        print("EDGE", mode, flush=True)
        start = time.perf_counter()
        with (out / "run.log").open("w") as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        (out / "status.json").write_text(json.dumps(dict(returncode=result.returncode,
            process_wall_seconds=time.perf_counter() - start, command=cmd), indent=2))
        result.check_returncode()
        start = time.perf_counter()
        with (out / "cold.log").open("w") as log:
            cold = subprocess.run(cmd[:-1] + [str(out / "cold"), "--cold-only"], env=env, stdout=log, stderr=subprocess.STDOUT)
        (out / "cold_total.json").write_text(json.dumps(dict(seconds=time.perf_counter() - start, returncode=cold.returncode), indent=2))
        cold.check_returncode()


def main():
    env = environment()
    start = time.perf_counter()
    suite(SCARCE, (42, 43), env)
    check_consistency()
    suite(["label_noise", "imbalance"], (42,), env)
    suite(LOSO, (42, 43), env)
    edge(env)
    print(f"DONE in {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
