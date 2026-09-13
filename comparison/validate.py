"""Consolidate the final integrity, test, CLI, and edge checks."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    runs = ROOT / "runs" / args.campaign
    results = ROOT / "results" / args.campaign
    tests_log = (runs / "tests.log").read_text()
    match = re.findall(r"(?m)^\d+ passed(?:, \d+ warnings?)? in [0-9.]+s$", tests_log)
    edge = []
    for path in sorted((runs / "edge").glob("*/benchmark.json")):
        benchmark = read(path)
        method = benchmark["method"]
        original = np.load(runs / f"full__{method}__seed42/eval/test_clean/predictions.npz")["y_pred"][:len(benchmark["predictions"])]
        edge.append(dict(method=method, mode=benchmark["mode"], samples=len(original), prediction_agreement=float(np.mean(original == benchmark["predictions"])), returncode=read(path.parent / "status.json", {}).get("returncode"), cold_returncode=read(path.parent / "cold_total.json", {}).get("returncode")))
    available_methods = [folder.name.removeprefix("full__").removesuffix("__seed42") for folder in runs.glob("full__*__seed42") if (folder / "model").exists()]
    expected_edge = sorted((method, mode) for method in available_methods for mode in (("cpu1", "cpu4") if method.startswith("polygarbor") else ("cpu1", "cpu4", "gpu")))
    observed_edge = {(row["method"], row["mode"]) for row in edge}
    missing_edge = [dict(method=method, mode=mode) for method, mode in expected_edge if (method, mode) not in observed_edge]
    cli = read(results / "cli_smoke/status.json", [])
    cli_failed = [row for row in cli if row.get("returncode") != 0]
    payload = dict(integrity_audit=read(ROOT / "integrity_audit.json"), tests_log=str((runs / "tests.log").relative_to(ROOT.parent)), tests_summary=match[-1] if match else "not found", cli_commands_expected=len(cli), cli_commands_passed=len(cli) - len(cli_failed), cli_commands_failed=cli_failed, edge_expected=len(expected_edge), edge_missing=missing_edge, edge=edge)
    (results / "validation_summary.json").write_text(json.dumps(payload, indent=2))
    if payload["integrity_audit"]["campaign"] != args.campaign or payload["integrity_audit"]["checked"] != payload["integrity_audit"]["valid"] or not match or cli_failed or missing_edge or any(row["returncode"] != 0 or row["cold_returncode"] != 0 or row["prediction_agreement"] < .99 for row in edge):
        raise RuntimeError("Final validation failed; inspect validation_summary.json")
    print(dict(runs_valid=payload["integrity_audit"]["valid"], tests=payload["tests_summary"], cli=payload["cli_commands_passed"], edge=len(edge)))


if __name__ == "__main__":
    main()
