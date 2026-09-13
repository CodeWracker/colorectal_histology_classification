"""Audit completed artifacts without re-running inference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    results = []
    for folder in sorted((ROOT / "runs" / args.campaign).iterdir()):
        status_path = folder / "status.json"
        if not status_path.exists() or not (folder / "config.json").exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            continue
        errors = []
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        selection = json.loads((folder / "selection.json").read_text())
        cache = ROOT / "cache" / config["scenario"] if config["scenario"].startswith("source_") else ROOT / "cache"
        for name, metrics in summary.items():
            data = np.load(folder / "eval" / name / "predictions.npz")
            if name == "train":
                expected = selection["labels"]
            elif name == "train_original_labels":
                expected = selection["original_labels"]
            else:
                expected = np.load(cache / ("val_labels.npy" if name.startswith("val_") else "test_labels.npy"))
            if not np.array_equal(data["y_true"], expected):
                errors.append(name + ": labels/order differ")
            if not np.isfinite(data["probabilities"]).all() or not np.allclose(data["probabilities"].sum(1), 1, atol=1e-5):
                errors.append(name + ": invalid scores")
            if not np.isclose(np.mean(data["y_pred"] == data["y_true"]), metrics["accuracy"]):
                errors.append(name + ": metric mismatch")
        for name in ("config.json", "selection.json", "timings.json", "resources.csv", "resources.json", "latency.json", "model/model.json"):
            if not (folder / name).is_file():
                errors.append("missing " + name)
        model_bytes = sum(p.stat().st_size for p in (folder / "model").rglob("*") if p.is_file())
        if model_bytes != json.loads((folder / "model_size.json").read_text())["bytes"]:
            errors.append("model size mismatch")
        results.append(dict(run=folder.name, evaluations=len(summary), errors=errors))
    payload = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checked=len(results),
                   valid=sum(not r["errors"] for r in results), runs=results)
    (ROOT / "integrity_audit.json").write_text(json.dumps(payload, indent=2))
    print({k: v for k, v in payload.items() if k != "runs"})
    if any(r["errors"] for r in results):
        raise RuntimeError("Artifact integrity failures; inspect integrity_audit.json")


if __name__ == "__main__":
    main()
