"""Add G-mean to existing results without any model inference or retraining."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))
import numpy as np
from cnn.evaluation import multiclass_gmean


def main():
    count = 0
    for folder in (ROOT / "runs").glob("*/*"):
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        for name, metrics in summary.items():
            path = folder / "eval" / name / "predictions.npz"
            data = np.load(path)
            metrics["multiclass_gmean"] = multiclass_gmean(data["y_true"], data["y_pred"], data["probabilities"].shape[1])
            metrics["score_argmax_agreement"] = float(np.mean(data["y_pred"] == data["probabilities"].argmax(1)))
            (path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))
            count += 1
        summary_path.write_text(json.dumps(summary, indent=2))
    print("Updated", count, "evaluations from saved predictions")


if __name__ == "__main__":
    main()
