"""Per-image classification metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvaluationResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    class_names: list[str]

    @property
    def accuracy(self) -> float:
        return float(accuracy_score(self.y_true, self.y_pred))

    @property
    def report_dict(self) -> dict:
        return classification_report(
            self.y_true, self.y_pred,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            output_dict=True, zero_division=0,
        )

    def report_frame(self):
        import pandas as pd

        return pd.DataFrame(self.report_dict).transpose().round(3)

    def report_text(self) -> str:
        return classification_report(
            self.y_true, self.y_pred,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            zero_division=0, digits=3,
        )

    def confusion(self, normalize: str | None = "true") -> np.ndarray:
        return confusion_matrix(
            self.y_true, self.y_pred,
            labels=range(len(self.class_names)),
            normalize=normalize,
        )

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        """Write the report (csv/txt) and the confusion matrix (csv)."""
        import pandas as pd

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "report_csv": out_dir / "classification_report.csv",
            "report_txt": out_dir / "classification_report.txt",
            "confusion_csv": out_dir / "confusion_matrix.csv",
        }
        self.report_frame().to_csv(paths["report_csv"])
        paths["report_txt"].write_text(self.report_text())
        pd.DataFrame(
            self.confusion(), index=self.class_names, columns=self.class_names
        ).round(4).to_csv(paths["confusion_csv"])
        return paths


def evaluate(
    classifier,
    samples,
    method: str = "patch",
    aggregation: str = "voting",
    progress=None,
) -> EvaluationResult:
    """Run the classifier over an ``(image, label)`` iterable."""
    y_true, y_pred = classifier.predict_many(
        samples, method=method, aggregation=aggregation, progress=progress
    )
    return EvaluationResult(y_true, y_pred, list(classifier.class_names))
