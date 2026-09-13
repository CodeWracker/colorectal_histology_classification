"""Metrics shared by the CLI and paired experiments."""
from pathlib import Path
import json

import numpy as np
from sklearn import metrics
from polygarbor.evaluation import EvaluationResult


def multiclass_gmean(y_true, y_pred, n_classes):
    """Geometric mean of class recalls; undefined when a class is absent."""
    cm = metrics.confusion_matrix(y_true, y_pred, labels=range(n_classes))
    support = cm.sum(axis=1)
    if np.any(support == 0):
        return None
    recalls = np.diag(cm) / support
    return 0.0 if np.any(recalls == 0) else float(np.exp(np.log(recalls).mean()))


def summarize(y_true, probabilities, class_names, y_pred=None):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), len(class_names)) or not len(y):
        raise ValueError("Nonempty matching labels and score matrix required")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Scores must be finite nonnegative normalized rows")
    p = p / p.sum(axis=1, keepdims=True)
    pred = p.argmax(1) if y_pred is None else np.asarray(y_pred)
    result = EvaluationResult(y, pred, list(class_names))
    confidence = p[np.arange(len(y)), pred]
    correct = pred == y
    bins = []
    for i in range(10):
        mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else (confidence <= 1))
        bins.append(dict(lower=i / 10, count=int(mask.sum()),
                         confidence=float(confidence[mask].mean()) if mask.any() else None,
                         accuracy=float(correct[mask].mean()) if mask.any() else None))
    ece = sum(b["count"] / len(y) * abs(b["confidence"] - b["accuracy"])
              for b in bins if b["count"])
    aucs = [metrics.roc_auc_score(y == i, p[:, i]) for i in range(len(class_names))
            if 0 < (y == i).sum() < len(y)]
    summary = dict(n=len(y), accuracy=result.accuracy,
                   multiclass_gmean=multiclass_gmean(y, pred, len(class_names)),
                   score_argmax_agreement=float(np.mean(pred == p.argmax(1))),
                   macro_f1=float(metrics.f1_score(y, pred, labels=range(len(class_names)),
                                                   average="macro", zero_division=0)),
                   balanced_accuracy=float(metrics.balanced_accuracy_score(y, pred)),
                   mcc=float(metrics.matthews_corrcoef(y, pred)),
                   kappa=float(metrics.cohen_kappa_score(y, pred)),
                   top2_accuracy=float(np.mean([t in row for t, row in zip(y, np.argsort(p, axis=1)[:, -2:])])),
                   nll=float(metrics.log_loss(y, p, labels=range(len(class_names)))),
                   brier=float(np.mean(np.sum((p - np.eye(len(class_names))[y]) ** 2, axis=1))),
                   ece=float(ece), auroc_macro=float(np.mean(aucs)) if aucs else None,
                   calibration_bins=bins, per_class=result.report_dict)
    return summary, result


def save_evaluation(y, probabilities, class_names, out_dir, y_pred=None):
    from polygarbor import visualize
    import matplotlib.pyplot as plt
    summary, result = summarize(y, probabilities, class_names, y_pred)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.save(out)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / "predictions.npz", y_true=y, y_pred=result.y_pred,
                        probabilities=probabilities)
    visualize.use_headless()
    fig = visualize.plot_confusion_matrix(result.confusion(), class_names)
    visualize.save_figure(fig, out / "confusion_matrix.png")
    plt.close(fig)
    return summary


def evaluate(classifier, samples, **kwargs):
    samples = list(samples)
    return EvaluationResult(np.array([int(y) for _, y in samples]),
                            classifier.predict_proba([x for x, _ in samples]).argmax(1),
                            classifier.class_names)
