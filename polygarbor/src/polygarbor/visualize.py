"""Pipeline figures: filter bank, patch decomposition, dense maps and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def use_headless() -> None:
    """Force the Agg backend (used when the CLI only writes PNGs)."""
    import matplotlib

    matplotlib.use("Agg")


def _plt():
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, path: str | Path, dpi: int = 130) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_dataset_grid(
    samples: Iterable[tuple[np.ndarray, int]],
    class_names: Sequence[str],
    n: int = 9,
):
    """Grid of dataset samples with each one's label."""
    plt = _plt()
    items = [s for _, s in zip(range(n), samples)]
    cols = int(np.ceil(np.sqrt(len(items)))) or 1
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (image, label) in zip(axes.ravel(), items):
        ax.imshow(image)
        ax.set_title(f"{class_names[label]} (ID: {label})", fontsize=9)
    fig.suptitle("Training set samples", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gabor_bank(bank):
    """Real part of every kernel in the bank."""
    plt = _plt()
    n = len(bank)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, ((k_real, _), meta) in enumerate(zip(bank.kernels, bank.labels)):
        ax = axes[i // cols, i % cols]
        ax.imshow(k_real, cmap="gray")
        ax.set_title(f"Kernel {i + 1}\n{meta}", fontsize=9)
    fig.suptitle(
        "Gabor filter bank (frequency x orientation)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_feature_matrix(features: np.ndarray, feature_names: Sequence[str]):
    """Heatmap of the descriptor matrix (rows: patches, columns: features)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(14, max(3.0, len(features) * 0.35 + 1.5)))
    im = ax.imshow(features, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Value")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([f"Patch {i}" for i in range(len(features))])
    ax.set_title(
        f"Hybrid descriptor matrix ({len(features)} patches x {features.shape[1]} features)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_patch_decomposition(view, kernel_labels: Sequence[str], title: str = ""):
    """Original patch -> grayscale -> Gabor energy maps."""
    plt = _plt()
    n_maps = len(view.energy)
    total = n_maps + 2
    cols = min(5, total)
    rows = int(np.ceil(total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.9 * cols, 3.3 * rows), squeeze=False)
    flat = axes.ravel()
    for ax in flat:
        ax.axis("off")

    flat[0].imshow(view.rgb)
    flat[0].set_title(f"Patch {view.index}", fontsize=9)
    flat[1].imshow(view.gray, cmap="gray")
    flat[1].set_title("Grayscale", fontsize=9)
    for i, (emap, meta) in enumerate(zip(view.energy, kernel_labels)):
        ax = flat[i + 2]
        ax.imshow(emap, cmap="inferno")
        ax.set_title(f"G{i + 1}: {meta}", fontsize=8)

    fig.suptitle(title or "Patch decomposition into Gabor channels", fontweight="bold")
    fig.tight_layout()
    return fig


def distances_to_similarity(
    distances: np.ndarray,
    gamma: float = 0.05,
    reference: tuple[float, float] | None = None,
) -> np.ndarray:
    """Turn distances into a [0, 255] similarity with exponential decay.

    Without ``reference`` it applies ``exp(-gamma * d)`` directly, as in the
    prototype. With ``reference=(lo, scale)`` the distance goes through ``log1p``
    and is rescaled first. That matters because polynomial distances span several
    orders of magnitude across classes (from ~1 to ~1e6): on a linear scale the
    nearby classes collapse into one saturated block. The reference must be the
    same for every class, otherwise the maps stop being comparable to each other.
    """
    d = np.asarray(distances, dtype=np.float64)
    if reference is not None:
        lo, scale = reference
        d = np.log1p(np.maximum(d, 0.0))
        d = (d - lo) / max(scale, 1e-9) * (4.0 / max(abs(gamma), 1e-9))
    sim = np.clip(np.exp(-abs(gamma) * d), 0.0, 1.0)
    return np.rint(sim * 255.0).astype(np.uint8)


def shared_reference(distances: np.ndarray) -> tuple[float, float]:
    """Robust log-space range (1st and 99th percentiles) shared by all maps."""
    log_d = np.log1p(np.maximum(np.asarray(distances, dtype=np.float64), 0.0))
    lo, hi = np.percentile(log_d, [1.0, 99.0])
    return float(lo), float(max(hi - lo, 1e-9))


def plot_similarity_maps(
    image: np.ndarray,
    prediction,
    class_names: Sequence[str],
    true_label: int | None = None,
    gamma: float = 0.05,
    alpha: float = 0.55,
    normalize: bool = True,
):
    """Dense per-class similarity maps plus the per-region decision map.

    Requires a prediction made with ``method='dense'`` (which carries the grid).
    With ``normalize=True`` the color scale comes from the percentiles of the
    distances themselves, which keeps the contrast useful even though the dense
    descriptor works at a different spatial scale than the one used in training.
    """
    if prediction.grid is None:
        raise ValueError(
            "plot_similarity_maps needs a dense prediction "
            "(clf.predict(img, method='dense'))."
        )
    plt = _plt()
    gh, gw = prediction.grid
    h, w = image.shape[:2]
    n_classes = len(class_names)
    total = 2 + n_classes

    fig, axes = plt.subplots(1, total, figsize=(3.6 * total, 4.4))
    is_correct = true_label is not None and prediction.label == true_label
    accent = "forestgreen" if is_correct else "crimson"

    axes[0].imshow(image)
    axes[0].axis("off")
    title = f"Pred: {prediction.class_name}"
    if true_label is not None:
        title += f"\nGT: {class_names[true_label]}"
        color = accent
    else:
        title += f"\nconfidence: {prediction.confidence:.1%}"
        color = "black"
    axes[0].set_title(title, fontsize=11, fontweight="bold", color=color)

    reference = shared_reference(prediction.distances) if normalize else None
    for idx, name in enumerate(class_names):
        ax = axes[1 + idx]
        heat = distances_to_similarity(
            prediction.distances[:, idx], gamma, reference
        ).reshape(gh, gw)
        heat_full = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
        ax.imshow(image)
        im = ax.imshow(heat_full, cmap="inferno", alpha=alpha, vmin=0, vmax=255)
        ax.set_title(f"Similarity:\n{name}", fontsize=10)
        ax.axis("off")
        if idx == prediction.label:
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(accent if true_label is not None else "royalblue")
                spine.set_linewidth(3.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_last = axes[-1]
    decision = prediction.unit_labels.reshape(gh, gw).astype(np.uint8)
    decision_full = cv2.resize(decision, (w, h), interpolation=cv2.INTER_NEAREST)
    cmap = plt.get_cmap("tab10", n_classes)
    ax_last.imshow(image)
    im_disc = ax_last.imshow(
        decision_full, cmap=cmap, alpha=0.6, vmin=0, vmax=n_classes - 1
    )
    ax_last.set_title(f"Winner per region\n({gh}x{gw})", fontsize=10)
    ax_last.axis("off")
    cbar = fig.colorbar(
        im_disc, ax=ax_last, ticks=range(n_classes), fraction=0.046, pad=0.04
    )
    cbar.ax.set_yticklabels(class_names, fontsize=7)

    fig.tight_layout()
    return fig


def plot_prediction_summary(
    image: np.ndarray,
    prediction,
    class_names: Sequence[str],
    true_label: int | None = None,
):
    """Image plus the similarity ranking and per-class votes."""
    plt = _plt()
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2]}
    )

    accent = "#4c72b0"
    if true_label is not None:
        accent = "#2e8b57" if prediction.label == true_label else "#c44e52"

    ax_img.imshow(image)
    ax_img.axis("off")
    header = (
        f"{prediction.class_name}\n"
        f"vote {prediction.confidence:.1%} · similarity "
        f"{prediction.similarity[prediction.label]:.3f}"
    )
    if true_label is not None:
        header += f"  |  GT: {class_names[true_label]}"
    ax_img.set_title(header, fontweight="bold", color=accent)

    order = np.argsort(-prediction.mean_distances)
    names = [class_names[i] for i in order]
    values = prediction.similarity[order]
    colors = [accent if i == prediction.label else "#9fb3d1" for i in order]
    bars = ax_bar.barh(names, values, color=colors)
    for bar, i in zip(bars, order):
        ax_bar.text(
            bar.get_width() + values.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prediction.similarity[i]:.3f}  ({int(prediction.votes[i])} votes)",
            va="center", fontsize=8,
        )
    ax_bar.set_xlim(0, values.max() * 1.35)
    ax_bar.set_xlabel("Normalized similarity")
    ax_bar.set_title(
        f"Class ranking ({prediction.method}/{prediction.aggregation})",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str = ""):
    """Row-normalized (ground truth) confusion matrix."""
    plt = _plt()
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ConfusionMatrixDisplay(cm, display_labels=list(class_names)).plot(
        cmap="Blues", values_format=".2f", ax=ax, colorbar=True
    )
    ax.set_title(
        title or "Normalized confusion matrix", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class (GT)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig
