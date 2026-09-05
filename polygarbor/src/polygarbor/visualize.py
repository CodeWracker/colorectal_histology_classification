"""Figuras do pipeline: banco de filtros, decomposicao de patch, mapas densos e metricas."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def use_headless() -> None:
    """Forca o backend Agg (usado quando o CLI apenas grava PNGs)."""
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
    """Grade de amostras do dataset com o rotulo de cada uma."""
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
    fig.suptitle("Amostras do conjunto de treino", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_gabor_bank(bank):
    """Parte real de cada kernel do banco."""
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
        "Banco de Filtros de Gabor (frequencia x orientacao)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_feature_matrix(features: np.ndarray, feature_names: Sequence[str]):
    """Heatmap da matriz descritora (linhas: patches, colunas: features)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(14, max(3.0, len(features) * 0.35 + 1.5)))
    im = ax.imshow(features, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Valor")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([f"Patch {i}" for i in range(len(features))])
    ax.set_title(
        f"Matriz Descritora Hibrida ({len(features)} patches x {features.shape[1]} features)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_patch_decomposition(view, kernel_labels: Sequence[str], title: str = ""):
    """Patch original -> escala de cinza -> mapas de energia de Gabor."""
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
    flat[1].set_title("Escala de cinza", fontsize=9)
    for i, (emap, meta) in enumerate(zip(view.energy, kernel_labels)):
        ax = flat[i + 2]
        ax.imshow(emap, cmap="inferno")
        ax.set_title(f"G{i + 1}: {meta}", fontsize=8)

    fig.suptitle(title or "Decomposicao do patch em canais de Gabor", fontweight="bold")
    fig.tight_layout()
    return fig


def distances_to_similarity(
    distances: np.ndarray,
    gamma: float = 0.05,
    reference: tuple[float, float] | None = None,
) -> np.ndarray:
    """Converte distancias em similaridade [0, 255] com decaimento exponencial.

    Sem ``reference`` aplica ``exp(-gamma * d)`` diretamente, como no prototipo.
    Com ``reference=(lo, escala)`` a distancia passa por ``log1p`` e e reescalada
    antes da exponencial. Isso importa porque as distancias polinomiais variam por
    varias ordens de grandeza entre as classes (de ~1 a ~1e6): na escala linear as
    classes proximas viram um bloco saturado. A referencia precisa ser a mesma
    para todas as classes, senao os mapas deixam de ser comparaveis entre si.
    """
    d = np.asarray(distances, dtype=np.float64)
    if reference is not None:
        lo, scale = reference
        d = np.log1p(np.maximum(d, 0.0))
        d = (d - lo) / max(scale, 1e-9) * (4.0 / max(abs(gamma), 1e-9))
    sim = np.clip(np.exp(-abs(gamma) * d), 0.0, 1.0)
    return np.rint(sim * 255.0).astype(np.uint8)


def shared_reference(distances: np.ndarray) -> tuple[float, float]:
    """Faixa robusta em escala log (percentis 1 e 99) compartilhada pelos mapas."""
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
    """Mapas densos de similaridade por classe + mapa de decisao por regiao.

    Requer uma predicao feita com ``method='dense'`` (que carrega a grade).
    Com ``normalize=True`` a escala de cor e derivada dos percentis das proprias
    distancias, garantindo contraste util mesmo quando o descritor denso opera
    numa escala espacial diferente da usada no treino.
    """
    if prediction.grid is None:
        raise ValueError(
            "plot_similarity_maps precisa de uma predicao densa "
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
        title += f"\nconfianca: {prediction.confidence:.1%}"
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
        ax.set_title(f"Similaridade:\n{name}", fontsize=10)
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
    ax_last.set_title(f"Vencedor por regiao\n({gh}x{gw})", fontsize=10)
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
    """Imagem + ranking de similaridade e votos por classe."""
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
        f"voto {prediction.confidence:.1%} · similaridade "
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
            f"{prediction.similarity[i]:.3f}  ({int(prediction.votes[i])} votos)",
            va="center", fontsize=8,
        )
    ax_bar.set_xlim(0, values.max() * 1.35)
    ax_bar.set_xlabel("Similaridade normalizada")
    ax_bar.set_title(
        f"Ranking de classes ({prediction.method}/{prediction.aggregation})",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], title: str = ""):
    """Matriz de confusao normalizada por linha (ground truth)."""
    plt = _plt()
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ConfusionMatrixDisplay(cm, display_labels=list(class_names)).plot(
        cmap="Blues", values_format=".2f", ax=ax, colorbar=True
    )
    ax.set_title(
        title or "Matriz de Confusao Normalizada", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real (GT)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig
