"""Learning curves, prediction ranking and class activation maps."""
from pathlib import Path
import numpy as np
from polygarbor.visualize import use_headless, save_figure, plot_confusion_matrix, plot_dataset_grid


def plot_history(history):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ("accuracy", "loss")):
        for key, label in ((metric, "treino"), ("val_" + metric, "validação")):
            if key in history:
                ax.plot(np.arange(1, len(history[key]) + 1), history[key], label=label)
        ax.set(xlabel="Época", ylabel=metric)
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    return fig


def activation_map(classifier, image, label=None):
    """Exact CAM for GAP + linear head, equivalent to Grad-CAM up to scale.

    Uses pre-softmax class weights; keeps the trained model unchanged.
    The native 4x4 map at 128px input is saved separately from interpolation.
    """
    if classifier.architecture != "resnet18":
        raise ValueError("CAM is available for the ResNet GAP head; use occlusion for YOLO")
    import tensorflow as tf
    from polygarbor.classifier import load_image
    rgb = load_image(image)
    x = tf.image.resize(rgb, (classifier.image_size, classifier.image_size))[None] / 255.
    if classifier.weights == "imagenet":
        x = (x - tf.constant([.485, .456, .406])) / tf.constant([.229, .224, .225])
    feature_model = tf.keras.Model(classifier.model.inputs,
                                   classifier.model.get_layer("spatial_features").output)
    features = feature_model(x, training=False).numpy()[0]
    if label is None:
        label = classifier.predict(rgb).label
    weights = classifier.model.get_layer("classifier").get_weights()[0][:, label]
    raw = np.maximum(features @ weights, 0)
    return raw / raw.max() if raw.max() > 0 else raw


def occlusion_map(classifier, image, label=None, grid=5):
    """Model-agnostic score drop when each tile is replaced by mean RGB."""
    from polygarbor.classifier import load_image
    rgb = load_image(image)
    base = classifier.predict(rgb)
    label = base.label if label is None else label
    images = []
    ys, xs = np.linspace(0, len(rgb), grid + 1, dtype=int), np.linspace(0, rgb.shape[1], grid + 1, dtype=int)
    for i in range(grid):
        for j in range(grid):
            altered = rgb.copy()
            altered[ys[i]:ys[i+1], xs[j]:xs[j+1]] = rgb.mean(axis=(0, 1)).astype(np.uint8)
            images.append(altered)
    return (base.probabilities[label] - classifier.predict_proba(images)[:, label]).reshape(grid, grid)


def plot_prediction_summary(image, prediction, class_names, true_label=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].axis("off")
    title = prediction.class_name + f" ({prediction.confidence:.1%})"
    if true_label is not None:
        title += f"; real: {class_names[true_label]}"
    axes[0].set_title(title)
    axes[1].barh(class_names, prediction.probabilities)
    axes[1].set(xlim=(0, 1), xlabel="Softmax (sem calibração)")
    fig.tight_layout()
    return fig


def plot_explanation(image, heatmap, title="CAM — evidência da classe predita"):
    import cv2
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(image)
    axes[0].set_title("Imagem")
    signed = float(heatmap.min()) < 0
    bound = max(float(np.abs(heatmap).max()), 1e-12)
    cmap = "coolwarm" if signed else "magma"
    scale = dict(vmin=-bound if signed else 0, vmax=bound)
    shown = axes[1].imshow(heatmap, cmap=cmap, **scale)
    axes[1].set_title(f"Mapa nativo {heatmap.shape}\nmin={heatmap.min():.3g}; max={heatmap.max():.3g}")
    fig.colorbar(shown, ax=axes[1], fraction=.046, pad=.04)
    axes[2].imshow(image)
    axes[2].imshow(cv2.resize(heatmap, (image.shape[1], image.shape[0])), cmap=cmap, alpha=.45, **scale)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
