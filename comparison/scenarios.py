"""Nested subsets and fixed, paired test perturbations."""
import cv2
import numpy as np

SIZES = (1, 2, 5, 10, 20, 50, 100)
CORRUPTIONS = ("clean", "blur", "noise", "brightness", "color", "jpeg", "resolution", "occlusion", "rotation")


def select_training(labels, scenario, seed):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    if scenario.startswith("few"):
        groups = [g[:int(scenario[3:])] for g in groups]
    elif scenario == "imbalance":
        groups = [g[:max(1, len(g) // 10)] if c in (0, 1, 2, 5) else g for c, g in enumerate(groups)]
    elif scenario not in ("full", "label_noise"):
        raise ValueError(scenario)
    indices = np.sort(np.concatenate(groups))
    selected = labels[indices].copy()
    if scenario == "label_noise":
        corrupt = rng.choice(len(indices), int(.2 * len(indices)), replace=False)
        selected[corrupt] = (selected[corrupt] + rng.integers(1, len(groups), len(corrupt))) % len(groups)
    return indices, selected


def perturb(image, kind, index):
    rng = np.random.default_rng(20260912 + index)
    if kind == "clean":
        return image
    if kind == "blur":
        return cv2.GaussianBlur(image, (9, 9), 2)
    if kind == "noise":
        return np.clip(image.astype(float) + rng.normal(0, 20, image.shape), 0, 255).astype(np.uint8)
    if kind == "brightness":
        return np.clip(image.astype(float) * 1.35, 0, 255).astype(np.uint8)
    if kind == "color":
        return np.clip(image.astype(float) * [1.2, .8, 1.1], 0, 255).astype(np.uint8)
    if kind == "jpeg":
        _, encoded = cv2.imencode(".jpg", image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 20])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1]
    if kind == "resolution":
        return cv2.resize(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA), (image.shape[1], image.shape[0]))
    if kind == "occlusion":
        result = image.copy()
        h, w = image.shape[:2]
        result[h//3:2*h//3, w//3:2*w//3] = 255
        return result
    if kind == "rotation":
        return np.rot90(image).copy()
    raise ValueError(kind)
