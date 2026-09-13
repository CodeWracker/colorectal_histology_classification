"""PolyGabor ablations: spatial granularity and train-only augmentation."""
import numpy as np

VARIANTS = {
    "polygarbor": (1, 0),
    "polygarbor_p3": (3, 0),
    "polygarbor_p5": (5, 0),
    "polygarbor_aug": (1, 3),
    "polygarbor_p3_aug": (3, 3),
}


def training_views(image, copies, seed, original_index):
    """Original plus three fixed draws of ResNet-style flips and brightness.

    This is not YOLO's RandAugment policy. Keep the policy explicit and compare
    augmentation vs no augmentation with the same descriptor and cap.
    """
    yield image
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(original_index)]))
    for _ in range(copies):
        view = image
        if rng.random() < .5:
            view = np.flip(view, axis=1)
        if rng.random() < .5:
            view = np.flip(view, axis=0)
        brightness = rng.uniform(-.1, .1) * 255
        yield np.clip(view.astype(float) + brightness, 0, 255).astype(np.uint8)
