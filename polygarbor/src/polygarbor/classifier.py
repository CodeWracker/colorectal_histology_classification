"""Classifier: one polynomial Mahalanobis subspace per class."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from polymahalanobis import PolyMahalanobis

from .features import (
    dense_features,
    feature_names,
    n_features,
    patch_features,
    resolve_patch_size,
)
from .gabor import GaborBank, GaborConfig

MODEL_FILE = "model.json"
SAMPLES_DIR = "samples"


@dataclass
class Prediction:
    """Outcome of classifying a single image."""

    label: int
    class_name: str
    method: str
    aggregation: str
    votes: np.ndarray            # (n_classes,) units won by each class
    mean_distances: np.ndarray   # (n_classes,) mean distance per class
    similarity: np.ndarray       # (n_classes,) normalized similarity (sums to 1)
    distances: np.ndarray        # (n_units, n_classes) raw distances
    unit_labels: np.ndarray      # (n_units,) winning class of each unit
    grid: tuple[int, int] | None = None  # grid shape, when applicable

    @property
    def confidence(self) -> float:
        """Fraction of units (patches or pixels) that voted for the winning class."""
        total = float(self.votes.sum())
        return float(self.votes[self.label] / total) if total else 0.0

    def ranking(
        self, class_names: Sequence[str]
    ) -> list[tuple[str, float, int, float]]:
        """Classes from most to least likely: ``(name, similarity, votes, distance)``.

        Ordering uses the mean distance rather than the similarity: for very
        distant classes the similarity saturates at zero and would stop breaking
        ties.
        """
        order = np.argsort(self.mean_distances)
        return [
            (
                class_names[i],
                float(self.similarity[i]),
                int(self.votes[i]),
                float(self.mean_distances[i]),
            )
            for i in order
        ]

    def to_dict(self, class_names: Sequence[str]) -> dict[str, Any]:
        return {
            "label": self.label,
            "class_name": self.class_name,
            "method": self.method,
            "aggregation": self.aggregation,
            "confidence": self.confidence,
            "votes": {class_names[i]: int(v) for i, v in enumerate(self.votes)},
            "mean_distances": {
                class_names[i]: float(d) for i, d in enumerate(self.mean_distances)
            },
            "similarity": {
                class_names[i]: float(s) for i, s in enumerate(self.similarity)
            },
        }


@dataclass
class PolyGaborClassifier:
    """Full Gabor + polynomial Mahalanobis pipeline.

    Typical use as a library::

        from polygarbor import PolyGaborClassifier

        clf = PolyGaborClassifier.load("model")
        pred = clf.predict("slide.png")
        print(pred.class_name, pred.confidence)
    """

    class_names: list[str]
    gabor: GaborConfig = field(default_factory=GaborConfig)
    patch_size: int | None = None
    patches_per_row: int = 1
    num_levels: int = 3
    max_samples_per_class: int = 350
    dense_grid: int = 75
    dense_window: tuple[int, int] = (15, 15)
    similarity_gamma: float = 0.05   # decay used by the heatmaps
    similarity_scale: float = 1.0    # decay (in log space) of the class ranking
    random_state: int = 42

    bank: GaborBank = field(init=False)
    models: dict[int, PolyMahalanobis] = field(init=False, default_factory=dict)
    train_samples: dict[int, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.bank = GaborBank(self.gabor)

    # -------------------------------------------------------------- plumbing

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return n_features(self.bank)

    @property
    def feature_names(self) -> list[str]:
        return feature_names(self.bank)

    @property
    def is_fitted(self) -> bool:
        return len(self.models) == self.n_classes

    def patch_size_for(self, image_size: int) -> int:
        return resolve_patch_size(image_size, self.patch_size, self.patches_per_row)

    # ------------------------------------------------------------ extraction

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Per-patch descriptors of an image, shaped (n_patches, n_features)."""
        from .features import to_uint8_rgb

        rgb = to_uint8_rgb(image)
        size = self.patch_size_for(min(rgb.shape[0], rgb.shape[1]))
        feats, _ = patch_features(rgb, self.bank, size)
        return feats

    def extract_with_views(self, image: np.ndarray):
        from .features import to_uint8_rgb

        rgb = to_uint8_rgb(image)
        size = self.patch_size_for(min(rgb.shape[0], rgb.shape[1]))
        return patch_features(rgb, self.bank, size, with_views=True)

    def extract_dense(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        return dense_features(image, self.bank, self.dense_grid, self.dense_window)

    # -------------------------------------------------------------- training

    def fit(
        self,
        samples: Iterable[tuple[np.ndarray, int]],
        progress: Any = None,
    ) -> "PolyGaborClassifier":
        """Extract features from an ``(image, label)`` iterable and fit the subspaces."""
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        iterator = progress(samples) if progress else samples
        for image, label in iterator:
            feats = self.extract(image)
            if len(feats):
                X_parts.append(feats)
                y_parts.append(np.full(len(feats), int(label), dtype=np.int32))
        if not X_parts:
            raise ValueError("No features extracted: check the input dataset.")
        return self.fit_features(np.vstack(X_parts), np.concatenate(y_parts))

    def fit_features(self, X: np.ndarray, y: np.ndarray) -> "PolyGaborClassifier":
        """Fit one polynomial subspace per class from precomputed features."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        rng = np.random.default_rng(self.random_state)

        self.models.clear()
        self.train_samples.clear()
        for idx in range(self.n_classes):
            samples = X[y == idx]
            if len(samples) == 0:
                raise ValueError(
                    f"Class '{self.class_names[idx]}' has no training samples."
                )
            if self.max_samples_per_class and len(samples) > self.max_samples_per_class:
                pick = rng.choice(len(samples), self.max_samples_per_class, replace=False)
                samples = samples[pick]
            self.train_samples[idx] = samples
            self.models[idx] = _make_space(samples, self.num_levels)
        return self

    # ------------------------------------------------------------- inference

    def distances(self, features: np.ndarray) -> np.ndarray:
        """Polynomial distance from each vector to each class, shaped (n, n_classes)."""
        self._check_fitted()
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        out = np.zeros((len(features), self.n_classes), dtype=np.float32)
        for idx, model in self.models.items():
            res = np.asarray(model.evaluate(features))
            out[:, idx] = res[:, -1] if res.ndim == 2 else res.ravel()
        return out

    def predict(
        self,
        image: np.ndarray | str | Path,
        method: str = "patch",
        aggregation: str = "voting",
    ) -> Prediction:
        """Classify one image.

        ``method='patch'`` uses the same patch grid as training (recommended).
        ``method='dense'`` uses the per-pixel descriptor, consistent with the
        similarity maps. ``aggregation`` picks between majority voting across
        units (``'voting'``) and smallest mean distance (``'mean'``).
        """
        image = load_image(image)
        if method == "patch":
            feats = self.extract(image)
            grid = None
        elif method == "dense":
            feats, grid = self.extract_dense(image)
        else:
            raise ValueError(f"invalid method: {method!r} (use 'patch' or 'dense')")

        if not len(feats):
            raise ValueError(
                "Image smaller than the configured patch: no descriptor extracted."
            )
        return self.predict_from_distances(
            self.distances(feats), method=method, aggregation=aggregation, grid=grid
        )

    def predict_from_distances(
        self,
        dist: np.ndarray,
        method: str = "patch",
        aggregation: str = "voting",
        grid: tuple[int, int] | None = None,
    ) -> Prediction:
        unit_labels = np.argmin(dist, axis=1)
        votes = np.bincount(unit_labels, minlength=self.n_classes)
        mean_dist = dist.mean(axis=0)

        if aggregation == "voting":
            label = int(np.argmax(votes))
        elif aggregation == "mean":
            label = int(np.argmin(mean_dist))
        else:
            raise ValueError(
                f"invalid aggregation: {aggregation!r} (use 'voting' or 'mean')"
            )

        # The log scale keeps distant classes (distances around 1e5) from all
        # collapsing to zero similarity and becoming indistinguishable.
        log_dist = np.log1p(np.maximum(mean_dist, 0.0))
        sim = np.exp(-abs(self.similarity_scale) * (log_dist - log_dist.min()))
        sim = sim / sim.sum() if sim.sum() else np.full(self.n_classes, 1 / self.n_classes)

        return Prediction(
            label=label,
            class_name=self.class_names[label],
            method=method,
            aggregation=aggregation,
            votes=votes,
            mean_distances=mean_dist,
            similarity=sim.astype(np.float32),
            distances=dist,
            unit_labels=unit_labels,
            grid=grid,
        )

    def predict_many(
        self,
        samples: Iterable[tuple[np.ndarray, int]],
        method: str = "patch",
        aggregation: str = "voting",
        progress: Any = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Classify an ``(image, label)`` iterable; returns ``(y_true, y_pred)``."""
        y_true, y_pred = [], []
        iterator = progress(samples) if progress else samples
        for image, label in iterator:
            try:
                pred = self.predict(image, method=method, aggregation=aggregation)
            except ValueError:
                continue
            y_true.append(int(label))
            y_pred.append(pred.label)
        return np.asarray(y_true), np.asarray(y_pred)

    # ----------------------------------------------------------- persistence

    def save(self, path: str | Path) -> Path:
        """Save config and training samples; the subspace is rebuilt on load."""
        self._check_fitted()
        path = Path(path)
        (path / SAMPLES_DIR).mkdir(parents=True, exist_ok=True)

        for idx, samples in self.train_samples.items():
            np.savetxt(
                path / SAMPLES_DIR / f"class_{idx:02d}.txt",
                samples, fmt="%.6f", delimiter=" ",
            )

        meta = {
            "format_version": 1,
            "class_names": self.class_names,
            "gabor": self.gabor.to_dict(),
            "patch_size": self.patch_size,
            "patches_per_row": self.patches_per_row,
            "num_levels": self.num_levels,
            "max_samples_per_class": self.max_samples_per_class,
            "dense_grid": self.dense_grid,
            "dense_window": list(self.dense_window),
            "similarity_gamma": self.similarity_gamma,
            "similarity_scale": self.similarity_scale,
            "random_state": self.random_state,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "train_samples_per_class": {
                self.class_names[i]: int(len(s)) for i, s in self.train_samples.items()
            },
        }
        (path / MODEL_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PolyGaborClassifier":
        """Reload a saved model, rebuilding the subspaces from the samples."""
        path = Path(path)
        meta_file = path / MODEL_FILE
        if not meta_file.exists():
            raise FileNotFoundError(f"'{meta_file}' not found: invalid model.")
        meta = json.loads(meta_file.read_text())

        clf = cls(
            class_names=list(meta["class_names"]),
            gabor=GaborConfig.from_dict(meta["gabor"]),
            patch_size=meta["patch_size"],
            patches_per_row=int(meta["patches_per_row"]),
            num_levels=int(meta["num_levels"]),
            max_samples_per_class=int(meta["max_samples_per_class"]),
            dense_grid=int(meta["dense_grid"]),
            dense_window=tuple(meta["dense_window"]),
            similarity_gamma=float(meta["similarity_gamma"]),
            similarity_scale=float(meta.get("similarity_scale", 1.0)),
            random_state=int(meta["random_state"]),
        )
        for idx in range(clf.n_classes):
            sample_file = path / SAMPLES_DIR / f"class_{idx:02d}.txt"
            if not sample_file.exists():
                raise FileNotFoundError(f"Missing samples for class {idx}: {sample_file}")
            samples = np.atleast_2d(np.loadtxt(sample_file, dtype=np.float32))
            clf.train_samples[idx] = samples
            clf.models[idx] = _make_space(samples, clf.num_levels, sample_file)
        return clf

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not trained. Use fit()/fit_features() or "
                "PolyGaborClassifier.load(<folder>)."
            )


def _make_space(
    samples: np.ndarray, num_levels: int, sample_file: Path | None = None
) -> PolyMahalanobis:
    """Fit a PolyMahalanobis model.

    The library only accepts samples coming from a text file, so when the array
    is already in memory we write a temporary file instead of duplicating the
    format handling.
    """
    import tempfile

    if sample_file is not None:
        model = PolyMahalanobis(str(sample_file), num_levels=num_levels)
    else:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            np.savetxt(tmp, samples, fmt="%.6f", delimiter=" ")
            tmp_path = tmp.name
        try:
            model = PolyMahalanobis(tmp_path, num_levels=num_levels)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    model.makeSpace()
    return model


def load_image(source: np.ndarray | str | Path) -> np.ndarray:
    """Accept a file path, a numpy array or a tensor and return RGB uint8."""
    from .features import to_uint8_rgb

    if isinstance(source, (str, Path)):
        import cv2

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not decode the image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return to_uint8_rgb(source)
