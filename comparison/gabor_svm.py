"""Gabor + CIELAB descriptor classified by an RBF support vector machine with fixed hyperparameters.

Registered in PROTOCOL.md ("RBF-SVM on the descriptor in the registered
scenarios") on 2026-09-15. The descriptor, its extraction and the cap of 350
vectors per class, with the same random draw, are those of
PolyGaborClassifier; only the decision function changes: per-feature
standardization estimated on the retained vectors, then SVC(C=1,
gamma="scale"). No validation data are used.

The interface mirrors the parts of PolyGaborClassifier used by run.py and
edge.py: extract, fit_features, train_samples, max_samples_per_class,
predict(image).label/.similarity, save and load.
"""
from __future__ import annotations

import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from polygarbor import PolyGaborClassifier
from polygarbor.classifier import GaborConfig

MODEL_FILE = "model.json"
ESTIMATOR_FILE = "estimator.pkl"


class GaborSVMClassifier:
    def __init__(self, class_names, random_state=42, max_samples_per_class=350, C=1.0, gamma="scale", gabor=None):
        self.class_names = list(class_names)
        self.random_state = random_state
        self.max_samples_per_class = max_samples_per_class
        self.C, self.gamma = C, gamma
        # Used for feature extraction only; its PMD models are never fitted.
        self.descriptor = PolyGaborClassifier(self.class_names, gabor=gabor or GaborConfig())
        self.train_samples: dict[int, np.ndarray] = {}
        self.scaler = self.svm = None

    def extract(self, image):
        return self.descriptor.extract(image)

    def fit_features(self, X, y):
        """Cap each class exactly as PolyGaborClassifier.fit_features, then fit scaler and SVM."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        rng = np.random.default_rng(self.random_state)
        self.train_samples.clear()
        for idx in range(len(self.class_names)):
            samples = X[y == idx]
            if len(samples) == 0:
                raise ValueError(f"Class '{self.class_names[idx]}' has no training samples.")
            if self.max_samples_per_class and len(samples) > self.max_samples_per_class:
                samples = samples[rng.choice(len(samples), self.max_samples_per_class, replace=False)]
            self.train_samples[idx] = samples
        fit_X = np.vstack(list(self.train_samples.values()))
        fit_y = np.concatenate([np.full(len(s), idx) for idx, s in self.train_samples.items()])
        self.scaler = StandardScaler().fit(fit_X)
        self.svm = SVC(C=self.C, gamma=self.gamma, kernel="rbf").fit(self.scaler.transform(fit_X), fit_y)
        return self

    def predict(self, image):
        Z = self.scaler.transform(self.extract(image))
        label = int(self.svm.predict(Z)[0])
        similarity = softmax(self.svm.decision_function(Z)[0]).astype(np.float32)
        return SimpleNamespace(label=label, class_name=self.class_names[label], similarity=similarity)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with (path / ESTIMATOR_FILE).open("wb") as f:
            pickle.dump(dict(scaler=self.scaler, svm=self.svm), f)
        meta = dict(format_version=1, class_names=self.class_names, gabor=self.descriptor.gabor.to_dict(),
                    C=self.C, gamma=self.gamma, max_samples_per_class=self.max_samples_per_class,
                    random_state=self.random_state,
                    train_samples_per_class={self.class_names[i]: int(len(s)) for i, s in self.train_samples.items()},
                    support_vectors_per_class={name: int(n) for name, n in zip(self.class_names, self.svm.n_support_)})
        (path / MODEL_FILE).write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / MODEL_FILE).read_text())
        clf = cls(meta["class_names"], random_state=meta["random_state"], max_samples_per_class=meta["max_samples_per_class"],
                  C=meta["C"], gamma=meta["gamma"], gabor=GaborConfig.from_dict(meta["gabor"]))
        with (path / ESTIMATOR_FILE).open("rb") as f:
            estimator = pickle.load(f)
        clf.scaler, clf.svm = estimator["scaler"], estimator["svm"]
        return clf
