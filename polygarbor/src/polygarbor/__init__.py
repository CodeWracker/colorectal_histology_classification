"""polygarbor - Gabor + Distancia de Mahalanobis Polinomial para histologia colorretal.

Uso como biblioteca::

    from polygarbor import PolyGaborClassifier

    clf = PolyGaborClassifier.load("artifacts/model")
    pred = clf.predict("minha_lamina.png")
    print(pred.class_name, pred.confidence)
"""

from .classifier import PolyGaborClassifier, Prediction, load_image
from .evaluation import EvaluationResult, evaluate
from .features import dense_features, feature_names, patch_features, to_uint8_rgb
from .gabor import GaborBank, GaborConfig

__version__ = "0.1.0"

__all__ = [
    "PolyGaborClassifier",
    "Prediction",
    "GaborBank",
    "GaborConfig",
    "EvaluationResult",
    "evaluate",
    "load_image",
    "patch_features",
    "dense_features",
    "feature_names",
    "to_uint8_rgb",
]
