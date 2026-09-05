"""Testes do pipeline com imagens sinteticas (sem dependencia do TFDS)."""

from __future__ import annotations

import numpy as np
import pytest

from polygarbor import PolyGaborClassifier, evaluate, visualize
from polygarbor.features import dense_features, patch_features, to_uint8_rgb
from polygarbor.gabor import GaborBank, GaborConfig

CLASSES = ["listras_h", "listras_v", "ruido"]
SIZE = 150


def synth(cls: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    if cls == 0:
        img[::6, :, :] = 255
        img[..., 0] = 200
    elif cls == 1:
        img[:, ::6, :] = 255
        img[..., 1] = 200
    else:
        img = rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)
    return np.clip(img.astype(float) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)


def make_set(n: int, seed: int) -> list[tuple[np.ndarray, int]]:
    rng = np.random.default_rng(seed)
    return [(synth(c, rng), c) for c in range(len(CLASSES)) for _ in range(n)]


@pytest.fixture(scope="module")
def fitted() -> PolyGaborClassifier:
    clf = PolyGaborClassifier(class_names=CLASSES, patches_per_row=3)
    return clf.fit(make_set(20, seed=0))


def test_bank_shape():
    bank = GaborBank(GaborConfig())
    assert len(bank) == 8
    assert bank.n_features == 16
    for k_real, k_imag in bank:
        assert k_real.shape == (21, 21) == k_imag.shape


def test_descriptor_is_22d_and_consistent():
    bank = GaborBank()
    img = synth(0, np.random.default_rng(1))
    feats, views = patch_features(img, bank, patch_size=50, with_views=True)
    assert feats.shape == (9, 22)
    assert len(views) == 9
    dense, grid = dense_features(img, bank, grid_size=75)
    assert grid == (75, 75)
    assert dense.shape == (75 * 75, 22)


def test_to_uint8_rgb_accepts_float_and_gray():
    assert to_uint8_rgb(np.zeros((10, 10, 3), np.float32)).dtype == np.uint8
    assert to_uint8_rgb(np.zeros((10, 10), np.uint8)).shape == (10, 10, 3)


def test_patch_smaller_than_image_yields_nothing():
    feats, _ = patch_features(np.zeros((10, 10, 3), np.uint8), GaborBank(), patch_size=50)
    assert feats.shape == (0, 22)


def test_fit_and_predict(fitted):
    for image, label in make_set(3, seed=7):
        assert fitted.predict(image).label == label


def test_dense_prediction_has_grid(fitted):
    pred = fitted.predict(synth(0, np.random.default_rng(3)), method="dense")
    assert pred.grid == (75, 75)
    assert pred.distances.shape == (75 * 75, 3)
    assert 0.0 <= pred.confidence <= 1.0


def test_similarity_sums_to_one(fitted):
    pred = fitted.predict(synth(1, np.random.default_rng(4)))
    assert pred.similarity.sum() == pytest.approx(1.0, abs=1e-5)


def test_save_load_roundtrip(fitted, tmp_path):
    path = fitted.save(tmp_path / "model")
    reloaded = PolyGaborClassifier.load(path)
    assert reloaded.class_names == fitted.class_names
    assert reloaded.patches_per_row == fitted.patches_per_row
    image = synth(2, np.random.default_rng(5))
    assert reloaded.predict(image).label == fitted.predict(image).label


def test_predict_requires_fitted_model():
    with pytest.raises(RuntimeError, match="nao treinado"):
        PolyGaborClassifier(class_names=CLASSES).predict(np.zeros((150, 150, 3), np.uint8))


def test_invalid_method(fitted):
    with pytest.raises(ValueError, match="method invalido"):
        fitted.predict(np.zeros((150, 150, 3), np.uint8), method="xyz")


def test_evaluation_report(fitted):
    result = evaluate(fitted, make_set(4, seed=11))
    assert result.accuracy == pytest.approx(1.0)
    assert set(result.report_dict) >= set(CLASSES)
    assert result.confusion().shape == (3, 3)


def test_figures_are_created(fitted, tmp_path):
    visualize.use_headless()
    image = synth(0, np.random.default_rng(9))
    pred = fitted.predict(image, method="dense")
    feats, views = fitted.extract_with_views(image)
    figures = [
        visualize.plot_gabor_bank(fitted.bank),
        visualize.plot_patch_decomposition(views[0], fitted.bank.labels),
        visualize.plot_feature_matrix(feats, fitted.feature_names),
        visualize.plot_similarity_maps(image, pred, CLASSES, true_label=0),
        visualize.plot_prediction_summary(image, pred, CLASSES),
    ]
    for i, fig in enumerate(figures):
        assert visualize.save_figure(fig, tmp_path / f"f{i}.png").stat().st_size > 0


def test_similarity_maps_reject_patch_prediction(fitted):
    pred = fitted.predict(synth(0, np.random.default_rng(2)))
    with pytest.raises(ValueError, match="predicao densa"):
        visualize.plot_similarity_maps(np.zeros((150, 150, 3), np.uint8), pred, CLASSES)
