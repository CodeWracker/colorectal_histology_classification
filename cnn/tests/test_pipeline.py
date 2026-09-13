"""Small behavioral checks: architecture, fitting, save/load, explanations, CLI."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pytest

from cnn.classifier import CNNClassifier, build_resnet18, configure_tensorflow
from cnn.evaluation import summarize
from cnn.cli import build_parser
from cnn.visualize import activation_map


def test_resnet_roundtrip_and_cam(tmp_path):
    configure_tensorflow("cpu", threads=2, seed=3)
    rng = np.random.default_rng(3)
    samples = [(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8), i % 2) for i in range(8)]
    clf = CNNClassifier(["a", "b"], image_size=32, batch_size=4, device="cpu", threads=2)
    clf.fit(samples, samples, epochs=1, out_dir=tmp_path / "history", verbose=0)
    x = samples[0][0]
    pred = clf.predict(x)
    assert pred.probabilities.sum() == pytest.approx(1, abs=1e-6)
    assert len([l for l in clf.model.layers if l.__class__.__name__ == "Conv2D"]) == 20
    before = clf.predict_proba([x])
    heat = activation_map(clf, x)
    assert heat.shape == (1, 1) and np.isfinite(heat).all()
    np.testing.assert_allclose(before, clf.predict_proba([x]), atol=1e-7)
    clf.save(tmp_path / "model")
    import zipfile, json
    with zipfile.ZipFile(tmp_path / "model/model.keras") as archive:
        assert not json.loads(archive.read("config.json")).get("compile_config")
    loaded = CNNClassifier.load(tmp_path / "model", device="cpu")
    np.testing.assert_allclose(before, loaded.predict_proba([x]), atol=1e-6)


def test_metrics_handle_missing_classes_and_validate():
    summary, result = summarize([0, 0], [[.8, .2], [.6, .4]], ["a", "b"])
    assert summary["accuracy"] == 1 and summary["macro_f1"] == .5
    assert summary["ece"] == pytest.approx(.3)
    with pytest.raises(ValueError):
        summarize([0], [[2., -1.]], ["a", "b"])


def test_cli_and_empty_data():
    parser = build_parser()
    for command in ("dataset", "train", "evaluate", "info"):
        assert parser.parse_args([command]).command == command
    assert parser.parse_args(["predict", "-i", "x.png", "--no-viz"]).no_viz
    assert parser.parse_args(["train", "--architecture", "resnet18", "--weights", "imagenet"]).weights == "imagenet"
    with pytest.raises(ValueError, match="Empty"):
        CNNClassifier(["a", "b"]).fit([], [])
    assert CNNClassifier(["a", "b"], architecture="resnet18").weights == "random"
    assert CNNClassifier(["a", "b"], architecture="resnet18", weights="imagenet").weights == "imagenet"
    assert CNNClassifier(["a", "b"], architecture="yolo11n", weights="random").weights == "yolo11n-cls.yaml"


def test_multiclass_gmean_zero_and_missing_class():
    from cnn.evaluation import multiclass_gmean
    assert multiclass_gmean([0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 0, 0], 2) == pytest.approx(.5)
    assert multiclass_gmean([0, 1], [0, 0], 2) == 0
    assert multiclass_gmean([0, 0], [0, 0], 2) is None
