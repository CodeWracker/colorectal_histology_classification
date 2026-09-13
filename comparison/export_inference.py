"""Migrate early Keras exports that cloned Adam; retain original checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cnn/src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="2026-09-12")
    args = parser.parse_args()
    import numpy as np
    from cnn.classifier import CNNClassifier
    for path in (ROOT / "runs" / args.campaign).glob("*/model/model.keras"):
        with zipfile.ZipFile(path) as archive:
            compiled = json.loads(archive.read("config.json")).get("compile_config")
        if not compiled:
            continue
        start = time.perf_counter()
        classifier = CNNClassifier.load(path.parent, device="cpu")
        image = np.array(np.load(ROOT / "cache/test_images.npy", mmap_mode="r")[0])
        before = classifier.predict_proba([image])
        checkpoint = path.parent.parent / "training_checkpoint.keras"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        path.rename(checkpoint)
        classifier.save(path.parent)
        reloaded = CNNClassifier.load(path.parent, device="cpu")
        after = reloaded.predict_proba([image])
        np.testing.assert_allclose(before, after, atol=1e-6)
        size = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
        (path.parent.parent / "model_size.json").write_text(json.dumps(dict(bytes=size, mb=size / 1e6,
            training_checkpoint_mb=checkpoint.stat().st_size / 1e6), indent=2))
        (path.parent.parent / "inference_export.json").write_text(json.dumps(dict(
            reason="Keras clone_model had copied optimizer state; exported an uncompiled Functional model",
            seconds=time.perf_counter() - start, max_probability_difference=float(np.max(np.abs(before-after))),
            original_checkpoint=str(checkpoint), inference_mb=size / 1e6), indent=2))
        print(path, size / 1e6, "MB", flush=True)


if __name__ == "__main__":
    main()
