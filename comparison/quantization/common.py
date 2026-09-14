"""LiteRT export, post-training quantization and inference shared by the quantization study."""
import json
import tempfile
from pathlib import Path

import numpy as np

COMPARISON = Path(__file__).resolve().parent.parent
REPO = COMPARISON.parent
METHODS = ("resnet18", "resnet18_imagenet", "yolo11n_random", "yolo11n")
VARIANTS = ("fp32", "fp16_weights", "int8_dynamic", "int8_static")
CALIBRATION_SEED = 20260914
CALIBRATION_PER_CLASS = 25
IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.array([.229, .224, .225], dtype=np.float32)


def source_run(campaign, method, seed):
    return COMPARISON / "runs" / campaign / f"full__{method}__seed{seed}"


def read_meta(run):
    return json.loads((run / "model/model.json").read_text())


def calibration_indices(labels, per_class=CALIBRATION_PER_CLASS, seed=CALIBRATION_SEED):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
                                   for c in np.unique(labels)]))


def preprocess(method, meta, images):
    """uint8 RGB images -> float32 model inputs, without TensorFlow or PyTorch."""
    size = meta["image_size"]
    if method.startswith("resnet18"):
        import cv2
        x = np.stack([cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR) for image in images]).astype(np.float32) / 255.
        return (x - IMAGENET_MEAN) / IMAGENET_STD if meta["weights"] == "imagenet" else x
    from PIL import Image
    # Equivalent to the checkpoint's Resize(bilinear, antialias) + CenterCrop + ToTensor on square inputs.
    x = np.stack([np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR)) for image in images])
    return (x.astype(np.float32) / 255.).transpose(0, 3, 1, 2).copy()


def export_fp32(method, run, out):
    meta = read_meta(run)
    size = meta["image_size"]
    if method.startswith("resnet18"):
        import tensorflow as tf
        model = tf.keras.models.load_model(run / "model/model.keras", compile=False)
        # A concrete function drops the Keras 3 variables; a SavedModel keeps them.
        with tempfile.TemporaryDirectory() as saved:
            model.export(saved, format="tf_saved_model", verbose=False,
                         input_signature=[tf.TensorSpec([1, size, size, 3], tf.float32)])
            out.write_bytes(tf.lite.TFLiteConverter.from_saved_model(saved).convert())
        return
    import torch
    import litert_torch
    from ultralytics import YOLO

    class Probabilities(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, x):
            output = self.network(x)
            return output[0] if isinstance(output, (list, tuple)) else output

    network = YOLO(str(run / "model/model.pt")).model.float().eval()
    litert_torch.convert(Probabilities(network).eval(), (torch.zeros(1, 3, size, size),)).export(str(out))


def quantize(fp32, variant, calibration, out):
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import qtyping, quantizer, recipe, recipe_manager
    qt = quantizer.Quantizer(str(fp32))
    if variant == "fp16_weights":
        manager = recipe_manager.RecipeManager()
        manager.add_weight_only_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED, num_bits=16,
                                       algorithm_key=recipe.AlgorithmName.FLOAT_CASTING)
        qt.load_quantization_recipe(manager.get_quantization_recipe())
        result = qt.quantize()
    elif variant == "int8_dynamic":
        qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        result = qt.quantize()
    elif variant == "int8_static":
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
        for operation in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
            qt.update_quantization_recipe(regex=".*", operation_name=operation, algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)
        signatures = Interpreter(model_path=str(fp32)).get_signature_list()
        key = next(iter(signatures))
        name = signatures[key]["inputs"][0]
        result = qt.quantize(qt.calibrate({key: [{name: calibration[i:i + 1]} for i in range(len(calibration))]}))
    else:
        raise ValueError(variant)
    result.export_model(str(out), overwrite=True)


class LiteRTModel:
    def __init__(self, path, threads):
        from ai_edge_litert.interpreter import Interpreter
        self.interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]["index"]
        self.output = self.interpreter.get_output_details()[0]["index"]

    def __call__(self, x):
        """Scores for one preprocessed input that already has the batch axis."""
        self.interpreter.set_tensor(self.input, x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output)[0].copy()


def normalize(scores):
    """Quantized softmax rows drift from 1; renormalize so shared metrics accept them."""
    scores = np.clip(scores, 0, None)
    totals = scores.sum(1, keepdims=True)
    return np.where(totals > 0, scores / np.where(totals > 0, totals, 1), 1 / scores.shape[1])
