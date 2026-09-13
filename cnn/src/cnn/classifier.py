"""ResNet-18 from cnn_test.ipynb; optional Ultralytics YOLO11 backend."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from polygarbor.classifier import load_image


def configure_tensorflow(device="auto", threads=4, seed=42):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("GPU requested but TensorFlow has no usable GPU")
    try:
        if device == "cpu":
            tf.config.set_visible_devices([], "GPU")
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError as exc:
        # A caller may configure once and build/load several models afterwards.
        visible = tf.config.get_visible_devices("GPU")
        if (device == "cpu" and visible) or (device == "gpu" and not visible):
            raise RuntimeError("Configure the device before creating TensorFlow tensors") from exc
    return tf


def build_resnet18(image_size=128, num_classes=8):
    from tensorflow.keras import layers, Model

    def conv(x, filters, kernel, strides=1):
        x = layers.Conv2D(filters, kernel, strides=strides, padding="same", use_bias=False)(x)
        return layers.ReLU()(layers.BatchNormalization()(x))

    def block(x, filters, strides=1):
        shortcut = x
        y = conv(x, filters, 3, strides)
        y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
        y = layers.BatchNormalization()(y)
        if strides != 1 or x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(x)
            shortcut = layers.BatchNormalization()(shortcut)
        return layers.ReLU()(layers.Add()([shortcut, y]))

    inputs = layers.Input((image_size, image_size, 3))
    x = conv(inputs, 64, 7, 2)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    for filters, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
        x = block(x, filters, stride)
        x = block(x, filters)
    x = layers.Activation("linear", name="spatial_features")(x)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return Model(inputs, outputs, name="resnet18")


@dataclass
class Prediction:
    label: int
    class_name: str
    probabilities: np.ndarray

    @property
    def confidence(self):
        return float(self.probabilities[self.label])

    def ranking(self, class_names):
        return [(class_names[i], float(self.probabilities[i]))
                for i in np.argsort(-self.probabilities)]

    def to_dict(self, class_names):
        return dict(label=self.label, class_name=self.class_name,
                    confidence=self.confidence, confidence_kind="softmax_uncalibrated",
                    probabilities=dict(zip(class_names, map(float, self.probabilities))))


class CNNClassifier:
    def __init__(self, class_names, architecture="resnet18", image_size=128,
                 batch_size=32, learning_rate=1e-3, random_state=42,
                 device="auto", threads=4, augment=True, weights="yolo11n-cls.pt"):
        if architecture not in ("resnet18", "yolo11n"):
            raise ValueError("architecture must be resnet18 or yolo11n")
        if len(class_names) < 2 or len(set(class_names)) != len(class_names):
            raise ValueError("At least two distinct class names are required")
        if architecture == "yolo11n" and (learning_rate != 1e-3 or not augment):
            raise ValueError("YOLO uses its backend learning-rate/augmentation policy; these overrides apply only to ResNet")
        if image_size < 32 or batch_size < 1 or learning_rate <= 0:
            raise ValueError("image_size >= 32, batch_size >= 1 and learning_rate > 0 required")
        self.class_names = list(class_names)
        self.architecture, self.image_size, self.batch_size = architecture, image_size, batch_size
        self.learning_rate, self.random_state = learning_rate, random_state
        self.device, self.threads, self.augment, self.weights = device, threads, augment, weights
        self.model = None
        self.history = {}
        self.epoch_seconds = []
        self._infer = None

    @property
    def n_classes(self):
        return len(self.class_names)

    def _pipeline(self, images, labels, training=False):
        import tensorflow as tf

        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        if training:
            ds = ds.shuffle(1000, seed=self.random_state)

        def preprocess(image, label):
            image = tf.cast(tf.image.resize(image, (self.image_size, self.image_size)), tf.float32) / 255.
            if training and self.augment:
                image = tf.image.random_flip_left_right(image)
                image = tf.image.random_flip_up_down(image)
                image = tf.image.random_brightness(image, max_delta=0.1)
            return image, label

        options = tf.data.Options()
        options.threading.private_threadpool_size = self.threads
        return (ds.map(preprocess, num_parallel_calls=1).batch(self.batch_size)
                .prefetch(1).with_options(options))

    def fit(self, samples, validation_data=None, epochs=100, patience=8,
            out_dir="artifacts/train", verbose=1, stage=None):
        from contextlib import nullcontext
        stage = stage or (lambda name: nullcontext())
        if epochs < 1 or patience < 0:
            raise ValueError("epochs >= 1 and patience >= 0 required")
        samples = list(samples)
        if not samples:
            raise ValueError("Empty training data")
        images = np.stack([load_image(x) for x, _ in samples])
        labels = np.asarray([int(y) for _, y in samples])
        if set(labels) != set(range(self.n_classes)):
            raise ValueError("Training data must include every configured class")
        val = list(validation_data) if validation_data is not None else []
        if not val:
            raise ValueError("A separate validation split is required for model selection")
        self._infer = None
        self.history = {}
        self.epoch_seconds = []
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.architecture == "yolo11n":
            return self._fit_yolo(samples, val, epochs, patience, out, verbose, stage)

        with stage("build_model"):
            tf = configure_tensorflow(self.device, self.threads, self.random_state)
            self.model = build_resnet18(self.image_size, self.n_classes)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        with stage("prepare_pipeline"):
            train_ds = self._pipeline(images, labels, True)
            val_ds = self._pipeline(np.stack([load_image(x) for x, _ in val]),
                                    np.array([int(y) for _, y in val]))
        owner = self

        class EpochLog(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.started = time.perf_counter()

            def on_epoch_end(self, epoch, logs=None):
                owner.epoch_seconds.append(time.perf_counter() - self.started)
                payload = dict(epoch=epoch + 1, seconds=owner.epoch_seconds[-1],
                               **{k: float(v) for k, v in (logs or {}).items()})
                with (out / "epochs.jsonl").open("a") as handle:
                    handle.write(json.dumps(payload) + "\n")
                print(json.dumps(payload), flush=True)

        with stage("fit"):
            history = self.model.fit(
                train_ds, validation_data=val_ds, epochs=epochs, verbose=verbose,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                                           restore_best_weights=True),
                           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.5,
                                                               patience=4, min_lr=1e-6),
                           tf.keras.callbacks.CSVLogger(str(out / "history.csv")), EpochLog()],
            )
        self.history = {k: list(map(float, v)) for k, v in history.history.items()}
        (out / "history.json").write_text(json.dumps(dict(history=self.history,
                                                         epoch_seconds=self.epoch_seconds), indent=2))
        return self

    def _fit_yolo(self, samples, val, epochs, patience, out, verbose, stage):
        from PIL import Image
        from ultralytics import YOLO
        import torch

        torch.set_num_threads(self.threads)
        root = out / "yolo_dataset"
        if root.exists():
            raise FileExistsError(f"Refusing to mix old training images in {root}; use a new out-dir")
        with stage("prepare_pipeline"):
            for split, data in (("train", samples), ("val", val)):
                for c in range(self.n_classes):
                    (root / split / f"{c:02d}").mkdir(parents=True, exist_ok=True)
                for i, (image, label) in enumerate(data):
                    folder = root / split / f"{int(label):02d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(load_image(image)).save(folder / f"{i:05d}.png")
        with stage("build_model"):
            self.model = YOLO(self.weights)
            self.model.add_callback("on_train_start", lambda trainer: torch.set_num_threads(self.threads))
        with stage("fit"):
            self.model.train(data=str(root.resolve()), epochs=epochs, imgsz=self.image_size,
                             batch=self.batch_size, patience=patience, seed=self.random_state,
                             device=self._yolo_device(), workers=0, project=str(out.resolve()),
                             name="yolo_training", exist_ok=False, verbose=bool(verbose),
                             plots=True)
        self.model = YOLO(str(self.model.trainer.best))
        import pandas as pd
        csv = self.model.ckpt_path and out / "yolo_training" / "results.csv"
        if csv and csv.exists():
            frame = pd.read_csv(csv)
            self.history = {k.strip(): frame[k].tolist() for k in frame}
            frame.to_csv(out / "history.csv", index=False)
        (out / "history.json").write_text(json.dumps(self.history, indent=2))
        return self

    def _yolo_device(self):
        return "cpu" if self.device == "cpu" else (0 if self.device == "gpu" else None)

    def predict_proba(self, images):
        if self.model is None:
            raise RuntimeError("Model is not trained; use fit() or load()")
        images = [load_image(x) for x in images]
        if not images:
            return np.empty((0, self.n_classes), dtype=np.float32)
        if self.architecture == "yolo11n":
            import torch
            if self.model.predictor is None:
                self.model.add_callback("on_predict_start", lambda predictor: torch.set_num_threads(self.threads))
            # Ultralytics expects BGR numpy images. Numeric folders fix class order.
            results = []
            for start in range(0, len(images), self.batch_size):
                results.extend(self.model.predict(
                    [x[..., ::-1].copy() for x in images[start:start + self.batch_size]],
                    imgsz=self.image_size, device=self._yolo_device(), verbose=False))
                import torch
                torch.set_num_threads(self.threads)
            order = [int(self.model.names[i]) for i in range(self.n_classes)]
            probs = np.zeros((len(images), self.n_classes), dtype=np.float32)
            probs[:, order] = np.stack([r.probs.data.cpu().numpy() for r in results])
            return probs
        import tensorflow as tf
        if self._infer is None:
            self._infer = tf.function(lambda x: self.model(x, training=False),
                input_signature=[tf.TensorSpec([None, self.image_size, self.image_size, 3], tf.float32)])
        result = []
        for start in range(0, len(images), self.batch_size):
            x = tf.stack([tf.image.resize(i, (self.image_size, self.image_size))
                          for i in images[start:start + self.batch_size]]) / 255.
            result.append(self._infer(x).numpy())  # synchronizes GPU
        return np.concatenate(result)

    def predict(self, image):
        p = self.predict_proba([image])[0]
        label = int(p.argmax())
        return Prediction(label, self.class_names[label], p)

    def predict_many(self, samples, **kwargs):
        samples = list(samples)
        if not samples:
            raise ValueError("Empty evaluation data")
        return (np.array([int(y) for _, y in samples]),
                self.predict_proba([x for x, _ in samples]).argmax(axis=1))

    def save(self, path):
        if self.model is None:
            raise RuntimeError("Model is not trained")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {k: getattr(self, k) for k in ("class_names", "architecture", "image_size",
                "batch_size", "learning_rate", "random_state", "threads", "augment", "weights")}
        meta["format_version"] = 1
        if self.architecture == "resnet18":
            import tensorflow as tf
            # Persist only inference state: Adam moments would triple disk size.
            inference = tf.keras.Model(self.model.inputs, self.model.output, name=self.model.name)
            inference.save(path / "model.keras")
            meta["parameters"] = int(inference.count_params())
        else:
            self.model.save(str(path / "model.pt"))
            meta["parameters"] = sum(p.numel() for p in self.model.model.parameters())
        (path / "model.json").write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, path, device="auto"):
        path = Path(path)
        meta = json.loads((path / "model.json").read_text())
        if meta.pop("format_version") != 1:
            raise ValueError("Unsupported model format")
        meta.pop("parameters", None)
        obj = cls(**meta, device=device)
        if obj.architecture == "resnet18":
            tf = configure_tensorflow(device, obj.threads, obj.random_state)
            obj.model = tf.keras.models.load_model(path / "model.keras", compile=False)
        else:
            from ultralytics import YOLO
            obj.model = YOLO(str(path / "model.pt"))
        return obj
