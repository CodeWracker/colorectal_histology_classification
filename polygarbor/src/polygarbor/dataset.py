"""Carga permanente do dataset via TensorFlow Datasets.

O TFDS baixa e materializa o dataset uma unica vez em ``data_dir``; execucoes
seguintes apenas leem o cache dessa pasta. TensorFlow so e importado aqui
(sob demanda), para que o fluxo de inferencia funcione sem ele instalado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

DEFAULT_DATASET = "colorectal_histology"
DEFAULT_SPLITS = ("train[:80%]", "train[80%:90%]", "train[90%:]")
SPLIT_NAMES = ("train", "val", "test")


def _import_tfds():
    # Silencia os logs de inicializacao do TensorFlow antes do import.
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        import tensorflow_datasets as tfds  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise ImportError(
            "O fluxo de dataset precisa do TensorFlow Datasets.\n"
            "Instale com:  uv sync --extra dataset"
        ) from exc
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
    tfds.core.utils.tqdm_utils.disable_progress_bar()
    return tfds


@dataclass
class DatasetBundle:
    """Splits carregados + metadados do dataset."""

    splits: dict[str, Any]
    class_names: list[str]
    image_shape: tuple[int, ...]
    num_examples: dict[str, int]
    data_dir: Path
    name: str

    def __getitem__(self, split: str) -> Any:
        return self.splits[split]

    @property
    def image_size(self) -> int:
        return int(min(self.image_shape[0], self.image_shape[1]))


def load(
    data_dir: str | Path,
    name: str = DEFAULT_DATASET,
    splits: Sequence[str] = DEFAULT_SPLITS,
    shuffle_files: bool = False,
) -> DatasetBundle:
    """Baixa (se necessario) e carrega o dataset em ``data_dir``, de forma permanente.

    ``shuffle_files`` fica desligado por padrao: a ordem de leitura muda quais
    vetores caem na amostragem de treino, e com ela o resultado do modelo.
    """
    tfds = _import_tfds()
    data_dir = Path(data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets, info = tfds.load(
        name,
        split=list(splits),
        data_dir=str(data_dir),
        as_supervised=True,
        with_info=True,
        shuffle_files=shuffle_files,
    )

    counts = {}
    for key, ds in zip(SPLIT_NAMES, datasets):
        counts[key] = int(ds.cardinality().numpy())

    return DatasetBundle(
        splits=dict(zip(SPLIT_NAMES, datasets)),
        class_names=list(info.features["label"].names),
        image_shape=tuple(info.features["image"].shape),
        num_examples=counts,
        data_dir=data_dir,
        name=name,
    )


def iter_numpy(dataset: Any, limit: int | None = None) -> Iterator[tuple[np.ndarray, int]]:
    """Itera um split como pares ``(imagem uint8 RGB, rotulo int)``."""
    if limit:
        dataset = dataset.take(limit)
    for image, label in dataset.as_numpy_iterator():
        yield np.asarray(image), int(label)


def export_samples(
    bundle: DatasetBundle,
    out_dir: str | Path,
    split: str = "test",
    per_class: int = 1,
) -> list[Path]:
    """Grava algumas imagens em PNG para servir de entrada ao fluxo de predicao."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = {i: per_class for i in range(len(bundle.class_names))}
    written: list[Path] = []

    for image, label in iter_numpy(bundle[split]):
        if remaining.get(label, 0) <= 0:
            if not any(v > 0 for v in remaining.values()):
                break
            continue
        remaining[label] -= 1
        name = bundle.class_names[label]
        path = out_dir / f"{label:02d}_{name}_{per_class - remaining[label]}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        written.append(path)
    return written
