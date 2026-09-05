"""Extracao do descritor hibrido: energia de Gabor (textura) + estatisticas Lab (cor).

Dois modos de extracao compartilham exatamente a mesma definicao de feature:

``patch_features``
    Estatisticas por bloco de uma grade regular. E o modo usado no treino e na
    classificacao, porque produz um vetor por regiao macroscopica.

``dense_features``
    As mesmas estatisticas calculadas por pixel com uma janela deslizante
    (via ``cv2.blur``), gerando um vetor por posicao. Serve para os mapas de
    similaridade, que precisam de resolucao espacial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .gabor import GaborBank

LAB_CHANNELS = ("L", "A", "B")


def feature_names(bank: GaborBank) -> list[str]:
    """Rotulos das colunas do descritor, na mesma ordem em que sao geradas."""
    names = [f"G{i + 1}_{s}" for i in range(len(bank)) for s in ("μ", "σ")]
    names += [f"{c}_{s}" for c in LAB_CHANNELS for s in ("μ", "σ")]
    return names


def n_features(bank: GaborBank) -> int:
    return bank.n_features + 2 * len(LAB_CHANNELS)


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Normaliza qualquer entrada (tensor TF, float [0,1], uint8) para RGB uint8."""
    if hasattr(image, "numpy"):
        image = image.numpy()
    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = (image * 255.0) if float(image.max(initial=0)) <= 1.0 else image
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


@dataclass
class ImageMaps:
    """Representacoes intermediarias calculadas uma unica vez por imagem."""

    rgb: np.ndarray
    gray: np.ndarray
    lab: np.ndarray
    energy: list[np.ndarray]


def image_maps(image: np.ndarray, bank: GaborBank) -> ImageMaps:
    rgb = to_uint8_rgb(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return ImageMaps(rgb=rgb, gray=gray, lab=lab, energy=bank.energy_maps(gray))


@dataclass
class PatchView:
    """Recortes de um patch, guardados apenas quando se quer visualizar."""

    index: int
    box: tuple[int, int, int, int]  # (x, y, x_end, y_end)
    rgb: np.ndarray
    gray: np.ndarray
    lab: np.ndarray
    energy: list[np.ndarray]


def resolve_patch_size(image_size: int, patch_size: int | None, patches_per_row: int) -> int:
    """Define o lado do patch: valor explicito ou divisao da imagem em N blocos."""
    if patch_size:
        return int(patch_size)
    return max(1, int(image_size // max(1, patches_per_row)))


def patch_features(
    image: np.ndarray,
    bank: GaborBank,
    patch_size: int,
    with_views: bool = False,
) -> tuple[np.ndarray, list[PatchView]]:
    """Filtra a imagem inteira e resume cada patch da grade em um vetor de features."""
    maps = image_maps(image, bank)
    h, w = maps.gray.shape[:2]
    dim = n_features(bank)

    vectors: list[list[float]] = []
    views: list[PatchView] = []

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            y_end, x_end = y + patch_size, x + patch_size

            feats: list[float] = []
            energy_crops = []
            for emap in maps.energy:
                crop = emap[y:y_end, x:x_end]
                feats += [float(np.mean(crop)), float(np.std(crop))]
                if with_views:
                    energy_crops.append(crop)

            lab_crop = maps.lab[y:y_end, x:x_end]
            for c in range(lab_crop.shape[2]):
                ch = lab_crop[:, :, c]
                feats += [float(np.mean(ch)), float(np.std(ch))]

            vectors.append(feats)
            if with_views:
                views.append(
                    PatchView(
                        index=len(views),
                        box=(x, y, x_end, y_end),
                        rgb=maps.rgb[y:y_end, x:x_end],
                        gray=maps.gray[y:y_end, x:x_end],
                        lab=lab_crop,
                        energy=energy_crops,
                    )
                )

    if not vectors:
        return np.empty((0, dim), dtype=np.float32), views
    return np.asarray(vectors, dtype=np.float32), views


def _local_mean_std(plane: np.ndarray, win: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Media e desvio padrao locais via filtro de caixa (O(1) por pixel)."""
    mean = cv2.blur(plane, win)
    mean_sq = cv2.blur(plane * plane, win)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return mean, std


def dense_features(
    image: np.ndarray,
    bank: GaborBank,
    grid_size: int = 75,
    win_size: Sequence[int] = (15, 15),
) -> tuple[np.ndarray, tuple[int, int]]:
    """Descritor por pixel numa grade fixa ``grid_size`` x ``grid_size``.

    Retorna ``(features (grid*grid, D), (altura, largura))``.
    """
    rgb = to_uint8_rgb(image)
    small = cv2.resize(rgb, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    win = (int(win_size[0]), int(win_size[1]))

    planes: list[np.ndarray] = []
    for emap in bank.energy_maps(gray):
        mean, std = _local_mean_std(emap, win)
        planes += [mean, std]
    for c in range(lab.shape[2]):
        mean, std = _local_mean_std(lab[:, :, c], win)
        planes += [mean, std]

    tensor = np.stack(planes, axis=-1).astype(np.float32)
    return tensor.reshape(-1, tensor.shape[-1]), (grid_size, grid_size)
