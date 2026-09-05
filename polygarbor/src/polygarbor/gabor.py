"""Banco de filtros de Gabor (canais de frequencia x orientacao)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class GaborConfig:
    """Parametros do banco de filtros de Gabor.

    O banco final tem ``len(lambdas) * len(thetas)`` filtros; cada filtro rende
    duas features por regiao (media e desvio padrao do mapa de energia).
    """

    ksize: int = 21
    sigma: float = 3.0
    gamma: float = 0.5
    lambdas: tuple[float, ...] = (4.0, 8.0)
    thetas: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ksize": self.ksize,
            "sigma": self.sigma,
            "gamma": self.gamma,
            "lambdas": list(self.lambdas),
            "thetas": list(self.thetas),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GaborConfig":
        return cls(
            ksize=int(data["ksize"]),
            sigma=float(data["sigma"]),
            gamma=float(data["gamma"]),
            lambdas=tuple(float(v) for v in data["lambdas"]),
            thetas=tuple(float(v) for v in data["thetas"]),
        )


@dataclass
class GaborBank:
    """Par de kernels (real, imaginario) por combinacao de lambda e theta."""

    config: GaborConfig = field(default_factory=GaborConfig)
    kernels: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.kernels:
            self.kernels, self.labels = _build(self.config)

    def __len__(self) -> int:
        return len(self.kernels)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return iter(self.kernels)

    @property
    def n_features(self) -> int:
        """Numero de features de textura geradas pelo banco (media + desvio)."""
        return 2 * len(self.kernels)

    def energy_maps(self, gray: np.ndarray) -> list[np.ndarray]:
        """Aplica todos os filtros na imagem inteira e devolve os mapas de energia.

        A convolucao e feita na imagem completa (nao por recorte) para evitar
        artefatos de borda nos patches.
        """
        gray = gray.astype(np.float32, copy=False)
        maps = []
        for k_real, k_imag in self.kernels:
            f_real = cv2.filter2D(gray, cv2.CV_32F, k_real)
            f_imag = cv2.filter2D(gray, cv2.CV_32F, k_imag)
            maps.append(cv2.magnitude(f_real, f_imag))
        return maps


def _build(cfg: GaborConfig) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    kernels: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[str] = []
    for lam in cfg.lambdas:
        for theta in cfg.thetas:
            k_real = cv2.getGaborKernel(
                (cfg.ksize, cfg.ksize), cfg.sigma, theta, lam, cfg.gamma,
                psi=0, ktype=cv2.CV_32F,
            )
            k_imag = cv2.getGaborKernel(
                (cfg.ksize, cfg.ksize), cfg.sigma, theta, lam, cfg.gamma,
                psi=np.pi / 2, ktype=cv2.CV_32F,
            )
            kernels.append((k_real, k_imag))
            labels.append(f"λ={lam:g}, θ={int(np.degrees(theta))}°")
    return kernels, labels
