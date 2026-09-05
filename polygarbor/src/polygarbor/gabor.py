"""Gabor filter bank (frequency x orientation channels)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class GaborConfig:
    """Parameters of the Gabor filter bank.

    The resulting bank holds ``len(lambdas) * len(thetas)`` filters; each filter
    contributes two features per region (mean and standard deviation of its
    energy map).
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
    """One (real, imaginary) kernel pair per lambda/theta combination."""

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
        """Number of texture features produced by the bank (mean + std)."""
        return 2 * len(self.kernels)

    def energy_maps(self, gray: np.ndarray) -> list[np.ndarray]:
        """Apply every filter to the whole image and return the energy maps.

        Convolution runs on the full image rather than on individual crops, so
        patches do not pick up border artifacts.
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
