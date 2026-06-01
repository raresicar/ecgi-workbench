"""Small value objects describing what to simulate and what came out."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InfarctSpec:
    """A single myocardial infarction: a ball on the epicardium where tau_out is
    reduced so the region cannot sustain activation. ``None`` centre = healthy."""

    centre_mm: tuple[float, float, float] | None = None
    radius_mm: float = 12.0

    @property
    def is_healthy(self) -> bool:
        return self.centre_mm is None

    @staticmethod
    def healthy() -> "InfarctSpec":
        return InfarctSpec(centre_mm=None)

    def centre(self) -> np.ndarray:
        if self.centre_mm is None:
            raise ValueError("healthy case has no infarct centre")
        return np.asarray(self.centre_mm, dtype=np.float64)


@dataclass
class ForwardResult:
    """Output of one forward simulation, sliced for both rendering and inversion.

    ``vm`` is per heart-mesh vertex (for the V_m animation); ``hsp`` is restricted
    to the outer EPI∪BASE interface (what the POD basis and inverse operate on).
    """

    times_ms: np.ndarray            # (n_snap,)
    vm: np.ndarray                  # (n_snap, n_heart_vertices)  transmembrane potential
    hsp: np.ndarray                 # (n_snap, n_outer)           heart-surface potential
    spec: InfarctSpec

    def snapshot_count(self) -> int:
        return int(self.times_ms.size)

    def peak_depolarisation_index(self) -> int:
        """Snapshot with the largest HSP RMS — the depolarisation dipole peak."""
        return int(np.argmax(np.sqrt((self.hsp ** 2).mean(axis=1))))


@dataclass
class InverseResult:
    """Output of one stabFEM inverse solve on a chosen HSP snapshot."""

    hsp_truth: np.ndarray           # (n_outer,)
    hsp_recovered: np.ndarray       # (n_outer,)
    clean_bsp: np.ndarray           # (n_body,)
    noisy_bsp: np.ndarray           # (n_body,)
    cosine: float
    rel_l2: float
    iterations: int
    n_modes: int
    snapshot_time_ms: float
