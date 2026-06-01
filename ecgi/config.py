"""Static configuration: where the vendored geometry/bases live, and forward defaults.

Meshes are *fixed inputs* (never regenerated). POD databases live under
``_vendor/ukb/database/<name>/`` and can be swapped or regenerated; each provides
a ``pod_basis.npz`` and an ``extended_pod_basis/`` directory that the inverse
solver consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "_vendor"


@dataclass(frozen=True)
class Paths:
    """Filesystem locations of the vendored, fixed geometry and shared tooling."""

    vendor: Path = VENDOR
    heart_msh: Path = VENDOR / "ukb" / "meshes" / "heart" / "ED_clipped.msh"
    torso_msh: Path = VENDOR / "ukb" / "meshes" / "torso" / "p001_ukb_torso.msh"
    params_json: Path = VENDOR / "ukb" / "pipeline" / "params.json"
    database_root: Path = VENDOR / "ukb" / "database"


@dataclass(frozen=True)
class Database:
    """A POD database the inverse can use as its prior (name + basis locations)."""

    name: str
    pod_basis: Path
    extended_dir: Path
    description: str = ""

    @property
    def is_available(self) -> bool:
        return self.pod_basis.exists() and self.extended_dir.exists()

    @property
    def n_extended(self) -> int:
        """How many torso-extended modes are available (caps usable n_modes)."""
        return len(list(self.extended_dir.glob("extended_pod_mode_*.npz")))


@dataclass(frozen=True)
class ForwardDefaults:
    """Default time integration window for an interactive forward run.

    Shorter than the database's 400 ms so a single live simulation stays snappy,
    while still spanning depolarisation and the early plateau where an infarct's
    signature appears.
    """

    t_end_ms: float = 200.0
    dt_ms: float = 0.5
    snapshot_every_ms: float = 5.0
    #: tau_out is divided by this inside a scar ball so it cannot stay activated
    infarct_tau_out_factor: float = 0.01


PATHS = Paths()
FORWARD = ForwardDefaults()


def available_databases() -> dict[str, Database]:
    """Discover vendored POD databases (those with a basis + extended modes)."""
    out: dict[str, Database] = {}
    specs = {
        "infarction": "Healthy beat + infarcts at many sites — the scar prior",
        "four_params": "Healthy beats, ±20% ionic-parameter variation — healthy prior",
    }
    for name, desc in specs.items():
        db = Database(
            name=name,
            pod_basis=PATHS.database_root / name / "pod_basis" / "pod_basis.npz",
            extended_dir=PATHS.database_root / name / "extended_pod_basis",
            description=desc,
        )
        if db.is_available:
            out[name] = db
    return out
