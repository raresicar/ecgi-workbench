"""Expose the vendored scientific code (copied from the stabFEM-cauchy repo).

The app deliberately reuses the thesis code *verbatim* under ``ecgi/_vendor/``
rather than reimplementing the numerics. Those modules locate their own meshes
and POD bases via paths relative to their location, so all we have to do here is
put the three import roots on ``sys.path``:

* ``ukb/pipeline``  -> ``common`` (mesh loading, tags, partitions, params)
* ``ukb/stabFEM``   -> ``stabfem`` (the data-enriched stabFEM inverse solver)
* ``solvers``       -> ``forward.*`` / ``ionic_models.*`` / ``utils.*`` (the
                        monodomain + Mitchell-Schaeffer forward building blocks)
* ``ukb/database``  -> ``build_pod_basis`` / ``extend_pod_to_torso`` (POD tooling)

Import this module once (``ecgi/__init__`` does) before importing any vendored
name. Meshes are treated as fixed inputs; everything else (databases, POD bases)
may be regenerated through the vendored tooling.
"""
from __future__ import annotations

import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "_vendor"

_IMPORT_ROOTS = (
    VENDOR / "ukb" / "pipeline",
    VENDOR / "ukb" / "stabFEM",
    VENDOR / "solvers",
    VENDOR / "ukb" / "database",
)


def install() -> None:
    """Idempotently prepend the vendored import roots to ``sys.path``."""
    for root in _IMPORT_ROOTS:
        s = str(root)
        if s not in sys.path:
            sys.path.insert(0, s)


def _allow_gmsh_off_main_thread() -> None:
    """Let ``gmsh.initialize()`` run inside a worker thread (e.g. Streamlit's).

    gmsh installs a SIGINT handler via ``signal.signal``, which raises
    ``ValueError: signal only works in main thread`` when the mesh is loaded off
    the main thread. We only lose Ctrl-C interruption *of gmsh*, which a web app
    never needs, so we swallow that specific failure and otherwise defer to the
    real ``signal.signal``.
    """
    import signal

    _orig = signal.signal

    def _thread_safe_signal(signalnum, handler):
        try:
            return _orig(signalnum, handler)
        except ValueError:
            return None

    if getattr(signal.signal, "__name__", "") != "_thread_safe_signal":
        signal.signal = _thread_safe_signal  # type: ignore[assignment]


install()
_allow_gmsh_off_main_thread()
