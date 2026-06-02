"""StabFEM ECGi Workbench — an interactive forward/inverse cardiac-electrophysiology app.

The package wraps the thesis' stabFEM ECGi pipeline (vendored under ``_vendor/``)
in a small, modular, object-oriented API:

* :class:`ecgi.geometry.Geometry`        — the fixed heart/torso meshes + surfaces
* :class:`ecgi.forward.ForwardSimulator` — monodomain V_m -> extracellular HSP
* :class:`ecgi.inverse.InverseSolver`    — the data-enriched stabFEM inverse
* :class:`ecgi.rendering.Renderer`       — Plotly 3D surface fields with a legend
* :class:`ecgi.cases.InfarctSpec`        — a user-placed scar (centre + radius)

Importing the package puts the vendored scientific code on ``sys.path``.
"""
from __future__ import annotations

from . import _bootstrap  # noqa: F401  (side effect: sys.path setup; must come first)

__all__ = ["Geometry", "ForwardSimulator", "InverseSolver", "Renderer", "InfarctSpec"]
__version__ = "0.1.0"
