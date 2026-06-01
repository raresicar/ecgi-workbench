"""Transmural cell-model heterogeneity: a per-dof ``tau_close`` field.

Generic and config-agnostic: every marker id, layer fraction, and tau_close
value is passed in explicitly (the UKB pipeline sources them from params.json;
the database samples them). Nothing here reads a config file.

``beat.utils.expand_layer_biv`` splits the ventricular wall into endo / mid /
epi layers using a normalised Laplace coordinate; it lumps LV and RV endo into
a single "endo-near" region, so the lumped endo shell uses the endo value.
"""
from __future__ import annotations

import dolfinx
import numpy as np
from beat.utils import expand_layer_biv

# Region markers used in the returned region field.
MID = 0
ENDO = 1
EPI = 2


def build_region_field(
    V: dolfinx.fem.FunctionSpace,
    facet_tags: dolfinx.mesh.MeshTags,
    *,
    lv_marker: int,
    rv_marker: int,
    epi_marker: int,
    endo_size: float,
    epi_size: float,
) -> tuple[dolfinx.fem.Function, dict]:
    """Split the wall into endo/mid/epi. Returns ``(region_function, counts)``.

    ``region_function`` is a CG1 Function with values 0=mid, 1=endo, 2=epi
    (see the module markers); ``counts`` is the per-region local dof count.
    """
    region = expand_layer_biv(
        V=V,
        ft=facet_tags,
        endo_lv_marker=lv_marker,
        endo_rv_marker=rv_marker,
        epi_marker=epi_marker,
        endo_size=endo_size,
        epi_size=epi_size,
        output_mid_marker=MID,
        output_endo_marker=ENDO,
        output_epi_marker=EPI,
    )
    marker = np.rint(region.x.array).astype(int)
    counts = {
        "endo": int((marker == ENDO).sum()),
        "mid": int((marker == MID).sum()),
        "epi": int((marker == EPI).sum()),
    }
    return region, counts


def build_tau_close_field(
    V: dolfinx.fem.FunctionSpace,
    facet_tags: dolfinx.mesh.MeshTags,
    *,
    lv_marker: int,
    rv_marker: int,
    epi_marker: int,
    endo_size: float,
    epi_size: float,
    tau_close_endo: float,
    tau_close_mid: float,
    tau_close_epi: float,
) -> tuple[np.ndarray, dolfinx.fem.Function, dict]:
    """Return ``(tau_close_array, region_function, metadata)``.

    ``tau_close_array`` has shape ``(n_dofs_local,)`` and is ready to pass to
    :func:`ionic_models.mitchell_schaeffer.make_step` as ``tau_close``.
    """
    region, counts = build_region_field(
        V, facet_tags,
        lv_marker=lv_marker, rv_marker=rv_marker, epi_marker=epi_marker,
        endo_size=endo_size, epi_size=epi_size,
    )
    marker = np.rint(region.x.array).astype(int)
    tau_close = np.full(marker.shape, tau_close_mid, dtype=np.float64)
    tau_close[marker == ENDO] = tau_close_endo
    tau_close[marker == EPI] = tau_close_epi

    metadata = {
        "expand_layer_biv": {
            "endo_size": float(endo_size),
            "epi_size": float(epi_size),
            "lv_marker": int(lv_marker),
            "rv_marker": int(rv_marker),
            "epi_marker": int(epi_marker),
        },
        "tau_close_per_region_ms": {
            "endo": float(tau_close_endo),
            "mid": float(tau_close_mid),
            "epi": float(tau_close_epi),
        },
        "endo_note": "LV and RV endo are lumped by expand_layer_biv; the "
                     "lumped endo shell uses the endo value.",
        "dof_counts": counts,
    }
    return tau_close, region, metadata
