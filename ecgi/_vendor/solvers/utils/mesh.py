"""Generic mesh helpers for the forward solver utilities.

Config-agnostic: pure dolfinx, no knowledge of the UKB pipeline or its
parameter files. (The UKB pipeline keeps its own copy in ``common`` for the
many scripts that import it from there; this is the canonical solver-side one.)
"""
from __future__ import annotations

import dolfinx
import numpy as np


def tagged_facet_vertices(
    mesh: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    tag: int,
) -> np.ndarray:
    """Unique vertex indices of all facets carrying ``tag``."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    facets = facet_tags.indices[facet_tags.values == tag]
    if facets.size == 0:
        return np.empty((0,), dtype=np.int64)
    verts = np.unique(np.concatenate([f2v.links(int(f)) for f in facets]))
    return verts.astype(np.int64)
