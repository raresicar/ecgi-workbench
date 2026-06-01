"""The fixed heart and torso geometry, loaded once and shared.

This wraps the vendored ``common`` helpers to build everything the forward
simulator, the inverse solver and the renderer need from the *fixed* meshes:
the CG1 function space, the endo/mid/epi region marker, and the triangle
surfaces (with their point arrays) for the heart boundary, the outer EPI∪BASE
interface, and the torso body surface.

Loading meshes and computing the region marker is the expensive part, so an
instance is meant to be built once (and cached by the app).
"""
from __future__ import annotations

import os
import tempfile
from functools import cached_property

import numpy as np
from mpi4py import MPI

# vendored scientific code (sys.path set up by ecgi._bootstrap)
from common import (  # type: ignore  # noqa: E402
    BASE, EPI, GAMMA_BODY, LV, RV,
    cg1_space, heart_boundary_partition, load_gmsh_mesh, load_params,
    tagged_facet_vertices, vertex_to_dof_map,
)
from utils.heterogeneity import build_region_field  # type: ignore  # noqa: E402

from .config import PATHS


def _triangle_faces(mesh, facet_indices: np.ndarray) -> np.ndarray:
    """(n,3) vertex-index triangles for the given boundary facets of ``mesh``."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    return np.asarray([f2v.links(int(f)) for f in facet_indices], dtype=np.int64)


def _reindex_faces(faces: np.ndarray, vertex_subset: np.ndarray):
    """Keep only faces fully inside ``vertex_subset`` and renumber them to the
    subset's local 0..k-1 indexing (so they index a sliced point array)."""
    row = np.full(int(faces.max()) + 1, -1, dtype=np.int64)
    row[vertex_subset] = np.arange(vertex_subset.size, dtype=np.int64)
    keep = np.all(row[faces] >= 0, axis=1)
    return row[faces[keep]]


class Geometry:
    """Fixed heart + torso geometry and the surfaces used for rendering/solving."""

    def __init__(self) -> None:
        self.params = load_params()

        # --- heart mesh + CG1 space ---
        self.heart_mesh, _, self.heart_tags = load_gmsh_mesh(PATHS.heart_msh, MPI.COMM_SELF)
        self.V = cg1_space(self.heart_mesh)
        self.v2d = vertex_to_dof_map(self.V)
        self.heart_points = self.heart_mesh.geometry.x[:, :3].copy()
        self.dof_coords = self.V.tabulate_dof_coordinates()[:, :3].copy()

        # --- endo/mid/epi region marker (1=endo, 2=epi, else mid) ---
        self.region_marker = self._region_marker()

        # --- boundary partitions: full boundary, outer EPI∪BASE, apex seed ---
        part = heart_boundary_partition(self.heart_mesh, self.heart_tags)
        self.outer_vertices = np.asarray(part["outer_vertices"], dtype=np.int64)
        self.union_vertices = np.asarray(part["union_vertices"], dtype=np.int64)
        self.outer_mask_in_union = np.asarray(part["outer_mask"], dtype=bool)
        self.boundary_vertices = self.union_vertices
        #: lowest-z dof, used as the apex ball-stimulus seed (matches the database)
        self.apex_seed = tuple(float(x) for x in self.dof_coords[int(np.argmin(self.dof_coords[:, 2]))])

        # --- torso body surface (for BSP rendering only) ---
        self.torso_mesh, _, self.torso_tags = load_gmsh_mesh(PATHS.torso_msh, MPI.COMM_SELF)
        self.body_vertices = tagged_facet_vertices(self.torso_mesh, self.torso_tags, GAMMA_BODY)

    # -- region marker (vendored expand_layer_biv writes a diagnostic file; run in a tempdir) --
    def _region_marker(self) -> np.ndarray:
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chdir(td)
                region_fun, _ = build_region_field(
                    V=self.V, facet_tags=self.heart_tags, lv_marker=LV, rv_marker=RV,
                    epi_marker=EPI, endo_size=self.params["endo_size"],
                    epi_size=self.params["epi_size"],
                )
                return np.rint(region_fun.x.array).astype(np.int32).copy()
            finally:
                os.chdir(cwd)

    # ------------------------------------------------------------------
    # Render surfaces: (points, faces) triples in their own local indexing
    # ------------------------------------------------------------------
    @cached_property
    def heart_surface(self) -> tuple[np.ndarray, np.ndarray]:
        """Whole heart boundary (for the V_m field)."""
        faces = _triangle_faces(self.heart_mesh, self.heart_tags.indices)
        return self.heart_points, faces

    @cached_property
    def outer_surface(self) -> tuple[np.ndarray, np.ndarray]:
        """EPI∪BASE interface only (for the HSP field; rows align with HSP vectors)."""
        epi_base = self.heart_tags.indices[np.isin(self.heart_tags.values, (EPI, BASE))]
        faces = _triangle_faces(self.heart_mesh, epi_base)
        pts = self.heart_points[self.outer_vertices]
        return pts, _reindex_faces(faces, self.outer_vertices)

    @cached_property
    def body_surface(self) -> tuple[np.ndarray, np.ndarray]:
        """Torso body surface (for the BSP field)."""
        body_facets = self.torso_tags.indices[self.torso_tags.values == GAMMA_BODY]
        faces = _triangle_faces(self.torso_mesh, body_facets)
        pts = self.torso_mesh.geometry.x[self.body_vertices, :3]
        return pts, _reindex_faces(faces, self.body_vertices)

    @property
    def n_outer(self) -> int:
        return int(self.outer_vertices.size)
