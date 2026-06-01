"""Shared helpers for the UKB clipped-heart / synthetic-torso pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path

import dolfinx
import numpy as np
from dolfinx.io import gmshio
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[2]
UKB_ROOT = REPO_ROOT / "ukb"
# Single source of truth for the forward-pipeline physical / simulation
# parameters (Boulakia et al.). See load_params() and params.json's "_about".
PARAMS_JSON = Path(__file__).resolve().parent / "params.json"
HEART_MSH = UKB_ROOT / "meshes" / "heart" / "ED_clipped.msh"
TORSO_MSH = UKB_ROOT / "meshes" / "torso" / "p001_ukb_torso.msh"
# UKB_RESULTS_ROOT lets the sweep harness redirect outputs to per-run dirs
# under ukb/experiments/runs/<run_id>/ without clobbering ukb/results/.
RESULTS = Path(os.environ.get("UKB_RESULTS_ROOT", UKB_ROOT / "results"))
STEP_01 = RESULTS / "01_transmembrane"
STEP_02 = RESULTS / "02_extracellular"
STEP_03 = RESULTS / "03_torso"

GAMMA_HEART_D = 1   # EPI + BASE: Dirichlet from u_e (real myocardium)
GAMMA_BODY = 2
GAMMA_HEART_N = 3   # LV/RV valve disks: Neumann no-flux (artificial closures)
TORSO = 10
# Back-compat alias: code written before the heart was closed reads GAMMA_HEART
# and gets the Dirichlet portion (EPI∪BASE), which is what it always wanted.
GAMMA_HEART = GAMMA_HEART_D

LV = 1
RV = 2
EPI = 3
BASE = 4
WALL = 5


def load_params(path: Path = PARAMS_JSON) -> dict:
    """Load the shared forward-pipeline parameters from ``params.json``.

    Keys whose name starts with ``_`` are comments and are dropped. Each
    remaining value can be overridden by an environment variable named after
    the UPPERCASE key (e.g. ``STIM_AMP`` overrides ``stim_amp``), parsed as a
    float -- this is the override layer the sweep harnesses use. The JSON is
    the source of truth for normal runs; edit it once to change every tool.
    """
    with open(path) as f:
        raw = json.load(f)
    params = {k: v for k, v in raw.items() if not k.startswith("_")}
    for key in list(params):
        env = os.environ.get(key.upper())
        if env is not None:
            params[key] = float(env)
    return params


def load_gmsh_mesh(path: Path, comm: MPI.Comm = MPI.COMM_SELF):
    return gmshio.read_from_msh(str(path), comm, 0, gdim=3)


def cg1_space(mesh: dolfinx.mesh.Mesh):
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def vertex_to_dof_map(V: dolfinx.fem.FunctionSpace) -> np.ndarray:
    mesh = V.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    dofs = V.dofmap.list
    n_vertices = mesh.topology.index_map(0).size_local
    out = np.full(n_vertices, -1, dtype=np.int64)
    for c in range(mesh.topology.index_map(tdim).size_local):
        verts = c2v.links(c)
        for v, d in zip(verts, dofs[c]):
            if v < n_vertices:
                out[int(v)] = int(d)
    if np.any(out < 0):
        raise RuntimeError("Some mesh vertices could not be mapped to CG1 dofs")
    return out


def tagged_facet_vertices(mesh: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags, tag: int) -> np.ndarray:
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    facets = facet_tags.indices[facet_tags.values == tag]
    if facets.size == 0:
        return np.empty((0,), dtype=np.int64)
    verts = np.unique(np.concatenate([f2v.links(int(f)) for f in facets]))
    return verts.astype(np.int64)


def heart_boundary_partition(mesh: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags) -> dict:
    """Split heart-mesh boundary vertices into outer (EPI∪BASE) and inner (LV∪RV only).

    Only the outer subset is part of the heart-torso interface used by ECGi
    (the cavity-facing LV/RV endo walls are sealed as no-flux in the torso
    Laplace, matching the Lagracie/Coudière/Weynans formulation). Valve-ring
    vertices shared between BASE and LV/RV stay on the outer side.
    """
    union = np.unique(np.concatenate([
        tagged_facet_vertices(mesh, facet_tags, t) for t in (LV, RV, EPI, BASE)
    ])).astype(np.int64)
    outer = np.unique(np.concatenate([
        tagged_facet_vertices(mesh, facet_tags, t) for t in (EPI, BASE)
    ])).astype(np.int64)
    outer_mask = np.isin(union, outer)
    return {
        "union_vertices": union,
        "outer_vertices": union[outer_mask],
        "inner_vertices": union[~outer_mask],
        "outer_mask": outer_mask,
    }


def torso_gamma_heart_facet_partition(
    torso_mesh: dolfinx.mesh.Mesh,
    torso_facet_tags: dolfinx.mesh.MeshTags,
    torso_vertex_is_outer: np.ndarray,
) -> dict:
    """Split torso GAMMA_HEART facets into outer / inner.

    `torso_vertex_is_outer` has length n_torso_vertices and is True where the
    torso vertex sits on the outer (EPI∪BASE) side of the heart-torso
    interface. A facet is outer iff all 3 vertices are outer; if any is inner,
    the facet is treated as inner (sealed). Returns the facet index arrays
    plus a count of mixed facets for diagnostic purposes.
    """
    fdim = torso_mesh.topology.dim - 1
    torso_mesh.topology.create_connectivity(fdim, 0)
    f2v = torso_mesh.topology.connectivity(fdim, 0)
    gamma_facets = torso_facet_tags.indices[torso_facet_tags.values == GAMMA_HEART]
    outer_facets = []
    inner_facets = []
    mixed = 0
    for f in gamma_facets:
        verts = f2v.links(int(f))
        outer_count = int(np.sum(torso_vertex_is_outer[verts]))
        if outer_count == verts.size:
            outer_facets.append(int(f))
        else:
            inner_facets.append(int(f))
            if outer_count != 0:
                mixed += 1
    return {
        "outer_facets": np.asarray(outer_facets, dtype=np.int32),
        "inner_facets": np.asarray(inner_facets, dtype=np.int32),
        "n_mixed": mixed,
        "n_gamma_facets": int(gamma_facets.size),
    }


def write_json(path: Path, payload: dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
