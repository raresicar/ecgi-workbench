"""Step 3 of the heart–torso uncoupled pipeline: torso forward problem (3D rabbit).

Given u_e on the heart,
solve the generalised Laplace problem in the passive conductor Ω_T:

    div(σ_T ∇ u_T)     = 0      in  Ω_T
                 u_T   = u_e    on  Γ_H     (heart-torso interface)
   σ_T ∇ u_T · n_T    = 0       on  Γ_B  (skin)

The 3D rabbit geometry from Zenodo 6340066 has a structural feature we
exploit ruthlessly: ``Mesh_Ven.vtu`` is *embedded* inside
``Mesh_Torso.vtu`` with byte-for-byte identical vertex coordinates in the
heart region (verified at load time via
:func:`load_meshes.build_ventricle_to_torso_map`). That means the
Dirichlet trace transfer from ventricles to torso is a *pure integer
permutation* — no spatial interpolation, no projection between function
spaces, no 2D-style angular hack. We just look up ``u_e[i]`` and write it
to ``u_T[ven2torso[i]]``.

Bookkeeping decision: we solve Laplace on the *full* torso mesh and impose
Dirichlet conditions at every vertex shared with the ventricle, not just
the epicardial interface. Mathematically equivalent to carving out a body
submesh, much simpler to implement. The heart-region dofs are simply
forced to their ventricular u_e values; the elliptic solver fills in u_T
everywhere else.

Module layout:

  * :func:`build_vertex_to_dof_map`      — CG1 helper: vertex index → dof
                                           index on a mesh.
  * :func:`heart_region_dirichlet_bc`    — Dirichlet BC pinning every
                                           torso vertex that has a
                                           ventricle counterpart, fed by a
                                           callback or a precomputed array.
  * :func:`solve_torso_potential`        — assembled Laplace solve.
  * :func:`outer_skin_dofs`              — body-surface CG1 dof indices
                                           (your "many electrodes" set).
  * :func:`sample_at_dofs`               — generic dof-array extractor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import dolfinx
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# Rabbit torso background conductivity from Moss et al. Table 1 (S/m). The
# validation path is homogeneous until material-tag mapping is implemented.
SIGMA_T_DEFAULT = 0.035


# ---------------------------------------------------------------------------
# CG1 vertex ↔ dof map
# ---------------------------------------------------------------------------
def build_vertex_to_dof_map(
    V: dolfinx.fem.FunctionSpace,
) -> np.ndarray:
    """Return ``v2d[i]`` = CG1 dof index of mesh vertex ``i``.

    For a degree-1 Lagrange space on a tet mesh, dofs sit at mesh vertices —
    but the dof ordering is *not* in general the same as the vertex
    ordering. We rebuild the bijection from the cell→vertices and
    cell→dofs connectivities: walking one cell tells us which dof lives at
    which vertex (since both share the cell-local index 0,…,3).

    Local-only: returns the mapping for vertices owned by *this* rank. The
    array has length ``mesh.topology.index_map(0).size_local``.
    """
    mesh = V.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    cells_array = c2v.array
    cells_offsets = c2v.offsets

    dofs_array = V.dofmap.list  # (n_cells, n_dofs_per_cell) for tet CG1 -> 4 cols

    n_vert_local = mesh.topology.index_map(0).size_local
    v2d = np.full(n_vert_local, -1, dtype=np.int64)

    n_cells = mesh.topology.index_map(tdim).size_local
    for c in range(n_cells):
        verts = cells_array[cells_offsets[c]:cells_offsets[c + 1]]
        dofs = dofs_array[c]
        for v, d in zip(verts, dofs):
            if v < n_vert_local:
                v2d[v] = d
    # Should be fully populated for any reasonable mesh.
    if (v2d == -1).any():
        n_miss = int((v2d == -1).sum())
        raise RuntimeError(
            f"build_vertex_to_dof_map: {n_miss} local vertices have no "
            "associated CG1 dof — mesh has isolated vertices?"
        )
    return v2d


# ---------------------------------------------------------------------------
# Dirichlet from a precomputed (vertex-index, value) pair
# ---------------------------------------------------------------------------
def heart_region_dirichlet_bc(
    V_torso: dolfinx.fem.FunctionSpace,
    torso_vertex_indices: np.ndarray,
    values: np.ndarray,
) -> dolfinx.fem.DirichletBC:
    """Build a Dirichlet BC pinning u_T on a set of torso vertices.

    Args:
        V_torso: CG1 function space on the torso mesh.
        torso_vertex_indices: ``(K,)`` int array of torso vertex indices
            (e.g. ``ven2torso`` for all heart vertices, or
            ``ven2torso[epi_mask]`` for just the epicardium).
        values: ``(K,)`` float array of Dirichlet values at those vertices.

    Implementation: looks up the CG1 dof index for each vertex and
    constructs a :class:`dolfinx.fem.dirichletbc`.
    """
    if torso_vertex_indices.shape != values.shape:
        raise ValueError(
            f"Index/value shape mismatch: "
            f"{torso_vertex_indices.shape} vs {values.shape}"
        )

    v2d = build_vertex_to_dof_map(V_torso)
    # Only keep vertices owned by this rank; in parallel, each rank pins
    # only the dofs it owns and PETSc broadcasts the rest.
    n_local = v2d.size
    owned_mask = torso_vertex_indices < n_local
    dofs = v2d[torso_vertex_indices[owned_mask]].astype(np.int32)
    vals_owned = values[owned_mask].astype(dolfinx.default_scalar_type)

    bc_fun = dolfinx.fem.Function(V_torso, name="u_T_bc")
    bc_fun.x.array[dofs] = vals_owned
    bc_fun.x.scatter_forward()
    return dolfinx.fem.dirichletbc(bc_fun, dofs)


# ---------------------------------------------------------------------------
# Generalised Laplace solve on Ω_T
# ---------------------------------------------------------------------------
def solve_torso_potential(
    torso_mesh: dolfinx.mesh.Mesh,
    heart_vertex_indices: np.ndarray,
    heart_values: np.ndarray,
    sigma_T: float = SIGMA_T_DEFAULT,
) -> dolfinx.fem.Function:
    """Solve div(σ_T ∇ u_T) = 0 in Ω_T with Dirichlet on the heart region.

    Args:
        torso_mesh: the full 3D torso mesh (Mesh_Torso.vtu loaded).
        heart_vertex_indices: ``(K,)`` torso vertex indices to pin.
            Typically ``ven2torso`` (all ventricle vertices, K=508 987) or
            a subset like ``ven2torso[epi_mask]``.
        heart_values: ``(K,)`` Dirichlet values to assign there.
            For the standard heart→torso forward solve this is just the
            ventricular u_e dof array (after the
            :func:`build_vertex_to_dof_map` permutation, if u_e is held on
            a CG1 space on the ventricle mesh).
        sigma_T: scalar isotropic torso conductivity. Homogeneous over all
            14 tissue tags for now — see SIGMA_T_DEFAULT comment.

    Γ_B (the outer skin) gets the natural condition σ_T ∇ u_T · n = 0
    for free (no surface term added to the weak form).

    Returns the CG1 :class:`dolfinx.fem.Function` u_T on the torso mesh.
    """
    V = dolfinx.fem.functionspace(torso_mesh, ("Lagrange", 1))
    bc = heart_region_dirichlet_bc(V, heart_vertex_indices, heart_values)

    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    a = ufl.inner(sigma_T * ufl.grad(u), ufl.grad(w)) * ufl.dx
    L = dolfinx.fem.Constant(torso_mesh, 0.0) * w * ufl.dx

    u_T = dolfinx.fem.Function(V, name="u_T")
    problem = LinearProblem(
        a, L, u=u_T, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "ksp_rtol": 1.0e-10,
        },
    )
    problem.solve()
    return u_T


# ---------------------------------------------------------------------------
# Body-surface (outer-skin) measurement extraction
# ---------------------------------------------------------------------------
def outer_skin_dofs(
    V_torso: dolfinx.fem.FunctionSpace,
    skin_vertex_indices: np.ndarray,
) -> np.ndarray:
    """Return CG1 dof indices for the given outer-skin torso vertices.

    Wraps :func:`build_vertex_to_dof_map` and slices it down to the skin
    vertex set returned by
    :func:`load_meshes.outer_torso_skin_vertices`. Use the result to read
    body-surface potentials from a solved ``u_T`` Function:

        skin_dofs = outer_skin_dofs(V, skin_vertex_indices)
        bsp = u_T.x.array[skin_dofs]
    """
    v2d = build_vertex_to_dof_map(V_torso)
    n_local = v2d.size
    owned_mask = skin_vertex_indices < n_local
    return v2d[skin_vertex_indices[owned_mask]].astype(np.int64)


def sample_at_dofs(
    u: dolfinx.fem.Function,
    dof_indices: np.ndarray,
) -> np.ndarray:
    """Plain numpy slice into a Function's dof array. Provided so callers
    don't have to remember the ``.x.array`` attribute path.
    """
    return u.x.array[dof_indices]
