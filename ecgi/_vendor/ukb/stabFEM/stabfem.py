"""Data-enriched stabFEM solver for the ECGi elliptic Cauchy problem.

Implements Method 1 from ``Lucrare_Licenta_Info.pdf`` §4.3: given body-surface
data e on Γ_B, recover the torso potential u and its trace u_H on the heart-
torso interface Γ_H via the stabilised FE saddle system

    L(u, z) = ½ γ_data ‖u − e‖²_{L²(Γ_B)}
            + a(u, z) + ½ S(u) − ½ S*(z) + ½ R_N(u),

where a(u, w) = σ_T (∇u, ∇w), S is a CIP jump penalty on interior facets,
S*(z) = γ_S* ‖∇z‖²_{L²(Ω_T)}, and R_N(u) = γ_reg ‖Q_T u‖²_{H¹(Ω_T)} is the
POD-database regulariser (Q_T = I − P_T projects out the extended-mode
subspace).

Geometry: the torso mesh has the heart wall cut out as a hole, with two
sub-pieces of GAMMA_HEART — outer (EPI∪BASE), where the unknown HSP lives
and z = 0; inner (LV/RV cavity walls), which are sealed (no-flux) and where
z is free. Matches the model of Lagracie, Coudière & Weynans (2026).

Efficient bits:
* MINRES on the symmetric indefinite saddle (no need for GMRES).
* Sparse + low-rank decomposition of the regulariser (Stokes-DB paper §3.3,
  eq. (39) trick): the regulariser is γ_reg H + low-rank correction, with
  the correction applied matrix-free using cached HΞ and the N×N Gram
  G = Ξ^T H Ξ. Per-matvec cost is O(nnz(sparse) + N · n_torso).
* Build-once / solve-many: the matrices depend only on the geometry and
  hyperparameters; only the RHS changes when the BSPM data changes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import ufl
from mpi4py import MPI
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
UKB_ROOT = HERE.parent
UKB_PIPELINE = UKB_ROOT / "pipeline"
if str(UKB_PIPELINE) not in sys.path:
    sys.path.insert(0, str(UKB_PIPELINE))

from common import (  # noqa: E402
    GAMMA_BODY,
    GAMMA_HEART,
    HEART_MSH,
    TORSO_MSH,
    cg1_space,
    heart_boundary_partition,
    load_gmsh_mesh,
    tagged_facet_vertices,
    torso_gamma_heart_facet_partition,
    vertex_to_dof_map,
)

DEFAULT_POD = UKB_ROOT / "database" / "pod_basis" / "pod_basis.npz"
DEFAULT_EXTENDED_DIR = UKB_ROOT / "database" / "extended_pod_basis"
DEFAULT_OUTPUT_DIR = HERE / "results" / "data_enriched_stabfem"
MATCH_TOL = 1.0e-8


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# stores all hyperaparameters
@dataclass
class Parameters:
    """Hyperparameters for the data-enriched stabFEM saddle system."""
    n_modes: int = 75
    sigma_t: float = 6.0e-4
    gamma_data: float = 1.0
    # Smaller stabilisation weights → answer is closer to the unstabilised
    # Cauchy minimiser. The thesis only requires γ_S, γ_S* > 0; values ~1e-6
    # were enough to reach machine precision on the in-basis smoke test.
    gamma_s: float = 1.0e-6
    gamma_s_star: float = 1.0e-6
    gamma_reg: float = 1.0
    # MINRES needs a tight tolerance: the saddle is ill-conditioned, so a
    # loose rtol (1e-6) can leave 10–20% relative error in the recovered HSP.
    minres_rtol: float = 1.0e-10
    minres_maxiter: int = 10000
    # When True, the preconditioner uses an exact sparse LU of the
    # sparse part of the (1,1) and (2,2) blocks plus a Sherman–Morrison–
    # Woodbury correction for the rank-2N regulariser piece. This is much
    # more effective than diagonal Jacobi for larger n_modes.
    use_smw_preconditioner: bool = True


# ---------------------------------------------------------------------------
# Matrix assembly helpers
# ---------------------------------------------------------------------------
# use SciPy CSR because later we use the solver scipy.sparse.linalg.minres
def _csr_from_petsc(A) -> sp.csr_matrix:
    indptr, indices, values = A.getValuesCSR()
    out = sp.csr_matrix((values, indices, indptr), shape=A.getSize()).copy()
    return out

# UFL weak form assembled in SciPy CSR: UFL form -> PETSc matrix -> SciPy CSR
def _assemble(form) -> sp.csr_matrix:
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(form))
    A.assemble()
    out = _csr_from_petsc(A)
    A.destroy() # destroy PETSc matrix to free memory
    return out

# return later
class _SMWBlockInverse:
    """Apply (A_sparse + U S U^T)^{-1} via the Sherman–Morrison–Woodbury formula:
    (A + U S Uᵀ)^{-1} = A^{-1} - A^{-1} U (S^{-1} + Uᵀ A^{-1} U)^{-1} Uᵀ A^{-1}

    A_sparse is factored once (sparse LU); the rank-2N correction is folded in
    by a small (2N × 2N) inverse. Each application is O(nnz(L) + 2N · n).
    Suitable for preconditioning MINRES on the stabFEM (1,1) block.
    """

    def __init__(self, A_sparse: sp.csr_matrix, U: np.ndarray, S: np.ndarray):
        self.lu = spla.splu(A_sparse.tocsc()) # LU factorisation of sparse part
        # we can now call A_sparse^{-1} via self.lu.solve
        # X = A_sparse^{-1} U, column by column.
        self.U = np.ascontiguousarray(U) # ensure C-contiguous for efficient column access in the loop below
        n, k = U.shape # n = number of rows in A_sparse, k = number of columns in U = 2N
        X = np.empty((n, k), dtype=np.float64)
        for j in range(k):
            X[:, j] = self.lu.solve(U[:, j]) # solves A_sparse x = U[:, j] for x, stores in X[:, j]
        self.X = X
        # Inner correction: (S^{-1} + U^T X)^{-1}.
        S_inv = np.linalg.inv(S)
        M_inner = S_inv + U.T @ X
        self.M_inv = np.linalg.inv(M_inner) # small (2N, 2N) inverse

    def __call__(self, v: np.ndarray) -> np.ndarray:
        w = self.lu.solve(v)
        Utw = self.U.T @ w
        return w - self.X @ (self.M_inv @ Utw) # efficient way to apply A_uu^{-1}


# build a MeshTags object for the outer part of Gamma_H
def _build_outer_gamma_meshtags(
    mesh: dolfinx.mesh.Mesh,
    outer_facet_indices: np.ndarray,
    tag_value: int = 100,
) -> dolfinx.mesh.MeshTags:
    """Wrap ``outer_facet_indices`` as a MeshTags so ufl ``ds`` can integrate
    over the outer subset of GAMMA_HEART only."""
    fdim = mesh.topology.dim - 1
    sorted_idx = np.sort(outer_facet_indices.astype(np.int32))
    values = np.full(sorted_idx.size, tag_value, dtype=np.int32)
    return dolfinx.mesh.meshtags(mesh, fdim, sorted_idx, values)

# possible translation between heart mesh coords and torso mesh coords
def _translated_hsp_points(hsp_points: np.ndarray) -> np.ndarray:
    """Apply the heart→torso translation, if recorded next to the torso mesh."""
    import json
    alignment_meta = TORSO_MSH.parent / "alignment_meta.json"
    if not alignment_meta.exists():
        return hsp_points
    meta = json.loads(alignment_meta.read_text())
    translation = np.asarray(meta.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    return hsp_points + translation


# ---------------------------------------------------------------------------
# Body-data loader (accepts several common formats)
# ---------------------------------------------------------------------------
# loads measured body-surface potential data and places it into a full torso DOF vector
def load_body_data(
    path: Path,
    *,
    body_vertices: np.ndarray,
    body_points: np.ndarray,
    n_torso_dofs: int,
    v2d: np.ndarray,
) -> np.ndarray:
    """Load a BSPM file and place its values onto the torso DOF vector.

    Accepted layouts:
      * ``vertex_indices`` + ``values`` (torso mesh vertex indices).
      * ``torso_body_vertex_indices`` + ``bsp_mode`` (extended POD per-mode files).
      * ``points`` + ``values`` (3D coordinates; nearest neighbour to body_vertices).
    Returns the dense torso-DOF vector with zeros outside Γ_B.
    """
    data = np.load(path)
    # these torso mesh vertices have these BSPM values
    if "vertex_indices" in data.files and "values" in data.files: 
        vertices = np.asarray(data["vertex_indices"], dtype=np.int64)
        values = np.asarray(data["values"], dtype=np.float64)
    # used for extended POD modes
    elif "torso_body_vertex_indices" in data.files and "bsp_mode" in data.files:
        vertices = np.asarray(data["torso_body_vertex_indices"], dtype=np.int64)
        values = np.asarray(data["bsp_mode"], dtype=np.float64)
    # 3D coordinates instead of vertex indices + values
    elif "points" in data.files and "values" in data.files:
        points = np.asarray(data["points"], dtype=np.float64)
        values = np.asarray(data["values"], dtype=np.float64)
        tree = cKDTree(body_points)
        dist, idx = tree.query(points, k=1)
        if float(dist.max()) > MATCH_TOL:
            raise RuntimeError(f"body-data point mismatch: max distance {float(dist.max()):.3e}")
        vertices = body_vertices[idx]
    else:
        raise ValueError(
            f"{path} must contain one of: (vertex_indices, values), "
            "(torso_body_vertex_indices, bsp_mode), or (points, values)"
        )

    if vertices.shape != values.shape:
        raise ValueError(f"body-data shape mismatch: {vertices.shape} vs {values.shape}")

    e = np.zeros(n_torso_dofs, dtype=np.float64)
    e[v2d[vertices]] = values # maps vertex indices to FE DOF indices because the vertex index is 
    # not always the same as the CG1 dof index
    return e


# ---------------------------------------------------------------------------
# The system itself: assemble once, solve many.
# ---------------------------------------------------------------------------
# stores all expensive assembled objects -> internal state of solver
@dataclass
class _AssembledOperators:
    A_uu_sparse: sp.csr_matrix # sparse part of primal block
    A_uz: sp.csr_matrix # coupling from z to u 
    A_zu: sp.csr_matrix # coupling from u to z
    A_zz: sp.csr_matrix # dual satbilisation block
    M_body: sp.csr_matrix # data mass matrix
    H: sp.csr_matrix # H^1 matrix = stiffness + mass -> used for R_N
    Psi: np.ndarray # extended POD modes in torso volume: n_torso × N 
    C: sp.csr_matrix # projection from torso vector to POD coeffs: N × n_torso 
    HPsi: np.ndarray # H @ Psi, cached (n_torso × N)
    G: np.ndarray # Psi^T @ H @ Psi, cached (N × N Gram)
    z_free: np.ndarray # list of DOF indices where z is unknown
    outer_gamma_dofs: np.ndarray # CG1 DOF indices on the outer part of the heart boundary
    # z[outer_gamma_dofs] = 0; z[z_free] = solved values
    inner_gamma_dofs: np.ndarray # CG1 DOF indices on the inner part of the heart boundary
    body_dofs: np.ndarray # CG1 DOF indices on the body surface Γ_B
    body_vertices: np.ndarray # mesh vertex indices on the body surface Γ_B
    outer_gamma_vertices: np.ndarray # mesh vertex indices on the outer part of the heart boundary
    hsp_points_outer: np.ndarray # HSP points coordinates loaded from POD modes file
    gamma_match_distance: float # how well do the heart and torso heart bdounary vertices match?
    diag_u: np.ndarray # diagonal of A_uu
    diag_z: np.ndarray # diagonal of A_zz
    params: Parameters = field(default_factory=Parameters) # Parameters object used to build system

# main class for stabFEM solver, used as:
# system = StabFEMSystem(...)
# e = load_body_data(...)
# result = system.solve(e)
class StabFEMSystem:
    """Build the stabFEM saddle once, then call ``solve(e)`` repeatedly.

    Construction is the expensive step (≈ a few seconds for the UKB torso);
    each subsequent ``solve`` runs MINRES on the matrix-free operator.
    """
    # Constructor stores all paths and hyperparams
    def __init__(
        self,
        *,
        pod_path: Path = DEFAULT_POD,
        extended_dir: Path = DEFAULT_EXTENDED_DIR,
        params: Parameters | None = None,
        measured_body_vertices: np.ndarray | None = None,
    ):
        self.params = params or Parameters()
        self.pod_path = Path(pod_path)
        self.extended_dir = Path(extended_dir)
        # Optional partial-surface measurements
        self._measured_body_vertices = (
            None if measured_body_vertices is None
            else np.asarray(measured_body_vertices, dtype=np.int64)
        )
        self._ops: _AssembledOperators | None = None # all assembled objects
        self._linop: spla.LinearOperator | None = None # saddle system op, used by MINRES
        self._precond: spla.LinearOperator | None = None # preconditioner for MINRES
        self._assemble() # calling Constructor means assembling all matrices

    # -----------------------------------------------------------------
    # Assembly
    # -----------------------------------------------------------------
    # this functions assembles all objects
    def _assemble(self) -> None:
        p = self.params # parameters for the system
        print(f"[stabFEM] loading torso mesh: {TORSO_MSH}", flush=True)
        # load torso mesh
        torso_mesh, _, facet_tags = load_gmsh_mesh(TORSO_MSH, MPI.COMM_SELF)
        V = cg1_space(torso_mesh)
        # n is the total number of local DOFs -> size of unknown vector u
        n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        # maps mesh vertex indices to CG1 DOF indices
        v2d = vertex_to_dof_map(V)

        # POD basis + extended modes (already restricted to outer Γ_H by the
        # build_pod_basis / extend_pod_to_torso scripts).
        print(f"[stabFEM] loading POD basis: {self.pod_path}", flush=True)
        # load POD basis
        pod = np.load(self.pod_path)
        if "outer_mask_in_union" not in pod.files:
            raise ValueError(
                f"{self.pod_path} is missing 'outer_mask_in_union'. Rebuild the "
                "POD basis with the current build_pod_basis.py so it includes "
                "the EPI∪BASE partition metadata."
            )
        # outer_mask_in_union: which heart bdry vertices belong to the real heart outer surface?
        outer_mask_in_union = np.asarray(pod["outer_mask_in_union"], dtype=bool)
        hsp_points = np.asarray(pod["hsp_points"], dtype=np.float64)
        # only the first n_modes columns
        pod_modes = np.asarray(pod["modes"][:, :p.n_modes], dtype=np.float64)
        if pod_modes.shape[1] < p.n_modes:
            raise ValueError(
                f"requested {p.n_modes} modes, but POD basis has only {pod_modes.shape[1]}"
            )

        print(f"[stabFEM] loading heart mesh: {HEART_MSH}", flush=True)
        # load heart mesh for consistency check of POD basis
        heart_mesh, _, heart_ft = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
        heart_part = heart_boundary_partition(heart_mesh, heart_ft)
        if not np.array_equal(heart_part["outer_mask"], outer_mask_in_union):
            raise RuntimeError(
                "heart partition disagrees with the POD basis 'outer_mask_in_union'; "
                "rebuild the POD basis with the current heart mesh."
            )
        union_points = heart_mesh.geometry.x[heart_part["union_vertices"], :3].copy()

        # GAMMA_HEART vertex split (outer = Dirichlet of forward = unknown
        # trace of inverse; inner = sealed cavity wall, free in u and in z).
        gamma_vertices = tagged_facet_vertices(torso_mesh, facet_tags, GAMMA_HEART)
        gamma_points = torso_mesh.geometry.x[gamma_vertices, :3]
        # find torso vertices on Gamma_H -> match to heart bdry vertices
        tree = cKDTree(_translated_hsp_points(union_points))
        dist, union_idx = tree.query(gamma_points, k=1)
        # union_idx[i] = index of heart boundary vertex corresponding to torso Γ_H vertex i
        gamma_match = float(dist.max())
        # match is geometrically exact?
        if gamma_match > MATCH_TOL:
            raise RuntimeError(f"GammaHeart/heart-mesh mismatch: max distance {gamma_match:.3e}")
        # match is injective?
        if np.unique(union_idx).size != union_idx.size:
            raise RuntimeError("GammaHeart to heart-mesh map is not bijective")

        is_outer_gamma = outer_mask_in_union[union_idx]
        # torso DOFs on real outer heart bdry
        outer_gamma_vertices = gamma_vertices[is_outer_gamma]
        # torso DOFs on sealed cavity wall
        inner_gamma_vertices = gamma_vertices[~is_outer_gamma]
        outer_gamma_dofs = v2d[outer_gamma_vertices].astype(np.int64)
        inner_gamma_dofs = v2d[inner_gamma_vertices].astype(np.int64)
        # torso outer Gamma_H DOF order - POD mode row order map
        union_to_outer = np.full(outer_mask_in_union.size, -1, dtype=np.int64)
        union_to_outer[outer_mask_in_union] = np.arange(int(outer_mask_in_union.sum()), dtype=np.int64)
        outer_hsp_idx = union_to_outer[union_idx[is_outer_gamma]]

        # classifying outer heart facets
        torso_vertex_is_outer = np.zeros(torso_mesh.topology.index_map(0).size_local, dtype=bool)
        torso_vertex_is_outer[outer_gamma_vertices] = True
        gamma_facet_part = torso_gamma_heart_facet_partition(torso_mesh, facet_tags, torso_vertex_is_outer)
        if gamma_facet_part["n_mixed"] > 0:
            print(
                f"[stabFEM] warning: {gamma_facet_part['n_mixed']} torso GAMMA_HEART facets "
                "have mixed outer/inner vertices and were classified inner (sealed).",
                flush=True,
            )

        # tag outer Γ_H facets so we can integrate over them only: tag value 100
        outer_gamma_meshtags = _build_outer_gamma_meshtags(
            torso_mesh, gamma_facet_part["outer_facets"], tag_value=100,
        )

        # body vertices and dofs - used to compare solution on measured data
        body_vertices = tagged_facet_vertices(torso_mesh, facet_tags, GAMMA_BODY)
        body_dofs = v2d[body_vertices].astype(np.int64)

        print(
            f"[stabFEM] n_torso_dofs={n}  n_outer_gamma={outer_gamma_dofs.size}  "
            f"n_inner_gamma={inner_gamma_dofs.size}  n_body={body_dofs.size}",
            flush=True,
        )

        # olume + surface forms assembly
        u = ufl.TrialFunction(V)
        v_ = ufl.TestFunction(V)
        n_facet = ufl.FacetNormal(torso_mesh)
        K_stiff = _assemble(ufl.inner(ufl.grad(u), ufl.grad(v_)) * ufl.dx) # stiffness matrix
        M_vol = _assemble(u * v_ * ufl.dx) # volume mass matrix
        H = (M_vol + K_stiff).tocsr() # H^1 inner product matrix
        # body surface measure
        ds_body = ufl.Measure("ds", domain=torso_mesh, subdomain_data=facet_tags)
        # heart outer surface measure (tag value 100)
        ds_outer = ufl.Measure("ds", domain=torso_mesh, subdomain_data=outer_gamma_meshtags)
        M_body = _assemble(u * v_ * ds_body(GAMMA_BODY)) # surface mass matrix
        M_gamma_outer = _assemble(u * v_ * ds_outer(100)) # surface mass matrix

        # Data mass matrix. Full Γ_B by default; restricted to an electrode
        # patch when measured_body_vertices is supplied (a body facet counts
        # as measured iff all of its vertices are in the measured set).
        if self._measured_body_vertices is None:
            M_data = M_body # data uses the whole surface
        else:
            n_verts = torso_mesh.topology.index_map(0).size_local
            measured_mask = np.zeros(n_verts, dtype=bool)
            measured_mask[self._measured_body_vertices] = True
            fdim = torso_mesh.topology.dim - 1
            torso_mesh.topology.create_connectivity(fdim, 0)
            f2v = torso_mesh.topology.connectivity(fdim, 0)
            body_facets = facet_tags.indices[facet_tags.values == GAMMA_BODY]
            # mark measured vertices -> select body facets whose all vertices are measured
            patch_facets = np.asarray(
                [int(f) for f in body_facets if measured_mask[f2v.links(int(f))].all()],
                dtype=np.int32,
            )
            # MeshTag for measured facets (tag value 200)
            measured_meshtags = _build_outer_gamma_meshtags(torso_mesh, patch_facets, tag_value=200)
            # measure for measurement surface
            ds_meas = ufl.Measure("ds", domain=torso_mesh, subdomain_data=measured_meshtags)
            M_data = _assemble(u * v_ * ds_meas(200)) # surface mass matrix
            print(
                f"[stabFEM] partial-surface data: {patch_facets.size}/{body_facets.size} "
                f"body facets measured ({self._measured_body_vertices.size} vertices requested)",
                flush=True,
            )

        # CIP jump penalty on interior facets (one-sided faces on ∂Ω contribute
        # nothing to ufl.dS, so this naturally restricts to internal facets).
        # ufl.dS = integration over interior facets
        # CG1 => u is continuous but it's gradient is piecewise constant
        S_jump = _assemble(
            ufl.jump(ufl.grad(u), n_facet) * ufl.jump(ufl.grad(v_), n_facet) * ufl.dS
        )

        # Build C: the (N × n) sparse projection; C: R^n -> R^N
        # maps a torso vector to it's POD coeffs
        # Implementation: phi_gamma[i, k] = φ_i (heart vertex k_outer); place
        # those values at outer_gamma_dofs rows in an n_torso × N matrix Φ.
        # Then (C x)_i = (Φ^T M_Γ_outer x)_i = sum_k φ_i(k) (M_Γ_outer x)[outer_gamma_dofs[k]].
        phi_gamma_outer = pod_modes[outer_hsp_idx, :].T  # (N, n_outer_gamma) -> on torso bdry DOFs
        # Row i: place φ_i(k) at column outer_gamma_dofs[k]. Construct
        # via COO for clarity.
        N = phi_gamma_outer.shape[0]
        n_outer = outer_gamma_dofs.size
        ii = np.repeat(np.arange(N), n_outer)
        jj = np.tile(outer_gamma_dofs, N)
        vv = phi_gamma_outer.reshape(-1)
        Phi_T = sp.csr_matrix((vv, (ii, jj)), shape=(N, n)) # (N, n) sparse
        C = (Phi_T @ M_gamma_outer).tocsr()  # (N, n) sparse

        # extended POD modes on the full torso
        Psi = self._load_extended_modes(p.n_modes, n)
        # we have that Q_T u = u - Psi C u
        # regulariser: R_N(u) = γ_reg ‖Q_T u‖²_H
        # = γ_reg (u - Ψ C u)^T H (u - Ψ C u)
        # = γ_reg (u^T - u^T C^T Ψ^T) H (u - ΨCu)
        # = γ_reg (u^T H u - u^T H Ψ C u - u^T C^T Ψ^T H u + u^T C^T Ψ^T H Ψ C u)
        # = γ_reg (u^T H u - 2 u^T H Ψ C u + u^T C^T Ψ^T H Ψ C u) 
        # same scalar term appears twice since H is symmetric (scalar = scalar^T)
        # cached low-rank pieces for the regulariser -> used multiple times
        HPsi = H @ Psi # (n, N)
        G = Psi.T @ HPsi # (N, N)

        # saddle blocks: unkwnon vector x = [u, z_free]
        # length n + n - n_outer_gamma = 2n - n_outer_gamma
        sigma = p.sigma_t
        K_sigma = sigma * K_stiff # n x n
        A_uz = K_sigma[:, np.zeros(0, dtype=np.int64)]  # placeholder, fixed below
        z_free = np.setdiff1d(np.arange(n, dtype=np.int64), outer_gamma_dofs, assume_unique=False)
        A_uz = K_sigma[:, z_free].tocsr() # n x n_z
        A_zu = K_sigma[z_free, :].tocsr() # n_z x n
        A_zz = (-p.gamma_s_star * K_stiff[z_free, :][:, z_free]).tocsr() # n_z x n_z

        # Sparse part of A_uu, i.e. γ_data M_B + γ_S * S_jump + γ_reg H.
        # The low-rank correction γ_reg (−HPsiC − C^T Psi^T H + C^T G C) is
        # applied matrix-free in the matvec below.
        A_uu_sparse = (
            p.gamma_data * M_data
            + p.gamma_s * S_jump
            + p.gamma_reg * H # only the first term of the regulariser
        ).tocsr()

        # Preconditioner blocks. The dual block is just the negative stiffness
        # restricted to z_free — sparse direct factorisation. The primal block
        # is sparse + rank-2N from the regulariser; we use Sherman–Morrison–
        # Woodbury with a sparse LU of the sparse part so the preconditioner
        # sees the regulariser exactly. Falls back to Jacobi if disabled.
        diag_u = np.asarray(A_uu_sparse.diagonal(), dtype=np.float64)
        diag_z = np.maximum(np.abs(np.asarray(A_zz.diagonal(), dtype=np.float64)), 1.0e-14)
        smw_uu: _SMWBlockInverse | None = None
        zz_lu: spla.SuperLU | None = None
        if p.use_smw_preconditioner:
            # The rank-2N correction is L = U_lr S_lr U_lr^T with
            # U_lr = [HPsi, C^T] and S_lr = γ_reg [[0, -I], [-I, G]].
            # Direct computations give γ_reg (−HPsiC − C^T Psi^T H + C^T G C) = 
            # = U_lr S_lr U_lr^T
            U_lr = np.column_stack([HPsi, C.T.toarray()])
            zero_NN = np.zeros((p.n_modes, p.n_modes), dtype=np.float64)
            neg_I_N = -np.eye(p.n_modes, dtype=np.float64)
            S_lr = p.gamma_reg * np.block([[zero_NN, neg_I_N], [neg_I_N, G]])
            print(f"[stabFEM] factoring sparse (1,1) for SMW preconditioner...", flush=True)
            smw_uu = _SMWBlockInverse(A_uu_sparse, U_lr, S_lr)
            print(f"[stabFEM] factoring (2,2) block (sparse LU)...", flush=True)
            zz_lu = spla.splu(A_zz.tocsc()) # Lu factorisation of dual block


        # store assemled objects in the dataclass for use in matvec and solve
        self._ops = _AssembledOperators(
            A_uu_sparse=A_uu_sparse,
            A_uz=A_uz,
            A_zu=A_zu,
            A_zz=A_zz,
            # M_body holds the DATA mass matrix used by the RHS in solve();
            # equals the full-Γ_B mass unless a measurement patch was set.
            M_body=M_data.tocsr(),
            H=H,
            Psi=np.ascontiguousarray(Psi),
            C=C,
            HPsi=np.ascontiguousarray(HPsi),
            G=np.ascontiguousarray(G),
            z_free=z_free,
            outer_gamma_dofs=outer_gamma_dofs,
            inner_gamma_dofs=inner_gamma_dofs,
            body_dofs=body_dofs,
            body_vertices=body_vertices,
            outer_gamma_vertices=outer_gamma_vertices,
            hsp_points_outer=hsp_points,
            gamma_match_distance=gamma_match,
            diag_u=diag_u,
            diag_z=diag_z,
            params=p,
        )

        # Total system: u (n) + z (|z_free|).
        N_total = n + z_free.size
        gamma_reg = p.gamma_reg

        def matvec(x: np.ndarray) -> np.ndarray:
            ops = self._ops
            xu = x[:n] # u DOFs
            xz = x[n:] # z DOFs
            # u-row: sparse part + low-rank correction + cross block.
            yu = ops.A_uu_sparse @ xu + ops.A_uz @ xz
            # Low-rank: γ_reg (−H Psi C xu − C^T Psi^T H xu + C^T G C xu)
            c = ops.C @ xu                         # (N,)
            HPsi_c = ops.HPsi @ c                  # (n,)
            PsiTH_x = ops.HPsi.T @ xu              # (N,)
            Gc = ops.G @ c                         # (N,)
            yu += gamma_reg * (-HPsi_c - (ops.C.T @ PsiTH_x) + (ops.C.T @ Gc))
            # z-row: A_zu xu + A_zz xz.
            yz = ops.A_zu @ xu + ops.A_zz @ xz
            return np.concatenate([yu, yz])

        # wrap the matvec as a LinearOperator for use in MINRES
        self._linop = spla.LinearOperator((N_total, N_total), matvec=matvec, dtype=np.float64)

        # preconditioner matvec
        if smw_uu is not None and zz_lu is not None:
            def precond_matvec(x, _smw=smw_uu, _zz=zz_lu, _n=n):
                yu = _smw(x[:_n]) # SMW for primal block
                yz = -_zz.solve(x[_n:])  # |A_zz|^{-1} = −A_zz^{-1} (A_zz negative def.) -> positive for MINRES precond.
                return np.concatenate([yu, yz])
            # P^{-1} =
            # [ A_uu^{-1}      0       ]
            # [    0       (-A_zz)^{-1}]
        else: # fallback -> in case gamma_reg = 0 or preconditioner disabled -> Jacobi
            diag_full = np.concatenate([
                np.where(np.abs(diag_u) > 1.0e-14, diag_u, 1.0),
                diag_z,
            ])
            def precond_matvec(x, _d=diag_full):
                return x / _d # every element is scaled by the inverse diagonal entry
        self._precond = spla.LinearOperator( # wrap as a LinearOperator for MINRES
            (N_total, N_total),
            matvec=precond_matvec,
            dtype=np.float64,
        )
        self._n_torso = n
        self._v2d = v2d
        self._mesh = torso_mesh
        self._facet_tags = facet_tags
        self._body_points = torso_mesh.geometry.x[body_vertices, :3].copy()

    # loads extended POD modes from .npz files
    def _load_extended_modes(self, n_modes: int, n_torso_dofs: int) -> np.ndarray:
        out = np.zeros((n_torso_dofs, n_modes), dtype=np.float64)
        for j in range(n_modes):
            path = self.extended_dir / f"extended_pod_mode_{j + 1:03d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"missing extended POD mode: {path}")
            data = np.load(path)
            mode = np.asarray(data["torso_mode"], dtype=np.float64)
            if mode.shape != (n_torso_dofs,):
                raise ValueError(f"{path}: torso_mode shape {mode.shape}, expected {(n_torso_dofs,)}")
            out[:, j] = mode
        return out

    # -----------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------
    def solve(self, e: np.ndarray, *, rtol: float | None = None, maxiter: int | None = None) -> dict:
        """Run MINRES on the saddle for the given body-data vector e.

        ``e`` is a full-length torso-DOF vector with the BSPM values placed at
        body DOFs and zeros elsewhere. Use :func:`load_body_data` to build it.
        """
        ops = self._ops
        if ops is None:
            raise RuntimeError("system was not assembled")
        if e.shape != (self._n_torso,):
            raise ValueError(f"e shape {e.shape}; expected ({self._n_torso},)")
        p = self.params
        rtol = float(p.minres_rtol if rtol is None else rtol)
        maxiter = int(p.minres_maxiter if maxiter is None else maxiter)

        # assemble RHS: [γ_data M_B e; 0]
        rhs_u = p.gamma_data * (ops.M_body @ e)
        rhs = np.concatenate([rhs_u, np.zeros(ops.z_free.size, dtype=np.float64)])

        residuals: list[float] = []
        iters = {"n": 0}

        def _cb(xk):
            iters["n"] += 1
            # Cheap residual proxy: |b - A xk| / |b|.
            r = rhs - self._linop @ xk
            residuals.append(float(np.linalg.norm(r) / max(np.linalg.norm(rhs), 1.0e-30)))

        # MINRES solve. Returns the full solution vector [u, z_free] and info flag
        # uses the two LinearOperators defined in _assemble for the matrix and preconditioner
        sol, info = spla.minres(
            self._linop, rhs, M=self._precond,
            rtol=rtol, maxiter=maxiter, callback=_cb,
        )

        u = sol[:self._n_torso] # primal solution: torso DOFs
        z_red = sol[self._n_torso:] # dual solution at free DOFs; z[outer_gamma_dofs] = 0 by construction
        z = np.zeros(self._n_torso, dtype=np.float64) # full dual vector, with zeros at outer Gamma_H DOFs and solved values at free DOFs
        z[ops.z_free] = z_red # place solved dual values at free DOFs; rest are zero by construction

        # Diagnostics on the POD-coefficient breakdown.
        c = ops.C @ u
        proj = ops.Psi @ c
        residual_h1 = u - proj

        # For comparison to data: torso potentials at HSP locations, torso potentials at body vertices vs data
        hsp_recovered = u[ops.outer_gamma_dofs]
        body_reconstructed = u[ops.body_dofs]
        body_data = e[ops.body_dofs]
        body_residual = body_reconstructed - body_data

        return {
            "u": u,
            "z": z,
            "info": int(info),
            "iterations": int(iters["n"]),
            "minres_residuals": np.asarray(residuals, dtype=np.float64),
            "pod_coefficients": c,
            "projected_torso": proj,
            "regularization_residual": residual_h1,
            "regularization_value": float(p.gamma_reg * residual_h1 @ (ops.H @ residual_h1)),
            "hsp_recovered": hsp_recovered,
            "body_reconstructed": body_reconstructed,
            "body_data": body_data,
            "body_residual": body_residual,
        }

    # -----------------------------------------------------------------
    # Public accessors (mostly for downstream scripts)
    # -----------------------------------------------------------------
    # these properties allow access to the assembled operators and metadata for use in downstream analysis 
    # and visualization scripts without exposing the internal state of the solver directly
    @property
    def ops(self) -> _AssembledOperators:
        if self._ops is None:
            raise RuntimeError("system was not assembled")
        return self._ops

    @property
    def n_torso(self) -> int:
        return self._n_torso

    @property
    def v2d(self) -> np.ndarray:
        return self._v2d

    @property
    def body_points(self) -> np.ndarray:
        return self._body_points


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--body-data", type=Path, required=True,
                        help="NPZ with the BSPM (vertex_indices/values, "
                             "torso_body_vertex_indices/bsp_mode, or points/values).")
    parser.add_argument("--pod", type=Path, default=DEFAULT_POD)
    parser.add_argument("--extended-dir", type=Path, default=DEFAULT_EXTENDED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-modes", type=int, default=75)
    parser.add_argument("--sigma-t", type=float, default=6.0e-4)
    parser.add_argument("--gamma-data", type=float, default=1.0)
    parser.add_argument("--gamma-s", type=float, default=1.0e-3)
    parser.add_argument("--gamma-s-star", type=float, default=1.0e-3)
    parser.add_argument("--gamma-reg", type=float, default=1.0)
    parser.add_argument("--minres-rtol", type=float, default=1.0e-8)
    parser.add_argument("--minres-maxiter", type=int, default=2000)
    args = parser.parse_args(argv)

    params = Parameters(
        n_modes=args.n_modes,
        sigma_t=args.sigma_t,
        gamma_data=args.gamma_data,
        gamma_s=args.gamma_s,
        gamma_s_star=args.gamma_s_star,
        gamma_reg=args.gamma_reg,
        minres_rtol=args.minres_rtol,
        minres_maxiter=args.minres_maxiter,
    )
    system = StabFEMSystem(pod_path=args.pod, extended_dir=args.extended_dir, params=params)
    ops = system.ops

    e = load_body_data(
        args.body_data,
        body_vertices=ops.body_vertices,
        body_points=system.body_points,
        n_torso_dofs=system.n_torso,
        v2d=system.v2d,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("[stabFEM] solving saddle system via MINRES", flush=True)
    result = system.solve(e)
    print(
        f"[stabFEM] info={result['info']}  iterations={result['iterations']}  "
        f"final residual={result['minres_residuals'][-1] if result['minres_residuals'].size else float('nan'):.3e}",
        flush=True,
    )

    np.savez(
        out / "solution.npz",
        u_torso=result["u"],
        z_lagrange=result["z"],
        hsp_recovered=result["hsp_recovered"],
        hsp_outer_vertex_indices=ops.outer_gamma_vertices,
        hsp_outer_dofs=ops.outer_gamma_dofs,
        body_vertex_indices=ops.body_vertices,
        body_dofs=ops.body_dofs,
        body_data=result["body_data"],
        body_reconstructed=result["body_reconstructed"],
        body_residual=result["body_residual"],
        pod_coefficients=result["pod_coefficients"],
        projected_torso=result["projected_torso"],
        regularization_residual=result["regularization_residual"],
        minres_residuals=result["minres_residuals"],
    )
    meta = {
        "method": "data-enriched stabFEM (MINRES)",
        "body_data": str(args.body_data),
        "pod_basis": str(args.pod),
        "extended_pod_dir": str(args.extended_dir),
        "torso_mesh": str(TORSO_MSH),
        "heart_mesh": str(HEART_MSH),
        "parameters": params.__dict__,
        "n_torso_dofs": int(system.n_torso),
        "n_outer_gamma_dofs": int(ops.outer_gamma_dofs.size),
        "n_inner_gamma_dofs": int(ops.inner_gamma_dofs.size),
        "n_body_dofs": int(ops.body_dofs.size),
        "gamma_heart_match_max_distance": float(ops.gamma_match_distance),
        "minres_info": int(result["info"]),
        "minres_iterations": int(result["iterations"]),
        "minres_final_residual": float(result["minres_residuals"][-1]) if result["minres_residuals"].size else None,
        "regularization_value": float(result["regularization_value"]),
        "body_residual_rms": float(np.sqrt(np.mean(result["body_residual"] ** 2))),
        "u_range": [float(result["u"].min()), float(result["u"].max())],
        "hsp_recovered_range": [float(result["hsp_recovered"].min()), float(result["hsp_recovered"].max())],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[stabFEM] wrote {out / 'solution.npz'} and {out / 'meta.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
