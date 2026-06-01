"""Build a mass-weighted POD basis from UKB HSP database samples.

Each ``sample_XXXXX.npz`` written by ``generate_database.py`` contains
``hsp_stack`` with shape ``(n_snapshots, n_boundary_vertices)``. This script
turns those into one snapshot matrix

    X[:, j] = one extracellular-potential boundary snapshot

with shape ``(n_boundary_vertices, n_samples * n_snapshots)``. It then
computes a POD basis that is orthonormal in the surface L2 inner product on
the heart boundary:

    Phi.T @ M @ Phi = I

where ``M`` is the CG1 boundary mass matrix assembled on the union of the
heart mesh boundary tags LV/RV/EPI/BASE.

The mass-weighted POD is computed through the snapshot Gram matrix

    C = X.T @ M @ X

instead of a dense Cholesky of ``M``. This is much cheaper here because the
boundary has about 10k dofs while the snapshot count is about 250 * 6.

Run from the repository root:

    python ukb/database/build_pod_basis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dolfinx
import dolfinx.fem.petsc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import ufl
from mpi4py import MPI
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
UKB_ROOT = HERE.parent
PIPELINE = UKB_ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from common import (  # noqa: E402
    BASE,
    EPI,
    HEART_MSH,
    LV,
    RV,
    cg1_space,
    heart_boundary_partition,
    load_gmsh_mesh,
    tagged_facet_vertices,
    vertex_to_dof_map,
)

DEFAULT_SAMPLES_DIR = HERE / "samples"
DEFAULT_OUTPUT_DIR = HERE / "pod_basis"
# Heart-side boundary tags forming the heart-torso interface (epicardium +
# basal cut). LV/RV endocardia are intentionally excluded — they face blood
# pools that are sealed (no-flux) in the torso Laplace, so their values are
# not part of the ECGi inverse problem.
BOUNDARY_TAGS = (EPI, BASE)
FULL_HEART_TAGS = (LV, RV, EPI, BASE)


def _sample_sort_key(path: Path) -> int:
    """Sort ``sample_00042.npz`` numerically instead of lexicographically."""
    return int(path.stem.split("_")[-1])


def load_snapshot_matrix(
    samples_dir: Path,
    pattern: str,
    outer_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load all sample snapshots and stack them as columns of one matrix.

    When ``outer_mask`` is supplied, rows are sliced to the outer (EPI∪BASE)
    subset of the heart-torso interface — LV/RV endocardial rows are dropped.
    The mask must be in the same row order as ``hsp_stack`` (i.e. sorted
    unique union vertices).

    Returns:
        ``X``:
            Shape ``(n_outer_dofs, n_total_snapshots)``. Column ``j`` is one
            HSP snapshot from one sample, restricted to the outer rows.
        ``hsp_points``:
            Coordinates of the kept rows.
        ``snapshot_times_ms``:
            Time associated with each matrix column.
        ``column_ids``:
            Human-readable column provenance strings.
    """
    sample_paths = sorted(samples_dir.glob(pattern), key=_sample_sort_key)
    if not sample_paths:
        raise FileNotFoundError(f"No sample files matched {samples_dir / pattern}")

    blocks: list[np.ndarray] = []
    all_times: list[np.ndarray] = []
    column_ids: list[str] = []
    hsp_points_ref: np.ndarray | None = None
    n_boundary = None

    for path in sample_paths:
        data = np.load(path)
        hsp_stack = np.asarray(data["hsp_stack"], dtype=np.float64)
        hsp_points = np.asarray(data["hsp_points"], dtype=np.float64)
        times = np.asarray(data["snapshot_times_ms"], dtype=np.float64)

        if hsp_stack.ndim != 2:
            raise ValueError(f"{path}: hsp_stack must be 2D, got {hsp_stack.shape}")
        if times.shape != (hsp_stack.shape[0],):
            raise ValueError(
                f"{path}: snapshot_times_ms shape {times.shape} does not match "
                f"hsp_stack snapshots {hsp_stack.shape[0]}"
            )
        if hsp_points.shape != (hsp_stack.shape[1], 3):
            raise ValueError(
                f"{path}: hsp_points shape {hsp_points.shape} does not match "
                f"hsp_stack boundary size {hsp_stack.shape[1]}"
            )

        if outer_mask is not None and outer_mask.size != hsp_stack.shape[1]:
            raise ValueError(
                f"{path}: outer_mask size {outer_mask.size} does not match "
                f"hsp_stack boundary size {hsp_stack.shape[1]}"
            )

        if outer_mask is not None:
            hsp_stack = hsp_stack[:, outer_mask]
            hsp_points = hsp_points[outer_mask, :]

        if hsp_points_ref is None:
            hsp_points_ref = hsp_points.copy()
            n_boundary = hsp_stack.shape[1]
        else:
            if hsp_stack.shape[1] != n_boundary:
                raise ValueError(f"{path}: boundary size changed from {n_boundary} to {hsp_stack.shape[1]}")
            if not np.allclose(hsp_points, hsp_points_ref, rtol=0.0, atol=1.0e-10):
                raise ValueError(f"{path}: hsp_points differ from the first sample")

        # hsp_stack is (time, boundary). Transpose it so each snapshot is a
        # column and each boundary dof is a row.
        blocks.append(hsp_stack.T)
        all_times.append(times)
        column_ids.extend(f"{path.stem}:t={t:g}ms" for t in times)

    X = np.concatenate(blocks, axis=1)
    snapshot_times_ms = np.concatenate(all_times)
    assert hsp_points_ref is not None
    return X, hsp_points_ref, snapshot_times_ms, column_ids


def assemble_boundary_mass_matrix(hsp_points: np.ndarray, atol: float = 1.0e-10) -> sp.csr_matrix:
    """Assemble the CG1 surface mass matrix on the outer heart-torso interface.

    ``hsp_points`` are the outer (EPI∪BASE) vertex coordinates in sorted
    union-vertex order. The form ``u v`` is integrated over the EPI and BASE
    boundary facets only — LV/RV endocardia are excluded — and the resulting
    matrix is sliced down to the outer DOFs.
    """
    mesh, _, facet_tags = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    V = cg1_space(mesh)

    part = heart_boundary_partition(mesh, facet_tags)
    outer_vertices = part["outer_vertices"]
    outer_points = mesh.geometry.x[outer_vertices, :3]
    if outer_points.shape != hsp_points.shape:
        raise ValueError(
            f"heart outer boundary has {outer_points.shape[0]} vertices, "
            f"but snapshot rows have {hsp_points.shape[0]}"
        )
    max_dist = float(np.max(np.linalg.norm(outer_points - hsp_points, axis=1)))
    if max_dist > atol:
        raise ValueError(f"snapshot row order does not match heart outer boundary; max distance {max_dist:.3e}")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    # Integrate basis_i * basis_j over the outer (EPI∪BASE) heart-torso
    # interface only.
    mass_form = dolfinx.fem.form(sum(ufl.inner(u, v) * ds(tag) for tag in BOUNDARY_TAGS))
    A = dolfinx.fem.petsc.assemble_matrix(mass_form)
    A.assemble()
    indptr, indices, values = A.getValuesCSR()
    n_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    full_mass = sp.csr_matrix((values, indices, indptr), shape=(n_dofs, n_dofs))
    A.destroy()

    # The snapshot rows are outer vertices; for CG1 we need the matching
    # finite-element dofs before slicing the assembled matrix.
    v2d = vertex_to_dof_map(V)
    outer_dofs = v2d[outer_vertices].astype(np.int64)
    M = full_mass[outer_dofs, :][:, outer_dofs].tocsr()

    # Remove tiny numerical asymmetry from PETSc/CSR conversion.
    M = 0.5 * (M + M.T)
    return M.tocsr()


def load_boundary_plot_faces(hsp_points: np.ndarray, atol: float = 1.0e-10) -> np.ndarray:
    """Return outer heart-boundary facets as triangles indexing ``hsp_points`` rows.

    Only EPI and BASE facets are returned — LV/RV endocardia are not part of
    the heart-torso interface and are excluded from the POD basis. Vertex
    indices on the facets are converted to row indices in the outer
    ``hsp_points`` array.
    """
    mesh, _, facet_tags = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)

    part = heart_boundary_partition(mesh, facet_tags)
    outer_vertices = part["outer_vertices"]
    outer_points = mesh.geometry.x[outer_vertices, :3]
    if outer_points.shape != hsp_points.shape:
        raise ValueError(
            f"heart outer boundary has {outer_points.shape[0]} vertices, "
            f"but hsp_points has {hsp_points.shape[0]}"
        )
    max_dist = float(np.max(np.linalg.norm(outer_points - hsp_points, axis=1)))
    if max_dist > atol:
        raise ValueError(f"hsp_points row order does not match heart outer boundary; max distance {max_dist:.3e}")

    vertex_to_row = np.full(mesh.topology.index_map(0).size_local, -1, dtype=np.int64)
    vertex_to_row[outer_vertices] = np.arange(outer_vertices.size, dtype=np.int64)

    faces = []
    for tag in BOUNDARY_TAGS:
        facets = facet_tags.indices[facet_tags.values == tag]
        for facet in facets:
            verts = f2v.links(int(facet))
            rows = vertex_to_row[verts]
            if np.any(rows < 0):
                raise RuntimeError("outer facet contains a vertex outside hsp_points")
            faces.append(rows)
    return np.asarray(faces, dtype=np.int64)


def write_diagnostic_plots(
    *,
    output_dir: Path,
    hsp_points: np.ndarray,
    modes: np.ndarray,
    singular_values: np.ndarray,
    energy_cumulative: np.ndarray,
    energy_plot_modes: int = 100,
) -> None:
    """Write singular-value, energy, and first-mode heart-surface plots."""
    mode_numbers = np.arange(1, singular_values.size + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.plot(mode_numbers, singular_values, color="#1f2937", linewidth=1.2)
    ax.scatter(mode_numbers, singular_values, s=7, color="#1f2937")
    ax.set_yscale("log")
    ax.set_xlabel("POD mode index")
    ax.set_ylabel("singular value")
    ax.set_title("Mass-weighted POD singular values")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "singular_values.png", facecolor="white")
    plt.close(fig)

    n_energy = min(int(energy_plot_modes), energy_cumulative.size)
    energy_modes = mode_numbers[:n_energy]
    energy_values = energy_cumulative[:n_energy]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    ax.plot(energy_modes, energy_values, color="#1f2937", linewidth=1.4)
    for level in (0.9, 0.95, 0.99, 0.999):
        ax.axhline(level, color="#94a3b8", linestyle="--", linewidth=0.8)
        idx = np.searchsorted(energy_values, level)
        if idx < energy_values.size:
            ax.axvline(idx + 1, color="#cbd5e1", linestyle=":", linewidth=0.8)
            ax.text(idx + 1, level, f" {idx + 1}", va="bottom", fontsize=8)
    ax.set_xlim(1, max(1, n_energy))
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("number of POD modes")
    ax.set_ylabel("cumulative energy")
    ax.set_title(f"Retained POD energy, first {n_energy} modes")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pod_modes_vs_energy.png", facecolor="white")
    plt.close(fig)

    faces = load_boundary_plot_faces(hsp_points)
    n_show = min(10, modes.shape[1])
    pts = hsp_points[:, :3]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * (maxs - mins).max()

    fig = plt.figure(figsize=(17, 7), dpi=160)
    for k in range(n_show):
        ax = fig.add_subplot(2, 5, k + 1, projection="3d")
        values = modes[:, k]
        facet_values = values[faces].mean(axis=1)
        vmax = float(np.max(np.abs(facet_values))) or 1.0
        coll = Poly3DCollection(pts[faces], linewidths=0.02, edgecolors="#111827")
        coll.set_array(facet_values)
        coll.set_cmap("RdBu_r")
        coll.set_clim(-vmax, vmax)
        ax.add_collection3d(coll)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=18, azim=-62)
        ax.set_axis_off()
        ax.set_title(f"mode {k + 1}", fontsize=10)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_dir / "first_10_pod_modes_heart_surface.png", facecolor="white")
    plt.close(fig)


def compute_mass_weighted_pod(
    X: np.ndarray,
    M: sp.csr_matrix,
    *,
    rank: int | None = None,
    center: bool = False,
    rtol: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute POD modes from ``C = X.T M X``.

    If ``center`` is true, the column mean is subtracted first and returned.
    The returned modes satisfy ``modes.T @ M @ modes = I`` up to numerical
    roundoff. For eigenpair ``C q_i = lambda_i q_i``, the mode is

        phi_i = X q_i / sqrt(lambda_i)

    which avoids factoring the large boundary mass matrix.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if M.shape != (X.shape[0], X.shape[0]):
        raise ValueError(f"M shape {M.shape} is incompatible with X shape {X.shape}")

    mean = X.mean(axis=1) if center else np.zeros(X.shape[0], dtype=np.float64)
    X_work = X - mean[:, None] if center else X

    MX = M @ X_work
    gram = X_work.T @ MX
    gram = 0.5 * (gram + gram.T)

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Very small eigenvalues are numerically dominated by roundoff. Keeping
    # them produces modes with poor M-orthonormality because we divide by
    # sqrt(lambda). The cutoff is relative to the leading eigenvalue.
    eigvals = np.maximum(eigvals, 0.0)
    total_energy = float(np.sum(eigvals))
    keep = eigvals > (float(eigvals[0]) * rtol if eigvals.size else 0.0)
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    if rank is not None:
        eigvals = eigvals[:rank]
        eigvecs = eigvecs[:, :rank]

    singular_values = np.sqrt(eigvals)
    modes = (X_work @ eigvecs) / singular_values[None, :]

    energy_cumulative = np.cumsum(eigvals) / total_energy if total_energy > 0.0 else np.zeros_like(eigvals)
    return modes, singular_values, energy_cumulative, mean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--sample-glob", default="sample_*.npz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank", type=int, default=None,
                        help="optional truncation rank; default keeps all retained modes")
    parser.add_argument("--rtol", type=float, default=1.0e-8,
                        help="relative eigenvalue cutoff for numerically stable modes")
    parser.add_argument("--center", action="store_true",
                        help="subtract the column mean before POD")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip diagnostic PNG generation")
    parser.add_argument("--energy-plot-modes", type=int, default=100,
                        help="number of leading modes shown in the energy plot")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("computing heart-mesh outer/inner partition")
    heart_mesh, _, heart_facet_tags = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    part = heart_boundary_partition(heart_mesh, heart_facet_tags)
    print(
        f"  union={part['union_vertices'].size}  "
        f"outer (EPI∪BASE)={part['outer_vertices'].size}  "
        f"inner (LV∪RV only)={part['inner_vertices'].size}"
    )

    print(f"loading samples from {args.samples_dir}")
    X, hsp_points, snapshot_times_ms, column_ids = load_snapshot_matrix(
        args.samples_dir, args.sample_glob, outer_mask=part["outer_mask"],
    )
    print(f"snapshot matrix X (outer rows only): {X.shape[0]} boundary dofs x {X.shape[1]} snapshots")

    print("assembling boundary mass matrix")
    M = assemble_boundary_mass_matrix(hsp_points)
    print(f"mass matrix M: {M.shape}, nnz={M.nnz}")

    print("computing mass-weighted POD")
    modes, singular_values, energy_cumulative, mean = compute_mass_weighted_pod(
        X, M, rank=args.rank, center=args.center, rtol=args.rtol,
    )
    gram = modes.T @ (M @ modes)
    ortho_error = float(np.max(np.abs(gram - np.eye(gram.shape[0])))) if modes.size else 0.0
    print(f"computed {modes.shape[1]} modes; max |Phi.T M Phi - I| = {ortho_error:.3e}")

    matrix_path = args.output_dir / "snapshot_matrix.npz"
    basis_path = args.output_dir / "pod_basis.npz"
    mass_path = args.output_dir / "boundary_mass_matrix.npz"
    meta_path = args.output_dir / "pod_basis_meta.json"

    np.savez_compressed(
        matrix_path,
        snapshot_matrix=X,
        hsp_points=hsp_points,
        snapshot_times_ms=snapshot_times_ms,
        column_ids=np.asarray(column_ids, dtype=str),
    )
    sp.save_npz(mass_path, M)
    np.savez_compressed(
        basis_path,
        modes=modes,
        singular_values=singular_values,
        energy_cumulative=energy_cumulative,
        mean=mean,
        hsp_points=hsp_points,
        centered=np.asarray(args.center),
        outer_mask_in_union=part["outer_mask"],
        outer_heart_vertices=part["outer_vertices"],
        union_heart_vertices=part["union_vertices"],
        boundary_tags=np.asarray(BOUNDARY_TAGS, dtype=np.int32),
    )
    meta_path.write_text(json.dumps({
        "samples_dir": str(args.samples_dir),
        "sample_glob": args.sample_glob,
        "heart_mesh": str(HEART_MSH),
        "boundary_tags": list(BOUNDARY_TAGS),
        "boundary_tags_note": "POD lives on EPI∪BASE only (heart-torso interface); LV/RV endo are excluded.",
        "n_union_dofs": int(part["union_vertices"].size),
        "n_boundary_dofs": int(X.shape[0]),
        "n_inner_dofs_excluded": int(part["inner_vertices"].size),
        "n_snapshots_total": int(X.shape[1]),
        "n_modes": int(modes.shape[1]),
        "rank_requested": args.rank,
        "eigenvalue_rtol": args.rtol,
        "centered": bool(args.center),
        "orthonormality_max_abs_error": ortho_error,
        "snapshot_matrix": str(matrix_path),
        "mass_matrix": str(mass_path),
        "pod_basis": str(basis_path),
    }, indent=2))

    if not args.no_plots:
        write_diagnostic_plots(
            output_dir=args.output_dir,
            hsp_points=hsp_points,
            modes=modes,
            singular_values=singular_values,
            energy_cumulative=energy_cumulative,
            energy_plot_modes=args.energy_plot_modes,
        )

    print(f"wrote {matrix_path}")
    print(f"wrote {mass_path}")
    print(f"wrote {basis_path}")
    print(f"wrote {meta_path}")
    if not args.no_plots:
        print(f"wrote {args.output_dir / 'singular_values.png'}")
        print(f"wrote {args.output_dir / 'pod_modes_vs_energy.png'}")
        print(f"wrote {args.output_dir / 'first_10_pod_modes_heart_surface.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
