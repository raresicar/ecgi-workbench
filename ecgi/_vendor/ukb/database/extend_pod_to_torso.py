"""Extend heart-surface POD modes through the torso Laplace problem.

The torso mesh has the heart wall cut out as a hole, but only the outer
(epi + base) part of that hole carries data — the LV/RV cavity-facing walls
are sealed (no-flux), matching the standard ECGi torso model from Lagracie /
Coudière / Weynans (2026).  For each POD mode Phi_i on the outer interface,
we solve

    -div(sigma_T grad u_T) = 0 in Omega_T
                      u_T = Phi_i on Gamma_outer  (EPI∪BASE side of the wall)
       sigma_T grad u_T . n = 0 on Gamma_inner    (LV/RV cavity walls)
       sigma_T grad u_T . n = 0 on Gamma_body

and store both the full torso solution and its body-surface trace.

Each mode is an independent Laplace solve, so ``--n-workers K`` fans the modes
out across K processes (spawn; one BLAS thread each). ``--n-workers 1`` keeps
the original in-order sequential behaviour.

Run from the repository root:

    python ukb/database/extend_pod_to_torso.py [--n-modes N] [--n-workers K]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Pin BLAS/OMP to one thread BEFORE numpy/petsc/dolfinx import so worker
# processes inherit a single thread each (no oversubscription with --n-workers).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dolfinx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
UKB_ROOT = HERE.parent
PIPELINE = UKB_ROOT / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from common import (  # noqa: E402
    GAMMA_BODY,
    GAMMA_HEART,
    HEART_MSH,
    TORSO_MSH,
    cg1_space,
    heart_boundary_partition,
    load_gmsh_mesh,
    tagged_facet_vertices,
    vertex_to_dof_map,
    write_json,
)

DEFAULT_POD_PATH = HERE / "pod_basis" / "pod_basis.npz"
DEFAULT_OUTPUT_DIR = HERE / "extended_pod_basis"
SIGMA_T = 6.0e-4
MATCH_TOL = 1.0e-8


def _translated_hsp_points(hsp_points: np.ndarray) -> np.ndarray:
    alignment_meta = TORSO_MSH.parent / "alignment_meta.json"
    if not alignment_meta.exists():
        return hsp_points
    meta = json.loads(alignment_meta.read_text())
    translation = np.asarray(meta.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    return hsp_points + translation


def _partition_gamma_heart_vertices(
    torso_mesh,
    torso_facet_tags,
    outer_hsp_points: np.ndarray,
    full_union_points: np.ndarray,
    outer_mask_in_union: np.ndarray,
) -> dict:
    """Split torso GAMMA_HEART vertices into outer (Dirichlet) and inner (Neumann).

    Each torso GAMMA_HEART vertex is bijectively matched to a heart-mesh
    union vertex via a cKDTree query against ``full_union_points``. Vertices
    whose heart-mesh partner is on EPI∪BASE go to the outer subset and are
    given a Dirichlet value from the POD mode; the rest stay inner and
    receive natural Neumann (no flux).
    """
    gamma_vertices = tagged_facet_vertices(torso_mesh, torso_facet_tags, GAMMA_HEART)
    gamma_points = torso_mesh.geometry.x[gamma_vertices, :3]

    tree = cKDTree(_translated_hsp_points(full_union_points))
    dist, union_idx = tree.query(gamma_points, k=1)
    max_dist = float(dist.max())
    if max_dist > MATCH_TOL:
        raise RuntimeError(f"GammaHeart/heart-mesh coordinate mismatch: max distance {max_dist:.3e}")
    if np.unique(union_idx).size != union_idx.size:
        raise RuntimeError("GammaHeart to heart-mesh union map is not bijective")

    is_outer = outer_mask_in_union[union_idx]
    outer_gamma_vertices = gamma_vertices[is_outer]
    inner_gamma_vertices = gamma_vertices[~is_outer]
    outer_union_idx = union_idx[is_outer]
    # Convert union-index -> outer-index by cumulative sum on the mask.
    union_to_outer = np.full(outer_mask_in_union.size, -1, dtype=np.int64)
    union_to_outer[outer_mask_in_union] = np.arange(int(outer_mask_in_union.sum()), dtype=np.int64)
    outer_hsp_idx = union_to_outer[outer_union_idx]

    # Cross-check coordinates of the outer matches.
    outer_match_check = float(np.max(np.linalg.norm(
        _translated_hsp_points(outer_hsp_points)[outer_hsp_idx]
        - torso_mesh.geometry.x[outer_gamma_vertices, :3], axis=1,
    ))) if outer_gamma_vertices.size else 0.0
    if outer_match_check > MATCH_TOL:
        raise RuntimeError(f"outer GammaHeart/POD mismatch: max distance {outer_match_check:.3e}")

    return {
        "gamma_vertices": gamma_vertices,
        "outer_gamma_vertices": outer_gamma_vertices,
        "inner_gamma_vertices": inner_gamma_vertices,
        "outer_hsp_idx": outer_hsp_idx.astype(np.int64),
        "max_match_distance": max_dist,
    }


def _dirichlet_on_vertices(V, vertex_indices: np.ndarray, values: np.ndarray):
    v2d = vertex_to_dof_map(V)
    dofs = v2d[vertex_indices].astype(np.int32)
    bc_fun = dolfinx.fem.Function(V, name="u_T_bc_GammaHeart")
    bc_fun.x.array[dofs] = values.astype(bc_fun.x.array.dtype, copy=False)
    bc_fun.x.scatter_forward()
    return dolfinx.fem.dirichletbc(bc_fun, dofs)


def _body_topology(mesh, facet_tags, V) -> dict:
    """Geometry-only body-surface info (independent of any mode solution)."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    v2d = vertex_to_dof_map(V)
    facets = facet_tags.indices[facet_tags.values == GAMMA_BODY]
    faces = np.asarray([f2v.links(int(f)) for f in facets], dtype=np.int64)
    body_vertices = tagged_facet_vertices(mesh, facet_tags, GAMMA_BODY)
    return {
        "faces": faces,                                   # (n_faces, 3) vertex indices
        "body_vertices": body_vertices,
        "body_dofs": v2d[body_vertices].astype(np.int64),
        "body_points": mesh.geometry.x[body_vertices, :3].copy(),
        "v2d": v2d,
    }


def _face_values(values: np.ndarray, faces: np.ndarray, v2d: np.ndarray) -> np.ndarray:
    """Mean of a dof field over each facet's vertices."""
    return values[v2d[faces]].mean(axis=1)


def _set_equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max())
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=18, azim=-62)
    ax.set_axis_off()


def _plot_bspm(mesh, faces: np.ndarray, face_values: np.ndarray, mode_number: int, out_png: Path) -> None:
    pts = mesh.geometry.x[:, :3]
    vmax = float(np.max(np.abs(face_values))) if face_values.size else 1.0
    if vmax == 0.0:
        vmax = 1.0

    fig = plt.figure(figsize=(7.0, 6.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(pts[faces], linewidths=0.025, edgecolors="#1f2937", alpha=0.92)
    coll.set_array(face_values)
    coll.set_cmap("RdBu_r")
    coll.set_clim(-vmax, vmax)
    ax.add_collection3d(coll)
    _set_equal_axes(ax, pts)
    ax.set_title(f"Extended POD mode {mode_number}: BSPM", fontsize=11)
    fig.colorbar(coll, ax=ax, shrink=0.62, label="body-surface potential / POD units")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


def _plot_montage(mesh, faces: np.ndarray, face_values_by_mode: np.ndarray, out_png: Path) -> None:
    pts = mesh.geometry.x[:, :3]
    n_modes = face_values_by_mode.shape[1]
    ncols = 5
    nrows = int(np.ceil(n_modes / ncols))
    global_vmax = float(np.max(np.abs(face_values_by_mode))) if face_values_by_mode.size else 1.0
    if global_vmax == 0.0:
        global_vmax = 1.0

    fig = plt.figure(figsize=(3.2 * ncols, 3.2 * nrows), dpi=170)
    mappable = None
    for j in range(n_modes):
        ax = fig.add_subplot(nrows, ncols, j + 1, projection="3d")
        coll = Poly3DCollection(pts[faces], linewidths=0.015, edgecolors="#1f2937", alpha=0.92)
        coll.set_array(face_values_by_mode[:, j])
        coll.set_cmap("RdBu_r")
        coll.set_clim(-global_vmax, global_vmax)
        ax.add_collection3d(coll)
        _set_equal_axes(ax, pts)
        ax.set_title(f"mode {j + 1}", fontsize=9)
        mappable = coll
    if mappable is not None:
        fig.colorbar(mappable, ax=fig.axes, shrink=0.72, label="body-surface potential / POD units")
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _should_plot(mode_number: int) -> bool:
    return mode_number <= 10 or mode_number % 10 == 0


# ---------------------------------------------------------------------------
# Setup (deterministic; built once in the main process and once per worker)
# ---------------------------------------------------------------------------
def build_setup(pod_path: Path, n_modes: int) -> dict:
    pod = np.load(pod_path)
    modes = np.asarray(pod["modes"], dtype=np.float64)
    hsp_points = np.asarray(pod["hsp_points"], dtype=np.float64)
    singular_values = np.asarray(pod["singular_values"], dtype=np.float64)
    if "outer_mask_in_union" not in pod.files:
        raise ValueError(
            f"{pod_path} is missing 'outer_mask_in_union' — regenerate the POD basis "
            "via ukb/database/build_pod_basis.py to add the EPI∪BASE partition info."
        )
    outer_mask_in_union = np.asarray(pod["outer_mask_in_union"], dtype=bool)
    if n_modes > modes.shape[1]:
        raise ValueError(f"requested {n_modes} modes, but POD basis has only {modes.shape[1]}")

    heart_mesh, _, heart_facet_tags = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    heart_part = heart_boundary_partition(heart_mesh, heart_facet_tags)
    if heart_part["outer_mask"].size != outer_mask_in_union.size or \
       not np.array_equal(heart_part["outer_mask"], outer_mask_in_union):
        raise RuntimeError(
            "Heart-mesh partition disagrees with the POD basis 'outer_mask_in_union'. "
            "Make sure the POD basis was built from the current heart mesh."
        )
    full_union_points = heart_mesh.geometry.x[heart_part["union_vertices"], :3].copy()

    mesh, _, facet_tags = load_gmsh_mesh(TORSO_MSH, MPI.COMM_SELF)
    V = cg1_space(mesh)
    n_torso_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs

    split = _partition_gamma_heart_vertices(
        torso_mesh=mesh, torso_facet_tags=facet_tags,
        outer_hsp_points=hsp_points, full_union_points=full_union_points,
        outer_mask_in_union=outer_mask_in_union,
    )
    body = _body_topology(mesh, facet_tags, V)

    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    a = SIGMA_T * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    L = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) * w * ufl.dx

    return {
        "mesh": mesh, "facet_tags": facet_tags, "V": V, "a": a, "L": L,
        "modes": modes, "hsp_points": hsp_points, "singular_values": singular_values,
        "n_modes": int(n_modes), "n_torso_dofs": int(n_torso_dofs),
        "gamma_vertices": split["gamma_vertices"],
        "outer_gamma_vertices": split["outer_gamma_vertices"],
        "inner_gamma_vertices": split["inner_gamma_vertices"],
        "outer_hsp_idx": split["outer_hsp_idx"],
        "max_match_distance": split["max_match_distance"],
        "body": body,
    }


def process_mode(setup: dict, j: int, output_dir: Path) -> dict:
    """Solve one mode, write its per-mode npz (+ PNG when applicable).

    Returns the small arrays the main process needs to assemble the combined
    outputs (bsp trace + facet values + scalar ranges)."""
    mode_number = j + 1
    V, modes, body = setup["V"], setup["modes"], setup["body"]
    outer_gamma_values = modes[setup["outer_hsp_idx"], j]
    bc = _dirichlet_on_vertices(V, setup["outer_gamma_vertices"], outer_gamma_values)
    u_T = dolfinx.fem.Function(V, name=f"u_T_pod_mode_{mode_number:02d}")
    problem = LinearProblem(
        setup["a"], setup["L"], u=u_T, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-10"},
    )
    problem.solve()
    u_T.x.scatter_forward()
    values = u_T.x.array.copy()

    faces = body["faces"]
    face_values = _face_values(values, faces, body["v2d"])
    bsp_values = values[body["body_dofs"]].copy()

    mode_file = output_dir / f"extended_pod_mode_{mode_number:03d}.npz"
    np.savez(
        mode_file,
        mode_number=np.asarray(mode_number, dtype=np.int64),
        torso_mode=values,
        bsp_mode=bsp_values,
        heart_mode=modes[:, j].copy(),
        hsp_points=setup["hsp_points"],
        torso_body_points=body["body_points"],
        torso_body_vertex_indices=body["body_vertices"],
        torso_body_dofs=body["body_dofs"],
        body_faces=faces,
        body_face_values=face_values,
        singular_value=np.asarray(setup["singular_values"][j], dtype=np.float64),
    )
    plotted = _should_plot(mode_number)
    if plotted:
        _plot_bspm(setup["mesh"], faces, face_values,
                   mode_number, output_dir / f"bspm_extended_pod_mode_{mode_number:02d}.png")
    print(f"mode {mode_number:02d}: wrote {mode_file}"
          + (" and its BSPM png" if plotted else ""), flush=True)
    return {
        "j": j, "mode_number": mode_number,
        "bsp_values": bsp_values, "face_values": face_values,
        "torso_range": [float(values.min()), float(values.max())],
        "bsp_range": [float(bsp_values.min()), float(bsp_values.max())],
        "singular_value": float(setup["singular_values"][j]),
        "plotted": plotted,
    }


# ---- worker plumbing (spawn) ----
_W: dict | None = None


def _init_worker(pod_path_str: str, n_modes: int, output_dir_str: str) -> None:
    global _W
    _W = {"setup": build_setup(Path(pod_path_str), n_modes), "output_dir": Path(output_dir_str)}


def _worker_task(j: int) -> dict:
    assert _W is not None, "_init_worker did not run"
    return process_mode(_W["setup"], j, _W["output_dir"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def extend_modes(pod_path: Path, output_dir: Path, n_modes: int, n_workers: int = 1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading heart mesh from {HEART_MSH}", flush=True)
    print(f"loading torso mesh from {TORSO_MSH}", flush=True)
    setup = build_setup(pod_path, n_modes)
    print(
        f"  torso GAMMA_HEART: {setup['gamma_vertices'].size} verts  "
        f"outer (Dirichlet)={setup['outer_gamma_vertices'].size}  "
        f"inner (Neumann, sealed)={setup['inner_gamma_vertices'].size}",
        flush=True,
    )

    results: list[dict] = []
    n_workers = max(1, int(n_workers))
    if n_workers == 1:
        for j in range(n_modes):
            results.append(process_mode(setup, j, output_dir))
    else:
        print(f"extending {n_modes} modes across {n_workers} workers ...", flush=True)
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx,
            initializer=_init_worker, initargs=(str(pod_path), int(n_modes), str(output_dir)),
        ) as pool:
            futures = {pool.submit(_worker_task, j): j for j in range(n_modes)}
            for fut in as_completed(futures):
                results.append(fut.result())

    # Assemble combined outputs in mode order.
    results.sort(key=lambda r: r["j"])
    body = setup["body"]
    bsp_modes = np.stack([r["bsp_values"] for r in results], axis=1)
    body_face_values = np.stack([r["face_values"] for r in results], axis=1)
    ranges = [{
        "mode": r["mode_number"], "torso_range": r["torso_range"],
        "bsp_range": r["bsp_range"], "singular_value": r["singular_value"],
    } for r in results]
    plotted_modes = [r["mode_number"] for r in results if r["plotted"]]

    np.savez(
        output_dir / "extended_pod_modes.npz",
        bsp_modes=bsp_modes,
        heart_modes=setup["modes"][:, :n_modes].copy(),
        hsp_points=setup["hsp_points"],
        torso_body_points=body["body_points"],
        torso_body_vertex_indices=body["body_vertices"],
        torso_body_dofs=body["body_dofs"],
        body_faces=body["faces"],
        body_face_values=body_face_values,
        singular_values=setup["singular_values"][:n_modes].copy(),
        mode_numbers=np.arange(1, n_modes + 1, dtype=np.int64),
        torso_outer_gamma_vertices=setup["outer_gamma_vertices"],
        torso_inner_gamma_vertices=setup["inner_gamma_vertices"],
        torso_outer_hsp_idx=setup["outer_hsp_idx"],
    )
    plotted_indices = [m - 1 for m in plotted_modes]
    montage_path = output_dir / "bspm_extended_pod_modes_plotted.png"
    if plotted_indices:
        _plot_montage(setup["mesh"], body["faces"], body_face_values[:, plotted_indices], montage_path)
    write_json(output_dir / "extended_pod_modes_meta.json", {
        "pod_basis": str(pod_path),
        "torso_mesh": str(TORSO_MSH),
        "heart_mesh": str(HEART_MSH),
        "n_modes_extended": int(n_modes),
        "n_workers": int(n_workers),
        "sigma_t": SIGMA_T,
        "n_torso_dofs": int(setup["n_torso_dofs"]),
        "n_gamma_heart_vertices": int(setup["gamma_vertices"].size),
        "n_gamma_heart_outer_vertices": int(setup["outer_gamma_vertices"].size),
        "n_gamma_heart_inner_vertices": int(setup["inner_gamma_vertices"].size),
        "gamma_heart_boundary_note": (
            "outer = Dirichlet from POD mode (EPI∪BASE side); "
            "inner = natural Neumann (LV/RV cavity walls sealed)."
        ),
        "n_gamma_body_vertices": int(body["body_vertices"].size),
        "hsp_match_max_distance": setup["max_match_distance"],
        "outputs": {
            "extended_modes_summary": str(output_dir / "extended_pod_modes.npz"),
            "per_mode_files": [str(output_dir / f"extended_pod_mode_{m:03d}.npz")
                               for m in range(1, n_modes + 1)],
            "bspm_montage": str(montage_path),
            "plotted_modes": plotted_modes,
        },
        "mode_ranges": ranges,
    })
    print(f"wrote {output_dir / 'extended_pod_modes.npz'}", flush=True)
    print(f"wrote {output_dir / 'extended_pod_modes_meta.json'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", type=Path, default=DEFAULT_POD_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-modes", type=int, default=10)
    parser.add_argument("--n-workers", type=int, default=1,
                        help="parallel worker processes (each rebuilds the torso setup "
                             "once and solves a subset of modes; one BLAS thread each)")
    args = parser.parse_args(argv)
    extend_modes(args.pod, args.output_dir, int(args.n_modes), int(args.n_workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
