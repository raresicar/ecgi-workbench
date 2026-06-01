"""Held-out infarction reconstruction with the data-enriched stabFEM solver.

End-to-end test for the infarction experiment:

1. Synthesise (or reuse) an **off-grid** infarction with ``generate.py --heldout``
   (a random visible-wall centre + radius, NOT on the training FPS grid).
2. Pick the most informative HSP snapshot, restrict it to the outer EPI∪BASE
   interface, and forward it through the torso (mixed-BC Laplace: outer Γ_H
   Dirichlet, inner Γ_H sealed Neumann, body Neumann) to a clean BSPM.
3. Add Gaussian noise at the requested levels (fraction of BSPM RMS).
4. Reconstruct the HSP with ``StabFEMSystem`` using THIS experiment's POD basis,
   and compare the recovered HSP to the held-out truth (rel-L2, cosine), with a
   true-vs-recovered surface figure per noise level.

Measurement region: ``--measured-region full`` uses the whole body surface (no
change to stabfem.py). ``--measured-region <vertices.npy>`` or ``anterior``
restricts the data term to a body-facet subset, which requires the
``measured_body_vertices`` option in ``StabFEMSystem`` (added separately).

Run from the repository root inside the scientific-python conda env.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
UKB_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
DATABASE = HERE.parent
sys.path.insert(0, str(UKB_ROOT / "pipeline"))
sys.path.insert(0, str(UKB_ROOT / "stabFEM"))
sys.path.insert(0, str(DATABASE))

from common import (  # noqa: E402
    GAMMA_BODY, GAMMA_HEART, HEART_MSH, TORSO_MSH,
    cg1_space, heart_boundary_partition, load_gmsh_mesh,
    tagged_facet_vertices, vertex_to_dof_map,
)
from stabfem import (  # noqa: E402
    MATCH_TOL, Parameters, StabFEMSystem, _translated_hsp_points,
)
from build_pod_basis import load_boundary_plot_faces  # noqa: E402

DEFAULT_POD = HERE / "pod_basis" / "pod_basis.npz"
DEFAULT_EXTENDED = HERE / "extended_pod_basis"
DEFAULT_OUTPUT = HERE / "results"
GENERATOR = HERE / "generate.py"


# ---------------------------------------------------------------------------
# Held-out sample
# ---------------------------------------------------------------------------
def generate_heldout(seed: int, out_dir: Path, overwrite: bool) -> Path:
    sample = out_dir / "sample_00000.npz"
    if sample.exists() and not overwrite:
        print(f"[inverse] reusing held-out sample {sample}", flush=True)
        return sample
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(GENERATOR), "--heldout", "--base-seed", str(seed),
           "--output-dir", str(out_dir), "--overwrite", "--n-workers", "1", "--temp-csv", ""]
    print(f"[inverse] generating off-grid held-out infarct (seed={seed})", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not sample.exists():
        raise FileNotFoundError(sample)
    return sample


# ---------------------------------------------------------------------------
# Torso partition (outer/inner Γ_H + body) shared by forward and inverse
# ---------------------------------------------------------------------------
def build_partition_maps(torso_mesh, torso_ft) -> dict:
    heart_mesh, _, heart_ft = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    part = heart_boundary_partition(heart_mesh, heart_ft)
    om = np.asarray(part["outer_mask"], dtype=bool)
    union_pts = heart_mesh.geometry.x[part["union_vertices"], :3].copy()

    gv = tagged_facet_vertices(torso_mesh, torso_ft, GAMMA_HEART)
    gp = torso_mesh.geometry.x[gv, :3]
    tree = cKDTree(_translated_hsp_points(union_pts))
    dist, uidx = tree.query(gp, k=1)
    if float(dist.max()) > MATCH_TOL:
        raise RuntimeError(f"GammaHeart/heart-mesh mismatch: max distance {float(dist.max()):.3e}")
    is_outer = om[uidx]
    u2o = np.full(om.size, -1, dtype=np.int64)
    u2o[np.where(om)[0]] = np.arange(int(om.sum()), dtype=np.int64)
    return {
        "outer_gamma_vertices": gv[is_outer],
        "outer_hsp_idx": u2o[uidx[is_outer]],
        "outer_mask_in_union": om,
        "body_vertices": tagged_facet_vertices(torso_mesh, torso_ft, GAMMA_BODY),
        "heart_mesh": heart_mesh,
        "heart_outer_points": heart_mesh.geometry.x[part["outer_vertices"], :3].copy(),
        "heart_facet_tags": heart_ft,
    }


def forward_hsp_to_bspm(hsp_on_outer, outer_hsp_idx, *, torso_mesh, outer_gamma_vertices,
                        body_vertices, sigma_t=6.0e-4) -> dict:
    """−div(σ_T ∇u)=0 with u=HSP on outer Γ_H, Neumann elsewhere → body trace."""
    V = cg1_space(torso_mesh)
    v2d = vertex_to_dof_map(V)
    outer_dofs = v2d[outer_gamma_vertices].astype(np.int32)
    body_dofs = v2d[body_vertices].astype(np.int64)

    bc_fun = dolfinx.fem.Function(V, name="heldout_outer_bc")
    bc_fun.x.array[outer_dofs] = hsp_on_outer[outer_hsp_idx].astype(bc_fun.x.array.dtype, copy=False)
    bc_fun.x.scatter_forward()
    bc = dolfinx.fem.dirichletbc(bc_fun, outer_dofs)

    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = sigma_t * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    L = dolfinx.fem.Constant(torso_mesh, dolfinx.default_scalar_type(0.0)) * w * ufl.dx
    u_T = dolfinx.fem.Function(V, name="u_T_heldout")
    LinearProblem(a, L, u=u_T, bcs=[bc],
                  petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-10"}).solve()
    u_T.x.scatter_forward()
    return {"bsp_values": u_T.x.array[body_dofs].copy(), "body_dofs": body_dofs}


# ---------------------------------------------------------------------------
# Plot: true vs recovered HSP on the outer heart surface
# ---------------------------------------------------------------------------
def plot_recon(hsp_points, faces, truth, recon, infarct_centre, out_png, title,
               infarct_radius=None, frame_pts=None) -> None:
    err = recon - truth
    vmax = float(np.max(np.abs(np.concatenate([truth, recon])))) or 1.0
    pts = hsp_points[:, :3]
    # Axis limits from the FULL heart-mesh bounding box, EXACTLY as the pipeline
    # step plots (e.g. 01_transmembrane _plot_surface) do — so the framing is
    # identical. frame_pts = mesh.geometry.x[:, :3]; falls back to the surface pts.
    box = (frame_pts[:, :3] if frame_pts is not None else pts)
    mins, maxs = box.min(axis=0), box.max(axis=0)
    ctr, rad = 0.5 * (mins + maxs), 0.5 * (maxs - mins).max()
    have_inf = infarct_centre is not None and np.all(np.isfinite(infarct_centre))
    # FIXED camera = the GIF view used everywhere (generator infarct_centres.png,
    # V_m GIFs, pipeline step plots). Infarcts are placed on this visible wall,
    # so the scar is always in frame and every plot is directly comparable.
    elev, azim = 18.0, -62.0
    # Scar footprint as a per-facet mask. We mark it through THIS SAME collection's
    # edge colours — a separate overlaid Poly3DCollection is unreliably composited
    # by matplotlib's 3D painter (it can vanish or render on the wrong face).
    scar_mask = None
    if have_inf and infarct_radius:
        foot_v = np.linalg.norm(pts - np.asarray(infarct_centre), axis=1) <= infarct_radius
        if foot_v.any():
            m = np.array([bool(foot_v[f].all()) for f in faces])
            scar_mask = m if m.any() else None
    panels = [("truth HSP", truth, vmax, "RdBu_r"),
              ("recovered HSP", recon, vmax, "RdBu_r"),
              ("error", err, float(np.max(np.abs(err))) or 1.0, "PuOr_r")]
    fig = plt.figure(figsize=(16, 5.5), dpi=160)
    for i, (lab, vals, vm, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        fv = vals[faces].mean(axis=1)
        coll = Poly3DCollection(pts[faces])
        coll.set_array(fv); coll.set_cmap(cmap); coll.set_clim(-vm, vm)
        if scar_mask is not None:
            ec = np.zeros((faces.shape[0], 4))           # transparent edges everywhere
            ec[scar_mask] = [0.09, 0.49, 0.09, 1.0]      # except a green scar marker
            lw = np.zeros(faces.shape[0]); lw[scar_mask] = 1.8
            coll.set_edgecolor(ec); coll.set_linewidth(lw)
        else:
            coll.set_edgecolor("none")
        ax.add_collection3d(coll)
        ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad); ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
        ax.view_init(elev=elev, azim=azim); ax.set_axis_off(); ax.set_title(lab, fontsize=11)
        fig.colorbar(coll, ax=ax, shrink=0.6)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", type=Path, default=DEFAULT_POD)
    ap.add_argument("--extended-dir", type=Path, default=DEFAULT_EXTENDED)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--heldout-seed", type=int, default=770001)
    ap.add_argument("--sample-path", type=Path, default=None, help="reuse an existing held-out npz")
    ap.add_argument("--snapshot-index", type=int, default=-1,
                    help="HSP snapshot to recover; -1 = pick the max outer-RMS snapshot")
    ap.add_argument("--n-modes", type=int, default=100)
    ap.add_argument("--gamma-reg", type=float, default=1.0)
    ap.add_argument("--gamma-data", type=float, default=1.0)
    ap.add_argument("--gamma-s", type=float, default=1.0e-6)
    ap.add_argument("--gamma-s-star", type=float, default=1.0e-6)
    ap.add_argument("--sigma-t", type=float, default=6.0e-4)
    ap.add_argument("--minres-rtol", type=float, default=1.0e-10)
    ap.add_argument("--minres-maxiter", type=int, default=10000)
    ap.add_argument("--no-smw", action="store_true",
                    help="use the Jacobi preconditioner instead of Sherman-Morrison-Woodbury "
                         "(required when gamma_reg=0, since the SMW correction is singular there)")
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    ap.add_argument("--noise-seed", type=int, default=20260601)
    ap.add_argument("--measured-region", default="full",
                    help="'full' (whole body) or path to a .npy of body vertex indices (patch)")
    ap.add_argument("--overwrite-sample", action="store_true")
    args = ap.parse_args(argv)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. held-out infarct ----
    sample_path = args.sample_path or generate_heldout(
        args.heldout_seed, out / "heldout", args.overwrite_sample)
    sample = np.load(sample_path)
    hsp_stack = np.asarray(sample["hsp_stack"], dtype=np.float64)        # (n_snap, n_union)
    times = np.asarray(sample["snapshot_times_ms"], dtype=np.float64)
    infarct_centre = np.asarray(sample["infarct_centre_mm"], dtype=np.float64)
    infarct_radius = float(sample["infarct_radius_mm"])

    # ---- 2. restrict to outer EPI∪BASE, pick a snapshot ----
    torso_mesh, _, torso_ft = load_gmsh_mesh(TORSO_MSH, MPI.COMM_SELF)
    part = build_partition_maps(torso_mesh, torso_ft)
    om = part["outer_mask_in_union"]
    hsp_outer_all = hsp_stack[:, om]                                     # (n_snap, n_outer)
    snap = (int(np.argmax(np.sqrt((hsp_outer_all ** 2).mean(axis=1))))
            if args.snapshot_index < 0 else int(args.snapshot_index))
    hsp_truth = hsp_outer_all[snap]
    print(f"[inverse] held-out infarct centre={infarct_centre.tolist()} r={infarct_radius:.1f}mm; "
          f"snapshot {snap} @ t={times[snap]:.1f}ms; |hsp_outer| in [{hsp_truth.min():.2f},{hsp_truth.max():.2f}]",
          flush=True)

    # ---- 3. forward HSP -> clean BSPM ----
    fwd = forward_hsp_to_bspm(hsp_truth, part["outer_hsp_idx"], torso_mesh=torso_mesh,
                              outer_gamma_vertices=part["outer_gamma_vertices"],
                              body_vertices=part["body_vertices"], sigma_t=args.sigma_t)
    clean_bsp = fwd["bsp_values"]
    rms = float(np.sqrt(np.mean(clean_bsp ** 2)))
    print(f"[inverse] clean BSPM rms={rms:.3e} range=[{clean_bsp.min():.3e},{clean_bsp.max():.3e}]", flush=True)

    # ---- measured region ----
    measured_vertices = None
    if args.measured_region != "full":
        measured_vertices = np.load(args.measured_region).astype(np.int64)
        print(f"[inverse] restricting data term to {measured_vertices.size}/"
              f"{part['body_vertices'].size} body vertices", flush=True)

    # ---- 4. build stabFEM system on THIS experiment's basis ----
    use_smw = (not args.no_smw) and (args.gamma_reg != 0.0)
    if args.gamma_reg == 0.0 and not args.no_smw:
        print("[inverse] gamma_reg=0: disabling SMW preconditioner (singular low-rank "
              "correction), falling back to Jacobi.", flush=True)
    params = Parameters(n_modes=args.n_modes, sigma_t=args.sigma_t, gamma_data=args.gamma_data,
                        gamma_s=args.gamma_s, gamma_s_star=args.gamma_s_star, gamma_reg=args.gamma_reg,
                        minres_rtol=args.minres_rtol, minres_maxiter=args.minres_maxiter,
                        use_smw_preconditioner=use_smw)
    sys_kwargs = dict(pod_path=args.pod, extended_dir=args.extended_dir, params=params)
    if measured_vertices is not None:
        sys_kwargs["measured_body_vertices"] = measured_vertices  # requires stabfem support
    system = StabFEMSystem(**sys_kwargs)
    ops = system.ops

    faces = load_boundary_plot_faces(np.asarray(sample["hsp_points"])[om])
    truth_norm = float(np.linalg.norm(hsp_truth))
    rng = np.random.default_rng(args.noise_seed)
    rows = []
    for nl in args.noise_levels:
        noisy = clean_bsp + (rng.normal(0.0, nl * rms, size=clean_bsp.shape) if nl > 0 else 0.0)
        e = np.zeros(system.n_torso, dtype=np.float64)
        e[system.v2d[part["body_vertices"]]] = noisy
        result = system.solve(e)
        aligned = np.zeros_like(hsp_truth)
        aligned[part["outer_hsp_idx"]] = result["hsp_recovered"]
        err = aligned - hsp_truth
        rel_l2 = float(np.linalg.norm(err) / max(truth_norm, 1e-30))
        cos = float(np.dot(aligned, hsp_truth) / max(np.linalg.norm(aligned) * truth_norm, 1e-30))
        tag = f"noise{int(round(100*nl)):02d}pct"
        plot_recon(np.asarray(sample["hsp_points"])[om], faces, hsp_truth, aligned,
                   infarct_centre, out / f"recon_{tag}.png",
                   f"held-out infarct  t={times[snap]:.0f}ms  n_modes={args.n_modes} γ_reg={args.gamma_reg:g}  "
                   f"noise={int(round(100*nl))}%  rel-L2={rel_l2:.3f} cos={cos:.3f}",
                   infarct_radius=infarct_radius,
                   frame_pts=part["heart_mesh"].geometry.x[:, :3])
        np.savez(out / f"recon_{tag}.npz", hsp_truth=hsp_truth, hsp_recovered=aligned,
                 clean_bsp=clean_bsp, noisy_bsp=noisy, pod_coefficients=result["pod_coefficients"],
                 minres_residuals=result["minres_residuals"], infarct_centre_mm=infarct_centre,
                 infarct_radius_mm=infarct_radius, snapshot_time_ms=times[snap])
        rows.append({"noise": nl, "minres_iters": int(result["iterations"]),
                     "minres_info": int(result["info"]), "hsp_rel_l2": rel_l2, "hsp_cosine": cos})
        print(f"[inverse] {tag}: iters={result['iterations']:>4} "
              f"hsp_rel_l2={rel_l2:.4f} cos={cos:.4f}", flush=True)

    (out / "summary.json").write_text(json.dumps({
        "sample_path": str(sample_path), "snapshot_index": snap, "snapshot_time_ms": float(times[snap]),
        "infarct_centre_mm": infarct_centre.tolist(), "infarct_radius_mm": infarct_radius,
        "n_modes": args.n_modes, "gamma_reg": args.gamma_reg,
        "measured_region": args.measured_region,
        "n_measured_body_vertices": int(measured_vertices.size) if measured_vertices is not None
        else int(part["body_vertices"].size),
        "clean_bsp_rms": rms, "results": rows,
    }, indent=2))
    print(f"[inverse] wrote {out / 'summary.json'} and recon_*.png/.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
