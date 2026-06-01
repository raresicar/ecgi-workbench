"""Held-out reconstruction with stabFEM on the four-parameter POD basis.

1. Synthesise an out-of-training sample with ``generate.py --n-samples 1
   --base-seed <seed>`` (a fresh tau_in/C_m/A_m/tau_close^RV combination).
2. Pick its highest-RMS HSP snapshot, restrict to the outer EPI∪BASE interface,
   and forward it through the torso (mixed-BC Laplace) to a clean BSPM.
3. Add Gaussian noise (fractions of BSPM RMS).
4. Reconstruct the HSP with ``StabFEMSystem`` using THIS experiment's basis
   (n_modes = the 99%-energy rank), and compare to the held-out truth.

Plots use the fixed GIF view (elev=18, azim=-62) with the full heart-mesh
bounding box, matching the pipeline step plots.
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
from stabfem import MATCH_TOL, Parameters, StabFEMSystem, _translated_hsp_points  # noqa: E402
from build_pod_basis import load_boundary_plot_faces  # noqa: E402

DEFAULT_POD = HERE / "pod_basis" / "pod_basis.npz"
DEFAULT_EXTENDED = HERE / "extended_pod_basis"
DEFAULT_OUTPUT = HERE / "results"
GENERATOR = HERE / "generate.py"


def generate_heldout(seed: int, out_dir: Path, overwrite: bool) -> Path:
    sample = out_dir / "sample_00000.npz"
    if sample.exists() and not overwrite:
        print(f"[inverse] reusing held-out sample {sample}", flush=True)
        return sample
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(GENERATOR), "--n-samples", "1", "--base-seed", str(seed),
           "--output-dir", str(out_dir), "--overwrite", "--n-workers", "1", "--temp-csv", ""]
    print(f"[inverse] generating out-of-training held-out sample (seed={seed})", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    if not sample.exists():
        raise FileNotFoundError(sample)
    return sample


def build_partition_maps(torso_mesh, torso_ft) -> dict:
    heart_mesh, _, heart_ft = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    part = heart_boundary_partition(heart_mesh, heart_ft)
    om = np.asarray(part["outer_mask"], dtype=bool)
    union_pts = heart_mesh.geometry.x[part["union_vertices"], :3].copy()
    gv = tagged_facet_vertices(torso_mesh, torso_ft, GAMMA_HEART)
    gp = torso_mesh.geometry.x[gv, :3]
    dist, uidx = cKDTree(_translated_hsp_points(union_pts)).query(gp, k=1)
    if float(dist.max()) > MATCH_TOL:
        raise RuntimeError(f"GammaHeart/heart mismatch: {float(dist.max()):.3e}")
    is_outer = om[uidx]
    u2o = np.full(om.size, -1, dtype=np.int64); u2o[np.where(om)[0]] = np.arange(int(om.sum()))
    return {"outer_gamma_vertices": gv[is_outer], "outer_hsp_idx": u2o[uidx[is_outer]],
            "outer_mask_in_union": om, "body_vertices": tagged_facet_vertices(torso_mesh, torso_ft, GAMMA_BODY),
            "heart_mesh": heart_mesh}


def forward_hsp_to_bspm(hsp_on_outer, outer_hsp_idx, *, torso_mesh, outer_gamma_vertices,
                        body_vertices, sigma_t=6.0e-4) -> np.ndarray:
    V = cg1_space(torso_mesh); v2d = vertex_to_dof_map(V)
    outer_dofs = v2d[outer_gamma_vertices].astype(np.int32); body_dofs = v2d[body_vertices].astype(np.int64)
    bc_fun = dolfinx.fem.Function(V); bc_fun.x.array[outer_dofs] = hsp_on_outer[outer_hsp_idx]
    bc_fun.x.scatter_forward(); bc = dolfinx.fem.dirichletbc(bc_fun, outer_dofs)
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = sigma_t * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
    L = dolfinx.fem.Constant(torso_mesh, dolfinx.default_scalar_type(0.0)) * w * ufl.dx
    uT = dolfinx.fem.Function(V)
    LinearProblem(a, L, u=uT, bcs=[bc], petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-10"}).solve()
    uT.x.scatter_forward()
    return uT.x.array[body_dofs].copy()


def plot_recon(hsp_points, faces, truth, recon, out_png, title, frame_pts) -> None:
    vmax = float(np.max(np.abs(np.concatenate([truth, recon])))) or 1.0
    pts = hsp_points[:, :3]; box = frame_pts[:, :3]
    mins, maxs = box.min(0), box.max(0); ctr, rad = 0.5 * (mins + maxs), 0.5 * (maxs - mins).max()
    panels = [("truth HSP", truth, vmax, "RdBu_r"), ("recovered HSP", recon, vmax, "RdBu_r"),
              ("error", recon - truth, float(np.max(np.abs(recon - truth))) or 1.0, "PuOr_r")]
    fig = plt.figure(figsize=(16, 5.5), dpi=160)
    for i, (lab, vals, vm, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        coll = Poly3DCollection(pts[faces]); coll.set_array(vals[faces].mean(1))
        coll.set_cmap(cmap); coll.set_clim(-vm, vm); coll.set_edgecolor("none")
        ax.add_collection3d(coll)
        ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad); ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
        ax.view_init(elev=18, azim=-62); ax.set_axis_off(); ax.set_title(lab, fontsize=11)
        fig.colorbar(coll, ax=ax, shrink=0.6)
    fig.suptitle(title, fontsize=12); fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, facecolor="white"); plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", type=Path, default=DEFAULT_POD)
    ap.add_argument("--extended-dir", type=Path, default=DEFAULT_EXTENDED)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--heldout-seed", type=int, default=880001)
    ap.add_argument("--sample-path", type=Path, default=None)
    ap.add_argument("--snapshot-index", type=int, default=-1, help="-1 = max outer-RMS snapshot")
    ap.add_argument("--n-modes", type=int, default=15)
    ap.add_argument("--gamma-reg", type=float, default=1.0)
    ap.add_argument("--gamma-data", type=float, default=1.0)
    ap.add_argument("--gamma-s", type=float, default=1.0e-6)
    ap.add_argument("--gamma-s-star", type=float, default=1.0e-6)
    ap.add_argument("--sigma-t", type=float, default=6.0e-4)
    ap.add_argument("--minres-rtol", type=float, default=1.0e-10)
    ap.add_argument("--minres-maxiter", type=int, default=10000)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    ap.add_argument("--noise-seed", type=int, default=20260601)
    ap.add_argument("--overwrite-sample", action="store_true")
    args = ap.parse_args(argv)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)

    sample_path = args.sample_path or generate_heldout(args.heldout_seed, out / "heldout", args.overwrite_sample)
    sample = np.load(sample_path)
    hsp_stack = np.asarray(sample["hsp_stack"], dtype=np.float64)
    times = np.asarray(sample["snapshot_times_ms"], dtype=np.float64)
    varied = json.loads(str(sample["varied_json"])) if "varied_json" in sample.files else {}

    torso_mesh, _, torso_ft = load_gmsh_mesh(TORSO_MSH, MPI.COMM_SELF)
    part = build_partition_maps(torso_mesh, torso_ft)
    om = part["outer_mask_in_union"]
    hsp_outer_all = hsp_stack[:, om]
    snap = (int(np.argmax(np.sqrt((hsp_outer_all ** 2).mean(1)))) if args.snapshot_index < 0
            else int(args.snapshot_index))
    hsp_truth = hsp_outer_all[snap]
    print(f"[inverse] held-out params={ {k: round(v,4) for k,v in varied.items()} }; "
          f"snapshot {snap} @ t={times[snap]:.1f}ms; |hsp_outer|max={np.abs(hsp_truth).max():.2f}", flush=True)

    clean_bsp = forward_hsp_to_bspm(hsp_truth, part["outer_hsp_idx"], torso_mesh=torso_mesh,
                                    outer_gamma_vertices=part["outer_gamma_vertices"],
                                    body_vertices=part["body_vertices"], sigma_t=args.sigma_t)
    rms = float(np.sqrt(np.mean(clean_bsp ** 2)))
    print(f"[inverse] clean BSPM rms={rms:.3e}", flush=True)

    params = Parameters(n_modes=args.n_modes, sigma_t=args.sigma_t, gamma_data=args.gamma_data,
                        gamma_s=args.gamma_s, gamma_s_star=args.gamma_s_star, gamma_reg=args.gamma_reg,
                        minres_rtol=args.minres_rtol, minres_maxiter=args.minres_maxiter, use_smw_preconditioner=True)
    system = StabFEMSystem(pod_path=args.pod, extended_dir=args.extended_dir, params=params)

    hp = np.asarray(sample["hsp_points"])[om]; faces = load_boundary_plot_faces(hp)
    frame = part["heart_mesh"].geometry.x[:, :3]
    truth_norm = float(np.linalg.norm(hsp_truth)); rng = np.random.default_rng(args.noise_seed)
    rows = []
    for nl in args.noise_levels:
        noisy = clean_bsp + (rng.normal(0.0, nl * rms, size=clean_bsp.shape) if nl > 0 else 0.0)
        e = np.zeros(system.n_torso); e[system.v2d[part["body_vertices"]]] = noisy
        res = system.solve(e)
        aligned = np.zeros_like(hsp_truth); aligned[part["outer_hsp_idx"]] = res["hsp_recovered"]
        rl = float(np.linalg.norm(aligned - hsp_truth) / max(truth_norm, 1e-30))
        cos = float(np.dot(aligned, hsp_truth) / max(np.linalg.norm(aligned) * truth_norm, 1e-30))
        tag = f"noise{int(round(100*nl)):02d}pct"
        plot_recon(hp, faces, hsp_truth, aligned, out / f"recon_{tag}.png",
                   f"four-params held-out  t={times[snap]:.0f}ms  n_modes={args.n_modes}  "
                   f"noise={int(round(100*nl))}%  rel-L2={rl:.3f} cos={cos:.3f}", frame)
        np.savez(out / f"recon_{tag}.npz", hsp_truth=hsp_truth, hsp_recovered=aligned,
                 clean_bsp=clean_bsp, noisy_bsp=noisy, snapshot_time_ms=times[snap], varied_json=json.dumps(varied))
        rows.append({"noise": nl, "iters": int(res["iterations"]), "hsp_rel_l2": rl, "hsp_cosine": cos})
        print(f"[inverse] {tag}: iters={res['iterations']:>4} rel_l2={rl:.4f} cos={cos:.4f}", flush=True)

    (out / "summary.json").write_text(json.dumps(
        {"sample_path": str(sample_path), "snapshot_index": snap, "snapshot_time_ms": float(times[snap]),
         "varied": varied, "n_modes": args.n_modes, "gamma_reg": args.gamma_reg,
         "clean_bsp_rms": rms, "results": rows}, indent=2))
    print(f"[inverse] wrote {out/'summary.json'} and recon_*.png/.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
