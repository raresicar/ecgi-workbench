"""Reconstruct the real P001 body-surface measurement with the four_params basis.

The 128-electrode body-surface potentials live in AllTrainingSignals/P001-ts.mat
(ts.potvals, 128 x 1001 time samples); the electrode positions are leadlinks in
TrainingGeometries/P001_torso.mat (which is the same torso as the solver mesh).
We map the 128 leads onto the solver body surface, take the QRS-peak time sample
as a 128-electrode partial measurement, and recover the HSP with the four_params
stabFEM basis (n_modes = the 99%-energy rank = 15). There is no ground-truth
epicardial potential, so we only plot the recovered HSP and report iterations.

    xvfb-run -a python ukb/database/four_params/run_p001_real.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (HERE, HERE.parent, HERE.parents[1] / "pipeline", HERE.parents[1] / "stabFEM"):
    sys.path.insert(0, str(p))

from mpi4py import MPI  # noqa: E402
from common import TORSO_MSH, load_gmsh_mesh  # noqa: E402
from stabfem import Parameters, StabFEMSystem  # noqa: E402
from build_pod_basis import load_boundary_plot_faces  # noqa: E402
import run_inverse as ri  # noqa: E402

POD = HERE / "pod_basis" / "pod_basis.npz"
EXTENDED = HERE / "extended_pod_basis"
OUT = HERE / "results" / "p001_real"
NM = 8
TORSO_MAT = REPO / "TrainingGeometries" / "P001_torso.mat"
TS_MAT = REPO / "AllTrainingSignals" / "P001-ts.mat"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- P001 geometry: 128 electrode positions ---
    torso = sio.loadmat(TORSO_MAT, squeeze_me=True, struct_as_record=False)["torso"]
    node = np.asarray(torso.node, float)
    leadlinks = np.asarray(torso.leadlinks, int) - 1          # MATLAB 1-based -> 0-based
    lead_xyz = node[leadlinks]                                 # (128,3)

    # --- P001 measurement: 128 leads x 1001 time samples ---
    ts = sio.loadmat(TS_MAT, squeeze_me=True, struct_as_record=False)["ts"]
    potvals = np.asarray(ts.potvals, float)                    # (128,1001)
    rms_t = np.sqrt((potvals ** 2).mean(axis=0))
    tstar = int(np.argmax(rms_t))                              # QRS peak
    meas = potvals[:, tstar]                                   # (128,)
    print(f"P001-ts potvals {potvals.shape}; QRS-peak sample {tstar}/{potvals.shape[1]} "
          f"(lead RMS {rms_t[tstar]:.3g}); meas range [{meas.min():.3g},{meas.max():.3g}]")

    # --- map leads onto the solver body surface ---
    tm, _, tf = load_gmsh_mesh(TORSO_MSH, MPI.COMM_SELF)
    part = ri.build_partition_maps(tm, tf)
    body_v = part["body_vertices"]
    body_xyz = tm.geometry.x[body_v, :3]

    # 128 isolated electrodes form no full facets, so the surface-integral data term
    # would be empty. Interpolate the electrode potentials onto every body vertex
    # (inverse-distance weighting, 6 nearest electrodes) -> a dense full-body BSP.
    dist, nbr = cKDTree(lead_xyz).query(body_xyz, k=6)
    dist = np.maximum(dist, 1e-9)
    w = 1.0 / dist ** 2
    bsp_body = (w * meas[nbr]).sum(1) / w.sum(1)              # (n_body,)
    bsp_body[dist[:, 0] < 1e-6] = meas[nbr[dist[:, 0] < 1e-6, 0]]   # exact electrode hits
    print(f"interpolated 128 electrodes -> {body_v.size} body vertices "
          f"(IDW k=6); body BSP range [{bsp_body.min():.3g},{bsp_body.max():.3g}]")

    # --- build stabFEM system on the four_params basis, full-body data ---
    system = StabFEMSystem(pod_path=POD, extended_dir=EXTENDED,
                           params=Parameters(n_modes=NM, use_smw_preconditioner=True))
    e = np.zeros(system.n_torso)
    e[system.v2d[body_v]] = bsp_body
    res = system.solve(e)
    iters = int(res["iterations"])
    hsp = res["hsp_recovered"]
    # data discrepancy at convergence: recovered torso solution's body trace vs the measured BSP
    br, bd, resid = res["body_reconstructed"], res["body_data"], res["body_residual"]
    rel = float(np.linalg.norm(resid) / max(np.linalg.norm(bd), 1e-30))
    rms = float(np.sqrt(np.mean(resid ** 2)))
    final_minres = float(res["minres_residuals"][-1]) if len(res["minres_residuals"]) else float("nan")
    print(f"RECOVERED HSP from real P001 measurement: n_modes={NM}, MINRES iterations={iters}, "
          f"info={res['info']}, HSP range [{hsp.min():.3g},{hsp.max():.3g}]")
    print(f"DATA DISCREPANCY at convergence (body trace vs measured BSP):")
    print(f"  relative L2 = {rel:.4f}  ({100*rel:.1f}% of the data norm)")
    print(f"  RMS misfit  = {rms:.4f}   (measured BSP RMS {np.sqrt(np.mean(bd**2)):.4f}, "
          f"range [{bd.min():.3g},{bd.max():.3g}])")
    print(f"  final MINRES linear residual = {final_minres:.2e}")

    # --- plot recovered HSP on the heart outer surface (no ground truth) ---
    pod = np.load(POD)
    outer_pts = pod["hsp_points"][:, :3]                       # already the EPI∪BASE outer points
    faces = load_boundary_plot_faces(outer_pts)
    aligned = np.zeros(outer_pts.shape[0])
    aligned[part["outer_hsp_idx"]] = hsp
    frame = part["heart_mesh"].geometry.x[:, :3]
    vmax = float(np.abs(aligned).max()) or 1.0
    fig = plt.figure(figsize=(7.5, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(outer_pts[faces]); coll.set_array(aligned[faces].mean(1))
    coll.set_cmap("RdBu_r"); coll.set_clim(-vmax, vmax); coll.set_edgecolor("none")
    ax.add_collection3d(coll)
    mn, mx = frame.min(0), frame.max(0); ct = 0.5 * (mn + mx); rad = 0.5 * (mx - mn).max()
    ax.set_xlim(ct[0]-rad, ct[0]+rad); ax.set_ylim(ct[1]-rad, ct[1]+rad); ax.set_zlim(ct[2]-rad, ct[2]+rad)
    ax.view_init(elev=18, azim=-62); ax.set_axis_off()
    ax.set_title(f"Recovered HSP from real P001 BSP\n(four_params basis, n_modes={NM}, 128 electrodes, "
                 f"QRS-peak t-sample {tstar}, MINRES {iters} iters)", fontsize=10)
    fig.colorbar(coll, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "recovered_hsp_p001.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT / 'recovered_hsp_p001.png'}")


if __name__ == "__main__":
    main()
