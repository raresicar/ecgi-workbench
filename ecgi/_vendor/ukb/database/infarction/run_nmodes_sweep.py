"""n_modes sweep (6..10) for the held-out infarct + the infarct-dominated case.

Three things, all on the SAME held-out off-grid infarct:

1. Reconstruct the held-out HSP at the standard (depolarisation, max-RMS) snapshot
   with n_modes = 6,7,8,9,10 and plot truth + the five reconstructions.
2. Take the best n_modes from (1) and reconstruct the *plateau* snapshot, where the
   wall sits at a near-uniform plateau so essentially only the infarct drives the
   HSP -- a localized, high-rank feature the global POD basis cannot represent.
3. The held-out ground-truth HSP is plotted alongside (the "truth" panels).

Reuses the held-out sample + machinery from run_inverse.py. Clean data (no noise),
so the only thing varying in (1) is the basis size.

    python ukb/database/infarction/run_nmodes_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))                      # for build_pod_basis
sys.path.insert(0, str(HERE.parents[1] / "pipeline"))
sys.path.insert(0, str(HERE.parents[1] / "stabFEM"))

from mpi4py import MPI  # noqa: E402
from common import TORSO_MSH, load_gmsh_mesh  # noqa: E402
from stabfem import Parameters, StabFEMSystem  # noqa: E402
from build_pod_basis import load_boundary_plot_faces  # noqa: E402
import run_inverse as ri  # noqa: E402

POD = HERE / "pod_basis" / "pod_basis.npz"
EXTENDED = HERE / "extended_pod_basis"
HELDOUT = HERE / "results" / "heldout" / "sample_00000.npz"
OUT = HERE / "results" / "nmodes_sweep"
N_MODES = [6, 7, 8, 9, 10]
ELEV, AZIM = 18.0, -62.0


def _panel(ax, pts, faces, vals, vmax, cmap, title, frame_pts, scar_mask=None):
    coll = Poly3DCollection(pts[faces])
    coll.set_array(vals[faces].mean(axis=1))
    coll.set_cmap(cmap)
    coll.set_clim(-vmax, vmax)
    if scar_mask is not None:
        ec = np.zeros((faces.shape[0], 4))
        ec[scar_mask] = [0.09, 0.49, 0.09, 1.0]
        lw = np.zeros(faces.shape[0]); lw[scar_mask] = 1.6
        coll.set_edgecolor(ec); coll.set_linewidth(lw)
    else:
        coll.set_edgecolor("none")
    ax.add_collection3d(coll)
    box = frame_pts
    mins, maxs = box.min(0), box.max(0)
    ctr, rad = 0.5 * (mins + maxs), 0.5 * (maxs - mins).max()
    ax.set_xlim(ctr[0]-rad, ctr[0]+rad); ax.set_ylim(ctr[1]-rad, ctr[1]+rad); ax.set_zlim(ctr[2]-rad, ctr[2]+rad)
    ax.view_init(elev=ELEV, azim=AZIM); ax.set_axis_off(); ax.set_title(title, fontsize=10)
    return coll


def reconstruct(system, e, outer_hsp_idx, hsp_truth):
    res = system.solve(e)
    aligned = np.zeros_like(hsp_truth)
    aligned[outer_hsp_idx] = res["hsp_recovered"]
    cos = float(np.dot(aligned, hsp_truth) / max(np.linalg.norm(aligned) * np.linalg.norm(hsp_truth), 1e-30))
    rel = float(np.linalg.norm(aligned - hsp_truth) / max(np.linalg.norm(hsp_truth), 1e-30))
    return aligned, cos, rel, int(res["iterations"])


def torso_rhs(system, body_vertices, noisy):
    e = np.zeros(system.n_torso)
    e[system.v2d[body_vertices]] = noisy
    return e


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sample = np.load(HELDOUT)
    hsp_stack = np.asarray(sample["hsp_stack"], float)
    times = np.asarray(sample["snapshot_times_ms"], float)
    centre = np.asarray(sample["infarct_centre_mm"], float)
    radius = float(sample["infarct_radius_mm"])

    torso_mesh, _, torso_ft = load_gmsh_mesh(TORSO_MSH, MPI.COMM_SELF)
    part = ri.build_partition_maps(torso_mesh, torso_ft)
    om = part["outer_mask_in_union"]
    hsp_outer_all = hsp_stack[:, om]                       # (n_snap, n_outer)
    outer_pts = np.asarray(sample["hsp_points"])[om][:, :3]
    faces = load_boundary_plot_faces(np.asarray(sample["hsp_points"])[om])
    frame_pts = part["heart_mesh"].geometry.x[:, :3]
    scar_mask_v = np.linalg.norm(outer_pts - centre, axis=1) <= radius
    scar_face = np.array([bool(scar_mask_v[f].all()) for f in faces]) if scar_mask_v.any() else None

    # snapshot choices: depol = max outer-RMS; plateau = late-window max scar contrast
    rms = np.sqrt((hsp_outer_all ** 2).mean(axis=1))
    depol = int(np.argmax(rms))
    late = np.where(times > 100.0)[0]
    contrast = np.array([
        abs(hsp_outer_all[s][scar_mask_v].mean() - hsp_outer_all[s][~scar_mask_v].mean())
        / (hsp_outer_all[s].std() + 1e-12) for s in late])
    plateau = int(late[int(np.argmax(contrast))])
    print(f"depol snapshot {depol} @ t={times[depol]:.1f}ms ; "
          f"plateau snapshot {plateau} @ t={times[plateau]:.1f}ms (scar contrast {contrast.max():.2f}σ)")

    # forward both snapshots -> clean BSPM
    def bspm(snap):
        return ri.forward_hsp_to_bspm(
            hsp_outer_all[snap], part["outer_hsp_idx"], torso_mesh=torso_mesh,
            outer_gamma_vertices=part["outer_gamma_vertices"],
            body_vertices=part["body_vertices"])["bsp_values"]

    # ---- (1) n_modes sweep on the depol snapshot ----
    truth_d = hsp_outer_all[depol]
    bspm_d = bspm(depol)
    e_d = None
    vmax = float(np.abs(truth_d).max()) or 1.0
    recons, cosines = [], []
    for nm in N_MODES:
        system = StabFEMSystem(pod_path=POD, extended_dir=EXTENDED,
                               params=Parameters(n_modes=nm, use_smw_preconditioner=True))
        if e_d is None:
            e_d = torso_rhs(system, part["body_vertices"], bspm_d)
        aligned, cos, rel, iters = reconstruct(system, e_d, part["outer_hsp_idx"], truth_d)
        recons.append(aligned); cosines.append(cos)
        print(f"  n_modes={nm:2d}  cos={cos:.4f}  rel-L2={rel:.4f}  iters={iters}")

    fig = plt.figure(figsize=(18, 6.2), dpi=150)
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    c0 = _panel(ax, outer_pts, faces, truth_d, vmax, "RdBu_r",
                f"held-out truth HSP  t={times[depol]:.0f}ms", frame_pts, scar_face)
    fig.colorbar(c0, ax=ax, shrink=0.6)
    for i, (nm, rec, cos) in enumerate(zip(N_MODES, recons, cosines)):
        ax = fig.add_subplot(2, 3, i + 2, projection="3d")
        c = _panel(ax, outer_pts, faces, rec, vmax, "RdBu_r",
                   f"recovered  n_modes={nm}  cos={cos:.3f}", frame_pts, scar_face)
        fig.colorbar(c, ax=ax, shrink=0.6)
    fig.suptitle("Held-out infarct: reconstructed HSP vs POD-basis size (depolarisation snapshot, no noise)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "nmodes_6_10_depol.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT / 'nmodes_6_10_depol.png'}")

    # ---- (1b) standalone triptych of the best n_modes on the depol snapshot ----
    best = int(np.argmax(cosines)); best_nm = N_MODES[best]
    rec_d = recons[best]; cos_d = cosines[best]
    rel_d = float(np.linalg.norm(rec_d - truth_d) / max(np.linalg.norm(truth_d), 1e-30))
    err_d = rec_d - truth_d
    fig = plt.figure(figsize=(18, 5.6), dpi=150)
    for i, (lab, vals, vm, cmap) in enumerate([
            (f"held-out truth HSP  t={times[depol]:.0f}ms (depolarisation)", truth_d, vmax, "RdBu_r"),
            (f"recovered  n_modes={best_nm}  cos={cos_d:.3f}  rel-L2={rel_d:.2f}", rec_d, vmax, "RdBu_r"),
            ("error", err_d, float(np.abs(err_d).max()) or 1.0, "PuOr_r")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        c = _panel(ax, outer_pts, faces, vals, vm, cmap, lab, frame_pts, scar_face)
        fig.colorbar(c, ax=ax, shrink=0.6)
    fig.suptitle("Depolarisation snapshot: the global activation dipole is recovered accurately", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "depol_best_nmodes.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT / 'depol_best_nmodes.png'}")

    # one system at the best n_modes, reused for the noisy depol case and the plateau
    system = StabFEMSystem(pod_path=POD, extended_dir=EXTENDED,
                           params=Parameters(n_modes=best_nm, use_smw_preconditioner=True))

    # ---- (1c) same depol triptych but with 10% measurement noise on the BSPM ----
    noise_frac = 0.10
    rms = float(np.sqrt(np.mean(bspm_d ** 2)))
    noisy = bspm_d + np.random.default_rng(20260601).normal(0.0, noise_frac * rms, size=bspm_d.shape)
    e_dn = torso_rhs(system, part["body_vertices"], noisy)
    rec_dn, cos_dn, rel_dn, _ = reconstruct(system, e_dn, part["outer_hsp_idx"], truth_d)
    err_dn = rec_dn - truth_d
    print(f"  DEPOL n_modes={best_nm} @ {int(noise_frac*100)}% noise: cos={cos_dn:.4f} rel-L2={rel_dn:.4f}")
    fig = plt.figure(figsize=(18, 5.6), dpi=150)
    for i, (lab, vals, vm, cmap) in enumerate([
            (f"held-out truth HSP  t={times[depol]:.0f}ms (depolarisation)", truth_d, vmax, "RdBu_r"),
            (f"recovered  n_modes={best_nm}  {int(noise_frac*100)}% noise  cos={cos_dn:.3f}  rel-L2={rel_dn:.2f}", rec_dn, vmax, "RdBu_r"),
            ("error", err_dn, float(np.abs(err_dn).max()) or 1.0, "PuOr_r")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        c = _panel(ax, outer_pts, faces, vals, vm, cmap, lab, frame_pts, scar_face)
        fig.colorbar(c, ax=ax, shrink=0.6)
    fig.suptitle(f"Depolarisation snapshot with {int(noise_frac*100)}% measurement noise (n_modes={best_nm})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "depol_best_nmodes_noise10pct.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT / 'depol_best_nmodes_noise10pct.png'}")

    # ---- (2) best n_modes on the infarct-dominated plateau snapshot ----
    truth_p = hsp_outer_all[plateau]
    e_p = torso_rhs(system, part["body_vertices"], bspm(plateau))
    aligned_p, cos_p, rel_p, _ = reconstruct(system, e_p, part["outer_hsp_idx"], truth_p)
    print(f"  PLATEAU n_modes={best_nm} (best on depol): cos={cos_p:.4f} rel-L2={rel_p:.4f}")

    vmp = float(np.abs(truth_p).max()) or 1.0
    err = aligned_p - truth_p
    fig = plt.figure(figsize=(18, 5.6), dpi=150)
    for i, (lab, vals, vm, cmap) in enumerate([
            (f"held-out truth HSP  t={times[plateau]:.0f}ms (infarct-dominated)", truth_p, vmp, "RdBu_r"),
            (f"recovered  n_modes={best_nm}  cos={cos_p:.3f}  rel-L2={rel_p:.2f}", aligned_p, vmp, "RdBu_r"),
            ("error", err, float(np.abs(err).max()) or 1.0, "PuOr_r")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        c = _panel(ax, outer_pts, faces, vals, vm, cmap, lab, frame_pts, scar_face)
        fig.colorbar(c, ax=ax, shrink=0.6)
    fig.suptitle("Infarct-dominated (plateau) snapshot: the scar is localized to the right place but blurred and under-amplified", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "plateau_best_nmodes.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT / 'plateau_best_nmodes.png'}")


if __name__ == "__main__":
    main()
