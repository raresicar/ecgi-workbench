"""Full-body vs partial anterior-chest-patch measurement comparison.

The held-out infarct's HSP is forwarded to a BSPM on the *whole* body surface, but
the stabFEM data term is then restricted to an anterior chest patch -- the body
vertices nearest the heart, which (since the heart sits anterior in the thorax)
form a front-of-chest region, ~27% of Gamma_B. We compare reconstruction quality
(and MINRES iterations) of full-body vs patch for the depolarisation snapshot and
the front-wraps-infarct snapshot, clean and at 10% noise.

    xvfb-run -a python ukb/database/infarction/run_partial_patch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parents[1] / "pipeline", HERE.parents[1] / "stabFEM"):
    sys.path.insert(0, str(p))

from mpi4py import MPI  # noqa: E402
from common import TORSO_MSH, load_gmsh_mesh  # noqa: E402
from stabfem import Parameters, StabFEMSystem  # noqa: E402
from build_pod_basis import load_boundary_plot_faces  # noqa: E402
import run_inverse as ri  # noqa: E402
import run_nmodes_sweep as sw  # noqa: E402

OUT = HERE / "results" / "partial_patch"
NM = 9
PATCH_FRAC = 0.27
NOISE_SEED = 20260601


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = np.load(HERE / "results" / "heldout" / "sample_00000.npz")
    hsp = np.asarray(s["hsp_stack"], float)
    times = np.asarray(s["snapshot_times_ms"], float)
    c = np.asarray(s["infarct_centre_mm"], float); r = float(s["infarct_radius_mm"])

    tm, _, tf = load_gmsh_mesh(TORSO_MSH, MPI.COMM_SELF)
    part = ri.build_partition_maps(tm, tf)
    om = part["outer_mask_in_union"]; ho = hsp[:, om]
    body_v = part["body_vertices"]
    body_pts = tm.geometry.x[body_v, :3]

    # heart centre in torso coordinates = centroid of the outer heart-boundary vertices
    heart_centre = tm.geometry.x[part["outer_gamma_vertices"], :3].mean(0)
    dist = np.linalg.norm(body_pts - heart_centre, axis=1)
    k = int(round(PATCH_FRAC * body_v.size))
    patch = body_v[np.argsort(dist)[:k]].astype(np.int64)
    np.save(OUT / "anterior_patch_vertices.npy", patch)
    # sanity: where is the patch vs the full body centroid?
    pc, bc = body_pts[np.argsort(dist)[:k]].mean(0), body_pts.mean(0)
    print(f"patch: {patch.size}/{body_v.size} body vertices ({100*patch.size/body_v.size:.0f}%)")
    print(f"  body centroid {bc.round(1)}  patch centroid {pc.round(1)}  heart centre {heart_centre.round(1)}")

    # torso panel: body points, patch highlighted
    fig = plt.figure(figsize=(6, 6), dpi=140); ax = fig.add_subplot(111, projection="3d")
    sel = np.zeros(body_v.size, bool); sel[np.argsort(dist)[:k]] = True
    ax.scatter(*body_pts[~sel].T, s=2, c="#cbd5e1", alpha=0.5)
    ax.scatter(*body_pts[sel].T, s=4, c="#dc2626")
    ax.scatter(*heart_centre, s=60, c="#1d4ed8", marker="*")
    ax.set_axis_off(); ax.set_title(f"anterior chest patch ({100*patch.size/body_v.size:.0f}% of body surface)")
    ax.view_init(elev=10, azim=-90)
    fig.tight_layout(); fig.savefig(OUT / "patch_location.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT / 'patch_location.png'}")

    sys_full = StabFEMSystem(pod_path=sw.POD, extended_dir=sw.EXTENDED,
                             params=Parameters(n_modes=NM, use_smw_preconditioner=True))
    sys_patch = StabFEMSystem(pod_path=sw.POD, extended_dir=sw.EXTENDED,
                              params=Parameters(n_modes=NM, use_smw_preconditioner=True),
                              measured_body_vertices=patch)

    def bspm(snap, noise):
        clean = ri.forward_hsp_to_bspm(
            ho[snap], part["outer_hsp_idx"], torso_mesh=tm,
            outer_gamma_vertices=part["outer_gamma_vertices"],
            body_vertices=body_v)["bsp_values"]
        if noise > 0:
            rms = float(np.sqrt(np.mean(clean ** 2)))
            clean = clean + np.random.default_rng(NOISE_SEED).normal(0, noise * rms, size=clean.shape)
        return clean

    def solve(system, snap, noise):
        e = sw.torso_rhs(system, body_v, bspm(snap, noise))
        aligned, cos, rel, it = sw.reconstruct(system, e, part["outer_hsp_idx"], ho[snap])
        return aligned, cos, rel, it

    cases = [("depol t68", 16, 0.0), ("depol t68 +10%", 16, 0.10),
             ("t96 wrap", 22, 0.0), ("t96 wrap +10%", 22, 0.10)]
    print(f"\n{'case':18s} {'full cos':9s} {'full it':7s} {'patch cos':10s} {'patch it':8s}")
    store = {}
    for lab, snap, noise in cases:
        af, cf, rf, itf = solve(sys_full, snap, noise)
        ap, cp, rp, itp = solve(sys_patch, snap, noise)
        store[(snap, noise)] = (af, ap, cf, cp)
        print(f"{lab:18s} {cf:^9.3f} {itf:^7d} {cp:^10.3f} {itp:^8d}")

    # comparison figure on the t96 wrap clean case: truth | full | patch
    faces = load_boundary_plot_faces(np.asarray(s["hsp_points"])[om])
    opts = np.asarray(s["hsp_points"])[om][:, :3]
    frame = part["heart_mesh"].geometry.x[:, :3]
    footv = np.linalg.norm(opts - c, axis=1) <= r
    scar = np.array([bool(footv[f].all()) for f in faces])
    af, ap, cf, cp = store[(22, 0.0)]
    truth = ho[22]; vmax = float(np.abs(truth).max())
    fig = plt.figure(figsize=(18, 5.6), dpi=150)
    for i, (lab, vals) in enumerate([
            (f"truth HSP  t=96ms", truth),
            (f"full body  cos={cf:.3f}", af),
            (f"anterior patch ({100*patch.size/body_v.size:.0f}%)  cos={cp:.3f}", ap)]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        cc = sw._panel(ax, opts, faces, vals, vmax, "RdBu_r", lab, frame, scar)
        fig.colorbar(cc, ax=ax, shrink=0.6)
    fig.suptitle(f"Full-body vs anterior-patch measurement (t=96ms, n_modes={NM}, clean)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "full_vs_patch_t96.png", facecolor="white"); plt.close(fig)
    print(f"wrote {OUT / 'full_vs_patch_t96.png'}")


if __name__ == "__main__":
    main()
