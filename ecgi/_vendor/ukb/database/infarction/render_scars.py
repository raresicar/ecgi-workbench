"""Boulakia-style infarct renders for the infarction experiment.

Produces front-facing, properly occluded views of the epicardium with the scar
regions painted ON the visible anterior wall (cf. Boulakia/Schenone/Gerbeau
arXiv:1111.5926, Figures 8 and 9) -- so the infarcts are shown directly on the
screen rather than bleeding through a translucent mesh.

Two figures are written next to the samples:
  * ``infarct_centres.png`` -- the opaque heart with the N training-infarct
    centres marked as dots on the front wall (Fig 9 style). If a held-out roster
    is given, its off-grid point P is added in gold with a label.
  * ``scar_example.png``    -- one representative infarct painted as a blue disk
    on the surface (Fig 8 style), to show the scar extent.

VTK needs a GL context, so run this under a virtual framebuffer:

    xvfb-run -a python ukb/database/infarction/render_scars.py \
        [--roster samples/roster.json] [--output-dir samples]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from mpi4py import MPI  # noqa: E402
from common import EPI, HEART_MSH, load_gmsh_mesh  # noqa: E402

import vtk  # noqa: E402
vtk.vtkObject.GlobalWarningDisplayOff()
try:
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
except Exception:  # noqa: BLE001
    pass

import pyvista as pv  # noqa: E402
pv.OFF_SCREEN = True

CORAL = "#d9544d"   # healthy myocardium
BLUE = "#2b6cb0"    # scar
GOLD = "#f6c343"    # held-out point P
WINDOW = (1000, 1050)


def epicardium_surface(mesh, facet_tags):
    """EPI triangle facets -> (pyvista PolyData, points, faces[int64, (nf,3)])."""
    pts = mesh.geometry.x[:, :3]
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)
    facets = facet_tags.indices[facet_tags.values == EPI]
    faces = np.asarray([f2v.links(int(f)) for f in facets], dtype=np.int64)
    pv_faces = np.hstack([np.full((faces.shape[0], 1), 3, np.int64), faces]).ravel()
    return pv.PolyData(pts, pv_faces), pts, faces


def _camera(plotter, pts, view_dir):
    ctr = pts.mean(0)
    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    plotter.camera.position = tuple(ctr + view_dir * diag * 1.9)
    plotter.camera.focal_point = tuple(ctr)
    plotter.camera.up = (0, 0, 1)        # apex (min z) points down
    plotter.camera.zoom(1.3)


def render_centres(surf, pts, view_dir, centres, out_png, *, p_point=None, label=True):
    p = pv.Plotter(off_screen=True, window_size=WINDOW)
    p.background_color = "white"
    p.add_mesh(surf, color=CORAL, smooth_shading=True, specular=0.15,
               show_scalar_bar=False)
    proud = [c + view_dir * 2.0 for c in centres]
    for q in proud:
        p.add_mesh(pv.Sphere(radius=2.6, center=q), color=BLUE)
    if label:
        p.add_point_labels(
            np.asarray(proud), [str(i + 1) for i in range(len(centres))],
            font_size=15, text_color="white", shape="rounded_rect",
            shape_color="black", show_points=False, always_visible=True,
        )
    if p_point is not None:
        pq = np.asarray(p_point) + view_dir * 2.0
        p.add_mesh(pv.Sphere(radius=3.4, center=pq), color=GOLD)
        p.add_point_labels([pq], ["P"], font_size=20, text_color="black",
                           shape="rounded_rect", shape_color=GOLD,
                           show_points=False, always_visible=True)
    _camera(p, pts, view_dir)
    p.screenshot(str(out_png))
    p.close()


def render_scar_example(surf, pts, faces, view_dir, centre, radius, out_png):
    face_ctr = pts[faces].mean(1)
    face_scar = (np.sum((face_ctr - centre) ** 2, axis=1) <= radius * radius).astype(float)
    s = surf.copy()
    s.cell_data["scar"] = face_scar
    p = pv.Plotter(off_screen=True, window_size=WINDOW)
    p.background_color = "white"
    p.add_mesh(s, scalars="scar", cmap=[CORAL, BLUE], clim=[0, 1],
               smooth_shading=True, specular=0.15, show_scalar_bar=False,
               interpolate_before_map=False)
    _camera(p, pts, view_dir)
    p.screenshot(str(out_png))
    p.close()
    return int(face_scar.sum())


def _infarct_rows(roster):
    return [r for r in roster if not r.get("is_healthy", False)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=HERE / "config.json")
    ap.add_argument("--roster", type=Path, default=HERE / "samples" / "roster.json")
    ap.add_argument("--heldout-roster", type=Path, default=None,
                    help="optional roster.json from a --heldout run; its infarct is drawn as P")
    ap.add_argument("--output-dir", type=Path, default=HERE / "samples")
    args = ap.parse_args(argv)

    cfg = {k: v for k, v in json.loads(args.config.read_text()).items()
           if not k.startswith("_")}
    view_dir = np.asarray(cfg["view_dir"], float)
    view_dir = view_dir / np.linalg.norm(view_dir)

    roster = json.loads(args.roster.read_text())
    infarcts = _infarct_rows(roster)
    centres = np.asarray([r["centre_mm"] for r in infarcts], float)
    radii = np.asarray([r["radius_mm"] for r in infarcts], float)

    p_point = None
    if args.heldout_roster is not None and args.heldout_roster.exists():
        hr = _infarct_rows(json.loads(args.heldout_roster.read_text()))
        if hr:
            p_point = np.asarray(hr[0]["centre_mm"], float)

    mesh, _, ft = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    surf, pts, faces = epicardium_surface(mesh, ft)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    centres_png = args.output_dir / "infarct_centres.png"
    render_centres(surf, pts, view_dir, centres, centres_png, p_point=p_point)
    print(f"wrote {centres_png}  ({len(centres)} centres"
          + (", + held-out P" if p_point is not None else "") + ")")

    # Representative scar: the centre closest to the medoid of all centres.
    medoid = int(np.argmin(np.linalg.norm(centres - centres.mean(0), axis=1)))
    example_png = args.output_dir / "scar_example.png"
    nf = render_scar_example(surf, pts, faces, view_dir,
                             centres[medoid], radii[medoid], example_png)
    print(f"wrote {example_png}  (centre #{medoid + 1}, r={radii[medoid]:.1f} mm, {nf} faces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
