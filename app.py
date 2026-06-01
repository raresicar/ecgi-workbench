"""ECGi Workbench — interactive forward/inverse cardiac electrophysiology.

Run live on the workstation (dolfinx/PETSc); VS Code forwards the port to your
laptop browser:

    streamlit run app.py

Workflow (Localisation lab):
  1. Choose tissue: an infarct placed *between* the training infarct sites
     (an in-distribution test), or a healthy beat.
  2. Simulate the beat (monodomain V_m -> extracellular HSP).
  3. Reconstruct the HSP across time frames with the stabFEM inverse, on the
     matching POD database (scar prior for infarcts, healthy prior for healthy).
  4. Inspect truth vs recovered per frame, the localisation error, and play the
     frames as an animation so the infarct's evolution is visible.

The app writes nothing to disk: the only file-writing step (the region marker)
runs in a temporary directory, and all 3D views render in the browser.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from ecgi.cases import InfarctSpec
from ecgi.rendering import Renderer
from ui import components as C

st.set_page_config(page_title="ECGi Workbench", page_icon="🫀", layout="wide")
W = "stretch"

geo = C.get_geometry()
sim = C.get_simulator(geo)
inv = C.get_inverse(geo)
dbs = C.databases()

# ----------------------------------------------------------------------------
# Sidebar: tissue (-> database) + inverse / simulation settings
# ----------------------------------------------------------------------------
st.sidebar.title("🫀 ECGi Workbench")
st.sidebar.caption("Live forward & inverse cardiac electrophysiology (stabFEM).")

tissue = st.sidebar.radio("Tissue", ["Infarct", "Healthy"], horizontal=True)
db_name = "infarction" if tissue == "Infarct" else "four_params"
database = dbs[db_name]
st.sidebar.caption(f"Prior database: **{db_name}** — {database.description}")

# clear stale results when the tissue (hence database) changes
if st.session_state.get("last_tissue") != tissue:
    for k in ("series", "result", "spec"):
        st.session_state.pop(k, None)
    st.session_state["last_tissue"] = tissue

st.sidebar.header("Inverse settings")
max_modes = max(2, database.n_extended)
n_modes = st.sidebar.slider("POD modes (n_modes)", 2, max_modes, min(9, max_modes),
                            help="The real regulariser: more modes = richer but noisier.")
noise = st.sidebar.slider("Measurement noise (% of BSP RMS)", 0, 20, 0, 1)
region = st.sidebar.radio("Electrodes", ["Full torso", "Anterior patch (~27%)"])
gamma_reg = st.sidebar.select_slider("γ_reg (conditioning)", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
st.sidebar.header("Simulation")
t_end = st.sidebar.slider("Window (ms)", 80, 400, 200, 10)
n_frames = st.sidebar.slider("Reconstructed frames", 6, 24, 12,
                             help="How many evenly-spaced snapshots to reconstruct/animate.")
measured = C.anterior_patch(0.27) if region.startswith("Anterior") else None

# ----------------------------------------------------------------------------
forward_tab, lab_tab = st.tabs(["① Forward viewer", "② Infarct-localisation lab"])

# ============================================================================
# Localisation lab
# ============================================================================
with lab_tab:
    spec = InfarctSpec.healthy()
    left, right = st.columns([3, 2])

    with left:
        if tissue == "Infarct":
            st.markdown("**Step 1 — position the infarct between two training sites**")
            sites = C.training_centres()
            pairs = C.adjacent_pairs()
            pi = st.selectbox("Between which two training sites?", range(len(pairs)),
                              format_func=lambda k: f"site {pairs[k][0] + 1}  ↔  site {pairs[k][1] + 1}")
            i, j = pairs[pi]
            blend = st.slider("Position along the segment", 0.0, 1.0, 0.5, 0.05,
                              help="0 = first site, 1 = second site, 0.5 = midway between them")
            radius = st.slider("Scar radius (mm)", 6.0, 16.0, 12.0, 0.5)
            centre = C.between_centre(i, j, blend)
            spec = InfarctSpec(centre_mm=tuple(float(x) for x in centre), radius_mm=radius)
            st.plotly_chart(Renderer.sites(geo, sites, selected=centre), width=W)
        else:
            st.markdown("**Step 1 — healthy beat (no infarct)**")
            st.info("A healthy beat will be simulated and reconstructed with the healthy prior.")
            st.plotly_chart(Renderer.sites(geo, C.training_centres()), width=W)

    with right:
        st.markdown("**Step 2 — simulate & reconstruct**")
        if tissue == "Infarct":
            st.success(f"Infarct between site {i + 1} and site {j + 1} "
                       f"({centre[0]:+.0f}, {centre[1]:+.0f}, {centre[2]:+.0f}) mm, r = {radius:.0f} mm")
        if st.button("▶ Simulate & reconstruct", type="primary", width=W):
            bar = st.progress(0.0, text="Forward: monodomain + extracellular…")
            res = sim.simulate(spec, t_end_ms=float(t_end),
                               progress=lambda f: bar.progress(min(0.5 * f, 0.5)))
            idx = np.unique(np.linspace(0, res.snapshot_count() - 1, n_frames).round().astype(int))
            bar.progress(0.5, text="Inverse: stabFEM reconstruction per frame…")
            series = inv.reconstruct_series(
                res.hsp[idx], res.times_ms[idx], database=database, n_modes=n_modes,
                noise_frac=noise / 100.0, gamma_reg=gamma_reg, measured_vertices=measured,
                progress=lambda f: bar.progress(0.5 + 0.5 * f))
            bar.empty()
            st.session_state.update(result=res, spec=spec, frame_idx=idx, series=series)

        series = st.session_state.get("series")
        if series:
            res = st.session_state["result"]; spec = st.session_state["spec"]
            idx = st.session_state["frame_idx"]; times = res.times_ms[idx]
            recovered = np.array([r.hsp_recovered for r in series])
            best = int(np.argmin([r.rel_l2 for r in series]))
            st.metric("lowest rel. L² error", f"{series[best].rel_l2:.3f}",
                      help=f"‖recovered − truth‖ / ‖truth‖, at t = {times[best]:.0f} ms")
            if not spec.is_healthy:
                errs = [C.localisation_error_mm(geo, recovered[f], spec.centre())[0]
                        for f in range(len(idx))]
                b = int(np.argmin(errs))
                st.metric("best localisation", f"{errs[b]:.0f} mm",
                          help=f"recovered extremum vs true scar, at t = {times[b]:.0f} ms")
            st.caption(f"database: {db_name} · n_modes={n_modes} · noise={noise}% · "
                       f"MINRES iters≈{series[best].iterations}")

    # ---- results: single-frame or animation -------------------------------
    series = st.session_state.get("series")
    if series:
        res = st.session_state["result"]; idx = st.session_state["frame_idx"]
        times = res.times_ms[idx]
        recovered = np.array([r.hsp_recovered for r in series])
        truth = np.array([r.hsp_truth for r in series])
        vmax = float(np.abs(np.concatenate([truth, recovered])).max()) or 1.0
        op, of = geo.outer_surface

        st.divider()
        view = st.radio("View", ["Single frame", "Animate truth HSP", "Animate recovered HSP"],
                        horizontal=True)
        if view == "Single frame":
            f = st.select_slider("Frame (ms)", options=list(range(len(idx))),
                                 format_func=lambda i: f"{times[i]:.0f}")
            g1, g2 = st.columns(2)
            g1.plotly_chart(Renderer.hsp(geo, truth[f], title=f"Truth HSP  t={times[f]:.0f} ms", vmax=vmax),
                            width=W)
            g2.plotly_chart(Renderer.hsp(geo, recovered[f],
                            title=f"Recovered  t={times[f]:.0f} ms  rel-L²={series[f].rel_l2:.2f}", vmax=vmax),
                            width=W)
        else:
            field = truth if "truth" in view else recovered
            label = "truth" if "truth" in view else "recovered"
            st.plotly_chart(
                Renderer.animation(op, of, field, times, colorscale="RdBu", reversescale=True,
                                   cmin=-vmax, cmax=vmax, colorbar_title="u_e [mV]",
                                   title=f"HSP ({label}) over time — press ▶"),
                width=W)

# ============================================================================
# Forward viewer
# ============================================================================
with forward_tab:
    st.subheader("Forward simulation viewer")
    res = st.session_state.get("result")
    if res is None:
        st.info("Use the **Localisation lab** to choose tissue and simulate; the beat shows here.")
    else:
        spec = st.session_state["spec"]
        kind = "healthy" if spec.is_healthy else f"infarct r={spec.radius_mm:.0f} mm"
        st.caption(f"{kind} — {res.snapshot_count()} snapshots over {t_end} ms.")
        quantity = st.radio("Field", ["Transmembrane", "Extracellular"], horizontal=True)
        fidx = np.unique(np.linspace(0, res.snapshot_count() - 1, min(res.snapshot_count(), 12))
                         .round().astype(int))
        times = res.times_ms[fidx]
        if quantity == "Transmembrane":
            pts, faces = geo.heart_surface
            st.plotly_chart(
                Renderer.animation(pts, faces, res.vm[fidx], times, colorscale="Turbo",
                                   cmin=-80.0, cmax=20.0, colorbar_title="V_m [mV]",
                                   title="Transmembrane over time — press ▶"), width=W)
        else:
            op, of = geo.outer_surface
            vmax = float(np.abs(res.hsp).max())
            st.plotly_chart(
                Renderer.animation(op, of, res.hsp[fidx], times, colorscale="RdBu", reversescale=True,
                                   cmin=-vmax, cmax=vmax, colorbar_title="u_e [mV]",
                                   title="Extracellular over time — press ▶"), width=W)
