"""ECGi Workbench — interactive forward/inverse cardiac electrophysiology.

Run live on the workstation (dolfinx/PETSc); VS Code forwards the port to your
laptop browser:

    streamlit run app.py

Workflow (Localisation lab):
  1. Click a point on the heart to place an infarct, set its radius.
  2. Simulate the beat (monodomain V_m -> extracellular HSP).
  3. Reconstruct the HSP across time frames with the stabFEM inverse on a chosen
     POD database, and see how/where the infarct is localised.
  4. Play the frames as an animation so the infarct's evolution is visible.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from ecgi.cases import InfarctSpec
from ecgi.rendering import Renderer
from ui import components as C

st.set_page_config(page_title="ECGi Workbench", page_icon="🫀", layout="wide")
W = "stretch"  # st width for full-width charts

geo = C.get_geometry()
sim = C.get_simulator(geo)
inv = C.get_inverse(geo)
dbs = C.databases()


def _clicked_points(event) -> list:
    """Pull clicked points out of st.plotly_chart's on_select return value,
    tolerating both dict-style and attribute-style state objects."""
    if not event:
        return []
    sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
    if sel is None:
        return []
    pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    return list(pts) if pts else []

# ----------------------------------------------------------------------------
# Sidebar: inverse / simulation settings (apply to the lab)
# ----------------------------------------------------------------------------
st.sidebar.title("🫀 ECGi Workbench")
st.sidebar.caption("Live forward & inverse cardiac electrophysiology (stabFEM).")

st.sidebar.header("Inverse settings")
db_name = st.sidebar.selectbox("POD database (prior)", list(dbs),
                               format_func=lambda n: f"{n}", help="The data-enriched prior basis.")
st.sidebar.caption(dbs[db_name].description)
n_modes = st.sidebar.slider("POD modes (n_modes)", 2, 30, 9,
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
    st.subheader("Place an infarct, simulate, and localise it")
    left, right = st.columns([3, 2])

    with left:
        st.markdown("**Step 1 — click a point on the heart to place the infarct**")
        centre = st.session_state.get("centre")
        radius = st.slider("Scar radius (mm)", 6.0, 16.0, 12.0, 0.5)
        fig = Renderer.picker(geo, C.candidate_points(),
                              selected=None if centre is None else np.asarray(centre),
                              radius_mm=radius)
        event = st.plotly_chart(fig, key="picker", on_select="rerun",
                                selection_mode="points", width=W)
        # capture the clicked point and snap it to the epicardium (the return type
        # supports both attribute and key access depending on Streamlit version)
        for p in _clicked_points(event):
            if all(k in p for k in ("x", "y", "z")):
                c = C.snap_to_epicardium((p["x"], p["y"], p["z"]))
                if st.session_state.get("centre") != tuple(float(x) for x in c):
                    st.session_state["centre"] = tuple(float(x) for x in c)
                    st.rerun()

        with st.expander("Click not registering? Pick a preset site instead"):
            sites = C.preset_sites()
            ch = st.selectbox("Preset infarct site", range(len(sites)),
                              format_func=lambda i: sites[i][0])
            if st.button("Use this site", width=W):
                st.session_state["centre"] = sites[ch][1]
                st.rerun()

    with right:
        st.markdown("**Step 2 — simulate & reconstruct**")
        if centre is None:
            st.info("Click a green candidate point on the heart to choose the infarct centre.")
        else:
            st.success(f"Infarct centre: ({centre[0]:+.0f}, {centre[1]:+.0f}, {centre[2]:+.0f}) mm, "
                       f"r = {radius:.0f} mm")
            if st.button("▶ Simulate & reconstruct", type="primary", width=W):
                spec = InfarctSpec(centre_mm=tuple(float(x) for x in centre), radius_mm=radius)
                bar = st.progress(0.0, text="Forward: monodomain + extracellular…")
                res = sim.simulate(spec, t_end_ms=float(t_end),
                                   progress=lambda f: bar.progress(min(0.5 * f, 0.5)))
                # evenly-spaced frames for reconstruction/animation
                idx = np.unique(np.linspace(0, res.snapshot_count() - 1, n_frames).round().astype(int))
                bar.progress(0.5, text="Inverse: stabFEM reconstruction per frame…")
                series = inv.reconstruct_series(
                    res.hsp[idx], res.times_ms[idx], database=dbs[db_name], n_modes=n_modes,
                    noise_frac=noise / 100.0, gamma_reg=gamma_reg, measured_vertices=measured,
                    progress=lambda f: bar.progress(0.5 + 0.5 * f))
                bar.empty()
                st.session_state.update(result=res, spec=spec, frame_idx=idx, series=series)

        series = st.session_state.get("series")
        if series:
            res = st.session_state["result"]; spec = st.session_state["spec"]
            idx = st.session_state["frame_idx"]
            times = res.times_ms[idx]
            recovered = np.array([r.hsp_recovered for r in series])
            truth = np.array([r.hsp_truth for r in series])
            # per-frame localisation error (recovered extremum vs true scar)
            errs = [C.localisation_error_mm(geo, recovered[f], spec.centre())[0] for f in range(len(idx))]
            best = int(np.argmin(errs))
            st.metric("best localisation", f"{errs[best]:.0f} mm",
                      help=f"at t = {times[best]:.0f} ms (recovered extremum vs true centre)")
            st.caption(f"cosine at that frame: {series[best].cosine:.3f} · "
                       f"MINRES iters: {series[best].iterations}")

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
                            title=f"Recovered  t={times[f]:.0f} ms  cos={series[f].cosine:.2f}", vmax=vmax),
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
        st.info("Use the **Localisation lab** to place an infarct and simulate; the beat shows here.")
    else:
        spec = st.session_state["spec"]
        st.caption(f"Infarct r={spec.radius_mm:.0f} mm — {res.snapshot_count()} snapshots over {t_end} ms.")
        quantity = st.radio("Field", ["V_m (transmembrane)", "u_e (heart surface)"], horizontal=True)
        # animate over a thinned set of frames (keep the figure light)
        fidx = np.unique(np.linspace(0, res.snapshot_count() - 1, min(res.snapshot_count(), 12))
                         .round().astype(int))
        times = res.times_ms[fidx]
        if quantity.startswith("V_m"):
            pts, faces = geo.heart_surface
            st.plotly_chart(
                Renderer.animation(pts, faces, res.vm[fidx], times, colorscale="Turbo",
                                   cmin=-80.0, cmax=20.0, colorbar_title="V_m [mV]",
                                   title="V_m over time — press ▶"), width=W)
        else:
            op, of = geo.outer_surface
            vmax = float(np.abs(res.hsp).max())
            st.plotly_chart(
                Renderer.animation(op, of, res.hsp[fidx], times, colorscale="RdBu", reversescale=True,
                                   cmin=-vmax, cmax=vmax, colorbar_title="u_e [mV]",
                                   title="u_e over time — press ▶"), width=W)
