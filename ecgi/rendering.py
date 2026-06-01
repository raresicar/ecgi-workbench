"""Plotly 3D surface rendering of fields on the heart/torso meshes.

Plotly's ``Mesh3d`` draws an interactive (rotate/zoom) WebGL surface in the
browser — which works perfectly over VS Code's SSH port forwarding — with a
colorbar that plays the role of the value legend in the thesis figures. V_m uses
the Turbo map over a fixed [V_rest, V_dep] range; extracellular potentials use a
symmetric red/blue (RdBu_r) map.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .geometry import Geometry

# anterior-ish camera, echoing the matplotlib elev=18 / azim=-62 used in the thesis
_HEART_CAMERA = dict(eye=dict(x=0.9, y=-1.7, z=0.6), up=dict(x=0, y=0, z=1))
_TORSO_CAMERA = dict(eye=dict(x=0.0, y=-1.9, z=0.4), up=dict(x=0, y=0, z=1))


class Renderer:
    """Builds Plotly figures for scalar fields defined on a triangle surface."""

    @staticmethod
    def surface_figure(
        points: np.ndarray,
        faces: np.ndarray,
        values: np.ndarray,
        *,
        colorscale: str,
        cmin: float,
        cmax: float,
        reversescale: bool = False,
        colorbar_title: str = "",
        title: str = "",
        camera: dict | None = None,
        height: int = 560,
    ) -> go.Figure:
        """A single ``Mesh3d`` surface coloured by per-vertex ``values``."""
        mesh = go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=values, intensitymode="vertex",
            colorscale=colorscale, reversescale=reversescale, cmin=cmin, cmax=cmax,
            colorbar=dict(title=colorbar_title, thickness=14, len=0.7),
            flatshading=False, lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1),
            showscale=True,
        )
        fig = go.Figure(mesh)
        fig.update_layout(
            title=title, height=height, margin=dict(l=0, r=0, t=36, b=0),
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode="data", camera=camera or _HEART_CAMERA,
            ),
        )
        return fig

    # -- convenience wrappers tied to the thesis quantities/legends --------

    @classmethod
    def vm(cls, geo: Geometry, vm_vertex: np.ndarray, *, title: str = "Transmembrane potential V_m") -> go.Figure:
        pts, faces = geo.heart_surface
        return cls.surface_figure(pts, faces, vm_vertex, colorscale="Turbo",
                                  cmin=-80.0, cmax=20.0, colorbar_title="V_m [mV]", title=title)

    @classmethod
    def hsp(cls, geo: Geometry, hsp_outer: np.ndarray, *, title: str = "Heart-surface potential u_e",
            vmax: float | None = None) -> go.Figure:
        pts, faces = geo.outer_surface
        vmax = float(np.max(np.abs(hsp_outer))) if vmax is None else vmax
        return cls.surface_figure(pts, faces, hsp_outer, colorscale="RdBu", reversescale=True,
                                  cmin=-vmax, cmax=vmax, colorbar_title="u_e [mV]", title=title)

    @classmethod
    def bsp(cls, geo: Geometry, bsp_body: np.ndarray, *, title: str = "Body-surface potential u_T",
            vmax: float | None = None) -> go.Figure:
        pts, faces = geo.body_surface
        vmax = float(np.max(np.abs(bsp_body))) if vmax is None else vmax
        return cls.surface_figure(pts, faces, bsp_body, colorscale="RdBu", reversescale=True,
                                  cmin=-vmax, cmax=vmax, colorbar_title="u_T [mV]",
                                  title=title, camera=_TORSO_CAMERA)

    # ------------------------------------------------------------------
    # Interactive infarct picker: an opaque heart + clickable candidate sites
    # ------------------------------------------------------------------
    @classmethod
    def picker(
        cls,
        geo: Geometry,
        candidates: np.ndarray,
        *,
        selected: np.ndarray | None = None,
        radius_mm: float | None = None,
        title: str = "Click a point to place the infarct",
    ) -> go.Figure:
        """Opaque epicardium + a clickable Scatter3d of candidate scar centres.

        Use with ``st.plotly_chart(fig, on_select="rerun")``; the clicked point's
        (x, y, z) is snapped to the nearest epicardial vertex by the caller.
        """
        pts, faces = geo.outer_surface
        heart = go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color="#d9a5a1", opacity=1.0, hoverinfo="skip",
            lighting=dict(ambient=0.75, diffuse=0.5, specular=0.1), showscale=False,
        )
        clickable = go.Scatter3d(
            x=candidates[:, 0], y=candidates[:, 1], z=candidates[:, 2],
            mode="markers", name="candidate sites",
            marker=dict(size=3, color="#1f6f54", opacity=0.45),
            hovertemplate="place infarct here<extra></extra>",
        )
        data = [heart, clickable]
        if selected is not None:
            data.append(go.Scatter3d(
                x=[selected[0]], y=[selected[1]], z=[selected[2]], mode="markers",
                name="infarct centre",
                marker=dict(size=9, color="#1d4ed8", symbol="circle"),
                hovertemplate="infarct centre<extra></extra>",
            ))
        fig = go.Figure(data)
        fig.update_layout(
            title=title, height=560, margin=dict(l=0, r=0, t=36, b=0),
            showlegend=False,
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                       zaxis=dict(visible=False), aspectmode="data", camera=_HEART_CAMERA),
        )
        return fig

    # ------------------------------------------------------------------
    # Time animation: play/slider over snapshots (the "render as a GIF" view)
    # ------------------------------------------------------------------
    @classmethod
    def animation(
        cls,
        points: np.ndarray,
        faces: np.ndarray,
        frame_values: np.ndarray,      # (n_frames, n_vertices)
        times_ms: np.ndarray,          # (n_frames,)
        *,
        colorscale: str,
        cmin: float,
        cmax: float,
        reversescale: bool = False,
        colorbar_title: str = "",
        title: str = "",
        camera: dict | None = None,
        height: int = 600,
    ) -> go.Figure:
        """A Plotly Mesh3d that animates ``frame_values`` over time with a play
        button + slider — the infarct's evolution rendered sequentially."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        def _mesh(vals):
            return go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k, intensity=vals, intensitymode="vertex",
                colorscale=colorscale, reversescale=reversescale, cmin=cmin, cmax=cmax,
                colorbar=dict(title=colorbar_title, thickness=14, len=0.7),
                lighting=dict(ambient=0.7, diffuse=0.5, specular=0.1), showscale=True,
            )

        frames = [go.Frame(data=[_mesh(frame_values[t])], name=f"{times_ms[t]:.0f}")
                  for t in range(len(times_ms))]
        fig = go.Figure(data=[_mesh(frame_values[0])], frames=frames)
        play = dict(
            type="buttons", showactive=False, x=0.05, y=0.05, xanchor="left", yanchor="bottom",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=250, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )
        slider = dict(
            active=0, x=0.15, len=0.8, y=0, yanchor="top",
            currentvalue=dict(prefix="t = ", suffix=" ms"),
            steps=[dict(method="animate", label=f"{times_ms[t]:.0f}",
                        args=[[f"{times_ms[t]:.0f}"],
                              dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                   for t in range(len(times_ms))],
        )
        fig.update_layout(
            title=title, height=height, margin=dict(l=0, r=0, t=36, b=0),
            updatemenus=[play], sliders=[slider],
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                       zaxis=dict(visible=False), aspectmode="data",
                       camera=camera or _HEART_CAMERA),
        )
        return fig
