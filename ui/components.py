"""Shared UI helpers: cached heavy resources, preset infarct sites, small utils.

The geometry / simulator / inverse objects load meshes and factor operators, so
they're cached once per session with ``st.cache_resource`` (the leading ``_`` on
arguments tells Streamlit not to hash the unhashable dolfinx objects).
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from ecgi.config import Database, available_databases
from ecgi.geometry import Geometry
from ecgi.forward import ForwardSimulator
from ecgi.inverse import InverseSolver


@st.cache_resource(show_spinner="Loading heart & torso meshes…")
def get_geometry() -> Geometry:
    return Geometry()


@st.cache_resource(show_spinner="Preparing the forward simulator…")
def get_simulator(_geo: Geometry) -> ForwardSimulator:
    return ForwardSimulator(_geo)


@st.cache_resource(show_spinner="Preparing the inverse solver…")
def get_inverse(_geo: Geometry) -> InverseSolver:
    return InverseSolver(_geo)


@st.cache_data(show_spinner=False)
def databases() -> dict[str, Database]:
    return available_databases()


@st.cache_data(show_spinner=False)
def candidate_points(max_points: int = 700) -> np.ndarray:
    """A subsample of EPI∪BASE points to draw as clickable scar-site markers.

    The full 4914-point cloud is too dense to click cleanly, so we evenly thin it
    to ``max_points`` while keeping the anterior (camera-facing) wall denser."""
    pts, _ = get_geometry().outer_surface
    if pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(0)
    view = np.array([0.45, -0.84, 0.31]); view /= np.linalg.norm(view)
    # weight anterior points higher so the visible wall is well covered
    w = (pts @ view - (pts @ view).min()) + 1.0
    idx = rng.choice(pts.shape[0], size=max_points, replace=False, p=w / w.sum())
    return pts[idx]


def snap_to_epicardium(xyz) -> np.ndarray:
    """Nearest EPI∪BASE vertex to a clicked 3D point (the actual scar centre)."""
    pts, _ = get_geometry().outer_surface
    return pts[int(np.argmin(np.linalg.norm(pts - np.asarray(xyz, float), axis=1)))]


@st.cache_data(show_spinner=False)
def anterior_patch(frac: float) -> np.ndarray:
    """The body vertices nearest the heart (≈ an anterior precordial electrode
    patch), as a fraction of the full body surface."""
    geo = get_geometry()
    body_xyz = geo.torso_mesh.geometry.x[geo.body_vertices, :3]
    heart_centre = geo.torso_mesh.geometry.x[
        get_inverse(geo)._partition["outer_gamma_vertices"], :3].mean(0)
    d = np.linalg.norm(body_xyz - heart_centre, axis=1)
    k = max(3, int(round(frac * geo.body_vertices.size)))
    return geo.body_vertices[np.argsort(d)[:k]].astype(np.int64)


def localisation_error_mm(geo: Geometry, recovered_hsp: np.ndarray,
                          true_centre: np.ndarray) -> tuple[float, np.ndarray]:
    """Distance from the recovered HSP's extremum (its most likely scar marker)
    to the true scar centre. Most meaningful on a plateau snapshot."""
    pts, _ = geo.outer_surface
    hot = pts[int(np.argmax(np.abs(recovered_hsp)))]
    return float(np.linalg.norm(hot - true_centre)), hot
