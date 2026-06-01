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


def snap_to_epicardium(xyz) -> np.ndarray:
    """Nearest EPI∪BASE vertex to a 3D point (so a chosen centre lies on the wall)."""
    pts, _ = get_geometry().outer_surface
    return pts[int(np.argmin(np.linalg.norm(pts - np.asarray(xyz, float), axis=1)))]


# The infarction database was trained with scars at these epicardial sites; we
# recompute them (farthest-point sampling over the anterior wall, the same recipe)
# so a held-out infarct can be placed *between* them — an in-distribution test.
_N_TRAIN_SITES = 18
_SCORE_PERCENTILE = 88.0


@st.cache_data(show_spinner=False)
def training_centres() -> np.ndarray:
    """The (n, 3) epicardial sites the scar database was trained on."""
    pts, _ = get_geometry().outer_surface
    view = np.array([0.45, -0.84, 0.31]); view /= np.linalg.norm(view)
    cand = pts[(pts @ view) >= np.percentile(pts @ view, _SCORE_PERCENTILE)]
    chosen = [int(np.argmax(cand @ view))]
    dist = np.linalg.norm(cand - cand[chosen[0]], axis=1)
    while len(chosen) < _N_TRAIN_SITES:
        nxt = int(np.argmax(dist)); chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(cand - cand[nxt], axis=1))
    return cand[chosen]


@st.cache_data(show_spinner=False)
def adjacent_pairs() -> list[tuple[int, int]]:
    """Each training site paired with its nearest neighbour (unique, sorted) —
    the segments a held-out infarct can sit *between*."""
    c = training_centres()
    seen: set[tuple[int, int]] = set()
    for i in range(len(c)):
        d = np.linalg.norm(c - c[i], axis=1); d[i] = np.inf
        j = int(np.argmin(d))
        seen.add((min(i, j), max(i, j)))
    return sorted(seen)


def between_centre(i: int, j: int, blend: float) -> np.ndarray:
    """A point on the segment between training sites ``i`` and ``j`` (``blend`` in
    [0, 1]), snapped to the epicardium."""
    c = training_centres()
    return snap_to_epicardium((1.0 - blend) * c[i] + blend * c[j])


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
