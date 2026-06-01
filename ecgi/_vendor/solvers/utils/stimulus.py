"""Shell (Purkinje-surrogate) stimulus from Boulakia et al., "Mathematical
Modeling of ECGs".

Activation is initiated by a time-dependent volume current on a thin
subendocardial layer:

    I_app(x,t) = I0(x) chi_S(x) chi_[0,tact](t) psi(theta(x), t)

with ``S = {c1 <= a dx^2 + b dy^2 + c dz^2 <= c2}`` a fitted ellipsoidal
shell, a linearly tapered amplitude across it, and a rotating angular gate
``alpha(t) = 2*pi*t/tact`` that sweeps the activation around the chamber.

Returned as a **UFL expression of the ``time`` Constant** so the solver drives
it implicitly -- no per-step update callback. The two spatial fields (taper and
angle) are precomputed once as CG1 Functions; the gate is applied at assembly
time via ``conditional(theta <= alpha(time), taper, 0)``.

Generic and config-agnostic: amplitude / tact / thickness / markers are passed
in explicitly (the UKB pipeline sources them from params.json).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import dolfinx
import numpy as np
import ufl

from utils.mesh import tagged_facet_vertices


@dataclass
class StimulusChamber:
    name: str
    marker: int
    centre: list[float]
    coefficients: list[float]
    c1: float
    c2: float
    angle_origin_xz: list[float]
    n_shell_dofs: int


@dataclass
class ShellStimulus:
    expr: ufl.core.expr.Expr            # I_app as a UFL expression of `time`
    seed_mask: np.ndarray               # all-false: current source, not a seed
    metadata: dict
    fields: list                        # keep CG1 coefficient Functions alive


def _fit_diagonal_ellipsoid(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit q(x)=a dx^2 + b dy^2 + c dz^2 ~= 1 on surface points."""
    centre = points.mean(axis=0)
    shifted = points - centre
    A = shifted**2
    coeffs, *_ = np.linalg.lstsq(A, np.ones(points.shape[0]), rcond=None)
    coeffs = np.maximum(coeffs, 1.0e-12)
    return centre, coeffs


def _shell_bounds(coeffs: np.ndarray, thickness_mm: float) -> tuple[float, float]:
    # For q=(r/R)^2, a layer of thickness d around q=1 has width ~ 2*d/R. Use
    # the geometric-mean radius of the fitted ellipsoid as a single scale.
    radii = 1.0 / np.sqrt(coeffs)
    r_eff = float(np.exp(np.mean(np.log(np.maximum(radii, 1.0e-12)))))
    dq = 2.0 * thickness_mm / r_eff
    c1 = max(0.0, 1.0 - 0.5 * dq)
    c2 = 1.0 + 0.5 * dq
    return c1, c2


def build_shell_stimulus(
    mesh: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    V: dolfinx.fem.FunctionSpace,
    time: dolfinx.fem.Constant,
    *,
    amplitude: float,
    tact_ms: float,
    layer_thickness_mm: float,
    lv_marker: int,
    rv_marker: int,
    chambers: tuple[str, ...] = ("lv", "rv"),
    centre_offset_mm: tuple[float, float, float] | np.ndarray | None = None,
) -> ShellStimulus:
    """Build the shell stimulus as a UFL expression of ``time``.

    The solver advances ``time``; the returned ``expr`` follows it. The
    ``seed_mask`` is all-false (the paper starts from a current source, not a
    pre-depolarised patch).
    """
    coords = V.tabulate_dof_coordinates()[:, :3]
    seed_mask = np.zeros(coords.shape[0], dtype=bool)
    centre_offset = np.zeros(3, dtype=np.float64)
    if centre_offset_mm is not None:
        centre_offset = np.asarray(centre_offset_mm, dtype=np.float64)
        if centre_offset.shape != (3,):
            raise ValueError("centre_offset_mm must contain three values")

    chamber_markers = {"lv": lv_marker, "rv": rv_marker}
    chamber_names = tuple(str(name).lower() for name in chambers)
    bad = sorted(set(chamber_names) - set(chamber_markers))
    if bad:
        raise ValueError(f"unsupported stimulus chambers: {bad}")
    if not chamber_names:
        raise ValueError("at least one stimulus chamber must be selected")

    # alpha(t) = 2*pi*t/tact, active only on the window [0, tact].
    alpha = 2.0 * np.pi * time / tact_ms
    time_gate = ufl.And(ufl.ge(time, 0.0), ufl.le(time, tact_ms))

    contribs: list[ufl.core.expr.Expr] = []
    fields: list[dolfinx.fem.Function] = []
    chamber_meta: list[StimulusChamber] = []

    for name in chamber_names:
        marker = chamber_markers[name]
        vertices = tagged_facet_vertices(mesh, facet_tags, marker)
        points = mesh.geometry.x[vertices, :3]
        centre, coeffs = _fit_diagonal_ellipsoid(points)
        centre = centre + centre_offset
        c1, c2 = _shell_bounds(coeffs, layer_thickness_mm)

        shifted = coords - centre
        q = np.einsum("ij,j,ij->i", shifted, coeffs, shifted)
        shell = (q >= c1) & (q <= c2)
        # Tapered amplitude across the shell, zero outside it.
        taper_arr = amplitude * (c2 - q) / (c2 - c1)
        taper_arr = np.where(shell, np.maximum(taper_arr, 0.0), 0.0)
        # Plane is drawn as (x, -z) in the paper; angle sweeps from the apex.
        theta_arr = np.mod(
            np.arctan2(coords[:, 0] - centre[0], -(coords[:, 2] - centre[2])),
            2.0 * np.pi,
        )

        taper_fn = dolfinx.fem.Function(V, name=f"stim_taper_{name}")
        taper_fn.x.array[:] = taper_arr
        taper_fn.x.scatter_forward()
        theta_fn = dolfinx.fem.Function(V, name=f"stim_theta_{name}")
        theta_fn.x.array[:] = theta_arr
        theta_fn.x.scatter_forward()
        fields.extend([taper_fn, theta_fn])

        # Active where the sweep has reached this dof; taper is already 0
        # outside the shell, so no separate shell mask is needed here.
        contribs.append(ufl.conditional(ufl.le(theta_fn, alpha), taper_fn, 0.0))

        chamber_meta.append(
            StimulusChamber(
                name=name,
                marker=int(marker),
                centre=[float(x) for x in centre],
                coefficients=[float(x) for x in coeffs],
                c1=float(c1),
                c2=float(c2),
                angle_origin_xz=[float(centre[0]), float(centre[2])],
                n_shell_dofs=int(shell.sum()),
            )
        )

    swept = contribs[0]
    for extra in contribs[1:]:
        swept = swept + extra
    expr = ufl.conditional(time_gate, swept, 0.0)

    metadata = {
        "source": "Boulakia et al. Mathematical Modeling of Electrocardiograms, Appendix A",
        "formula": "I_app = I0 chi_S chi_[0,tact] psi, alpha(t)=2*pi*t/tact",
        "note": "Paper ellipsoid constants are not published; constants are "
                "fitted to the LV/RV endocardial facets. Driven as a UFL "
                "expression of the time Constant (gate evaluated at quadrature "
                "points).",
        "amplitude": float(amplitude),
        "tact_ms": float(tact_ms),
        "layer_thickness_mm": float(layer_thickness_mm),
        "centre_offset_mm": [float(x) for x in centre_offset],
        "selected_chambers": list(chamber_names),
        "chambers": [asdict(item) for item in chamber_meta],
    }
    return ShellStimulus(
        expr=expr, seed_mask=seed_mask, metadata=metadata, fields=fields,
    )
