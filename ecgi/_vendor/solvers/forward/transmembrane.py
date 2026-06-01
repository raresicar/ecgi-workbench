"""Step 1 of the monodomain + heart–torso uncoupled pipeline: transmembrane potential.

We obtain V_m on the heart by solving the monodomain equation — the
standard single-variable reduction of the bidomain model (valid when
σ_i and σ_e are proportional, i.e. equal anisotropy ratios).

    A_m (C_m ∂V_m/∂t + I_ion(V_m, w)) − div(M ∇V_m) = A_m I_stim     in  Ω_H
                            M ∇V_m · n      = 0          on  Gamma_H
    with  M = σ_i σ_e / (σ_i + σ_e)  (the "harmonic-mean" effective tensor).

We integrate in time with Godunov (first-order) operator splitting:
  (a) an ODE step that advances the membrane state via a pointwise ionic
      model (Mitchell–Schaeffer, FitzHugh–Nagumo, etc. — passed in as a
      callable, see :mod:`cell_models.mitchell_schaeffer`);
  (b) a PDE step that advances the diffusive part using a backward-Euler
      scheme. fenicsx-beat's :class:`beat.MonodomainSplittingSolver`
      orchestrates both.

File pipeline:
mesh + time
    ↓
stimulus expression + seed mask
    ↓
ionic model initial state
    ↓
fenicsx-beat MonodomainModel
    ↓
fenicsx-beat DolfinODESolver
    ↓
MonodomainSplittingSolver
    ↓
V_m(x,t)

This module exposes two building blocks:

  * :func:`build_stimulus`  — disc-indicator stimulus current with a
    matching boolean seed mask for pre-depolarising the cell-model initial state.
  * :func:`build_solver`    — wires up the PDE, ODE, and splitting solver
    around a user-supplied ionic-step closure and returns
    ``(splitting_solver, v_pde_function)``. The caller drives the time
    loop and reads V_m from ``v_pde.x.array``.
"""
from __future__ import annotations

from dataclasses import dataclass

import dolfinx
import numpy as np
import ufl

import beat

# Solver-side utilities (config-agnostic; the pipeline feeds them params.json
# values). Imported here so build_monodomain_solver can assemble the whole
# problem and the pipeline only has to drive the time loop.
from ionic_models.mitchell_schaeffer import initial_state, make_step
from utils.heterogeneity import build_tau_close_field
from utils.stimulus import build_shell_stimulus

# No default physical parameters live here on purpose: every caller passes
# them explicitly (the pipeline pulls them from params.json via
# common.load_params; the database samples them per-sample). The stimulus
# current applied by build_stimulus is
#     I_stim(x,t) = A · 1_{dist(x, centre) ≤ radius} · 1_{t < duration}


def build_stimulus(
    mesh: dolfinx.mesh.Mesh,
    time: dolfinx.fem.Constant,
    centre: tuple[float, float, float],
    radius: float,
    amplitude: float,
    duration: float,
) -> tuple[ufl.core.expr.Expr, np.ndarray]:
    """Disc-indicator stimulus current and matching seed mask.

    Args:
        mesh: heart mesh.
        time: a :class:`dolfinx.fem.Constant` that the time loop
            advances each step. Used to gate the stimulus.
        centre: (x, y) of the stimulus disc.
        radius: disc radius — should be larger than the propagating
            wavefront width δ ≈ √(M / |I_ion'|), otherwise the seed
            cannot sustain a self-propagating wave.
        amplitude: current density inside the disc during the pulse.
        duration: ms, length of the on-pulse.

    Returns:
        ``(stim_expr, seed_mask)``:
          - ``stim_expr`` is a UFL expression for the stimulus current,
            ready to plug into ``beat.MonodomainModel(I_s=...)``. -> applied current
            during the first few ms
          - ``seed_mask`` is a boolean array of shape (n_dofs_local,)
            marking the CG1 dofs inside the disc. The caller uses this
            to pre-depolarise the corresponding cell-model states. -> pre-depolarises
            the same region at t=0
    """
    return build_multi_stimulus(
        mesh, time,
        [{
            "centre": centre, "radius": radius, "amplitude": amplitude,
            "duration": duration, "onset": 0.0,
        }],
    )


def build_multi_stimulus(
    mesh: dolfinx.mesh.Mesh,
    time: dolfinx.fem.Constant,
    stims: list[dict],
) -> tuple[ufl.core.expr.Expr, np.ndarray]:
    """Sum of disc-indicator stimulus currents, each with its own onset.

    Each entry of ``stims`` is a dict with keys:
      ``centre``    — ``(x, y)`` of the disc.
      ``radius``    — disc radius.
      ``amplitude`` — current density inside the disc during the pulse.
                      If ``<= 0``, the stim is treated as inactive and
                      skipped entirely.
      ``duration``  — ms, on-pulse length.
      ``onset``     — ms, start time of the pulse.

    The combined stimulus is

        I_stim(x, t) = Σ_k  A_k · 1_{disc_k}(x) · 1_{onset_k ≤ t < onset_k + dur_k}

    The seed mask returned pre-depolarises only those discs whose
    ``onset`` is essentially zero (so the cell-model initial state is
    consistent with the first wave starting at t=0). Stims with later
    onset rely entirely on the pulse current to depolarise their region
    when the time loop reaches their window.
    """
    cg1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1)) # Lagrange P1 on heart mesh
    # 3D tetrahedral CG1: dofs sit at the mesh vertices, full (x,y,z) needed.
    coords = cg1.tabulate_dof_coordinates()

    total_expr: ufl.core.expr.Expr | None = None
    seed_mask = np.zeros(coords.shape[0], dtype=bool)
    n_active = 0

    for k, stim in enumerate(stims):
        amp = float(stim["amplitude"])
        if amp <= 0.0:
            continue
        n_active += 1
        cx = float(stim["centre"][0])
        cy = float(stim["centre"][1])
        cz = float(stim["centre"][2])
        r2 = float(stim["radius"]) ** 2
        onset = float(stim["onset"])
        dur = float(stim["duration"])

        ind = dolfinx.fem.Function(cg1, name=f"stim_region_{k}")
        # indicator function of the 3D ball
        ind.interpolate(
            lambda x, cx=cx, cy=cy, cz=cz, r2=r2: (
                ((x[0] - cx) ** 2 + (x[1] - cy) ** 2 + (x[2] - cz) ** 2) <= r2
            ).astype(np.float64)
        )

        if onset <= 1.0e-9:
            # Fires immediately: simple "time < duration" gate
            pulse = ufl.conditional(ufl.lt(time, dur), amp, 0.0)
            # Pre-depolarise the cell-model initial state inside this ball.
            ball_mask = (
                (coords[:, 0] - cx) ** 2
                + (coords[:, 1] - cy) ** 2
                + (coords[:, 2] - cz) ** 2
                <= r2
            )
            seed_mask |= ball_mask
        else:
            # Fires later: gate by (onset ≤ t < onset + duration).
            in_window = ufl.And(
                ufl.ge(time, onset), ufl.lt(time, onset + dur)
            )
            pulse = ufl.conditional(in_window, amp, 0.0)

        contrib = ind * pulse
        total_expr = contrib if total_expr is None else total_expr + contrib

    if total_expr is None:
        # No active stim — fall back to a zero current. Useful so callers
        # don't crash on all-zero amplitude draws.
        total_expr = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

    return total_expr, seed_mask


def build_solver(
    mesh: dolfinx.mesh.Mesh,
    time: dolfinx.fem.Constant,
    sigma_i: float,
    sigma_e: float,
    c_m: float,
    stim_expr: ufl.core.expr.Expr,
    cell_step_fun,
    init_states: np.ndarray,
    num_states: int,
    a_m: float,
) -> tuple[beat.MonodomainSplittingSolver, dolfinx.fem.Function]:
    """Assemble the monodomain splitting solver.

    Args:
        mesh: heart mesh.
        time: ``Constant`` advanced by the caller's time loop.
        sigma_i, sigma_e: bulk intra-/extracellular conductivities.
            The harmonic-mean conductivity is σ_i σ_e / (σ_i + σ_e); the
            effective monodomain tensor passed to beat is this divided by
            the surface-to-volume ratio ``a_m`` (see ``a_m`` below).
        c_m: membrane capacitance per unit area.
        a_m: surface-to-volume ratio χ (cm⁻¹). The full monodomain equation
            is χ(C_m ∂V/∂t + I_ion) − div(M ∇V) = χ I_stim; dividing by χ
            and handling I_ion in the operator-split ODE step leaves a PDE
            with effective diffusivity M/χ. beat's MonodomainModel solves
            C_m ∂V/∂t − div(M_eff ∇V) = I_stim with no χ term of its own, so
            we fold χ into the tensor here as M_eff = σ_iσ_e/(σ_i+σ_e)/a_m.
            Required (no default): pass 1.0 to disable χ-scaling when using
            already-effective conductivities.
        stim_expr: UFL expression for I_stim (from :func:`build_stimulus`).
        cell_step_fun: pointwise ionic-model step callable, of signature
            ``fun(t, states, parameters, dt) -> states``. See
            :func:`ionic_models.mitchell_schaeffer.make_step` for an example.
        init_states: ``(num_states, n_dofs_local)`` initial state array.
        num_states: number of state variables in the cell model.

    Returns:
        ``(solver, v_pde)`` — call ``solver.step((t0, t1))`` to advance,
        read V_m from ``v_pde.x.array`` between steps.
    """
    # Isotropic harmonic-mean conductivity, divided by the surface-to-volume
    # ratio χ=a_m so beat's χ-free weak form reproduces the χ-normalised
    # monodomain equation (effective diffusivity M/χ). See the a_m docstring.
    m_mono = 1000 * sigma_i * sigma_e / (sigma_i + sigma_e) / a_m

    # if the stimulus needs to be of type beat.Stimulus:
    # stimulus = beat.Stimulus(expr=stim_expr, dZ=ufl.dx(domain=mesh)) & I_s=stimulus
    pde = beat.MonodomainModel(
        time=time, mesh=mesh, M=m_mono, I_s=stim_expr, C_m=c_m,
    )
    v_pde = pde.state # PDE variable for v_m
    v_ode = dolfinx.fem.Function(v_pde.function_space) # ODE variable for v_m

    # parameters argument is unused — cell-model params are baked into
    # the closure step function cell_step_fun (e.g. make_step) -> dummy array for API satisfaction
    parameters = np.array([0.0])

    ode = beat.odesolver.DolfinODESolver(
        v_ode=v_ode,
        v_pde=v_pde,
        init_states=init_states,
        parameters=parameters,
        fun=cell_step_fun,
        num_states=num_states,
        v_index=0, # v_m is in states[0]
    )
    solver = beat.MonodomainSplittingSolver(pde=pde, ode=ode)
    return solver, v_pde


@dataclass
class MonodomainProblem:
    """Everything a thin pipeline needs after build_monodomain_solver.

    Drive ``solver.step((t0, t1))`` in a loop (advancing ``time``); read V_m
    from ``v_m``. ``region`` / ``tau_close`` / the metadata dicts are exposed
    for the pipeline to save and plot.
    """
    solver: beat.MonodomainSplittingSolver
    v_m: dolfinx.fem.Function
    region: dolfinx.fem.Function
    tau_close: np.ndarray
    heterogeneity_metadata: dict
    stimulus_metadata: dict
    stimulus_fields: list  # keep stimulus coefficient Functions referenced


def build_monodomain_solver(
    mesh: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    V: dolfinx.fem.FunctionSpace,
    time: dolfinx.fem.Constant,
    *,
    # conductivity / membrane
    sigma_i: float,
    sigma_e: float,
    c_m: float,
    a_m: float,
    # Mitchell-Schaeffer kinetics
    tau_in: float,
    tau_out: float,
    tau_open: float,
    v_gate: float,
    # transmural tau_close heterogeneity
    tau_close_endo: float,
    tau_close_mid: float,
    tau_close_epi: float,
    endo_size: float,
    epi_size: float,
    # region markers (also used to fit the stimulus shell)
    lv_marker: int,
    rv_marker: int,
    epi_marker: int,
    # stimulus
    stim_amplitude: float,
    stim_tact_ms: float,
    stim_layer_thickness_mm: float,
    stim_chambers: tuple[str, ...] = ("lv", "rv"),
    stimulus_mode: str = "shell",
    stim_radius_mm: float = 8.0,
) -> MonodomainProblem:
    """Assemble the full Mitchell-Schaeffer monodomain problem.

    Builds, from explicit scalar parameters (the pipeline pulls them from
    params.json): the transmural ``tau_close`` field, the pointwise M-S cell
    step, the resting initial state, and either the paper shell stimulus or a
    simple apex-like ball stimulus. Then wires the generic :func:`build_solver`.
    Returns a :class:`MonodomainProblem`.
    """
    n_dofs_local = mesh.topology.index_map(0).size_local

    # 1. Transmural tau_close field (endo / mid / epi).
    tau_close, region, het_meta = build_tau_close_field(
        V=V, facet_tags=facet_tags,
        lv_marker=lv_marker, rv_marker=rv_marker, epi_marker=epi_marker,
        endo_size=endo_size, epi_size=epi_size,
        tau_close_endo=tau_close_endo, tau_close_mid=tau_close_mid,
        tau_close_epi=tau_close_epi,
    )

    # 2. Pointwise Mitchell-Schaeffer step + resting initial state.
    cell_step = make_step(
        tau_in=tau_in, tau_out=tau_out, tau_open=tau_open,
        tau_close=tau_close, v_gate=v_gate,
    )

    # 3. Stimulus as a UFL expression of `time` (no per-step update).
    mode = stimulus_mode.lower()
    if mode == "shell":
        stim = build_shell_stimulus(
            mesh=mesh, facet_tags=facet_tags, V=V, time=time,
            amplitude=stim_amplitude, tact_ms=stim_tact_ms,
            layer_thickness_mm=stim_layer_thickness_mm,
            lv_marker=lv_marker, rv_marker=rv_marker, chambers=stim_chambers,
        )
        stim_expr = stim.expr
        seed_mask = stim.seed_mask
        stimulus_metadata = stim.metadata
        stimulus_fields = stim.fields
    elif mode == "ball":
        coords = V.tabulate_dof_coordinates()[:, :3]
        centre = tuple(float(x) for x in coords[np.argmin(coords[:, 2])])
        stim_expr, seed_mask = build_stimulus(
            mesh=mesh, time=time, centre=centre, radius=stim_radius_mm,
            amplitude=stim_amplitude, duration=stim_tact_ms,
        )
        stimulus_metadata = {
            "type": "ball",
            "centre_mm": list(centre),
            "radius_mm": float(stim_radius_mm),
            "amplitude": float(stim_amplitude),
            "duration_ms": float(stim_tact_ms),
            "n_seed_dofs": int(seed_mask.sum()),
        }
        stimulus_fields = []
    else:
        raise ValueError(f"unsupported stimulus_mode={stimulus_mode!r}")
    init = initial_state(n_dofs_local, seed_mask=seed_mask)

    # 4. Wire the generic low-level solver.
    solver, v_pde = build_solver(
        mesh=mesh, time=time,
        sigma_i=sigma_i, sigma_e=sigma_e, c_m=c_m, a_m=a_m,
        stim_expr=stim_expr, cell_step_fun=cell_step,
        init_states=init, num_states=2,
    )
    # The splitting solver does not reliably copy the ODE voltage row into the
    # PDE state before the first PDE step (the shell stimulus is a current
    # source with no pre-depolarised patch). Start V_m from the resting state.
    v_pde.x.array[:] = init[0]
    v_pde.x.scatter_forward()

    return MonodomainProblem(
        solver=solver,
        v_m=v_pde,
        region=region,
        tau_close=tau_close,
        heterogeneity_metadata=het_meta,
        stimulus_metadata=stimulus_metadata,
        stimulus_fields=stimulus_fields,
    )
