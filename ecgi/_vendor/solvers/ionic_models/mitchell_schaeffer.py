"""Mitchell–Schaeffer cardiac cell model.
-> does not model many individual ion channels
-> captures the main shape of cardiac excitation with 2 variables

The model uses the normalised voltage v ∈ [0, 1]:

    J_in(v, h)  =  h · v² · (1 − v) / τ_in     (fast inward, depolarising) -> makes the cell fire
    J_out(v)    = − v / τ_out                      (slow outward, repolarising) -> brings the cell back down
    dv/dt       =  J_in + J_out
    dh/dt       = (1 − h) / τ_open    if v < v_gate    (gate recovers)
                = − h / τ_close       if v ≥ v_gate    (gate inactivates)
    
-> h controls if the cell is ready to fire again

To match the convention expected by :func:`forward.transmembrane.build_solver`
(and any other cell-model "step" callables one might plug in), we carry
V_m in physical mV in ``states[0]`` and convert v ↔ V_m at the boundary
of the step:

    v = (V_m − V_rest) / (V_dep − V_rest)
    V_m = V_rest + (V_dep − V_rest) · v

``states[1]`` carries the gate h ∈ [0, 1].  ``num_states = 2`` therefore.
Each point in the heart mesh stores:
states[0, i] = V_m at point i in mV
states[1, i] = h at point i in [0, 1] -> dimensionless gate variable
=> model state has shape (2, num_points) where num_points is the number of points in the heart mesh.
"""
from __future__ import annotations

import numpy as np

# Physical voltage range. These are the canonical V_m bounds for the
# whole pipeline: u_e recovery and torso forward both assume V_m is in
# mV on this scale, so callers that swap in a different cell model
# should keep the same (V_REST, V_DEP) convention.
V_REST = -80.0   # mV — resting membrane potential
V_DEP = 20.0     # mV — depolarised plateau target

# Normalised gating threshold v_gate. Mitchell–Schaeffer uses 0.13,
# which translates to V_gate ≈ -67 mV = -80 + 0.13 * (20 - (-80))
V_GATE_DEFAULT = 0.13

# Default time constants (ms). τ_in is the fast inward current scale and
# sets the forward-Euler stability bound dt ≲ τ_in. τ_close drives APD;
# making it spatially heterogeneous is the standard knob for T-wave shape.
TAU_IN_DEFAULT = 4.5 # speed of the fast depolarising current
TAU_OUT_DEFAULT = 90 # speed of the slow repolarising current
TAU_OPEN_DEFAULT = 100 # speed of h recovery when voltage is low (cell ready to fire)
TAU_CLOSE_DEFAULT = 130 # speed of h recovery when voltage is high (cell just fired, gate inactivates)


def make_step(
    tau_in=TAU_IN_DEFAULT,
    tau_out=TAU_OUT_DEFAULT,
    tau_open=TAU_OPEN_DEFAULT,
    tau_close=TAU_CLOSE_DEFAULT,
    v_rest: float = V_REST,
    v_dep: float = V_DEP,
    v_gate=V_GATE_DEFAULT,
):
    """Build a forward-Euler Mitchell–Schaeffer step closure.

    The returned ``step(t, states, parameters, dt)`` callable matches the
    protocol expected by :class:`beat.odesolver.DolfinODESolver.fun`:
    ``states`` has shape ``(num_states, num_points)`` with ``num_states = 2``
    (rows: V_m in mV, then h in [0, 1]). ``t`` and ``parameters`` are
    unused (the parameters are baked into the closure) but kept in the
    signature for protocol compatibility.

    Each tau parameter and ``v_gate`` may be either a scalar or a per-dof
    ``np.ndarray`` of shape ``(num_points,)``. Arrays let callers pass
    spatial heterogeneity fields (e.g. paper-style tau_close = endo/mid/epi
    via ``utils.heterogeneity.build_tau_close_field``) without changing the
    step function — NumPy broadcasting does the rest.

    Clamps both v and h to [0, 1] each step. Without that, forward Euler
    near v ≈ 0 or v ≈ 1 can produce a tiny negative / >1 value that the
    cubic J_in term then amplifies into a numerical blow-up — a known
    quirk of M–S with naïve explicit time stepping.
    """
    v_span = v_dep - v_rest
    inv_tau_in = 1.0 / np.asarray(tau_in, dtype=np.float64)
    inv_tau_out = 1.0 / np.asarray(tau_out, dtype=np.float64)
    inv_tau_open = 1.0 / np.asarray(tau_open, dtype=np.float64)
    inv_tau_close = 1.0 / np.asarray(tau_close, dtype=np.float64)

    def step(t, states, parameters, dt):
        """
        Compute the next state of the system at time t + dt given the current
        states and parameters at time t.

        Args:
            t (float): Current time in milliseconds.
            states (np.ndarray): Current state array of shape (2, num_points),
                where states[0] is V_m in mV and states[1] is h in [0, 1].
            parameters (np.ndarray): Unused in this model, but included for
                compatibility with the expected protocol.
            dt (float): Time step in milliseconds.
        Returns:
            np.ndarray: Updated state array of shape (2, num_points) after            
                applying the Mitchell–Schaeffer dynamics for one time step.
        """
        v_m = states[0]
        h = states[1]
        v = np.clip((v_m - v_rest) / v_span, 0.0, 1.0) # prevents tiny numerical overshoots

        j_in = h * v * v * (1.0 - v) * inv_tau_in # > 0, pushes upward
        j_out = -v * inv_tau_out # < 0, pushes downward
        v_new = np.clip(v + dt * (j_in + j_out), 0.0, 1.0) # explicit foward Euler

        # h-dynamics with a single np.where (vectorised): branch on whether the cell
        # has crossed the gating threshold this step.
        dh_dt = np.where(
            v < v_gate,
            (1.0 - h) * inv_tau_open, # cell is resting/recovering -> h increases toward 1 -> cell becomes ready to fire
            -h * inv_tau_close, # cell is depolarised -> h decreases toward 0 -> cell just fired, gate inactivates
        )
        h_new = np.clip(h + dt * dh_dt, 0.0, 1.0) # explicit forward Euler

        states[0] = v_rest + v_span * v_new
        states[1] = h_new
        return states

    return step


def initial_state(
    n_points: int, seed_mask: np.ndarray | None = None
) -> np.ndarray:
    """Build the (2, n_points) initial state array for a Mitchell–Schaeffer run.

    Returns:
        ``init[0]`` — V_m in mV, set to V_REST everywhere, then to V_DEP on
        ``seed_mask`` (a boolean mask of length n_points) if provided.
        ``init[1]`` — h, set to 1.0 everywhere (gate fully recovered: the
        cell is ready to fire).

    The seed pattern lets us start a propagating wavefront from a disc
    of pre-depolarised cells — :func:`forward.transmembrane.build_stimulus`
    returns exactly such a mask aligned with its stimulus disc.
    """
    init = np.empty((2, n_points), dtype=np.float64)
    init[0, :] = V_REST
    init[1, :] = 1.0 # set gate to recovered -> cell ready to fire
    if seed_mask is not None: 
        init[0, seed_mask] = V_DEP # initial excited regionto trigger the wavefront
    return init
