"""Live forward simulation: monodomain V_m, then the extracellular HSP.

Given an :class:`~ecgi.cases.InfarctSpec`, run the Mitchell-Schaeffer monodomain
model on the fixed heart mesh and, at each snapshot, solve the pure-Neumann
extracellular problem for u_e and read its trace on the EPI∪BASE interface. This
mirrors the database generator's per-sample forward exactly, but as a reusable
object driven by the interactive UI. Heavy (a few seconds), so it runs on the
workstation where dolfinx/PETSc live.
"""
from __future__ import annotations

import numpy as np
import ufl
import dolfinx
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc

# vendored forward building blocks (sys.path via ecgi._bootstrap)
from forward.transmembrane import build_solver, build_stimulus  # type: ignore
from ionic_models.mitchell_schaeffer import initial_state, make_step  # type: ignore

from .cases import ForwardResult, InfarctSpec
from .config import FORWARD
from .geometry import Geometry


def _build_ue_solver(V, sigma_i: float, sigma_e: float, v_m: dolfinx.fem.Function):
    """Pure-Neumann extracellular solver (A built once, RHS re-assembled per call).

    Solves ``-div((sigma_i+sigma_e) grad u_e) = div(sigma_i grad v_m)`` with the
    constant nullspace removed by enforcing zero mean. Returns ``(solve, destroy)``.
    """
    u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = dolfinx.fem.form((sigma_i + sigma_e) * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx)
    L = dolfinx.fem.form(-sigma_i * ufl.inner(ufl.grad(v_m), ufl.grad(w)) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    nullspace = PETSc.NullSpace().create(constant=True, comm=V.mesh.comm)
    A.setNullSpace(nullspace)
    A.setNearNullSpace(nullspace)
    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1.0e-10)
    one = dolfinx.fem.Constant(V.mesh, dolfinx.default_scalar_type(1.0))
    volume = V.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * ufl.dx)), op=MPI.SUM)
    u_e = dolfinx.fem.Function(V, name="u_e")

    def solve() -> dolfinx.fem.Function:
        b = dolfinx.fem.petsc.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        nullspace.remove(b)
        u_e.x.array[:] = 0.0
        ksp.solve(b, u_e.x.petsc_vec)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"u_e KSP failed: reason={ksp.getConvergedReason()}")
        u_e.x.scatter_forward()
        mean = V.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_e * ufl.dx)), op=MPI.SUM)
        u_e.x.array[:] -= mean / volume
        u_e.x.scatter_forward()
        b.destroy()
        return u_e

    def destroy() -> None:
        ksp.destroy(); A.destroy(); nullspace.destroy()

    return solve, destroy


class ForwardSimulator:
    """Runs the monodomain + extracellular forward problem for one case.

    Holds a reference to the shared :class:`Geometry` (mesh, space, region marker)
    so repeated simulations reuse the loaded geometry.
    """

    def __init__(self, geometry: Geometry) -> None:
        self.geo = geometry
        self.p = geometry.params

    def _tau_fields(self, spec: InfarctSpec):
        """Heterogeneous tau_close (endo/mid/epi) and a scar-reduced tau_out."""
        rm = self.geo.region_marker
        tau_close = np.where(rm == 1, self.p["tau_close_endo"],
                    np.where(rm == 2, self.p["tau_close_epi"], self.p["tau_close_mid"])
                    ).astype(np.float64)
        tau_out = np.full(self.geo.dof_coords.shape[0], self.p["tau_out"], dtype=np.float64)
        n_scar = 0
        if not spec.is_healthy:
            d2 = np.sum((self.geo.dof_coords - spec.centre()) ** 2, axis=1)
            mask = d2 <= spec.radius_mm ** 2
            tau_out[mask] *= FORWARD.infarct_tau_out_factor
            n_scar = int(mask.sum())
        return tau_close, tau_out, n_scar

    def simulate(
        self,
        spec: InfarctSpec,
        *,
        t_end_ms: float | None = None,
        dt_ms: float | None = None,
        snapshot_every_ms: float | None = None,
        progress=None,
    ) -> ForwardResult:
        """Time-step V_m and collect (V_m, HSP) snapshots.

        ``progress`` is an optional ``callable(fraction: float)`` for a UI bar.
        """
        geo, p = self.geo, self.p
        t_end = FORWARD.t_end_ms if t_end_ms is None else float(t_end_ms)
        dt = FORWARD.dt_ms if dt_ms is None else float(dt_ms)
        snap_every = FORWARD.snapshot_every_ms if snapshot_every_ms is None else float(snapshot_every_ms)

        tau_close, tau_out, _ = self._tau_fields(spec)

        # apex ball stimulus (matches the database's stimulus_mode="ball")
        time_const = dolfinx.fem.Constant(geo.heart_mesh, dolfinx.default_scalar_type(0.0))
        stim_expr, seed_mask = build_stimulus(
            mesh=geo.heart_mesh, time=time_const, centre=geo.apex_seed,
            radius=p.get("stim_radius_mm", 8.0), amplitude=p["stim_amp"],
            duration=p["stim_dur_ms"],
        )
        cell_step = make_step(tau_in=p["tau_in"], tau_out=tau_out, tau_open=p["tau_open"],
                              tau_close=tau_close, v_gate=p["v_gate"])
        init = initial_state(geo.dof_coords.shape[0], seed_mask=seed_mask)
        solver, v_pde = build_solver(
            mesh=geo.heart_mesh, time=time_const,
            sigma_i=p["sigma_i"], sigma_e=p["sigma_e"], c_m=p["c_m"], a_m=p["a_m"],
            stim_expr=stim_expr, cell_step_fun=cell_step, init_states=init, num_states=2,
        )
        v_pde.x.array[:] = init[0]
        v_pde.x.scatter_forward()
        ue_solve, ue_destroy = _build_ue_solver(geo.V, p["sigma_i"], p["sigma_e"], v_pde)

        # snapshot schedule: every snap_every ms on the dt grid
        snap_times = np.arange(snap_every, t_end + 1e-9, snap_every)
        n_steps = int(round(t_end / dt))
        v2d, outer = geo.v2d, geo.outer_vertices
        vm_snaps, hsp_snaps, taken = [], [], []
        nxt = 0
        try:
            for k in range(n_steps):
                t = k * dt
                time_const.value = t
                solver.step((t, t + dt))
                t_after = t + dt
                while nxt < snap_times.size and snap_times[nxt] <= t_after + 1e-9:
                    v_pde.x.scatter_forward()
                    vm_snaps.append(v_pde.x.array[v2d].copy())          # vertex-ordered V_m
                    u_e = ue_solve()
                    hsp_snaps.append(u_e.x.array[v2d[outer]].copy())     # HSP on EPI∪BASE
                    taken.append(t_after)
                    nxt += 1
                if progress is not None:
                    progress((k + 1) / n_steps)
        finally:
            ue_destroy()

        return ForwardResult(
            times_ms=np.asarray(taken, dtype=np.float64),
            vm=np.asarray(vm_snaps, dtype=np.float64),
            hsp=np.asarray(hsp_snaps, dtype=np.float64),
            spec=spec,
        )
