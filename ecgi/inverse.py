"""The data-enriched stabFEM inverse, wrapped for interactive use.

Takes a heart-surface potential (HSP) snapshot, projects it forward through the
torso to a body-surface potential (BSP), optionally adds measurement noise and/or
restricts to an electrode patch, and recovers the HSP with the vendored
:class:`stabfem.StabFEMSystem` using a chosen POD database as the prior.

Assembling a stabFEM system factors a preconditioner (a few seconds), so systems
are cached by (database, n_modes, gamma_reg, measured-region). Noise only changes
the right-hand side, so the noise slider re-solves instantly.
"""
from __future__ import annotations

import numpy as np
import ufl
import dolfinx
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from scipy.spatial import cKDTree

# vendored scientific code
from common import (  # type: ignore
    GAMMA_BODY, GAMMA_HEART, cg1_space, tagged_facet_vertices, vertex_to_dof_map,
)
from stabfem import (  # type: ignore
    MATCH_TOL, Parameters, StabFEMSystem, _translated_hsp_points,
)

from .cases import InverseResult
from .config import Database
from .geometry import Geometry


class InverseSolver:
    """Live stabFEM ECGi inversion against a selectable POD database."""

    def __init__(self, geometry: Geometry) -> None:
        self.geo = geometry
        self._partition = self._build_partition()
        self._systems: dict[tuple, StabFEMSystem] = {}

    # ------------------------------------------------------------------
    # Torso partition: map torso heart-boundary vertices to outer HSP indices
    # ------------------------------------------------------------------
    def _build_partition(self) -> dict:
        geo = self.geo
        union_pts = geo.heart_points[geo.union_vertices]
        gv = tagged_facet_vertices(geo.torso_mesh, geo.torso_tags, GAMMA_HEART)
        gp = geo.torso_mesh.geometry.x[gv, :3]
        # the torso's heart boundary is a translated copy of the heart mesh boundary
        dist, uidx = cKDTree(_translated_hsp_points(union_pts)).query(gp, k=1)
        if float(dist.max()) > MATCH_TOL:
            raise RuntimeError(f"torso/heart boundary mismatch: max dist {dist.max():.3e}")
        om = geo.outer_mask_in_union
        is_outer = om[uidx]
        u2o = np.full(om.size, -1, dtype=np.int64)
        u2o[np.where(om)[0]] = np.arange(int(om.sum()), dtype=np.int64)
        V = cg1_space(geo.torso_mesh)
        return {
            "V": V,
            "v2d": vertex_to_dof_map(V),
            "outer_gamma_vertices": gv[is_outer],
            "outer_hsp_idx": u2o[uidx[is_outer]],     # -> index into the outer HSP array
            "body_dofs": vertex_to_dof_map(V)[geo.body_vertices].astype(np.int64),
        }

    @property
    def body_vertices(self) -> np.ndarray:
        return self.geo.body_vertices

    # ------------------------------------------------------------------
    # Forward HSP (EPI∪BASE) -> clean BSP on the body surface
    # ------------------------------------------------------------------
    def forward_to_bsp(self, hsp_outer: np.ndarray, sigma_t: float = 6.0e-4) -> np.ndarray:
        """Mixed-BC Laplace: u = HSP on the outer heart boundary (Dirichlet),
        Neumann elsewhere; return the body-surface trace."""
        part, geo = self._partition, self.geo
        V = part["V"]
        outer_dofs = part["v2d"][part["outer_gamma_vertices"]].astype(np.int32)
        bc_fun = dolfinx.fem.Function(V, name="hsp_bc")
        bc_fun.x.array[outer_dofs] = hsp_outer[part["outer_hsp_idx"]].astype(bc_fun.x.array.dtype)
        bc_fun.x.scatter_forward()
        bc = dolfinx.fem.dirichletbc(bc_fun, outer_dofs)
        u, w = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = sigma_t * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
        L = dolfinx.fem.Constant(geo.torso_mesh, dolfinx.default_scalar_type(0.0)) * w * ufl.dx
        u_T = dolfinx.fem.Function(V, name="u_T")
        LinearProblem(a, L, u=u_T, bcs=[bc],
                      petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-10"}).solve()
        u_T.x.scatter_forward()
        return u_T.x.array[part["body_dofs"]].copy()

    # ------------------------------------------------------------------
    # stabFEM system (cached) + solve
    # ------------------------------------------------------------------
    def _system(self, db: Database, n_modes: int, gamma_reg: float,
                measured_vertices: np.ndarray | None) -> StabFEMSystem:
        key = (db.name, int(n_modes), float(gamma_reg),
               None if measured_vertices is None else tuple(measured_vertices.tolist()))
        if key not in self._systems:
            params = Parameters(n_modes=int(n_modes), gamma_reg=float(gamma_reg),
                                use_smw_preconditioner=True)
            kwargs = dict(pod_path=db.pod_basis, extended_dir=db.extended_dir, params=params)
            if measured_vertices is not None:
                kwargs["measured_body_vertices"] = measured_vertices
            self._systems[key] = StabFEMSystem(**kwargs)
        return self._systems[key]

    def solve(
        self,
        hsp_truth: np.ndarray,
        *,
        database: Database,
        n_modes: int = 9,
        noise_frac: float = 0.0,
        gamma_reg: float = 1.0,
        measured_vertices: np.ndarray | None = None,
        snapshot_time_ms: float = 0.0,
        noise_seed: int = 20260601,
    ) -> InverseResult:
        """Forward the truth HSP, add noise, recover it, and score the result."""
        clean = self.forward_to_bsp(hsp_truth)
        rms = float(np.sqrt(np.mean(clean ** 2)))
        noisy = clean.copy()
        if noise_frac > 0:
            noisy = clean + np.random.default_rng(noise_seed).normal(0.0, noise_frac * rms, clean.shape)

        system = self._system(database, n_modes, gamma_reg, measured_vertices)
        e = np.zeros(system.n_torso, dtype=np.float64)
        e[system.v2d[self.body_vertices]] = noisy
        res = system.solve(e)

        aligned = np.zeros_like(hsp_truth)
        aligned[self._partition["outer_hsp_idx"]] = res["hsp_recovered"]
        truth_norm = float(np.linalg.norm(hsp_truth)) or 1.0
        cos = float(np.dot(aligned, hsp_truth) / (np.linalg.norm(aligned) * truth_norm + 1e-30))
        rel = float(np.linalg.norm(aligned - hsp_truth) / truth_norm)
        return InverseResult(
            hsp_truth=hsp_truth, hsp_recovered=aligned, clean_bsp=clean, noisy_bsp=noisy,
            cosine=cos, rel_l2=rel, iterations=int(res["iterations"]),
            n_modes=int(n_modes), snapshot_time_ms=float(snapshot_time_ms),
        )

    def reconstruct_series(
        self,
        hsp_frames: np.ndarray,         # (n_frames, n_outer)
        times_ms: np.ndarray,
        *,
        database: Database,
        n_modes: int = 9,
        noise_frac: float = 0.0,
        gamma_reg: float = 1.0,
        measured_vertices: np.ndarray | None = None,
        progress=None,
    ) -> list[InverseResult]:
        """Recover every supplied HSP snapshot (the stabFEM system is built once
        and reused, so only the first frame pays the assembly cost)."""
        out: list[InverseResult] = []
        n = len(hsp_frames)
        for idx in range(n):
            out.append(self.solve(
                hsp_frames[idx], database=database, n_modes=n_modes, noise_frac=noise_frac,
                gamma_reg=gamma_reg, measured_vertices=measured_vertices,
                snapshot_time_ms=float(times_ms[idx]),
            ))
            if progress is not None:
                progress((idx + 1) / n)
        return out
