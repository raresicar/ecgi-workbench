"""Infarction-experiment HSP database generator.

Follows Boulakia/Schenone/Gerbeau (arXiv:1111.5926v2, section 3.3): one healthy
case plus several myocardial infarctions placed at different points. Inside each
scar ball ``tau_out`` is divided by 100 (the paper uses 10) so the region cannot
sustain activation. For every case we run the monodomain V_m solve followed by
the pure-Neumann extracellular u_e solve on the UKB heart, and save a stratified
set of HSP snapshots on the heart boundary. The torso step is NOT run here: the
downstream use is a POD basis of HSPs (``build_pod_basis.py``) feeding the
stabFEM inverse.

This generator is specific to the infarction experiment — each stabFEM test case
gets its own generator because different parameters vary. Physical / Mitchell-
Schaeffer parameters come from ``ukb/pipeline/params.json``; this experiment's
knobs (infarct geometry, snapshot schedule) live in ``config.json`` next to it.

Sample roster (sample_00000.npz ... sample_NNNNN.npz):
  * sample 0           : healthy (no scar), ``healthy_snapshots`` count.
  * sample 1 .. N      : one infarct each, ``infarct_snapshots`` count, at the
                         farthest-point-sampled visible-wall centres with a
                         per-sample random radius.

Run from the repository root inside the scientific-python conda env:

    python ukb/database/infarction/generate.py [--config config.json] [--n-workers K]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Pin BLAS/OMP BEFORE importing numpy/petsc/dolfinx so worker processes inherit
# one thread each — otherwise --n-workers oversubscribes the host.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parents[1] / "pipeline"
SOLVERS_DIR = HERE.parents[2] / "solvers"
for p in (PIPELINE, SOLVERS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import (  # noqa: E402
    EPI, HEART_MSH, LV, RV,
    cg1_space, load_gmsh_mesh, load_params, tagged_facet_vertices,
    vertex_to_dof_map,
)
from forward.transmembrane import build_solver, build_stimulus  # noqa: E402
from ionic_models.mitchell_schaeffer import (  # noqa: E402
    initial_state, make_step,
)
from utils.heterogeneity import build_region_field  # noqa: E402
from utils.stimulus import build_shell_stimulus  # noqa: E402

DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT_DIR = HERE / "samples"
DEPOL_THRESHOLD_MV = -50.0

# Transmural layer fractions for the endo/mid/epi tau_close split come from
# params.json (endo_size / epi_size); the region marker is the same across all
# samples, so it is precomputed once in the main process.


# ---------------------------------------------------------------------------
# Visible-wall infarct centre selection (deterministic, main process)
# ---------------------------------------------------------------------------
def select_infarct_centres(
    mesh, facet_tags, *, n_centers: int, view_dir, score_percentile: float,
) -> np.ndarray:
    """Pick ``n_centers`` epicardial points on the camera-facing wall.

    The heart GIFs use ``view_init(elev=18, azim=-62)``; front-facing points
    have a large dot product with ``view_dir``. We keep EPI vertices whose score
    is above ``score_percentile`` and farthest-point-sample ``n_centers`` of them
    so the lesions spread out over the visible free wall instead of clustering.
    """
    epi_vertices = tagged_facet_vertices(mesh, facet_tags, EPI)
    pts = mesh.geometry.x[epi_vertices, :3]
    view = np.asarray(view_dir, dtype=np.float64)
    view = view / np.linalg.norm(view)
    score = pts @ view
    keep = score >= np.percentile(score, score_percentile)
    cand = pts[keep]
    if cand.shape[0] < n_centers:
        raise ValueError(
            f"only {cand.shape[0]} candidate epi vertices above the "
            f"{score_percentile}th score percentile, but {n_centers} centres "
            f"requested; lower score_percentile or n_centers"
        )
    # Farthest-point sampling, seeded at the most front-facing candidate.
    chosen = [int(np.argmax(cand @ view))]
    dist = np.linalg.norm(cand - cand[chosen[0]], axis=1)
    while len(chosen) < n_centers:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(cand - cand[nxt], axis=1))
    return cand[chosen]


def build_sample_roster(
    centres: np.ndarray, *, rng: np.random.Generator,
    radius_min_mm: float, radius_max_mm: float,
) -> list[dict]:
    """sample 0 = healthy; samples 1..N = one infarct per centre with a radius."""
    roster = [{"is_healthy": True, "centre_mm": None, "radius_mm": None}]
    for centre in centres:
        roster.append({
            "is_healthy": False,
            "centre_mm": [float(x) for x in centre],
            "radius_mm": float(rng.uniform(radius_min_mm, radius_max_mm)),
        })
    return roster


def near_grid_centre(
    mesh, facet_tags, *, view_dir, score_percentile: float, n_centers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """A held-out epicardial point *between* two adjacent training centres.

    The held-out infarct must be a fair interpolation test: off the
    farthest-point-sampled training grid, but surrounded by it. We take the 18
    training centres, pick one from the cluster interior (nearest the centroid),
    join it to its nearest neighbour, and snap their midpoint to the closest
    epicardial vertex. The result is genuinely off-grid yet sits among the
    simulated scars (cf. Boulakia's point P in Fig 9).
    """
    centres = select_infarct_centres(
        mesh, facet_tags, n_centers=n_centers, view_dir=view_dir,
        score_percentile=score_percentile,
    )
    # bias to the interior: choose the seed among the half closest to the centroid
    to_centroid = np.linalg.norm(centres - centres.mean(0), axis=1)
    interior = np.argsort(to_centroid)[: max(2, n_centers // 2)]
    i = int(rng.choice(interior))
    d = np.linalg.norm(centres - centres[i], axis=1)
    d[i] = np.inf
    j = int(np.argmin(d))                       # nearest training neighbour of i
    midpoint = 0.5 * (centres[i] + centres[j])
    # snap to the nearest epicardial vertex so the centre lies on the surface
    epi_pts = mesh.geometry.x[tagged_facet_vertices(mesh, facet_tags, EPI), :3]
    return epi_pts[int(np.argmin(np.linalg.norm(epi_pts - midpoint, axis=1)))]


# ---------------------------------------------------------------------------
# Stratified snapshot times
# ---------------------------------------------------------------------------
def stratified_snapshot_times(
    rng: np.random.Generator, *, dt_ms: float, t_end_ms: float,
    window_split_ms: float, counts: tuple[int, int],
) -> list[float]:
    """Draw snapshot times on the dt grid: ``counts[0]`` from the early window
    ``(0, split]`` and ``counts[1]`` from the late window ``(split, t_end]``."""
    def _draw(lo: float, hi: float, n: int) -> list[float]:
        if n <= 0:
            return []
        first = max(1, int(np.ceil((lo - 1.0e-12) / dt_ms)))
        last = int(np.floor((hi + 1.0e-12) / dt_ms))
        grid = np.arange(first, last + 1, dtype=np.int64)
        if grid.size < n:
            raise ValueError(
                f"window ({lo}, {hi}] ms has only {grid.size} dt-grid times "
                f"but {n} snapshots requested"
            )
        return [float(s * dt_ms) for s in rng.choice(grid, size=n, replace=False)]

    early = _draw(0.0, window_split_ms, counts[0])
    late = _draw(window_split_ms, t_end_ms, counts[1])
    return sorted(early + late)


# ---------------------------------------------------------------------------
# u_e solver (pure-Neumann, constant-nullspace removal), reused per sample
# ---------------------------------------------------------------------------
def build_ue_solver(V, sigma_i: float, sigma_e: float, v_m: dolfinx.fem.Function):
    """A = ((sigma_i+sigma_e) <grad u, grad w>) assembled once; each solve()
    re-assembles only L = -sigma_i <grad v_m, grad w> and reuses A + KSP.
    The mean is removed to fix the constant nullspace."""
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)
    a = dolfinx.fem.form((sigma_i + sigma_e) * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx)
    L = dolfinx.fem.form(-sigma_i * ufl.inner(ufl.grad(v_m), ufl.grad(w)) * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a)
    A.assemble()
    nullspace = PETSc.NullSpace().create(constant=True, comm=V.mesh.comm)
    A.setNullSpace(nullspace); A.setNearNullSpace(nullspace)
    ksp = PETSc.KSP().create(V.mesh.comm)
    ksp.setOperators(A); ksp.setType(PETSc.KSP.Type.CG); ksp.getPC().setType(PETSc.PC.Type.HYPRE)
    ksp.setTolerances(rtol=1.0e-10)

    one_const = dolfinx.fem.Constant(V.mesh, dolfinx.default_scalar_type(1.0))
    volume = V.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(one_const * ufl.dx)), op=MPI.SUM,
    )
    u_e = dolfinx.fem.Function(V, name="u_e")

    def solve() -> dolfinx.fem.Function:
        b = dolfinx.fem.petsc.assemble_vector(L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        nullspace.remove(b)
        u_e.x.array[:] = 0.0
        ksp.solve(b, u_e.x.petsc_vec)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"u_e ksp failed: reason={ksp.getConvergedReason()}")
        u_e.x.scatter_forward()
        mean = V.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_e * ufl.dx)), op=MPI.SUM,
        )
        u_e.x.array[:] -= mean / volume
        u_e.x.scatter_forward()
        b.destroy()
        return u_e

    def destroy():
        ksp.destroy(); A.destroy(); nullspace.destroy()

    return solve, destroy


# ---------------------------------------------------------------------------
# One forward run (healthy or one infarct) -> stratified HSP snapshots
# ---------------------------------------------------------------------------
def run_one_sample(*, state: dict, spec: dict, snapshot_times_ms: list[float]) -> dict:
    mesh = state["heart_mesh"]
    facet_tags = state["heart_facet_tags"]
    V = state["V_heart"]
    p = state["params"]
    coords = state["dof_coords"]
    region_marker = state["region_marker"]

    # Heterogeneous tau_close from params.json (region marker shared across samples).
    tau_close = np.where(
        region_marker == 1, p["tau_close_endo"],
        np.where(region_marker == 2, p["tau_close_epi"], p["tau_close_mid"]),
    ).astype(np.float64)

    # tau_out: constant baseline, divided by 100 inside the scar ball (if any).
    tau_out = np.full(coords.shape[0], p["tau_out"], dtype=np.float64)
    if not spec["is_healthy"]:
        centre = np.asarray(spec["centre_mm"], dtype=np.float64)
        r2 = float(spec["radius_mm"]) ** 2
        mask = np.sum((coords - centre) ** 2, axis=1) <= r2
        tau_out[mask] *= state["tau_out_factor"]
        n_scar = int(mask.sum())
    else:
        n_scar = 0

    time_const = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    mode = str(p.get("stimulus_mode", "ball")).lower()
    if mode == "shell":
        stim = build_shell_stimulus(
            mesh=mesh, facet_tags=facet_tags, V=V, time=time_const,
            amplitude=p["stim_amp"], tact_ms=p["stim_dur_ms"],
            layer_thickness_mm=p["stim_layer_thickness_mm"],
            lv_marker=LV, rv_marker=RV, chambers=("lv", "rv"),
        )
        stim_expr, seed_mask, stim_fields = stim.expr, stim.seed_mask, stim.fields
    elif mode == "ball":
        seed_centre = tuple(float(x) for x in coords[int(np.argmin(coords[:, 2]))])
        stim_expr, seed_mask = build_stimulus(
            mesh=mesh, time=time_const, centre=seed_centre,
            radius=p.get("stim_radius_mm", 8.0), amplitude=p["stim_amp"],
            duration=p["stim_dur_ms"],
        )
        stim_fields = []
    else:
        raise ValueError(f"unsupported stimulus_mode={mode!r}")

    cell_step = make_step(
        tau_in=p["tau_in"], tau_out=tau_out, tau_open=p["tau_open"],
        tau_close=tau_close, v_gate=p["v_gate"],
    )
    init = initial_state(coords.shape[0], seed_mask=seed_mask)
    solver, v_pde = build_solver(
        mesh=mesh, time=time_const,
        sigma_i=p["sigma_i"], sigma_e=p["sigma_e"], c_m=p["c_m"], a_m=p["a_m"],
        stim_expr=stim_expr, cell_step_fun=cell_step,
        init_states=init, num_states=2,
    )
    _keep_alive = stim_fields  # noqa: F841 — keep UFL stimulus coefficients alive
    v_pde.x.array[:] = init[0]; v_pde.x.scatter_forward()

    ue_solve, ue_destroy = build_ue_solver(V, p["sigma_i"], p["sigma_e"], v_pde)

    boundary_vertices = state["boundary_vertices"]
    v2d = state["v2d_heart"]
    snapshot_times = sorted(float(t) for t in snapshot_times_ms)
    n_snap, n_hsp = len(snapshot_times), boundary_vertices.size
    hsp_stack = np.zeros((n_snap, n_hsp), dtype=np.float64)
    actual_times = np.zeros(n_snap, dtype=np.float64)
    v_m_max_per_snap = np.zeros(n_snap, dtype=np.float64)

    next_idx = 0
    n_steps = int(round(state["t_end_ms"] / state["dt_ms"]))
    v_min_g, v_max_g = float("inf"), float("-inf")
    for k in range(n_steps):
        t = k * state["dt_ms"]
        time_const.value = t
        solver.step((t, t + state["dt_ms"]))
        t_after = t + state["dt_ms"]
        while next_idx < n_snap and snapshot_times[next_idx] <= t_after + 1.0e-9:
            v_pde.x.scatter_forward()
            v_arr = v_pde.x.array
            v_min_g = min(v_min_g, float(v_arr.min()))
            v_max_g = max(v_max_g, float(v_arr.max()))
            u_e = ue_solve()
            hsp_stack[next_idx] = u_e.x.array[v2d[boundary_vertices]]
            actual_times[next_idx] = t_after
            v_m_max_per_snap[next_idx] = float(v_arr.max())
            next_idx += 1
    if next_idx < n_snap:
        raise RuntimeError(f"only {next_idx}/{n_snap} snapshots taken (t_end too small)")

    ue_destroy()
    final_v = v_pde.x.array
    return {
        "hsp_stack": hsp_stack,
        "snapshot_times_ms": actual_times,
        "v_m_max_per_snap": v_m_max_per_snap,
        "v_m_min_global": v_min_g,
        "v_m_max_global": v_max_g,
        "depol_frac_final": float((final_v > DEPOL_THRESHOLD_MV).mean()),
        "n_scar_dofs": n_scar,
    }


# ---------------------------------------------------------------------------
# Per-worker state + worker entry point
# ---------------------------------------------------------------------------
_WORKER_STATE: dict | None = None


def _precompute_region_marker() -> np.ndarray:
    """Run expand_layer_biv ONCE in the main process (it writes a diagnostic
    endo_epi_biv.xdmf to cwd; running it concurrently in 16 workers races on the
    HDF5 file lock). Ship the integer region marker to workers via initargs."""
    import tempfile
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        try:
            os.chdir(td)
            mesh, _, ft = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
            V = cg1_space(mesh)
            p = load_params()
            region_fun, _ = build_region_field(
                V=V, facet_tags=ft, lv_marker=LV, rv_marker=RV, epi_marker=EPI,
                endo_size=p["endo_size"], epi_size=p["epi_size"],
            )
            return np.rint(region_fun.x.array).astype(np.int32).copy()
        finally:
            os.chdir(cwd)


def _build_state(config: dict, region_marker: np.ndarray) -> dict:
    mesh, _, facet_tags = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    V = cg1_space(mesh)
    if region_marker.shape[0] != V.dofmap.index_map.size_local:
        raise RuntimeError("region_marker length != V_heart local dofs")
    v2d = vertex_to_dof_map(V)
    boundary_vertices = np.unique(np.concatenate([
        tagged_facet_vertices(mesh, facet_tags, tag) for tag in (1, 2, 3, 4)
    ])).astype(np.int64)
    return {
        "heart_mesh": mesh, "heart_facet_tags": facet_tags, "V_heart": V,
        "v2d_heart": v2d, "boundary_vertices": boundary_vertices,
        "hsp_points": mesh.geometry.x[boundary_vertices, :3].copy(),
        "dof_coords": V.tabulate_dof_coordinates()[:, :3].copy(),
        "region_marker": region_marker.astype(int, copy=False),
        "params": load_params(),
        "t_end_ms": float(config["t_end_ms"]),
        "dt_ms": float(config["dt_ms"]),
        "tau_out_factor": float(config["tau_out_factor"]),
        "window_split_ms": float(config["window_split_ms"]),
        "base_seed": int(config.get("base_seed", 20260529)),
        "healthy_snapshots": tuple(int(x) for x in config["healthy_snapshots"]),
        "infarct_snapshots": tuple(int(x) for x in config["infarct_snapshots"]),
    }


def _init_worker(config: dict, region_marker: np.ndarray, roster: list[dict]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = _build_state(config, region_marker)
    _WORKER_STATE["roster"] = roster


def _process_sample(args: tuple[int, str, bool]) -> dict:
    sample_idx, output_dir_str, overwrite = args
    state = _WORKER_STATE
    assert state is not None, "_init_worker did not run"
    output_dir = Path(output_dir_str)
    sample_id = f"sample_{sample_idx:05d}"
    out_npz = output_dir / f"{sample_id}.npz"
    if out_npz.exists() and not overwrite:
        return {"sample_id": sample_id, "ok": True, "skipped": True}

    spec = state["roster"][sample_idx]
    counts = state["healthy_snapshots"] if spec["is_healthy"] else state["infarct_snapshots"]
    rng = np.random.default_rng(state["base_seed"] + sample_idx)
    snapshot_times = stratified_snapshot_times(
        rng, dt_ms=state["dt_ms"], t_end_ms=state["t_end_ms"],
        window_split_ms=state["window_split_ms"], counts=counts,
    )
    t0 = time.perf_counter()
    try:
        result = run_one_sample(state=state, spec=spec, snapshot_times_ms=snapshot_times)
    except Exception as exc:  # noqa: BLE001
        return {"sample_id": sample_id, "ok": False, "error": repr(exc),
                "wall_s": time.perf_counter() - t0, "is_healthy": spec["is_healthy"]}
    wall = time.perf_counter() - t0
    params_json = json.dumps({
        **state["params"],
        "is_healthy": spec["is_healthy"],
        "infarct_centre_mm": spec["centre_mm"],
        "infarct_radius_mm": spec["radius_mm"],
        "tau_out_factor": state["tau_out_factor"],
    })
    np.savez(
        out_npz,
        hsp_points=state["hsp_points"],
        hsp_stack=result["hsp_stack"],
        snapshot_times_ms=result["snapshot_times_ms"],
        v_m_max_per_snap=result["v_m_max_per_snap"],
        params_json=params_json,
        is_healthy=spec["is_healthy"],
        infarct_centre_mm=(np.asarray(spec["centre_mm"], dtype=np.float64)
                           if spec["centre_mm"] is not None else np.full(3, np.nan)),
        infarct_radius_mm=(spec["radius_mm"] if spec["radius_mm"] is not None else np.nan),
        tau_out_factor=state["tau_out_factor"],
        t_end_ms=state["t_end_ms"], dt_ms=state["dt_ms"],
        seed=state["base_seed"] + sample_idx,
        v_m_min_global=result["v_m_min_global"],
        v_m_max_global=result["v_m_max_global"],
        depol_frac_final=result["depol_frac_final"],
    )
    return {
        "sample_id": sample_id, "ok": True, "wall_s": wall,
        "is_healthy": spec["is_healthy"], "n_scar_dofs": result["n_scar_dofs"],
        "radius_mm": spec["radius_mm"], "n_snapshots": len(result["snapshot_times_ms"]),
        "v_m_min_global": result["v_m_min_global"],
        "v_m_max_global": result["v_m_max_global"],
        "depol_frac_final": result["depol_frac_final"],
        "hsp_abs_max": float(np.abs(result["hsp_stack"]).max()),
    }


# ---------------------------------------------------------------------------
# CPU temperature monitor (thermal caution: long CPU-heavy generation)
# ---------------------------------------------------------------------------
class CpuTempMonitor(threading.Thread):
    def __init__(self, csv_path: Path, interval_s: float):
        super().__init__(name="cpu-temp-monitor", daemon=True)
        self.csv_path = csv_path
        self.interval_s = interval_s
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    @staticmethod
    def _flatten(temps: dict) -> dict[str, float]:
        out = {}
        for source, readings in temps.items():
            for i, r in enumerate(readings):
                label = (r.label or f"sensor{i}").replace(" ", "_").replace("/", "_")
                out[f"{source}_{label}_C"] = float(r.current)
        return out

    def _poll_once(self):
        try:
            import psutil
            return self._flatten(psutil.sensors_temperatures())
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def run(self) -> None:
        first = self._poll_once()
        header = ["timestamp"] + sorted(first.keys())
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            if new_file:
                writer.writeheader()
            writer.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), **first})
            fh.flush()
            while not self._stop_evt.wait(self.interval_s):
                sample = self._poll_once()
                writer.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"),
                                 **{k: sample.get(k, "") for k in header[1:]}})
                fh.flush()


# ---------------------------------------------------------------------------
# Centre-overview diagnostic plot (Boulakia-style, front-facing, occluded)
# ---------------------------------------------------------------------------
def _render_figures(output_dir: Path, config_path: Path) -> None:
    """Best-effort scar render via render_scars.py.

    The opaque, depth-buffered pyvista render needs a GL context, so it runs in
    its own process under xvfb (if available) rather than in this solve process.
    Reads ``roster.json`` from ``output_dir`` and writes ``infarct_centres.png``
    + ``scar_example.png`` there. A failure here never aborts generation.
    """
    import shutil
    import subprocess
    script = HERE / "render_scars.py"
    cmd = [sys.executable, str(script), "--config", str(config_path),
           "--roster", str(output_dir / "roster.json"),
           "--output-dir", str(output_dir)]
    if shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a"] + cmd
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  (scar render skipped: {exc!r})", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-centers", type=int, default=None, help="override config n_centers")
    parser.add_argument("--t-end-ms", type=float, default=None, help="override config t_end_ms (smoke tests)")
    parser.add_argument("--base-seed", type=int, default=None, help="override config base_seed")
    parser.add_argument("--heldout", action="store_true",
                        help="emit a single off-grid infarct (random visible-wall centre + radius) "
                             "as sample_00000.npz; no healthy case. For stabFEM held-out tests.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--temp-csv", default=None,
                        help="CPU-temperature CSV (default <output-dir>/cpu_temps.csv; empty to disable)")
    parser.add_argument("--temp-interval-s", type=float, default=300.0)
    args = parser.parse_args(argv)

    config = {k: v for k, v in json.loads(args.config.read_text()).items() if not k.startswith("_")}
    if args.n_centers is not None:
        config["n_centers"] = args.n_centers
    if args.t_end_ms is not None:
        config["t_end_ms"] = args.t_end_ms
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    n_centers = int(config["n_centers"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"

    temp_csv = (args.output_dir / "cpu_temps.csv" if args.temp_csv is None
                else Path(args.temp_csv) if str(args.temp_csv) else None)
    monitor = None
    if temp_csv is not None:
        monitor = CpuTempMonitor(temp_csv, float(args.temp_interval_s))
        monitor.start()
        print(f"cpu temp monitor -> {temp_csv} every {args.temp_interval_s:.0f}s", flush=True)

    print("precomputing cell-region marker + infarct centres in main process ...", flush=True)
    region_marker = _precompute_region_marker()
    mesh, _, facet_tags = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    rng = np.random.default_rng(int(config.get("base_seed", 20260529)))
    if args.heldout:
        centre = near_grid_centre(
            mesh, facet_tags, view_dir=config["view_dir"],
            score_percentile=float(config["score_percentile"]),
            n_centers=n_centers, rng=rng,
        )
        roster = [{
            "is_healthy": False, "centre_mm": [float(x) for x in centre],
            "radius_mm": float(rng.uniform(float(config["radius_min_mm"]), float(config["radius_max_mm"]))),
        }]
        print(f"held-out roster: 1 off-grid infarct (between training centres) at "
              f"{roster[0]['centre_mm']} r={roster[0]['radius_mm']:.1f}mm", flush=True)
    else:
        centres = select_infarct_centres(
            mesh, facet_tags, n_centers=n_centers,
            view_dir=config["view_dir"], score_percentile=float(config["score_percentile"]),
        )
        roster = build_sample_roster(
            centres, rng=rng,
            radius_min_mm=float(config["radius_min_mm"]), radius_max_mm=float(config["radius_max_mm"]),
        )
        print(f"roster: 1 healthy + {n_centers} infarct centres "
              f"(radius {config['radius_min_mm']}-{config['radius_max_mm']} mm, "
              f"tau_out x{config['tau_out_factor']})", flush=True)
    (args.output_dir / "roster.json").write_text(json.dumps(roster, indent=2))
    if not args.heldout:
        _render_figures(args.output_dir, args.config)

    work_items = [(k, str(args.output_dir), bool(args.overwrite)) for k in range(len(roster))]
    n_workers = max(1, int(args.n_workers))
    print(f"launching {n_workers} workers for {len(work_items)} samples "
          f"(t_end={config['t_end_ms']} ms, dt={config['dt_ms']} ms) ...", flush=True)

    t_start = time.perf_counter()
    n_done = n_failed = 0
    try:
        if n_workers == 1:
            _init_worker(config, region_marker, roster)
            for item in work_items:
                entry = _process_sample(item)
                with index_path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
                _log_entry(entry); n_done += 1
                n_failed += 0 if entry.get("ok", False) else 1
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                     initializer=_init_worker,
                                     initargs=(config, region_marker, roster)) as pool:
                futures = {pool.submit(_process_sample, item): item[0] for item in work_items}
                for fut in as_completed(futures):
                    entry = fut.result()
                    with index_path.open("a") as f:
                        f.write(json.dumps(entry) + "\n")
                    _log_entry(entry); n_done += 1
                    n_failed += 0 if entry.get("ok", False) else 1
    finally:
        if monitor is not None:
            monitor.stop(); monitor.join(timeout=2.0)

    wall = time.perf_counter() - t_start
    print(f"\ndone in {wall:.1f}s. {n_done} samples ({n_failed} failed). "
          f"log at {index_path}", flush=True)
    return 0 if n_failed == 0 else 1


def _log_entry(entry: dict) -> None:
    if entry.get("skipped"):
        print(f"  {entry['sample_id']} skipped (already exists)"); return
    if not entry.get("ok", False):
        print(f"  {entry['sample_id']} FAILED in {entry.get('wall_s', 0):.1f}s: {entry.get('error', '?')}")
        return
    kind = "healthy" if entry["is_healthy"] else f"infarct r={entry['radius_mm']:.1f}mm scar={entry['n_scar_dofs']}"
    print(f"  {entry['sample_id']}  {entry['wall_s']:6.1f}s  {kind:<28}  "
          f"N_snap={entry['n_snapshots']}  depol_final={entry['depol_frac_final']:.3f}  "
          f"V_m=[{entry['v_m_min_global']:+.1f},{entry['v_m_max_global']:+.1f}]  "
          f"|hsp|max={entry['hsp_abs_max']:.2e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
