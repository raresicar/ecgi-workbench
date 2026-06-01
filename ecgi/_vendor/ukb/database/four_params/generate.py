"""Four-parameter box database generator: tau_in, C_m, A_m, tau_close^RV.

Boulakia §3.1/§4.2 identifies these four as the parameters the ECG is most
sensitive to. Each sample draws them uniformly from a small box around the
``ukb/pipeline/params.json`` values; everything else is fixed. Healthy
monodomain (no infarct). For each sample we run V_m -> u_e on the UKB heart and
save ``n_snapshots`` HSP snapshots evenly spaced in ``(0, t_end]``, in the same
POD-compatible ``sample_NNNNN.npz`` format the shared build_pod_basis.py reads.

tau_close^RV needs the RV as its own region (the transmural endo/mid/epi split
lumps it into "endo"), so each myocardial vertex is classified LV vs RV by its
nearest endocardial facet (tags LV=1 / RV=2): RV vertices take the (varied)
tau_close_rv, LV vertices keep the fixed transmural endo/mid/epi values.

Run from the repository root inside the scientific-python conda env::

    python ukb/database/four_params/generate.py --n-workers 16
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

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
from ionic_models.mitchell_schaeffer import initial_state, make_step  # noqa: E402
from utils.heterogeneity import build_region_field  # noqa: E402
from utils.stimulus import build_shell_stimulus  # noqa: E402

DEFAULT_CONFIG = HERE / "config.json"
DEFAULT_OUTPUT_DIR = HERE / "samples"
DEPOL_THRESHOLD_MV = -50.0
VARIED = ("tau_in", "c_m", "a_m", "tau_close_rv")


def sample_params(rng: np.random.Generator, base: dict, frac: float) -> dict:
    """Draw the four varied params uniformly in ±frac around their base value."""
    p = dict(base)
    for k in VARIED:
        v = float(base[k])
        p[k] = float(rng.uniform(v * (1.0 - frac), v * (1.0 + frac)))
    return p


def even_snapshot_times(t_end_ms: float, dt_ms: float, n: int) -> list[float]:
    """``n`` evenly spaced snapshot times on the dt grid in ``(0, t_end]``."""
    steps = np.unique(np.round(np.linspace(t_end_ms / n, t_end_ms, n) / dt_ms).astype(np.int64))
    steps = steps[steps >= 1]
    return [float(s * dt_ms) for s in steps]


def build_ue_solver(V, sigma_i: float, sigma_e: float, v_m: dolfinx.fem.Function):
    """Pure-Neumann u_e solver (A assembled once, RHS re-assembled per solve)."""
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
            raise RuntimeError(f"u_e ksp failed: reason={ksp.getConvergedReason()}")
        u_e.x.scatter_forward()
        mean = V.mesh.comm.allreduce(
            dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_e * ufl.dx)), op=MPI.SUM)
        u_e.x.array[:] -= mean / volume
        u_e.x.scatter_forward()
        b.destroy()
        return u_e

    def destroy():
        ksp.destroy(); A.destroy(); nullspace.destroy()

    return solve, destroy


def run_one_sample(*, state: dict, params: dict, snapshot_times_ms: list[float]) -> dict:
    mesh = state["heart_mesh"]; facet_tags = state["heart_facet_tags"]; V = state["V_heart"]
    region_marker = state["region_marker"]; rv_mask = state["rv_mask"]; coords = state["dof_coords"]

    # tau_close: LV transmural (fixed endo/mid/epi) + RV uniform (varied tau_close_rv).
    tau_close = np.where(region_marker == 1, params["tau_close_endo"],
                np.where(region_marker == 2, params["tau_close_epi"], params["tau_close_mid"])
                ).astype(np.float64)
    tau_close[rv_mask] = params["tau_close_rv"]

    time_const = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))
    mode = str(params.get("stimulus_mode", "ball")).lower()
    if mode == "shell":
        stim = build_shell_stimulus(
            mesh=mesh, facet_tags=facet_tags, V=V, time=time_const,
            amplitude=params["stim_amp"], tact_ms=params["stim_dur_ms"],
            layer_thickness_mm=params["stim_layer_thickness_mm"],
            lv_marker=LV, rv_marker=RV, chambers=("lv", "rv"))
        stim_expr, seed_mask, stim_fields = stim.expr, stim.seed_mask, stim.fields
    elif mode == "ball":
        seed_centre = tuple(float(x) for x in coords[int(np.argmin(coords[:, 2]))])
        stim_expr, seed_mask = build_stimulus(
            mesh=mesh, time=time_const, centre=seed_centre,
            radius=params.get("stim_radius_mm", 8.0), amplitude=params["stim_amp"],
            duration=params["stim_dur_ms"])
        stim_fields = []
    else:
        raise ValueError(f"unsupported stimulus_mode={mode!r}")

    cell_step = make_step(tau_in=params["tau_in"], tau_out=params["tau_out"],
                          tau_open=params["tau_open"], tau_close=tau_close, v_gate=params["v_gate"])
    init = initial_state(coords.shape[0], seed_mask=seed_mask)
    solver, v_pde = build_solver(
        mesh=mesh, time=time_const, sigma_i=params["sigma_i"], sigma_e=params["sigma_e"],
        c_m=params["c_m"], a_m=params["a_m"], stim_expr=stim_expr, cell_step_fun=cell_step,
        init_states=init, num_states=2)
    _keep = stim_fields  # noqa: F841
    v_pde.x.array[:] = init[0]; v_pde.x.scatter_forward()

    ue_solve, ue_destroy = build_ue_solver(V, params["sigma_i"], params["sigma_e"], v_pde)
    bverts = state["boundary_vertices"]; v2d = state["v2d_heart"]
    times = sorted(float(t) for t in snapshot_times_ms)
    n_snap, n_hsp = len(times), bverts.size
    hsp_stack = np.zeros((n_snap, n_hsp)); actual = np.zeros(n_snap); vmax_snap = np.zeros(n_snap)
    nxt = 0; n_steps = int(round(state["t_end_ms"] / state["dt_ms"]))
    vmn, vmx = float("inf"), float("-inf")
    for k in range(n_steps):
        t = k * state["dt_ms"]; time_const.value = t; solver.step((t, t + state["dt_ms"]))
        ta = t + state["dt_ms"]
        while nxt < n_snap and times[nxt] <= ta + 1.0e-9:
            v_pde.x.scatter_forward(); va = v_pde.x.array
            vmn = min(vmn, float(va.min())); vmx = max(vmx, float(va.max()))
            hsp_stack[nxt] = ue_solve().x.array[v2d[bverts]]; actual[nxt] = ta
            vmax_snap[nxt] = float(va.max()); nxt += 1
    if nxt < n_snap:
        raise RuntimeError(f"only {nxt}/{n_snap} snapshots taken (t_end too small)")
    ue_destroy(); fv = v_pde.x.array
    return {"hsp_stack": hsp_stack, "snapshot_times_ms": actual, "v_m_max_per_snap": vmax_snap,
            "v_m_min_global": vmn, "v_m_max_global": vmx,
            "depol_frac_final": float((fv > DEPOL_THRESHOLD_MV).mean())}


# ---------------------------------------------------------------------------
# Per-worker state
# ---------------------------------------------------------------------------
_WORKER_STATE: dict | None = None


def _precompute_region_and_rv() -> tuple[np.ndarray, np.ndarray]:
    """3-region transmural marker + per-dof RV mask (nearest endo facet is RV).
    Run once in the main process (expand_layer_biv writes a diagnostic file)."""
    import tempfile
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        try:
            os.chdir(td)
            mesh, _, ft = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
            V = cg1_space(mesh); p = load_params()
            region_fun, _ = build_region_field(
                V=V, facet_tags=ft, lv_marker=LV, rv_marker=RV, epi_marker=EPI,
                endo_size=p["endo_size"], epi_size=p["epi_size"])
            region_marker = np.rint(region_fun.x.array).astype(np.int32)
            coords = V.tabulate_dof_coordinates()[:, :3]
            fdim = mesh.topology.dim - 1
            mesh.topology.create_connectivity(fdim, 0)
            f2v = mesh.topology.connectivity(fdim, 0)

            def centroids(tag):
                fac = ft.indices[ft.values == tag]
                return np.array([mesh.geometry.x[f2v.links(int(f)), :3].mean(0) for f in fac])
            lv_c, rv_c = centroids(LV), centroids(RV)
            allc = np.vstack([lv_c, rv_c])
            is_rv = np.concatenate([np.zeros(len(lv_c), bool), np.ones(len(rv_c), bool)])
            _, idx = cKDTree(allc).query(coords, k=1)
            rv_mask = is_rv[idx]
            return region_marker.copy(), rv_mask.copy()
        finally:
            os.chdir(cwd)


def _build_state(config: dict, region_marker: np.ndarray, rv_mask: np.ndarray) -> dict:
    mesh, _, facet_tags = load_gmsh_mesh(HEART_MSH, MPI.COMM_SELF)
    V = cg1_space(mesh)
    v2d = vertex_to_dof_map(V)
    bverts = np.unique(np.concatenate([
        tagged_facet_vertices(mesh, facet_tags, t) for t in (1, 2, 3, 4)])).astype(np.int64)
    return {
        "heart_mesh": mesh, "heart_facet_tags": facet_tags, "V_heart": V, "v2d_heart": v2d,
        "boundary_vertices": bverts, "hsp_points": mesh.geometry.x[bverts, :3].copy(),
        "dof_coords": V.tabulate_dof_coordinates()[:, :3].copy(),
        "region_marker": region_marker.astype(int, copy=False), "rv_mask": rv_mask.astype(bool, copy=False),
        "base_params": load_params(),
        "t_end_ms": float(config["t_end_ms"]), "dt_ms": float(config["dt_ms"]),
        "box_frac": float(config["box_half_width_frac"]), "base_seed": int(config.get("base_seed", 20260530)),
        "snapshot_times": even_snapshot_times(
            float(config["t_end_ms"]), float(config["dt_ms"]), int(config["n_snapshots"])),
    }


def _init_worker(config: dict, region_marker: np.ndarray, rv_mask: np.ndarray) -> None:
    global _WORKER_STATE
    _WORKER_STATE = _build_state(config, region_marker, rv_mask)


def _process_sample(args: tuple[int, str, bool]) -> dict:
    idx, out_str, overwrite = args
    st = _WORKER_STATE; assert st is not None
    out_dir = Path(out_str); sid = f"sample_{idx:05d}"; out_npz = out_dir / f"{sid}.npz"
    if out_npz.exists() and not overwrite:
        return {"sample_id": sid, "ok": True, "skipped": True}
    rng = np.random.default_rng(st["base_seed"] + idx)
    params = sample_params(rng, st["base_params"], st["box_frac"])
    t0 = time.perf_counter()
    try:
        r = run_one_sample(state=st, params=params, snapshot_times_ms=st["snapshot_times"])
    except Exception as exc:  # noqa: BLE001
        return {"sample_id": sid, "ok": False, "error": repr(exc), "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    np.savez(out_npz, hsp_points=st["hsp_points"], hsp_stack=r["hsp_stack"],
             snapshot_times_ms=r["snapshot_times_ms"], v_m_max_per_snap=r["v_m_max_per_snap"],
             params_json=json.dumps(params),
             varied_json=json.dumps({k: params[k] for k in VARIED}),
             t_end_ms=st["t_end_ms"], dt_ms=st["dt_ms"], seed=st["base_seed"] + idx,
             v_m_min_global=r["v_m_min_global"], v_m_max_global=r["v_m_max_global"],
             depol_frac_final=r["depol_frac_final"])
    return {"sample_id": sid, "ok": True, "wall_s": wall, "n_snapshots": len(r["snapshot_times_ms"]),
            "tau_in": params["tau_in"], "c_m": params["c_m"], "a_m": params["a_m"],
            "tau_close_rv": params["tau_close_rv"], "depol_frac_final": r["depol_frac_final"],
            "v_m_min_global": r["v_m_min_global"], "v_m_max_global": r["v_m_max_global"],
            "hsp_abs_max": float(np.abs(r["hsp_stack"]).max())}


class CpuTempMonitor(threading.Thread):
    def __init__(self, csv_path: Path, interval_s: float):
        super().__init__(name="cpu-temp-monitor", daemon=True)
        self.csv_path = csv_path; self.interval_s = interval_s; self._stop_evt = threading.Event()

    def stop(self): self._stop_evt.set()

    @staticmethod
    def _flat(temps):
        return {f"{s}_{(r.label or f'sensor{i}').replace(' ', '_').replace('/', '_')}_C": float(r.current)
                for s, rs in temps.items() for i, r in enumerate(rs)}

    def _poll(self):
        try:
            import psutil; return self._flat(psutil.sensors_temperatures())
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def run(self):
        first = self._poll(); header = ["timestamp"] + sorted(first.keys())
        self.csv_path.parent.mkdir(parents=True, exist_ok=True); new = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            if new: w.writeheader()
            w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), **first}); fh.flush()
            while not self._stop_evt.wait(self.interval_s):
                s = self._poll()
                w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"),
                            **{k: s.get(k, "") for k in header[1:]}}); fh.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--base-seed", type=int, default=None,
                    help="override config base_seed (use a value outside the training "
                         "range to synthesise an out-of-training held-out sample)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--temp-csv", default=None)
    ap.add_argument("--temp-interval-s", type=float, default=60.0)
    args = ap.parse_args(argv)

    config = {k: v for k, v in json.loads(args.config.read_text()).items() if not k.startswith("_")}
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    n_samples = args.n_samples if args.n_samples is not None else int(config["n_samples"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "index.jsonl"

    temp_csv = (args.output_dir / "cpu_temps.csv" if args.temp_csv is None
                else Path(args.temp_csv) if str(args.temp_csv) else None)
    monitor = None
    if temp_csv is not None:
        monitor = CpuTempMonitor(temp_csv, float(args.temp_interval_s)); monitor.start()
        print(f"cpu temp monitor -> {temp_csv} every {args.temp_interval_s:.0f}s", flush=True)

    print("precomputing region marker + RV mask ...", flush=True)
    region_marker, rv_mask = _precompute_region_and_rv()
    base = load_params()
    print(f"RV dofs={int(rv_mask.sum())} / {rv_mask.size}; varying {VARIED} "
          f"±{100*float(config['box_half_width_frac']):.0f}% around "
          f"{ {k: round(float(base[k]),4) for k in VARIED} }", flush=True)
    snaps = even_snapshot_times(float(config["t_end_ms"]), float(config["dt_ms"]), int(config["n_snapshots"]))
    print(f"{len(snaps)} snapshots in (0,{config['t_end_ms']}] ms, e.g. {snaps[:3]}...{snaps[-1]}", flush=True)

    items = [(k, str(args.output_dir), bool(args.overwrite)) for k in range(n_samples)]
    nw = max(1, int(args.n_workers))
    print(f"launching {nw} workers for {n_samples} samples ...", flush=True)
    t_start = time.perf_counter(); n_done = n_fail = 0
    try:
        if nw == 1:
            _init_worker(config, region_marker, rv_mask)
            for it in items:
                e = _process_sample(it)
                index_path.open("a").write(json.dumps(e) + "\n"); _log(e); n_done += 1
                n_fail += 0 if e.get("ok") else 1
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=nw, mp_context=ctx, initializer=_init_worker,
                                     initargs=(config, region_marker, rv_mask)) as pool:
                futs = {pool.submit(_process_sample, it): it[0] for it in items}
                for fut in as_completed(futs):
                    e = fut.result()
                    with index_path.open("a") as f: f.write(json.dumps(e) + "\n")
                    _log(e); n_done += 1; n_fail += 0 if e.get("ok") else 1
    finally:
        if monitor is not None: monitor.stop(); monitor.join(timeout=2.0)
    print(f"\ndone in {time.perf_counter()-t_start:.1f}s. {n_done} samples ({n_fail} failed).", flush=True)
    return 0 if n_fail == 0 else 1


def _log(e: dict) -> None:
    if e.get("skipped"): print(f"  {e['sample_id']} skipped"); return
    if not e.get("ok"): print(f"  {e['sample_id']} FAILED: {e.get('error','?')}"); return
    print(f"  {e['sample_id']}  {e['wall_s']:5.1f}s  tau_in={e['tau_in']:.3f} c_m={e['c_m']:.4f} "
          f"a_m={e['a_m']:.1f} tauRV={e['tau_close_rv']:.1f}  depol={e['depol_frac_final']:.3f} "
          f"|hsp|max={e['hsp_abs_max']:.2e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
