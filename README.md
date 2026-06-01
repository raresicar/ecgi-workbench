# ECGi Workbench

An interactive **forward / inverse cardiac-electrophysiology** app built on the
stabFEM ECGi pipeline from the thesis. You place an infarct on the heart,
simulate the beat, and watch the data-enriched stabFEM inverse try to recover the
heart-surface potential and localise the scar — across time, with adjustable
noise, electrode coverage, and POD-basis size.

It runs **live on the workstation** (where dolfinx/PETSc/gmsh are installed); the
browser UI is forwarded to your laptop by VS Code's SSH port forwarding.

## Run

```bash
# 1. activate the thesis scientific environment (provides dolfinx, petsc, gmsh, …)
conda activate scientific-python

# 2. install the small UI extras (once)
pip install -r requirements.txt

# 3. launch
streamlit run app.py
```

Open the `localhost` URL Streamlit prints — VS Code forwards the port to your
laptop automatically.

## Workflow

**② Infarct-localisation lab**
1. **Click a point on the heart** to place an infarct; set its **radius**.
2. **Simulate & reconstruct** — runs the monodomain V_m → extracellular HSP
   forward solve, then the stabFEM inverse on every time frame.
3. Read the **localisation error**, **cosine** and **MINRES iterations**; compare
   **truth vs recovered** HSP per frame.
4. **Animate** truth or recovered HSP (play button) to see the infarct evolve.

**① Forward viewer** animates V_m or u_e for the last simulated beat.

Sidebar controls (apply to the inverse): POD **database** (prior), **n_modes**,
**measurement noise**, **electrodes** (full torso vs anterior patch), **γ_reg**,
and the simulation window / number of reconstructed frames.

## Architecture

```
app.py                 # Streamlit UI (forward viewer + localisation lab)
ui/components.py       # cached resources, clickable candidate points, helpers
ecgi/
  geometry.py          # Geometry      — the FIXED heart/torso meshes + surfaces
  forward.py           # ForwardSimulator — monodomain V_m -> extracellular HSP
  inverse.py           # InverseSolver — the data-enriched stabFEM inverse
  rendering.py         # Renderer      — Plotly 3D fields, picker, animation
  cases.py             # InfarctSpec / ForwardResult / InverseResult
  config.py            # paths + POD database discovery
  _vendor/             # scientific code copied verbatim from the thesis repo
```

`_vendor/` holds the thesis code (`common`, `stabfem`, the forward solvers, the
POD/database tooling) and the data the app needs: the **fixed meshes** (heart +
torso, never regenerated) and the prebuilt **POD databases** (`infarction`,
`four_params`). New databases can still be generated through the vendored tooling
— only the meshes are treated as fixed inputs.
