# Infarction experiment

A self-contained stabFEM test case: build a POD basis that spans infarct-induced
HSPs, then recover a **new** infarction from its body-surface measurements with the
shared solver in `ukb/stabFEM/`.

Follows Boulakia/Schenone/Gerbeau (arXiv:1111.5926v2 §3.3): a healthy-only POD basis
cannot represent an infarct's sharp transmembrane variation, so the snapshot set is
enriched with infarctions at many locations. Inside each scar ball `tau_out` is
divided by **100** (the paper uses 10).

## Pipeline (the same five steps every experiment follows)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scientific-python

# 1. Generate the database: 1 healthy + n_centers infarct runs -> HSP snapshots.
#    (check cpu_temps.csv if running many workers; see CPU-thermal caution)
python ukb/database/infarction/generate.py --n-workers 8

# 2. Build the POD basis on EPI∪BASE + all diagnostic plots.
python ukb/database/build_pod_basis.py \
  --samples-dir ukb/database/infarction/samples \
  --output-dir  ukb/database/infarction/pod_basis

# 3. Inspect pod_basis/{singular_values_log,pod_modes_vs_energy}.png and pick a
#    truncation rank R (target ~99.9% energy; Boulakia used 100).

# 4. Extend ONLY the truncated basis through the torso to BSPM modes.
python ukb/database/extend_pod_to_torso.py \
  --pod        ukb/database/infarction/pod_basis/pod_basis.npz \
  --n-modes    R \
  --output-dir ukb/database/infarction/extended_pod_basis

# 5. Recover a held-out infarction with stabFEM, pointing the solver at THIS
#    experiment's basis. Synthesise an off-grid held-out infarct first:
python ukb/database/infarction/generate.py --heldout --base-seed 770001 \
  --output-dir ukb/database/infarction/heldout
# then forward its HSP -> BSPM and run the inverse with:
python ukb/stabFEM/stabfem.py \
  --body-data    <held-out BSPM .npz> \
  --pod          ukb/database/infarction/pod_basis/pod_basis.npz \
  --extended-dir ukb/database/infarction/extended_pod_basis \
  --n-modes      R \
  --output-dir   ukb/database/infarction/results
```

(The HSP→BSPM forward step + noise sweep is driven by your own stabFEM tooling;
`run_heldout_sweep.py` is not part of this experiment folder.)

## Files

| path | what |
|---|---|
| `config.json` | experiment knobs: window, infarct geometry, snapshot schedule |
| `generate.py` | bespoke generator (healthy + infarcts at varied positions/radii) |
| `samples/` | `sample_NNNNN.npz` (0 = healthy, 1..N = infarcts) + `index.jsonl`, `roster.json`, `infarct_centres.png`, `cpu_temps.csv` |
| `pod_basis/` | `build_pod_basis.py` output (HSP modes on EPI∪BASE + plots) |
| `extended_pod_basis/` | `extend_pod_to_torso.py` output (truncated BSPM modes) |
| `results/` | stabFEM held-out reconstruction(s) |

## Parameters

Physical / Mitchell-Schaeffer values come from `ukb/pipeline/params.json` (the shared
source of truth). With `tau_out=70`, the scar uses `tau_out=0.7`. Defaults in
`config.json`: `t_end=400 ms`, `dt=0.5 ms`, `n_centers=18`, radius `[6,16] mm`,
snapshots `[50,50]` (healthy) / `[25,25]` (infarct) split at `100 ms`. Infarct centres
are farthest-point-sampled over the **front/visible epicardial wall** (the GIF camera),
so every scar is visible in the standard heart view.

## Sanity check

A POD basis built from the healthy case alone should reconstruct a held-out infarct
**markedly worse** than this infarct-enriched basis — reproducing the paper's Fig 7
conclusion and confirming the database design matters.
