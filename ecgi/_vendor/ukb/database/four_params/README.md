# Four-parameter box experiment

A separate stabFEM test case (own generator / POD basis / extended basis;
only `ukb/stabFEM/` is shared). Varies the four parameters the ECG is most
sensitive to (Boulakia §3.1/§4.2): **τ_in, C_m, A_m, τ_close^RV**, each drawn
uniformly from a small box (±`box_half_width_frac`) around its
`ukb/pipeline/params.json` value; all other params fixed. Healthy monodomain.

`τ_close^RV` is applied by classifying each myocardial vertex LV vs RV by its
nearest endocardial facet (tags LV=1 / RV=2): RV vertices take the varied
`tau_close_rv`, LV vertices keep the fixed transmural endo/mid/epi split.

## Pipeline (same five steps as every experiment)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scientific-python

# 1. Database: 100 samples, ~40 HSP snapshots each in (0,150] ms.
python ukb/database/four_params/generate.py --n-workers 16

# 2. POD basis (+ plots) on EPI∪BASE.
python ukb/database/build_pod_basis.py \
  --samples-dir ukb/database/four_params/samples \
  --output-dir  ukb/database/four_params/pod_basis

# 3. Truncation rank R = the mode where cumulative energy first hits 99%
#    (here R=15; read from pod_basis/pod_modes_vs_energy.png).

# 4. Extend only the truncated basis to BSPM (parallel).
python ukb/database/extend_pod_to_torso.py \
  --pod ukb/database/four_params/pod_basis/pod_basis.npz \
  --n-modes 15 --n-workers 12 \
  --output-dir ukb/database/four_params/extended_pod_basis

# 5. stabFEM inverse: point --pod / --extended-dir at this experiment's basis.
```

Config in `config.json`: `t_end_ms=150`, `dt_ms=0.5`, `n_samples=100`,
`n_snapshots=40` (evenly spaced in (0,150], same times for all samples),
`box_half_width_frac=0.2`.
