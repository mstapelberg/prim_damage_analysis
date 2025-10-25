
# Cascade Radiation Analysis Pipeline

**File:** `cascade_rri_pipeline.py`

Usage examples:

```bash
# Analyze all runs under listed composition roots and save outputs
python cascade_rri_pipeline.py --roots /data/V-4Cr-4Ti /data/V-2Cr-2Ti-2W-2Zr --job-meta job_meta.json --out ./out

# Analyze a single run directory directly
python cascade_rri_pipeline.py --runs /data/V-4Cr-4Ti/3685636/analysis_production_eigen_greedy1/run_0 --job-meta job_meta.json --out ./out_one
```

The pipeline computes the metrics discussed in our thread and assembles two RRI scores:
- `RRI_phys`: physics-guided, aligned with the "PEL heterogeneity -> restricted crowdions & 3D dumbbell diffusion -> more recombination" hypothesis.
- `RRI_ml`: data-driven residual from a linear model of recombination vs features+temperature (positive residual = more recombination than expected).

**Notes**
- Momentum uses an *effective mass* (default Vanadium: 50.94 amu). Change via `--eff-mass-amu`.
- Direction-change threshold is configurable via `--angle-thresh-deg` (default 30°).

Outputs:
- `aggregated_metrics.csv` — one row per run (ready for plotting/stats).
- `per_run_detailed.json` — full nested metrics per run.
- `rri_model_info.json` — coefficients for the simple data-driven model (if enough runs).
