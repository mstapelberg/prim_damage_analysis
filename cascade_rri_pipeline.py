
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cascade Radiation Analysis Pipeline
-----------------------------------
Scans cascade analysis outputs and computes:
  • Recombination kinetics (overall and windowed rates after t>=t0)
  • Velocity statistics (total and ⟨111⟩ components) by defect type
  • Path tortuosity, mean free path, net displacement, and direction-change counts
  • Momentum-like statistics (requires a chosen effective mass per species)
  • Cluster-size statistics and defect-type fractions
  • Optional extras: orientation order parameter S111, turning-angle distribution,
    rotation/event rates, diffusion dimensionality index (1D↔3D)
  • Composite Radiation Resistance Index (RRI) with two flavors:
      - RRI_phys: physics-driven weighted z-sum aligned with the hypothesis
      - RRI_ml:   data-driven linear model predicting recombination after removing T
Usage
-----
python cascade_rri_pipeline.py --roots ROOT1 ROOT2 ... --job-meta job_meta.json --out OUTDIR
  or
python cascade_rri_pipeline.py --runs /abs/path/to/.../run_0 /abs/path/to/.../run_7 --out OUTDIR
The job_meta.json should map job_id -> { "temperature_K": <float>, "composition": "<string>" }
Example:
{
  "3685519": {"temperature_K": 800, "composition": "V-4Cr-4Ti"},
  "3685636": {"temperature_K": 600, "composition": "V-4Cr-4Ti"}
}
Author: GPT-5 Pro (analysis pipeline generator)
"""
from __future__ import annotations
import argparse, json, os, re, sys, math, glob, io, tarfile, warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Iterable, Any
import numpy as np
import pandas as pd

# -------- Utility --------------------------------------------------------------------------------

def robust_read_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, **kwargs)
        return df
    except Exception as e:
        warnings.warn(f"Failed to read CSV {path}: {e}")
        return None

def robust_read_json(path: str) -> Optional[Any]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        warnings.warn(f"Failed to read JSON {path}: {e}")
        return None

def detect_job_id(path: str) -> Optional[str]:
    # Return last numeric directory name in path (job id), if any
    parts = os.path.abspath(path).split(os.sep)
    for p in reversed(parts):
        if re.fullmatch(r"\d{6,}", p or ""):
            return p
    return None

def find_run_dirs(roots: List[str]) -> List[str]:
    """
    Find run directories following pattern: <root>/<job_id>/analysis_production_eigen_greedy1/run_X
    """
    run_dirs = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.exists(root):
            warnings.warn(f"Root missing: {root}")
            continue
        # job id dirs are numeric
        for job_dir in glob.glob(os.path.join(root, "*")):
            job_id = os.path.basename(job_dir)
            if not re.fullmatch(r"\d{6,}", job_id):
                continue
            base = os.path.join(job_dir, "analysis_production_eigen_greedy1")
            for run_dir in glob.glob(os.path.join(base, "run_*")):
                run_dirs.append(os.path.abspath(run_dir))
    return sorted(run_dirs)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# -------- Metrics --------------------------------------------------------------------------------

@dataclass
class RecombinationMetrics:
    start_time_ps: Optional[float] = None
    rate_overall_defects_per_ps: Optional[float] = None
    initial_defects: Optional[float] = None
    final_defects: Optional[float] = None
    total_recombined: Optional[float] = None
    time_span_ps: Optional[float] = None
    windowed: List[Dict[str, float]] = field(default_factory=list)

@dataclass
class VelocityMetrics:
    crowdion_mean_v_total: Optional[float] = None
    crowdion_std_v_total: Optional[float] = None
    crowdion_mean_v111: Optional[float] = None
    crowdion_std_v111: Optional[float] = None
    crowdion_n_tracks: Optional[int] = None
    dumbbell_mean_v_total: Optional[float] = None
    dumbbell_std_v_total: Optional[float] = None
    dumbbell_mean_v111: Optional[float] = None
    dumbbell_std_v111: Optional[float] = None
    dumbbell_n_tracks: Optional[int] = None

@dataclass
class PathMetrics:
    crowdion_mean_free_path_A: Optional[float] = None
    crowdion_mean_direction_changes: Optional[float] = None
    crowdion_mean_total_path_length_A: Optional[float] = None
    crowdion_mean_net_displacement_A: Optional[float] = None
    crowdion_mean_tortuosity: Optional[float] = None
    crowdion_n_tracks: Optional[int] = None
    dumbbell_mean_free_path_A: Optional[float] = None
    dumbbell_mean_direction_changes: Optional[float] = None
    dumbbell_mean_total_path_length_A: Optional[float] = None
    dumbbell_mean_net_displacement_A: Optional[float] = None
    dumbbell_mean_tortuosity: Optional[float] = None
    dumbbell_n_tracks: Optional[int] = None

@dataclass
class MomentumMetrics:
    crowdion_mean_p_amu_Aps: Optional[float] = None
    crowdion_std_p_amu_Aps: Optional[float] = None
    crowdion_mean_pmax_amu_Aps: Optional[float] = None
    crowdion_n_tracks: Optional[int] = None
    dumbbell_mean_p_amu_Aps: Optional[float] = None
    dumbbell_std_p_amu_Aps: Optional[float] = None
    dumbbell_mean_pmax_amu_Aps: Optional[float] = None
    dumbbell_n_tracks: Optional[int] = None

@dataclass
class FractionMetrics:
    crowdion_fraction: Optional[float] = None
    dumbbell_fraction: Optional[float] = None
    total_observations: Optional[int] = None

@dataclass
class ClusterMetrics:
    q10: Optional[float] = None
    q50: Optional[float] = None
    q90: Optional[float] = None
    mean: Optional[float] = None
    n_clusters: Optional[int] = None
    crowdions_count: Optional[int] = None  # counts of atoms in 1-atom objects
    dumbbells_count: Optional[int] = None  # counts of atoms in 2-atom objects *2
    other_count: Optional[int] = None      # counts of atoms in >2 objects
    total_interstitial_atoms: Optional[int] = None

@dataclass
class OrientationMetrics:
    # Optional extras
    S111_crowdion: Optional[float] = None
    S111_dumbbell: Optional[float] = None
    turn_rate_crowdion_per_ps: Optional[float] = None
    turn_rate_dumbbell_per_ps: Optional[float] = None

@dataclass
class RunSummary:
    run_dir: str
    job_id: Optional[str]
    composition: Optional[str] = None
    temperature_K: Optional[float] = None
    recombination: RecombinationMetrics = field(default_factory=RecombinationMetrics)
    velocity: VelocityMetrics = field(default_factory=VelocityMetrics)
    path: PathMetrics = field(default_factory=PathMetrics)
    momentum: MomentumMetrics = field(default_factory=MomentumMetrics)
    fractions: FractionMetrics = field(default_factory=FractionMetrics)
    clusters: ClusterMetrics = field(default_factory=ClusterMetrics)
    orientation: OrientationMetrics = field(default_factory=OrientationMetrics)
    key_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

# ---------- Computation helpers ------------------------------------------------------------------

def _orient_order_parameter_cos2(theta_rad: np.ndarray) -> float:
    # S = (3 <cos^2 θ> - 1)/2 ; θ is angle to ⟨111⟩ axis
    if theta_rad.size == 0:
        return np.nan
    c2 = np.cos(theta_rad)**2
    return float((3.0*np.mean(c2) - 1.0)/2.0)

def compute_recombination(defect_csv: str, t0: float = 0.25, window_ps: float = 1.0) -> RecombinationMetrics:
    rec = RecombinationMetrics(start_time_ps=t0)
    df = robust_read_csv(defect_csv)
    if df is None:
        return rec
    
    # Handle different time column names
    time_col = None
    for time_cand in ["time_ps", "Time_ps", "time", "Time"]:
        if time_cand in df.columns:
            time_col = time_cand
            break
    
    if time_col is None:
        return rec
    
    # pick defect count column heuristically
    count_col = None
    for cand in ["n_total", "n_defects", "n_interstitials_plus_vacancies", "n"]:
        if cand in df.columns:
            count_col = cand; break
    if count_col is None:
        # try sum of possible columns
        cands = [c for c in df.columns if re.search("defect|interstitial|vacanc", c, re.I)]
        if cands:
            df["n_sum"] = df[cands].sum(axis=1, min_count=1)
            count_col = "n_sum"
        else:
            return rec
    
    dfa = df[df[time_col] >= t0].copy()
    if dfa.empty:
        return rec
    t = dfa[time_col].to_numpy()
    n = dfa[count_col].to_numpy()
    rec.initial_defects = float(n[0])
    rec.final_defects = float(n[-1])
    rec.time_span_ps = float(t[-1] - t[0])
    rec.total_recombined = float(rec.initial_defects - rec.final_defects) if (rec.initial_defects is not None and rec.final_defects is not None) else None
    if rec.time_span_ps and rec.total_recombined is not None:
        rec.rate_overall_defects_per_ps = (rec.final_defects - rec.initial_defects)/rec.time_span_ps
    # windowed rates via linear fits in sliding windows
    wp = window_ps
    i0 = 0
    while i0 < len(t):
        t_start = t[i0]
        t_end = t_start + wp
        mask = (t >= t_start) & (t < t_end)
        if mask.sum() >= 3:
            tt = t[mask]; nn = n[mask]
            # fit slope dn/dt
            A = np.vstack([tt, np.ones_like(tt)]).T
            slope, intercept = np.linalg.lstsq(A, nn, rcond=None)[0]
            rec.windowed.append({"time_start_ps": float(t_start), "time_end_ps": float(t_end), "rate_defects_per_ps": float(slope), "n_points": int(mask.sum())})
        i0 += max(1, int(0.33*mask.sum()) if mask.sum() else 1)
    return rec

def compute_velocity_metrics(vel_csv: Optional[str], traj_json: Optional[str]) -> VelocityMetrics:
    vm = VelocityMetrics()
    # Priority: per-track in JSON if present, else per-frame CSV
    # We expect in JSON: tracks[].velocities_magnitude_Aps, velocities_parallel_Aps, types[]
    if traj_json and os.path.exists(traj_json):
        data = robust_read_json(traj_json)
        if data and "tracks" in data:
            v_tot = {"crowdion": [], "dumbbell": []}
            v_111 = {"crowdion": [], "dumbbell": []}
            for tr in data["tracks"]:
                typ = tr.get("types", [])
                vmag = tr.get("velocities_magnitude_Aps", [])
                vpar = tr.get("velocities_parallel_Aps", [])
                if not (typ and vmag and vpar) or len(typ) != len(vmag) or len(typ) != len(vpar):
                    continue
                # majority type per track
                vals = list(zip(typ, vmag, vpar))
                for tp, vm_, vp_ in vals:
                    if tp not in ("crowdion", "dumbbell"):
                        continue
                    v_tot[tp].append(float(vm_))
                    v_111[tp].append(float(vp_))
            def stats(arr):
                if len(arr)==0: return (np.nan, np.nan, 0)
                return (float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1) if len(arr)>1 else 0.0), len(arr))
            vm.crowdion_mean_v_total, vm.crowdion_std_v_total, vm.crowdion_n_tracks = stats(v_tot["crowdion"])
            vm.crowdion_mean_v111, vm.crowdion_std_v111, _ = stats(v_111["crowdion"])
            vm.dumbbell_mean_v_total, vm.dumbbell_std_v_total, vm.dumbbell_n_tracks = stats(v_tot["dumbbell"])
            vm.dumbbell_mean_v111, vm.dumbbell_std_v111, _ = stats(v_111["dumbbell"])
            return vm
    # Fallback CSV per-frame (columns must contain type and velocities)
    if vel_csv and os.path.exists(vel_csv):
        df = robust_read_csv(vel_csv)
        if df is not None:
            # heuristics for columns
            typ_col = next((c for c in df.columns if re.search("type", c, re.I)), None)
            vtot_col = next((c for c in df.columns if re.search("v.*mag|total_vel|speed", c, re.I)), None)
            v111_col = next((c for c in df.columns if re.search("111|parallel", c, re.I)), None)
            if typ_col and vtot_col and v111_col:
                def stats_on(tp):
                    arr_tot = df.loc[df[typ_col].str.lower()==tp, vtot_col].astype(float).to_numpy()
                    arr_111 = df.loc[df[typ_col].str.lower()==tp, v111_col].astype(float).to_numpy()
                    return (float(np.nanmean(arr_tot)) if arr_tot.size else np.nan,
                            float(np.nanstd(arr_tot, ddof=1)) if arr_tot.size>1 else 0.0,
                            float(np.nanmean(arr_111)) if arr_111.size else np.nan,
                            float(np.nanstd(arr_111, ddof=1)) if arr_111.size>1 else 0.0,
                            int(len(arr_tot)))
                c_mt, c_st, c_m111, c_s111, c_n = stats_on("crowdion")
                d_mt, d_st, d_m111, d_s111, d_n = stats_on("dumbbell")
                vm.crowdion_mean_v_total, vm.crowdion_std_v_total = c_mt, c_st
                vm.crowdion_mean_v111, vm.crowdion_std_v111 = c_m111, c_s111
                vm.crowdion_n_tracks = c_n
                vm.dumbbell_mean_v_total, vm.dumbbell_std_v_total = d_mt, d_st
                vm.dumbbell_mean_v111, vm.dumbbell_std_v111 = d_m111, d_s111
                vm.dumbbell_n_tracks = d_n
    return vm

def compute_path_metrics(traj_json: Optional[str], angle_thresh_deg: float = 30.0) -> PathMetrics:
    pm = PathMetrics()
    data = robust_read_json(traj_json) if traj_json and os.path.exists(traj_json) else None
    if not data or "tracks" not in data:
        return pm
    def per_type(tp: str):
        free_paths, nturns, path_lens, disp, torts = [], [], [], [], []
        ntracks = 0
        for tr in data["tracks"]:
            types = tr.get("types", [])
            if not types: 
                continue
            # majority type in this track
            # If types change, use sample-wise filter
            vpar = tr.get("velocities_parallel_Aps", [])
            pos = tr.get("positions_A", [])
            ang = tr.get("angles_111_deg", [])
            if not pos or not ang: 
                continue
            n = min(len(types), len(pos), len(ang))
            typ_mask = [types[i]==tp for i in range(n)]
            if not any(typ_mask):
                continue
            ntracks += 1
            # Direction changes: count increments where |Δangle| > threshold
            ang_arr = np.asarray([ang[i] for i in range(n) if typ_mask[i]], dtype=float)
            if ang_arr.size < 2:
                continue
            dtheta = np.abs(np.diff(ang_arr))
            turns = int(np.sum(dtheta > angle_thresh_deg))
            nturns.append(turns)
            # Path length and net displacement
            pos_arr = np.asarray([pos[i] for i in range(n) if typ_mask[i]], dtype=float)
            # pos entries are 3D coordinates
            p = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1).sum() if pos_arr.shape[0] >= 2 else 0.0
            path_lens.append(p)
            net = float(np.linalg.norm(pos_arr[-1]-pos_arr[0])) if pos_arr.shape[0] >= 2 else 0.0
            disp.append(net)
            torts.append((p/net) if (net>0) else np.nan)
            # Approximate mean free path: average step length between turns (or overall if no turns)
            if turns>0 and pos_arr.shape[0] > turns:
                # split into (turns+1) segments
                # approximate uniform segmentation
                seg_len = p / (turns+1)
                free_paths.append(seg_len)
            else:
                # no turns: free path ~ path length
                free_paths.append(p)
        def fstats(a):
            if len(a)==0: return (np.nan,)
            return (float(np.nanmean(a)),)
        return dict(mean_free_path_A=fstats(free_paths)[0],
                    mean_direction_changes=(float(np.nanmean(nturns)) if nturns else np.nan),
                    mean_total_path_length_A=fstats(path_lens)[0],
                    mean_net_displacement_A=fstats(disp)[0],
                    mean_tortuosity=fstats(torts)[0],
                    n_tracks=ntracks)
    c = per_type("crowdion"); d = per_type("dumbbell")
    for k,v in c.items(): setattr(pm, f"crowdion_{k}", v)
    for k,v in d.items(): setattr(pm, f"dumbbell_{k}", v)
    return pm

def compute_momentum_metrics(traj_json: Optional[str], effective_mass_amu: float = 50.94) -> MomentumMetrics:
    """
    Momentum-like proxy: p = m_eff * v_total. If velocity is per-atom or per-cluster,
    set m_eff accordingly in the caller (default is vanadium atomic mass).
    """
    mm = MomentumMetrics()
    data = robust_read_json(traj_json) if traj_json and os.path.exists(traj_json) else None
    if not data or "tracks" not in data:
        return mm
    mass = float(effective_mass_amu)
    def per_type(tp: str):
        vmax_list, v_list = [], []
        ntracks = 0
        for tr in data["tracks"]:
            types = tr.get("types", [])
            vmag = tr.get("velocities_magnitude_Aps", [])
            if not types or not vmag: 
                continue
            n = min(len(types), len(vmag))
            # sample-wise filter
            vm = [float(vmag[i]) for i in range(n) if types[i]==tp]
            if not vm:
                continue
            ntracks += 1
            v_list.extend(vm)
            vmax_list.append(max(vm))
        if not v_list:
            return (np.nan, np.nan, np.nan, 0)
        p = np.asarray(v_list) * mass
        pmax = np.asarray(vmax_list) * mass if vmax_list else np.array([])
        return (float(np.nanmean(p)),
                float(np.nanstd(p, ddof=1) if len(p)>1 else 0.0),
                float(np.nanmean(pmax) if pmax.size else np.nan),
                ntracks)
    cm, cs, cM, cn = per_type("crowdion")
    dm, ds, dM, dn = per_type("dumbbell")
    mm.crowdion_mean_p_amu_Aps, mm.crowdion_std_p_amu_Aps, mm.crowdion_mean_pmax_amu_Aps, mm.crowdion_n_tracks = cm, cs, cM, cn
    mm.dumbbell_mean_p_amu_Aps, mm.dumbbell_std_p_amu_Aps, mm.dumbbell_mean_pmax_amu_Aps, mm.dumbbell_n_tracks = dm, ds, dM, dn
    return mm

def compute_fraction_metrics(summary_csv: Optional[str], counts_csv: Optional[str]) -> FractionMetrics:
    fm = FractionMetrics()
    # Prefer counts (per frame) if available
    df = None
    if counts_csv and os.path.exists(counts_csv):
        df = robust_read_csv(counts_csv)
    elif summary_csv and os.path.exists(summary_csv):
        df = robust_read_csv(summary_csv)
    if df is None:
        return fm
    # Heuristics: look for counts of each type per frame
    lc = {c.lower(): c for c in df.columns}
    ccols = [lc[c] for c in lc if re.search("crowdion", c)]
    dcols = [lc[c] for c in lc if re.search("dumbbell", c)]
    if ccols and dcols:
        csum = df[ccols].sum(numeric_only=True).sum()
        dsum = df[dcols].sum(numeric_only=True).sum()
        tot = csum + dsum
        if tot>0:
            fm.crowdion_fraction = float(csum/tot)
            fm.dumbbell_fraction = float(dsum/tot)
            fm.total_observations = int(tot)
    return fm

def compute_cluster_metrics(clusters_csv: Optional[str], cd_summary_csv: Optional[str]) -> ClusterMetrics:
    cm = ClusterMetrics()
    dfc = robust_read_csv(clusters_csv) if clusters_csv and os.path.exists(clusters_csv) else None
    if dfc is not None:
        # assume "size" column in atoms
        size_col = next((c for c in dfc.columns if re.search("size", c, re.I)), None)
        if size_col:
            sizes = dfc[size_col].astype(float).to_numpy()
            if sizes.size:
                cm.q10 = float(np.nanpercentile(sizes, 10))
                cm.q50 = float(np.nanpercentile(sizes, 50))
                cm.q90 = float(np.nanpercentile(sizes, 90))
                cm.mean = float(np.nanmean(sizes))
                cm.n_clusters = int(np.isfinite(sizes).sum())
    # per-type atom counts from crowdion_dumbbell_summary
    dfs = robust_read_csv(cd_summary_csv) if cd_summary_csv and os.path.exists(cd_summary_csv) else None
    if dfs is not None:
        lc = {c.lower(): c for c in dfs.columns}
        # attempt to infer counts of atoms in each object type
        crowdion_atoms = dfs[lc.get("crowdions", next((c for c in dfs.columns if "crowdion" in c.lower()), None))].sum() if any("crowdion" in c.lower() for c in dfs.columns) else np.nan
        dumbbell_atoms = dfs[lc.get("dumbbells", next((c for c in dfs.columns if "dumbbell" in c.lower()), None))].sum() if any("dumbbell" in c.lower() for c in dfs.columns) else np.nan
        other_atoms = np.nan
        # "other" may be provided, else infer from total interstitial atoms per frame if available
        maybe_other = next((c for c in dfs.columns if "other" in c.lower()), None)
        if maybe_other:
            other_atoms = dfs[maybe_other].sum()
        # try total interstitial atoms column
        total_atoms_col = next((c for c in dfs.columns if re.search("total.*interstitial.*atoms", c, re.I)), None)
        total_atoms = float(dfs[total_atoms_col].sum()) if total_atoms_col else np.nan
        cm.crowdions_count = int(crowdion_atoms) if np.isfinite(crowdion_atoms) else None
        cm.dumbbells_count = int(dumbbell_atoms) if np.isfinite(dumbbell_atoms) else None
        cm.other_count = int(other_atoms) if np.isfinite(other_atoms) else None
        cm.total_interstitial_atoms = int(total_atoms) if np.isfinite(total_atoms) else None
    return cm

def compute_orientation_extras(traj_json: Optional[str], angle_key: str = "angles_111_deg", angle_thresh_deg: float = 30.0) -> OrientationMetrics:
    om = OrientationMetrics()
    data = robust_read_json(traj_json) if traj_json and os.path.exists(traj_json) else None
    if not data or "tracks" not in data:
        return om
    def per_type(tp: str):
        thetas = []
        turn_rates = []
        for tr in data["tracks"]:
            types = tr.get("types", [])
            ang = tr.get(angle_key, [])
            times = tr.get("times_ps", [])
            if not types or not ang or not times:
                continue
            n = min(len(types), len(ang), len(times))
            # sample-wise filter
            idx = [i for i in range(n) if types[i]==tp]
            if len(idx) < 2:
                continue
            a = np.radians(np.asarray([ang[i] for i in idx], dtype=float))
            thetas.append(a)
            # turn rate per ps
            dth = np.abs(np.diff(np.degrees(a)))
            turns = int(np.sum(dth > angle_thresh_deg))
            dt = float(times[idx[-1]] - times[idx[0]])
            tr_rate = (turns/dt) if dt>0 else np.nan
            turn_rates.append(tr_rate)
        if thetas:
            cat = np.concatenate(thetas)
            S = _orient_order_parameter_cos2(cat)
        else:
            S = np.nan
        trate = float(np.nanmean(turn_rates)) if turn_rates else np.nan
        return S, trate
    Sc, rc = per_type("crowdion")
    Sd, rd = per_type("dumbbell")
    om.S111_crowdion, om.turn_rate_crowdion_per_ps = Sc, rc
    om.S111_dumbbell, om.turn_rate_dumbbell_per_ps = Sd, rd
    return om

# ---------- RRI ----------------------------------------------------------------------------------

def robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median & MAD; returns zeros if degenerate."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x, dtype=float)
    return (x - med) / (1.4826 * mad)

def compute_RRI_phys(df_runs: pd.DataFrame) -> pd.Series:
    """
    Physics-guided RRI consistent with hypothesis:
      – larger (more negative) recombination rate  → higher RRI
      – lower crowdion fraction & lower v111(crowdion) → higher RRI
      – more direction changes & higher tortuosity (esp. dumbbells) → higher RRI
      – lower net displacement (esp. dumbbells) → higher RRI
    Implementation: robust-z each feature across runs, then weighted sum.
    """
    # define columns (with sign so that larger means more resistant)
    # Handle None values by replacing them with NaN before negation
    recomb_rate = df_runs["recombination_rate_defects_per_ps"].fillna(np.nan)
    cols_pos = {
        "recomb_rate_pos": -recomb_rate, # negate (more negative is better)
        "turn_rate_dumb": df_runs.get("turn_rate_dumbbell_per_ps"),
        "tortuosity_dumb": df_runs.get("dumbbell_mean_tortuosity"),
    }
    cols_neg = {
        "crowdion_frac": df_runs.get("crowdion_fraction"),
        "v111_crowdion": df_runs.get("crowdion_mean_v111_Aps"),
        "net_disp_dumb": df_runs.get("dumbbell_mean_net_displacement_A"),
    }
    # assemble feature matrix
    feats = {}
    for k, s in cols_pos.items():
        if s is not None:
            feats[k] = robust_z(np.asarray(s))
    for k, s in cols_neg.items():
        if s is not None:
            feats[k] = -robust_z(np.asarray(s))  # negative contribution
    if not feats:
        return pd.Series(np.zeros(len(df_runs)), index=df_runs.index)
    X = np.vstack([feats[k] for k in feats]).T
    # default weights
    keys = list(feats.keys())
    w = np.array([
        0.35 if "recomb_rate_pos" in k else
        0.20 if "turn_rate_dumb" in k else
        0.15 if "tortuosity_dumb" in k else
        0.15 if "crowdion_frac" in k else
        0.10 if "v111_crowdion" in k else
        0.05  # net_disp_dumb
        for k in keys
    ], dtype=float)
    w = w / w.sum()
    score = X @ w
    return pd.Series(score, index=df_runs.index)

def compute_RRI_ml(df_runs: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Data-driven linear model predicting recombination from features + temperature.
    Returns standardized residual (positive = better-than-expected recombination).
    """
    # Minimal dependency; use numpy lstsq
    cols = [
        "crowdion_fraction",
        "crowdion_mean_v111_Aps",
        "dumbbell_mean_net_displacement_A",
        "dumbbell_mean_tortuosity",
        "turn_rate_dumbbell_per_ps",
        "temperature_K",
    ]
    X_list, y = [], []
    for _, row in df_runs.iterrows():
        if not np.isfinite(row.get("recombination_rate_defects_per_ps", np.nan)):
            continue
        feat = []
        ok = True
        for c in cols:
            v = row.get(c, np.nan)
            if not np.isfinite(v):
                ok = False; break
            feat.append(float(v))
        if ok:
            X_list.append([1.0] + feat)  # intercept
            y.append(float(row["recombination_rate_defects_per_ps"]))
    if len(X_list) < 3:
        # not enough data
        return pd.Series(np.zeros(len(df_runs)), index=df_runs.index), {}
    X = np.asarray(X_list); y = np.asarray(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    # predict and residuals for rows with full features
    preds = np.full(len(df_runs), np.nan)
    idx_map = []
    for i, (_, row) in enumerate(df_runs.iterrows()):
        feat = []
        ok = True
        for c in cols:
            v = row.get(c, np.nan)
            if not np.isfinite(v):
                ok = False; break
            feat.append(float(v))
        if ok:
            preds[i] = np.array([1.0] + feat) @ beta
            idx_map.append(i)
    resid = y - (np.asarray([np.array([1.0] + [df_runs.iloc[i][c] for c in cols]) @ beta for i in idx_map]))
    # standardize residuals (robust)
    r_full = np.full(len(df_runs), np.nan)
    rz = robust_z(resid)
    for j, i in enumerate(idx_map):
        r_full[i] = rz[j]
    # coefficients
    coefs = {"intercept": float(beta[0])}
    for j, c in enumerate(cols, start=1):
        coefs[c] = float(beta[j])
    return pd.Series(r_full, index=df_runs.index), coefs

# ---------- Run analysis -------------------------------------------------------------------------

def analyze_run(run_dir: str, job_meta: Dict[str, Dict[str, Any]], angle_thresh_deg: float = 30.0, effective_mass_amu: float = 50.94) -> RunSummary:
    rs = RunSummary(run_dir=run_dir, job_id=detect_job_id(run_dir))
    # attach meta if available
    jm = job_meta.get(rs.job_id, {}) if rs.job_id else {}
    rs.temperature_K = jm.get("temperature_K") or jm.get("T_K") or jm.get("temperature") or None
    rs.composition = jm.get("composition") or jm.get("alloy") or None

    # expected files
    p = lambda f: os.path.join(run_dir, f)
    defect_csv = p("defect_summary.csv")
    vel_csv    = p("crowdion_dumbbell_velocity_per_frame.csv")
    counts_csv = p("crowdion_dumbbell_counts.csv")
    summary_csv= p("crowdion_dumbbell_summary.csv")
    clusters_csv = p("custom_clusters_per_frame.csv")
    traj_json = p("track_trajectories.json")  # preferred for rich kinematics
    # core metrics
    rs.recombination = compute_recombination(defect_csv)
    rs.velocity = compute_velocity_metrics(vel_csv, traj_json)
    rs.path = compute_path_metrics(traj_json, angle_thresh_deg=angle_thresh_deg)
    rs.momentum = compute_momentum_metrics(traj_json, effective_mass_amu=effective_mass_amu)
    rs.fractions = compute_fraction_metrics(summary_csv, counts_csv)
    rs.clusters = compute_cluster_metrics(clusters_csv, summary_csv)
    rs.orientation = compute_orientation_extras(traj_json, angle_thresh_deg=angle_thresh_deg)

    # key metrics collected in a flat dict (for tables)
    rs.key_metrics = {
        "recombination_rate_defects_per_ps": rs.recombination.rate_overall_defects_per_ps,
        "crowdion_fraction": rs.fractions.crowdion_fraction,
        "dumbbell_fraction": rs.fractions.dumbbell_fraction,
        "median_cluster_size_atoms": rs.clusters.q50,
        "crowdion_mean_v111_Aps": rs.velocity.crowdion_mean_v111,
        "dumbbell_mean_v111_Aps": rs.velocity.dumbbell_mean_v111,
        "crowdion_mean_net_displacement_A": rs.path.crowdion_mean_net_displacement_A,
        "dumbbell_mean_net_displacement_A": rs.path.dumbbell_mean_net_displacement_A,
        "dumbbell_mean_tortuosity": rs.path.dumbbell_mean_tortuosity,
        "turn_rate_dumbbell_per_ps": rs.orientation.turn_rate_dumbbell_per_ps,
        "temperature_K": rs.temperature_K,
    }
    # sanity notes
    if (rs.velocity.crowdion_n_tracks or 0) < 30:
        rs.notes.append("Few crowdion samples; statistics may be noisy.")
    if rs.fractions.crowdion_fraction is not None and rs.fractions.crowdion_fraction < 0.01:
        rs.notes.append("Crowdion fraction <1%; dynamics dominated by dumbbells.")
    return rs

def summarize_runs(run_dirs: List[str], job_meta: Dict[str, Dict[str, Any]], outdir: str, angle_thresh_deg: float = 30.0, effective_mass_amu: float = 50.94) -> Dict[str, Any]:
    ensure_dir(outdir)
    rows = []
    per_run_json = {}
    for r in run_dirs:
        rs = analyze_run(r, job_meta, angle_thresh_deg=angle_thresh_deg, effective_mass_amu=effective_mass_amu)
        d = asdict(rs)
        per_run_json[r] = d
        row = {"run_dir": r, "job_id": rs.job_id, "composition": rs.composition, "temperature_K": rs.temperature_K}
        row.update({k:v for k,v in rs.key_metrics.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    # RRI scores
    if not df.empty:
        df["RRI_phys"] = compute_RRI_phys(df)
        df["RRI_ml"], coefs = compute_RRI_ml(df)
    else:
        coefs = {}
    # save outputs
    df.to_csv(os.path.join(outdir, "aggregated_metrics.csv"), index=False)
    with open(os.path.join(outdir, "per_run_detailed.json"), "w") as f:
        json.dump(per_run_json, f, indent=2)
    with open(os.path.join(outdir, "rri_model_info.json"), "w") as f:
        json.dump(coefs, f, indent=2)
    return {"aggregated_csv": os.path.join(outdir, "aggregated_metrics.csv"),
            "per_run_json": os.path.join(outdir, "per_run_detailed.json"),
            "rri_info": os.path.join(outdir, "rri_model_info.json")}

# ---------- CLI ----------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Cascade Radiation Analysis Pipeline")
    ap.add_argument("--roots", nargs="*", help="Root folders like V-4Cr-4Ti (containing <job_id>/analysis_production_eigen_greedy1/run_X)", default=[])
    ap.add_argument("--runs", nargs="*", help="Direct list of run_* folders to analyze", default=[])
    ap.add_argument("--job-meta", type=str, default=None, help="Path to job_meta.json mapping job_id -> {temperature_K, composition}")
    ap.add_argument("--angle-thresh-deg", type=float, default=30.0, help="Turning angle threshold for direction changes")
    ap.add_argument("--eff-mass-amu", type=float, default=50.94, help="Effective mass (amu) for momentum proxy (default=V atomic mass)")
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    args = ap.parse_args()

    job_meta = {}
    if args.job_meta and os.path.exists(args.job_meta):
        with open(args.job_meta, "r") as f:
            job_meta = json.load(f)

    run_dirs = []
    if args.roots:
        run_dirs = find_run_dirs(args.roots)
    if args.runs:
        run_dirs.extend([os.path.abspath(r) for r in args.runs if os.path.exists(r)])
    run_dirs = sorted(set(run_dirs))
    if not run_dirs:
        print("No run directories found. Provide --roots or --runs.")
        sys.exit(2)

    paths = summarize_runs(run_dirs, job_meta, outdir=args.out, angle_thresh_deg=args.angle_thresh_deg, effective_mass_amu=args.eff_mass_amu)
    print(json.dumps(paths, indent=2))

if __name__ == "__main__":
    main()
