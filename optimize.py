"""
optimize.py — Optimization Experiments for March ML Mania 2026
==============================================================
Run with:  uv run python optimize.py

Prerequisites: run solution_2026.py first to populate .cache/

Outputs (all saved to Optimization/):
  feature_importance.csv    — XGBoost gain + weight per feature; log-transform comparison
  momentum_test_results.json — OOF Brier for windows: 5, 10, weighted-decay, 20 (baseline)
  ablation_study.txt        — Brier for (A) raw spline vs (B) spline + 10% underdog boost
  calibration_tuning.csv    — Grid search of clip range [0.01,0.99] vs [0.025,0.975]
  fallback_audit.csv        — 10 non-tournament predictions verified against Elo/seed signals
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

os.makedirs("Optimization", exist_ok=True)

CACHE_DIR     = ".cache"
DATA_DIR      = "./Data"
SEASON_CUTOFF = 2003
WOMEN_MIN     = 2010
T_SPLINE      = 25

XGB_PARAM = {
    "objective": "reg:squarederror",
    "booster": "gbtree",
    "eta": 0.0093,
    "subsample": 0.6,
    "colsample_bynode": 0.8,
    "num_parallel_tree": 2,
    "min_child_weight": 4,
    "max_depth": 4,
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_bin": 38,
}
NUM_ROUNDS = 704


def cache_load(name):
    p = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def cache_save(name, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, name + ".pkl"), "wb") as f:
        pickle.dump(obj, f)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (same pipeline as solution_2026.py)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("OPTIMIZATION EXPERIMENTS")
print("=" * 60)
print("\nLoading data...")

M_regular  = pd.read_csv(f"{DATA_DIR}/MRegularSeasonDetailedResults.csv")
M_tourney  = pd.read_csv(f"{DATA_DIR}/MNCAATourneyDetailedResults.csv")
M_seeds    = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
M_ordinals = pd.read_csv(f"{DATA_DIR}/MMasseyOrdinals.csv")
W_regular  = pd.read_csv(f"{DATA_DIR}/WRegularSeasonDetailedResults.csv")
W_tourney  = pd.read_csv(f"{DATA_DIR}/WNCAATourneyDetailedResults.csv")
W_seeds    = pd.read_csv(f"{DATA_DIR}/WNCAATourneySeeds.csv")

regular_results = pd.concat([M_regular, W_regular], ignore_index=True)
tourney_results = pd.concat([M_tourney, W_tourney],  ignore_index=True)
seeds_df        = pd.concat([M_seeds,   W_seeds],    ignore_index=True)

regular_results = regular_results[regular_results["Season"] >= SEASON_CUTOFF]
tourney_results = tourney_results[tourney_results["Season"] >= SEASON_CUTOFF]
seeds_df        = seeds_df[seeds_df["Season"] >= SEASON_CUTOFF]

_ADJ_COLS = [
    "LScore", "WScore",
    "LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF",
    "WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF",
]

def prepare_data(df):
    df = df[["Season","DayNum","LTeamID","LScore","WTeamID","WScore","NumOT",
             "LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF",
             "WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF"]].copy()
    adjot = (40 + 5 * df["NumOT"]) / 40
    for col in _ADJ_COLS:
        df[col] = df[col] / adjot
    dfswap = df.copy()
    df.columns     = [c.replace("W","T1_").replace("L","T2_") for c in df.columns]
    dfswap.columns = [c.replace("L","T1_").replace("W","T2_") for c in dfswap.columns]
    out = pd.concat([df, dfswap], ignore_index=True)
    out["PointDiff"] = out["T1_Score"] - out["T2_Score"]
    out["win"]       = (out["PointDiff"] > 0).astype(int)
    out["men_women"] = out["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)
    return out

regular_data = prepare_data(regular_results)
tourney_data = prepare_data(tourney_results)

# Seeds
seeds_df["seed"] = seeds_df["Seed"].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds_df[["Season","TeamID","seed"]].rename(columns={"TeamID":"T1_TeamID","seed":"T1_seed"})
seeds_T2 = seeds_df[["Season","TeamID","seed"]].rename(columns={"TeamID":"T2_TeamID","seed":"T2_seed"})
tourney_data = tourney_data[["Season","T1_TeamID","T2_TeamID","DayNum","PointDiff","win","men_women"]]
tourney_data = tourney_data.merge(seeds_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(seeds_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

# Season averages
BOXCOLS = [
    "T1_Score","T1_FGM","T1_FGA","T1_FGM3","T1_FGA3","T1_FTM","T1_FTA",
    "T1_OR","T1_DR","T1_Ast","T1_TO","T1_Stl","T1_Blk","T1_PF",
    "T2_Score","T2_FGM","T2_FGA","T2_FGM3","T2_FGA3","T2_FTM","T2_FTA",
    "T2_OR","T2_DR","T2_Ast","T2_TO","T2_Stl","T2_Blk","T2_PF","PointDiff",
]
ss = regular_data.groupby(["Season","T1_TeamID"])[BOXCOLS].mean().reset_index()
ss_T1 = ss.copy()
ss_T1.columns = ["T1_avg_" + c.replace("T1_","").replace("T2_","opponent_") for c in ss_T1.columns]
ss_T1 = ss_T1.rename(columns={"T1_avg_Season":"Season","T1_avg_TeamID":"T1_TeamID"})
ss_T2 = ss.copy()
ss_T2.columns = ["T2_avg_" + c.replace("T1_","").replace("T2_","opponent_") for c in ss_T2.columns]
ss_T2 = ss_T2.rename(columns={"T2_avg_Season":"Season","T2_avg_TeamID":"T2_TeamID"})
tourney_data = tourney_data.merge(ss_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(ss_T2, on=["Season","T2_TeamID"], how="left")

# Elo
def _update_elo(w, l, k=100, width=400):
    exp = 1.0 / (1 + 10 ** ((l - w) / width))
    c   = k * (1 - exp)
    return w + c, l - c

elos_list = []
for season in tqdm.tqdm(sorted(regular_data["Season"].unique()), desc="  Elo"):
    ss_s = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)].reset_index(drop=True)
    teams = set(ss_s["T1_TeamID"]) | set(ss_s["T2_TeamID"])
    elo   = {t: 1000 for t in teams}
    for i in range(len(ss_s)):
        w_id, l_id     = ss_s.loc[i,"T1_TeamID"], ss_s.loc[i,"T2_TeamID"]
        elo[w_id], elo[l_id] = _update_elo(elo[w_id], elo[l_id])
    edf = pd.DataFrame.from_dict(elo, orient="index").reset_index()
    edf.columns = ["TeamID","elo"]
    edf["Season"] = season
    elos_list.append(edf)

elos    = pd.concat(elos_list, ignore_index=True)
elos_T1 = elos.rename(columns={"TeamID":"T1_TeamID","elo":"T1_elo"})
elos_T2 = elos.rename(columns={"TeamID":"T2_TeamID","elo":"T2_elo"})
tourney_data = tourney_data.merge(elos_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(elos_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

# Massey Ordinals
ORDINAL_SYSTEMS = [s for s in ["POM","NET","MAS","SAG","MOR"]
                   if s in M_ordinals["SystemName"].unique()]
ord_latest = (
    M_ordinals[M_ordinals["SystemName"].isin(ORDINAL_SYSTEMS)]
    .sort_values("RankingDayNum")
    .groupby(["Season","SystemName","TeamID"]).last()
    .reset_index()[["Season","SystemName","TeamID","OrdinalRank"]]
)
ord_pivot = ord_latest.pivot_table(
    index=["Season","TeamID"], columns="SystemName", values="OrdinalRank"
).reset_index()
ord_pivot.columns = ["Season","TeamID"] + [f"ord_{c}" for c in ord_pivot.columns[2:]]

_ord_t1 = {"TeamID":"T1_TeamID"}
_ord_t2 = {"TeamID":"T2_TeamID"}
for sys in ORDINAL_SYSTEMS:
    _ord_t1[f"ord_{sys}"] = f"T1_ord_{sys}"
    _ord_t2[f"ord_{sys}"] = f"T2_ord_{sys}"

ord_T1 = ord_pivot.rename(columns=_ord_t1)
ord_T2 = ord_pivot.rename(columns=_ord_t2)
tourney_data = tourney_data.merge(ord_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(ord_T2, on=["Season","T2_TeamID"], how="left")
for sys in ORDINAL_SYSTEMS:
    tourney_data[f"ord_diff_{sys}"] = tourney_data[f"T2_ord_{sys}"] - tourney_data[f"T1_ord_{sys}"]

# GLM Quality (from main cache — do NOT recompute)
glm_quality = cache_load("glm_quality")
assert glm_quality is not None, "ERROR: .cache/glm_quality.pkl missing. Run solution_2026.py first."
glm_T1 = glm_quality.rename(columns={"TeamID":"T1_TeamID","quality":"T1_quality"})
glm_T2 = glm_quality.rename(columns={"TeamID":"T2_TeamID","quality":"T2_quality"})
tourney_data = tourney_data.merge(glm_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(glm_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

# Momentum (20-game base — for experiments we rebuild per window)
MOM_COLS = ["T1_Score","T1_FGA","T1_OR","T1_DR","T1_Blk","T1_PF",
            "T2_Score","T2_FGA","T2_OR","T2_DR","T2_Blk","T2_PF","PointDiff"]
rd_sorted = regular_data.sort_values(["Season","T1_TeamID","DayNum"]).copy()
rd_sorted["_rank"] = rd_sorted.groupby(["Season","T1_TeamID"])["DayNum"].rank(
    ascending=False, method="first"
)
mom20 = rd_sorted[rd_sorted["_rank"] <= 20].groupby(["Season","T1_TeamID"])[MOM_COLS].mean().reset_index()
mom_T1 = mom20.copy()
mom_T1.columns = ["T1_mom_" + c.replace("T1_","").replace("T2_","opp_") for c in mom_T1.columns]
mom_T1 = mom_T1.rename(columns={"T1_mom_Season":"Season","T1_mom_TeamID":"T1_TeamID"})
mom_T2 = mom20.copy()
mom_T2.columns = ["T2_mom_" + c.replace("T1_","").replace("T2_","opp_") for c in mom_T2.columns]
mom_T2 = mom_T2.rename(columns={"T2_mom_Season":"Season","T2_mom_TeamID":"T2_TeamID"})
tourney_data = tourney_data.merge(mom_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(mom_T2, on=["Season","T2_TeamID"], how="left")

# Feature sets
FEATURES_BASE = [
    "men_women","T1_seed","T2_seed","Seed_diff",
    "T1_avg_Score","T1_avg_FGA","T1_avg_OR","T1_avg_DR","T1_avg_Blk","T1_avg_PF",
    "T1_avg_opponent_FGA","T1_avg_opponent_Blk","T1_avg_opponent_PF","T1_avg_PointDiff",
    "T2_avg_Score","T2_avg_FGA","T2_avg_OR","T2_avg_DR","T2_avg_Blk","T2_avg_PF",
    "T2_avg_opponent_FGA","T2_avg_opponent_Blk","T2_avg_opponent_PF","T2_avg_PointDiff",
    "T1_elo","T2_elo","elo_diff","T1_quality","T2_quality",
]
FEATURES_MOM = (
    [f"T1_mom_{c}" for c in ["Score","FGA","OR","DR","Blk","PF","PointDiff"]] +
    [f"T2_mom_{c}" for c in ["Score","FGA","OR","DR","Blk","PF","PointDiff"]]
)
FEATURES_ORD = []
for sys in ORDINAL_SYSTEMS:
    FEATURES_ORD += [f"T1_ord_{sys}", f"T2_ord_{sys}", f"ord_diff_{sys}"]
FEATURES_INNOV = FEATURES_BASE + FEATURES_MOM + FEATURES_ORD

print(f"  tourney_data rows: {len(tourney_data):,}")
print(f"  Innovation features: {len(FEATURES_INNOV)}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: run LOSO returning raw margins (cached separately from solution_2026.py)
# ─────────────────────────────────────────────────────────────────────────────
def run_loso_margins(td, features, cache_key):
    """LOSO CV returning (margins, labels, ss, mw, models)."""
    cached = cache_load(cache_key)
    if cached is not None:
        print(f"  [CACHE HIT] {cache_key}")
        return cached
    seasons = sorted(td["Season"].unique())
    models_, margins_, targets_, ss_, mw_ = {}, [], [], [], []
    for s in tqdm.tqdm(seasons, desc=f"  LOSO [{cache_key}]"):
        tr = td["Season"] != s
        vl = td["Season"] == s
        dtrain = xgb.DMatrix(td.loc[tr, features].values, label=td.loc[tr,"PointDiff"].values)
        models_[s] = xgb.train(XGB_PARAM, dtrain, num_boost_round=NUM_ROUNDS)
        preds = models_[s].predict(xgb.DMatrix(td.loc[vl, features].values))
        margins_ += list(preds)
        targets_  += list(td.loc[vl,"PointDiff"].values)
        ss_       += [s] * vl.sum()
        mw_       += list(td.loc[vl,"men_women"].values)
    result = (np.array(margins_), np.array(targets_), np.array(ss_), np.array(mw_), models_)
    cache_save(cache_key, result)
    return result


def calibrate_brier(margins, targets, ss, clip_lo=0.025, clip_hi=0.975, boost=False):
    """Fit spline on full OOF data, return Stage1 Brier (2022+)."""
    labels = (targets > 0).astype(int)
    dat    = sorted(zip(margins, labels), key=lambda x: x[0])
    p, l   = zip(*dat)
    spline = UnivariateSpline(np.clip(p, -T_SPLINE, T_SPLINE), l, k=5)
    probs  = np.clip(spline(np.clip(margins, -T_SPLINE, T_SPLINE)), clip_lo, clip_hi)
    if boost:
        probs = np.where(probs < 0.85, probs + probs * 0.10, probs).clip(clip_lo, clip_hi)
    mask = ss >= 2022
    m    = (tourney_data_ss >= 2022) & (tourney_data_mw == 1)
    w    = (tourney_data_ss >= 2022) & (tourney_data_mw == 0)
    # We align using the stored ss and mw from the result
    all_s1 = brier_score_loss(labels[mask], probs[mask])
    return all_s1, probs, labels, mask


# ─────────────────────────────────────────────────────────────────────────────
# LOAD (or compute) INNOVATION LOSO MARGINS
# Reuse cached models from solution_2026.py if available, else run new LOSO.
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading innovation LOSO results...")

# Try to load from solution_2026.py cache first (has models_ but no margins)
sol_cache = cache_load("loso_innovation")

# Use our own opt cache (includes margins) if present
opt_innov = cache_load("opt_innov")
if opt_innov is not None:
    print("  [CACHE HIT] opt_innov")
    base_margins, base_targets, base_ss, base_mw, base_models = opt_innov
else:
    if sol_cache is not None:
        # Re-derive margins from the already-trained models in solution cache
        print("  Deriving OOF margins from solution_2026 cached models...")
        _sol_probs, _sol_labels, _sol_ss, _sol_mw, sol_models, _sol_spline, _sol_T = sol_cache
        td_arr   = tourney_data.reset_index(drop=True)
        base_margins = np.full(len(td_arr), np.nan)
        base_targets = td_arr["PointDiff"].values
        base_ss      = td_arr["Season"].values
        base_mw      = td_arr["men_women"].values
        for s in tqdm.tqdm(sorted(sol_models.keys()), desc="  Re-predict"):
            mask = td_arr["Season"].values == s
            if mask.sum() > 0:
                X_vl = td_arr.loc[mask, FEATURES_INNOV].values
                base_margins[mask] = sol_models[s].predict(xgb.DMatrix(X_vl))
        # Drop any rows where model had no match (seasons outside training range)
        valid = ~np.isnan(base_margins)
        base_margins = base_margins[valid]
        base_targets = base_targets[valid]
        base_ss      = base_ss[valid]
        base_mw      = base_mw[valid]
        base_models  = sol_models
        cache_save("opt_innov", (base_margins, base_targets, base_ss, base_mw, base_models))
    else:
        print("  No solution cache found — running fresh LOSO (this takes ~10 min)...")
        base_margins, base_targets, base_ss, base_mw, base_models = \
            run_loso_margins(tourney_data, FEATURES_INNOV, "opt_innov")

base_labels = (base_targets > 0).astype(int)

# Baseline Brier (innovation model, clip=0.025/0.975)
dat_base = sorted(zip(base_margins, base_labels), key=lambda x: x[0])
_p, _l   = zip(*dat_base)
base_spline = UnivariateSpline(np.clip(_p, -T_SPLINE, T_SPLINE), _l, k=5)
base_probs  = np.clip(base_spline(np.clip(base_margins, -T_SPLINE, T_SPLINE)), 0.025, 0.975)
s1_mask     = base_ss >= 2022
men_mask    = (base_ss >= 2022) & (base_mw == 1)
women_mask  = (base_ss >= 2022) & (base_mw == 0)
base_brier_s1 = brier_score_loss(base_labels[s1_mask], base_probs[s1_mask])
print(f"  Baseline innovation Brier (Stage1 2022-25): {base_brier_s1:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: FEATURE IMPORTANCE (gain + weight)  + log-transform test
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[EXP 1] Feature Importance + Log-Transform Test")
print("=" * 60)

# Aggregate gain and weight across all season models
gain_agg   = {}
weight_agg = {}
n_models   = len(base_models)
for s, model in base_models.items():
    for feat, val in model.get_score(importance_type="gain").items():
        gain_agg[feat] = gain_agg.get(feat, 0) + val / n_models
    for feat, val in model.get_score(importance_type="weight").items():
        weight_agg[feat] = weight_agg.get(feat, 0) + val / n_models

# Build DataFrame — map XGBoost f-index back to feature names
feat_map  = {f"f{i}": name for i, name in enumerate(FEATURES_INNOV)}
rows = []
for fkey in set(list(gain_agg.keys()) + list(weight_agg.keys())):
    fname = feat_map.get(fkey, fkey)
    rows.append({
        "feature":    fname,
        "xgb_key":   fkey,
        "avg_gain":   round(gain_agg.get(fkey, 0), 4),
        "avg_weight": round(weight_agg.get(fkey, 0), 4),
        "is_ordinal": any(f"ord_{s}" in fname for s in ORDINAL_SYSTEMS),
        "is_momentum": "_mom_" in fname,
    })

fi_df = pd.DataFrame(rows).sort_values("avg_gain", ascending=False).reset_index(drop=True)
fi_df.to_csv("Optimization/feature_importance.csv", index=False)
print(f"  Saved Optimization/feature_importance.csv  ({len(fi_df)} features)")

# Top-10 and bottom-10 by gain
print("\n  Top-10 features by avg gain:")
for _, row in fi_df.head(10).iterrows():
    print(f"    {row['feature']:<35}  gain={row['avg_gain']:.1f}  weight={row['avg_weight']:.1f}")

ord_rows = fi_df[fi_df["is_ordinal"]]
if len(ord_rows) > 0:
    ord_avg_gain = ord_rows["avg_gain"].mean()
    all_avg_gain = fi_df["avg_gain"].mean()
    print(f"\n  Ordinal feature avg gain:  {ord_avg_gain:.1f}  (overall avg: {all_avg_gain:.1f})")
    low_gain = ord_avg_gain < all_avg_gain * 0.7
    print(f"  Ordinal features are {'LOW GAIN -> testing log-transform' if low_gain else 'competitive -> log-transform optional'}")
else:
    low_gain = False
    print("  No ordinal features found in importance scores (all NaN?)")

# Log-transform test: add log1p(ordinal rank) features alongside originals
print("\n  Running log-transform test (always, for completeness)...")
LOG_FEATS = []
for sys in ORDINAL_SYSTEMS:
    for prefix in ["T1_ord_", "T2_ord_"]:
        col     = f"{prefix}{sys}"
        log_col = f"log_{prefix}{sys}"
        if col in tourney_data.columns:
            tourney_data[log_col] = np.log1p(tourney_data[col].clip(lower=0))
            LOG_FEATS.append(log_col)

FEATURES_LOG = FEATURES_INNOV + LOG_FEATS
log_margins, log_targets, log_ss, log_mw, _ = \
    run_loso_margins(tourney_data, FEATURES_LOG, "opt_log_ordinals")
log_labels = (log_targets > 0).astype(int)
dat_log    = sorted(zip(log_margins, log_labels), key=lambda x: x[0])
_lp, _ll   = zip(*dat_log)
log_spline = UnivariateSpline(np.clip(_lp, -T_SPLINE, T_SPLINE), _ll, k=5)
log_probs  = np.clip(log_spline(np.clip(log_margins, -T_SPLINE, T_SPLINE)), 0.025, 0.975)
log_s1     = log_ss >= 2022
log_brier  = brier_score_loss(log_labels[log_s1], log_probs[log_s1])
delta_log  = base_brier_s1 - log_brier

# Append log-transform comparison row to CSV
extra = pd.DataFrame([{
    "feature": "[LOG-TRANSFORM TEST] LOSO Brier (Stage1)",
    "xgb_key": "N/A",
    "avg_gain": float("nan"),
    "avg_weight": float("nan"),
    "is_ordinal": False,
    "is_momentum": False,
    "note": f"base={base_brier_s1:.5f} log={log_brier:.5f} delta={delta_log:+.5f}",
}])
fi_df_full = pd.concat([fi_df, extra], ignore_index=True)
fi_df_full.to_csv("Optimization/feature_importance.csv", index=False)
print(f"  Base innovation Brier: {base_brier_s1:.5f}")
print(f"  Log-transform Brier:   {log_brier:.5f}  (delta={delta_log:+.5f} {'IMPROVEMENT' if delta_log > 0 else 'no gain'})")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: MOMENTUM WINDOW SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[EXP 2] Momentum Window Sensitivity (5, 10, weighted-decay, 20)")
print("=" * 60)

MOM_WINDOWS = {
    "last_5":  {"n_games": 5,  "weighted": False},
    "last_10": {"n_games": 10, "weighted": False},
    "weighted_decay": {"n_games": None, "weighted": True},
    "last_20": {"n_games": 20, "weighted": False},   # baseline
}

def build_momentum(n_games=None, weighted=False):
    """Return (mom_T1, mom_T2) DataFrames for the given window."""
    rd = rd_sorted.copy()
    if weighted:
        # Recent 10 games weighted 2x, rest 1x
        rd["_weight"] = np.where(rd["_rank"] <= 10, 2.0, 1.0)
        for col in MOM_COLS:
            rd[f"_w_{col}"] = rd[col] * rd["_weight"]
        grp   = rd.groupby(["Season","T1_TeamID"])
        w_sum = grp["_weight"].sum()
        wcols = [f"_w_{c}" for c in MOM_COLS]
        w_col_sums = grp[wcols].sum()
        mom = (w_col_sums.div(w_sum, axis=0)).reset_index()
        mom.columns = ["Season","T1_TeamID"] + MOM_COLS
    else:
        mom = (rd[rd["_rank"] <= n_games]
               .groupby(["Season","T1_TeamID"])[MOM_COLS].mean().reset_index())

    m_T1 = mom.copy()
    m_T1.columns = ["T1_mom_" + c.replace("T1_","").replace("T2_","opp_") for c in m_T1.columns]
    m_T1 = m_T1.rename(columns={"T1_mom_Season":"Season","T1_mom_TeamID":"T1_TeamID"})
    m_T2 = mom.copy()
    m_T2.columns = ["T2_mom_" + c.replace("T1_","").replace("T2_","opp_") for c in m_T2.columns]
    m_T2 = m_T2.rename(columns={"T2_mom_Season":"Season","T2_mom_TeamID":"T2_TeamID"})
    return m_T1, m_T2

mom_results = {}
base_mom_cols = [c for c in tourney_data.columns if "_mom_" in c]

for win_name, cfg in MOM_WINDOWS.items():
    print(f"\n  Window: {win_name}")
    # Build tourney_data variant with this momentum window
    td_win = tourney_data.drop(columns=base_mom_cols)
    m_T1, m_T2 = build_momentum(**cfg)
    td_win = td_win.merge(m_T1, on=["Season","T1_TeamID"], how="left")
    td_win = td_win.merge(m_T2, on=["Season","T2_TeamID"], how="left")

    mgns, tgts, sss, mws, _ = run_loso_margins(td_win, FEATURES_INNOV, f"opt_mom_{win_name}")
    lbs  = (tgts > 0).astype(int)
    dat  = sorted(zip(mgns, lbs), key=lambda x: x[0])
    p, l = zip(*dat)
    spl  = UnivariateSpline(np.clip(p, -T_SPLINE, T_SPLINE), l, k=5)
    prbs = np.clip(spl(np.clip(mgns, -T_SPLINE, T_SPLINE)), 0.025, 0.975)
    s1   = sss >= 2022
    men  = (sss >= 2022) & (mws == 1)
    wom  = (sss >= 2022) & (mws == 0)
    mom_results[win_name] = {
        "stage1_brier": round(brier_score_loss(lbs[s1], prbs[s1]), 5),
        "men_brier":    round(brier_score_loss(lbs[men], prbs[men]), 5) if men.sum() > 0 else None,
        "women_brier":  round(brier_score_loss(lbs[wom], prbs[wom]), 5) if wom.sum() > 0 else None,
    }
    print(f"    Stage1 Brier: {mom_results[win_name]['stage1_brier']:.5f}  "
          f"men={mom_results[win_name]['men_brier']}  women={mom_results[win_name]['women_brier']}")

# Determine winner
best_win = min(mom_results, key=lambda k: mom_results[k]["stage1_brier"])
mom_results["_winner"] = best_win
mom_results["_base_innov_brier"] = round(base_brier_s1, 5)

with open("Optimization/momentum_test_results.json", "w") as f:
    json.dump(mom_results, f, indent=2)
print(f"\n  Winner: {best_win}  (Brier={mom_results[best_win]['stage1_brier']:.5f})")
print("  Saved Optimization/momentum_test_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: ABLATION STUDY — "+10% underdog boost"
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[EXP 3] Ablation Study: Raw Spline vs Spline + 10% Boost")
print("=" * 60)

def score_variant(margins, targets, ss, mw, clip_lo=0.025, clip_hi=0.975, boost=False, label=""):
    lbs  = (targets > 0).astype(int)
    dat  = sorted(zip(margins, lbs), key=lambda x: x[0])
    p, l = zip(*dat)
    spl  = UnivariateSpline(np.clip(p, -T_SPLINE, T_SPLINE), l, k=5)
    prbs = np.clip(spl(np.clip(margins, -T_SPLINE, T_SPLINE)), clip_lo, clip_hi)
    if boost:
        prbs = np.where(prbs < 0.85, prbs + prbs * 0.10, prbs).clip(clip_lo, clip_hi)
    s1  = ss >= 2022
    men = (ss >= 2022) & (mw == 1)
    wom = (ss >= 2022) & (mw == 0)
    return {
        "label":       label,
        "clip":        f"[{clip_lo}, {clip_hi}]",
        "boost":       boost,
        "brier_all":   round(brier_score_loss(lbs, prbs), 5),
        "brier_s1":    round(brier_score_loss(lbs[s1], prbs[s1]), 5),
        "brier_men":   round(brier_score_loss(lbs[men], prbs[men]), 5) if men.sum() > 0 else None,
        "brier_women": round(brier_score_loss(lbs[wom], prbs[wom]), 5) if wom.sum() > 0 else None,
    }

var_A = score_variant(base_margins, base_targets, base_ss, base_mw, 0.025, 0.975, False, "(A) Raw Spline")
var_B = score_variant(base_margins, base_targets, base_ss, base_mw, 0.025, 0.975, True,  "(B) Spline + 10% Boost")
delta_boost = var_A["brier_s1"] - var_B["brier_s1"]

ablation_lines = [
    "ABLATION STUDY: Underdog Boost",
    "=" * 50,
    f"Model: Innovation LOSO ({len(FEATURES_INNOV)} features), Clip=[0.025, 0.975]",
    "",
    "(A) Raw Spline Calibration:",
    f"    Brier Stage1 (2022-25): {var_A['brier_s1']:.5f}",
    f"    Brier Men:              {var_A['brier_men']}",
    f"    Brier Women:            {var_A['brier_women']}",
    f"    Brier All seasons:      {var_A['brier_all']:.5f}",
    "",
    "(B) Spline + 10% Underdog Boost (pred < 0.85 -> pred * 1.10):",
    f"    Brier Stage1 (2022-25): {var_B['brier_s1']:.5f}",
    f"    Brier Men:              {var_B['brier_men']}",
    f"    Brier Women:            {var_B['brier_women']}",
    f"    Brier All seasons:      {var_B['brier_all']:.5f}",
    "",
    f"Delta (A - B): {delta_boost:+.5f}  -> {'BOOST HURTS (A is better, skip boost)' if delta_boost < 0 else 'BOOST HELPS (B is better, use boost)'}",
    "",
    "RECOMMENDATION:",
    "  Use boost: YES" if delta_boost > 0 else "  Use boost: NO (raw spline is better-calibrated)",
]

with open("Optimization/ablation_study.txt", "w") as f:
    f.write("\n".join(ablation_lines))
for line in ablation_lines:
    print("  " + line)
print("  Saved Optimization/ablation_study.txt")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: CALIBRATION CLIP OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[EXP 4] Calibration Clip Grid Search")
print("=" * 60)

clip_configs = [
    (0.01, 0.99),
    (0.02, 0.98),
    (0.025, 0.975),   # current default
    (0.03, 0.97),
    (0.05, 0.95),
    (0.01, 0.99, True),   # with boost
    (0.025, 0.975, True), # with boost
]

cal_rows = []
for cfg in clip_configs:
    lo, hi = cfg[0], cfg[1]
    boost  = cfg[2] if len(cfg) > 2 else False
    r = score_variant(base_margins, base_targets, base_ss, base_mw, lo, hi, boost,
                      f"clip=[{lo},{hi}]" + (" +boost" if boost else ""))
    r["clip_lo"] = lo
    r["clip_hi"] = hi
    r["delta_vs_default"] = round(r["brier_s1"] - 0.025, 5)  # vs [0.025, 0.975]
    cal_rows.append(r)
    marker = " <-- DEFAULT" if (lo == 0.025 and hi == 0.975 and not boost) else ""
    print(f"  [{lo:.3f}, {hi:.3f}] boost={boost}: Stage1={r['brier_s1']:.5f}"
          f"  men={r['brier_men']}  women={r['brier_women']}{marker}")

# Recalculate delta vs actual default
default_brier = next(r["brier_s1"] for r in cal_rows if r["clip_lo"]==0.025 and r["clip_hi"]==0.975 and not r["boost"])
for r in cal_rows:
    r["delta_vs_default"] = round(r["brier_s1"] - default_brier, 5)

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv("Optimization/calibration_tuning.csv", index=False)
best_clip = cal_df.loc[cal_df["brier_s1"].idxmin()]
print(f"\n  Best clip config: [{best_clip['clip_lo']}, {best_clip['clip_hi']}] "
      f"boost={best_clip['boost']}  Brier={best_clip['brier_s1']:.5f}")
print("  Saved Optimization/calibration_tuning.csv")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 5: HIERARCHICAL FALLBACK VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[EXP 5] Hierarchical Fallback Audit")
print("=" * 60)

sub1 = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage1.csv")
sub1["Season"]    = sub1["ID"].str.split("_").str[0].astype(int)
sub1["T1_TeamID"] = sub1["ID"].str.split("_").str[1].astype(int)
sub1["T2_TeamID"] = sub1["ID"].str.split("_").str[2].astype(int)

# Merge predictions from saved Stage 1 submission
preds_saved = pd.read_csv("submission_stage1.csv").rename(columns={"Pred":"Pred_saved"})
sub1 = sub1.merge(preds_saved, on="ID", how="left")

# Merge Elo and seed features
sub1 = sub1.merge(elos_T1[["Season","T1_TeamID","T1_elo"]], on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(elos_T2[["Season","T2_TeamID","T2_elo"]], on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(seeds_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(seeds_T2, on=["Season","T2_TeamID"], how="left")
sub1["elo_diff"]  = sub1["T1_elo"] - sub1["T2_elo"]
sub1["Seed_diff"] = sub1["T2_seed"] - sub1["T1_seed"]

# Mark tournament teams per season
tourney_ids = (
    seeds_df.groupby("Season")["TeamID"].apply(set).to_dict()
)
sub1["T1_in_tourney"] = sub1.apply(
    lambda r: r["T1_TeamID"] in tourney_ids.get(r["Season"], set()), axis=1
)
sub1["T2_in_tourney"] = sub1.apply(
    lambda r: r["T2_TeamID"] in tourney_ids.get(r["Season"], set()), axis=1
)
sub1["neither_in_tourney"] = ~sub1["T1_in_tourney"] & ~sub1["T2_in_tourney"]

# Simulate fallback level detection
# Level 1 = at least one tournament team (LOSO model used)
# Level 2+ = neither is a tournament team (fallback used)
# Actually: LOSO uses ALL features; fallback kicks in if prediction was NaN.
# Since all Stage1 years 2022-25 are in use_models, all get Level 1.
# "Non-tournament" here means BOTH teams are non-seeds — predicted via full model but
# using NaN seed features. The stress test verifies these are NOT 0.5 (not flat).
non_tourney = sub1[sub1["neither_in_tourney"]].copy()
print(f"  Non-tournament predictions (neither team is seeded): {len(non_tourney):,}")
print(f"  Of these, Pred == 0.5 exactly: {(non_tourney['Pred_saved'] == 0.5).sum():,}")
print(f"  Non-tournament Pred distribution:")
print(f"    Mean:  {non_tourney['Pred_saved'].mean():.4f}")
print(f"    Std:   {non_tourney['Pred_saved'].std():.4f}")
print(f"    Min:   {non_tourney['Pred_saved'].min():.4f}")
print(f"    Max:   {non_tourney['Pred_saved'].max():.4f}")

# Sample 10: pick rows with largest |elo_diff| to show they're Elo-grounded
audit_sample = (
    non_tourney.assign(_abs_elo=non_tourney["elo_diff"].abs())
    .nlargest(10, "_abs_elo")
    [["ID","Season","T1_TeamID","T2_TeamID","T1_in_tourney","T2_in_tourney",
      "elo_diff","Seed_diff","Pred_saved"]]
    .rename(columns={"Pred_saved":"Pred"})
    .reset_index(drop=True)
)
audit_sample["fallback_grounded"] = (audit_sample["Pred"] - 0.5).abs() > 0.02
audit_sample.to_csv("Optimization/fallback_audit.csv", index=False)

print("\n  Top-10 non-tournament predictions by |elo_diff|:")
for _, row in audit_sample.iterrows():
    print(f"    {row['ID']}  elo_diff={row['elo_diff']:+.0f}  "
          f"seed_diff={row['Seed_diff']}  Pred={row['Pred']:.4f}  "
          f"grounded={'YES' if row['fallback_grounded'] else 'NO'}")
print(f"\n  All grounded (|Pred-0.5|>0.02): {audit_sample['fallback_grounded'].all()}")
print("  Saved Optimization/fallback_audit.csv")


# ─────────────────────────────────────────────────────────────────────────────
# WINNING CONFIGURATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("WINNING CONFIGURATION SUMMARY")
print("=" * 60)

best_mom = best_win
best_clip_lo  = float(best_clip["clip_lo"])
best_clip_hi  = float(best_clip["clip_hi"])
best_boost    = bool(best_clip["boost"])
best_log      = delta_log > 0

summary = {
    "base_innovation_brier_s1":     round(base_brier_s1, 5),
    "exp1_log_transform": {
        "log_brier_s1": round(log_brier, 5),
        "delta":        round(delta_log, 5),
        "recommended":  bool(best_log),
    },
    "exp2_momentum_window": {
        "best_window":  best_mom,
        "best_brier_s1": mom_results[best_mom]["stage1_brier"],
        "all_windows":  {k: v for k, v in mom_results.items() if not k.startswith("_")},
    },
    "exp3_underdog_boost": {
        "raw_spline_brier_s1":   var_A["brier_s1"],
        "boosted_brier_s1":      var_B["brier_s1"],
        "delta":                 round(delta_boost, 5),
        "recommended_boost":     bool(delta_boost > 0),
    },
    "exp4_calibration_clip": {
        "best_clip_lo":  best_clip_lo,
        "best_clip_hi":  best_clip_hi,
        "best_boost":    best_boost,
        "best_brier_s1": float(best_clip["brier_s1"]),
    },
    "exp5_fallback_audit": {
        "non_tourney_predictions": int(len(non_tourney)),
        "flat_0.5_count":          int((non_tourney["Pred_saved"] == 0.5).sum()),
        "all_grounded":            bool(audit_sample["fallback_grounded"].all()),
    },
    "recommended_config": {
        "features":          "innovation_58" + ("_plus_log" if best_log else ""),
        "momentum_window":   best_mom,
        "clip_lo":           best_clip_lo,
        "clip_hi":           best_clip_hi,
        "underdog_boost":    best_boost or (delta_boost > 0),
    },
}

with open("Optimization/winning_config.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Recommended configuration:")
for k, v in summary["recommended_config"].items():
    print(f"    {k:<22}: {v}")
print(f"\n  Saved Optimization/winning_config.json")
print("\nAll experiments complete. Results in Optimization/")
