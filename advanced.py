"""
advanced.py — Phase 1-3 Advanced Architecture Experiments
==========================================================
Phase 1: Golden Config  (log-ordinals + clip[0.01,0.99] + no boost)
Phase 2A: Gender-Split Model  (separate men/women LOSO)
Phase 2B: Ridge Quality Scores  (sklearn Ridge vs GLM OLS)
Phase 3: submission_2026_advanced.csv  (Stage 1, 519,144 rows)

Run with: uv run python advanced.py
Prerequisites: run solution_2026.py AND optimize.py first.
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
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

os.makedirs("Optimization", exist_ok=True)

CACHE_DIR     = ".cache"
DATA_DIR      = "./Data"
SEASON_CUTOFF = 2003
WOMEN_MIN     = 2010
T_SPLINE      = 25
RIDGE_ALPHA   = 1.0
CLIP_LO, CLIP_HI = 0.01, 0.99

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
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def cache_save(name, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, name + ".pkl"), "wb") as f:
        pickle.dump(obj, f)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ADVANCED EXPERIMENTS — Golden Config + Option A/B")
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
    "LScore","WScore",
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

# GLM Quality (from main solution cache)
glm_quality = cache_load("glm_quality")
assert glm_quality is not None, "glm_quality cache missing. Run solution_2026.py first."
glm_T1 = glm_quality.rename(columns={"TeamID":"T1_TeamID","quality":"T1_quality"})
glm_T2 = glm_quality.rename(columns={"TeamID":"T2_TeamID","quality":"T2_quality"})
tourney_data = tourney_data.merge(glm_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(glm_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

# Momentum (last 20 — Golden Config winner)
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

# FEATURE SETS
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

# Phase 1: Add log-ordinal features (Golden Config innovation)
LOG_FEATS = []
for sys in ORDINAL_SYSTEMS:
    for prefix in ["T1_ord_", "T2_ord_"]:
        col     = f"{prefix}{sys}"
        log_col = f"log_{prefix}{sys}"
        if col in tourney_data.columns:
            tourney_data[log_col] = np.log1p(tourney_data[col].clip(lower=0))
            LOG_FEATS.append(log_col)

FEATURES_GOLDEN = FEATURES_INNOV + LOG_FEATS
print(f"\n  Golden Config features: {len(FEATURES_GOLDEN)}")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: LOSO runner (raw margins)
# ─────────────────────────────────────────────────────────────────────────────
def run_loso(td, features, cache_key):
    """LOSO CV. Returns (margins, targets, ss, mw, models)."""
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


def calibrate(margins, targets, clip_lo=CLIP_LO, clip_hi=CLIP_HI):
    """Fit spline on OOF data. Returns (spline, probs, labels)."""
    labels = (targets > 0).astype(int)
    dat    = sorted(zip(margins, labels), key=lambda x: x[0])
    p, l   = zip(*dat)
    spline = UnivariateSpline(np.clip(p, -T_SPLINE, T_SPLINE), l, k=5)
    probs  = np.clip(spline(np.clip(margins, -T_SPLINE, T_SPLINE)), clip_lo, clip_hi)
    return spline, probs, labels


def stage1_brier_report(probs, labels, ss, mw, title):
    s1  = ss >= 2022
    men = (ss >= 2022) & (mw == 1)
    wom = (ss >= 2022) & (mw == 0)
    b_all = brier_score_loss(labels[s1], probs[s1])
    b_men = brier_score_loss(labels[men], probs[men]) if men.sum() > 0 else float("nan")
    b_wom = brier_score_loss(labels[wom], probs[wom]) if wom.sum() > 0 else float("nan")
    print(f"  {title}")
    print(f"    Stage1 (2022-25): {b_all:.5f}  |  Men: {b_men:.5f}  |  Women: {b_wom:.5f}")
    return b_all, b_men, b_wom


def fi_summary(models_dict, features, label, top_n=12):
    """Return averaged feature importance (gain) across all season models."""
    gain_agg = {}
    n = len(models_dict)
    for m in models_dict.values():
        for k, v in m.get_score(importance_type="gain").items():
            gain_agg[k] = gain_agg.get(k, 0) + v / n
    feat_map = {f"f{i}": nm for i, nm in enumerate(features)}
    rows = [(feat_map.get(k, k), round(v, 2)) for k, v in gain_agg.items()]
    rows.sort(key=lambda x: -x[1])
    print(f"\n  --- Feature Importance ({label}) top {top_n} by avg gain ---")
    for name, gain in rows[:top_n]:
        tag = ""
        if "_mom_" in name: tag = "[MOM]"
        elif "ord_" in name: tag = "[ORD]"
        elif "quality" in name: tag = "[GLM]"
        elif "elo" in name: tag = "[ELO]"
        elif "seed" in name.lower(): tag = "[SEED]"
        print(f"    {name:<38} {gain:>10.1f}  {tag}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: GOLDEN CONFIG BASELINE
# Load opt_log_ordinals (from optimize.py) and recalibrate with [0.01, 0.99]
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 1: Golden Config Baseline")
print("=" * 60)

golden_cache = cache_load("opt_log_ordinals")
assert golden_cache is not None, "opt_log_ordinals cache missing. Run optimize.py first."
g_margins, g_targets, g_ss, g_mw, g_models = golden_cache
g_spline, g_probs, g_labels = calibrate(g_margins, g_targets, CLIP_LO, CLIP_HI)
golden_brier, g_brier_men, g_brier_women = stage1_brier_report(
    g_probs, g_labels, g_ss, g_mw, "Golden Config (68 feat, clip=[0.01,0.99], no boost)"
)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2A: GENDER-SPLIT MODEL
# Train separate XGBoost for men and women
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2A: Gender-Split Model")
print("=" * 60)

td_men   = tourney_data[tourney_data["men_women"] == 1].copy().reset_index(drop=True)
td_women = tourney_data[tourney_data["men_women"] == 0].copy().reset_index(drop=True)
print(f"  Men's tourney rows: {len(td_men):,}   Women's tourney rows: {len(td_women):,}")

men_margins, men_targets, men_ss, men_mw, men_models = \
    run_loso(td_men,   FEATURES_GOLDEN, "opt_men_golden")
women_margins, women_targets, women_ss, women_mw, women_models = \
    run_loso(td_women, FEATURES_GOLDEN, "opt_women_golden")

# Calibrate each gender separately
men_spline,   men_probs,   men_labels   = calibrate(men_margins,   men_targets,   CLIP_LO, CLIP_HI)
women_spline, women_probs, women_labels = calibrate(women_margins, women_targets, CLIP_LO, CLIP_HI)

# Combined Stage1 Brier (men + women predictions merged)
men_s1   = men_ss   >= 2022
women_s1 = women_ss >= 2022
combined_probs  = np.concatenate([men_probs[men_s1],   women_probs[women_s1]])
combined_labels = np.concatenate([men_labels[men_s1],  women_labels[women_s1]])
gender_split_brier = brier_score_loss(combined_labels, combined_probs)
men_s1_brier   = brier_score_loss(men_labels[men_s1],   men_probs[men_s1])
women_s1_brier = brier_score_loss(women_labels[women_s1], women_probs[women_s1])

print(f"  Gender-Split combined Stage1: {gender_split_brier:.5f}")
print(f"    Men:   {men_s1_brier:.5f}  |  Women: {women_s1_brier:.5f}")
print(f"  vs Golden Config:             {golden_brier:.5f}")
option_a_improves = gender_split_brier < golden_brier
print(f"  Option A {'IMPROVES' if option_a_improves else 'does NOT improve'} over Golden Config "
      f"(delta={golden_brier - gender_split_brier:+.5f})")

# Save Option A metrics
gender_metrics = {
    "golden_brier_s1":       round(golden_brier, 5),
    "gender_split_brier_s1": round(gender_split_brier, 5),
    "men_s1_brier":          round(men_s1_brier, 5),
    "women_s1_brier":        round(women_s1_brier, 5),
    "delta_vs_golden":       round(golden_brier - gender_split_brier, 5),
    "option_a_improves":     option_a_improves,
    "per_year": {},
}
for yr in [2022, 2023, 2024, 2025]:
    m_mask = (men_ss == yr)
    w_mask = (women_ss == yr)
    if m_mask.sum() > 0 or w_mask.sum() > 0:
        yr_probs  = np.concatenate([men_probs[m_mask],  women_probs[w_mask]])
        yr_labels = np.concatenate([men_labels[m_mask], women_labels[w_mask]])
        gender_metrics["per_year"][str(yr)] = round(brier_score_loss(yr_labels, yr_probs), 5)

with open("Optimization/gender_split_metrics.json", "w") as f:
    json.dump(gender_metrics, f, indent=2)
print("  Saved Optimization/gender_split_metrics.json")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2B: RIDGE QUALITY SCORES
# Replace GLM (OLS) quality with Ridge(alpha=1.0) quality
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 2B: Ridge Quality Scores")
print("=" * 60)

def _ridge_quality(season, men_women_flag, data_df, alpha=RIDGE_ALPHA):
    """Compute team quality via Ridge regression instead of OLS GLM."""
    sub = data_df[(data_df["Season"] == season) & (data_df["men_women"] == men_women_flag)].copy()
    if len(sub) < 20:
        return pd.DataFrame(columns=["TeamID","quality","Season"])
    t1_ids    = sub["T1_TeamID"].astype(str).values
    t2_ids    = sub["T2_TeamID"].astype(str).values
    all_teams = np.unique(np.concatenate([t1_ids, t2_ids]))
    t_to_idx  = {t: i for i, t in enumerate(all_teams)}
    n_teams   = len(all_teams)
    n_games   = len(sub)
    # Design: T1 block (cols 0..n-1) + T2 block (cols n..2n-1)
    t1_cols = np.array([t_to_idx[t] for t in t1_ids])
    t2_cols = np.array([t_to_idx[t] for t in t2_ids]) + n_teams
    rows    = np.concatenate([np.arange(n_games), np.arange(n_games)])
    cols    = np.concatenate([t1_cols, t2_cols])
    X       = csr_matrix((np.ones(2 * n_games), (rows, cols)),
                         shape=(n_games, 2 * n_teams))
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, sub["PointDiff"].values)
    q = pd.DataFrame({
        "TeamID":  [int(t) for t in all_teams],
        "quality": ridge.coef_[:n_teams],
        "Season":  season,
    })
    return q


# Compute Ridge quality (same tournament-connected filter as GLM)
ridge_quality_cache = cache_load("ridge_quality")
if ridge_quality_cache is not None:
    print("  [CACHE HIT] ridge_quality")
    ridge_quality = ridge_quality_cache
else:
    # Tournament-connected set (from solution_2026.py logic)
    regular_data["ST1"] = regular_data["Season"].astype(str) + "/" + regular_data["T1_TeamID"].astype(str)
    regular_data["ST2"] = regular_data["Season"].astype(str) + "/" + regular_data["T2_TeamID"].astype(str)
    seeds_T1["ST1"]     = seeds_T1["Season"].astype(str) + "/" + seeds_T1["T1_TeamID"].astype(str)
    seeds_T2["ST2"]     = seeds_T2["Season"].astype(str) + "/" + seeds_T2["T2_TeamID"].astype(str)
    st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
    st = st | set(regular_data.loc[
        (regular_data["T1_Score"] > regular_data["T2_Score"]) & regular_data["ST2"].isin(st), "ST1"
    ])

    dt_hist = regular_data[
        (regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)) &
        (regular_data["Season"] <= 2025)
    ].copy()
    dt_hist["T1_TeamID"] = dt_hist["T1_TeamID"].astype(str)
    dt_hist["T2_TeamID"] = dt_hist["T2_TeamID"].astype(str)
    dt_hist.loc[~dt_hist["ST1"].isin(st), "T1_TeamID"] = "0000"
    dt_hist.loc[~dt_hist["ST2"].isin(st), "T2_TeamID"] = "0000"

    ridge_parts = []
    hist_seasons = sorted(seeds_df["Season"].unique())
    for s in tqdm.tqdm(hist_seasons, desc="  Ridge quality hist"):
        if s >= WOMEN_MIN:
            ridge_parts.append(_ridge_quality(s, 0, dt_hist))
        if s >= SEASON_CUTOFF:
            ridge_parts.append(_ridge_quality(s, 1, dt_hist))

    # 2026: all active teams (no tourney filter yet)
    dt_2026 = regular_data[regular_data["Season"] == 2026].copy()
    dt_2026["T1_TeamID"] = dt_2026["T1_TeamID"].astype(str)
    dt_2026["T2_TeamID"] = dt_2026["T2_TeamID"].astype(str)
    game_counts  = dt_2026.groupby("T1_TeamID").size()
    active_teams = set(game_counts[game_counts >= 10].index)
    dt_2026 = dt_2026[dt_2026["T1_TeamID"].isin(active_teams) & dt_2026["T2_TeamID"].isin(active_teams)]
    ridge_parts.append(_ridge_quality(2026, 0, dt_2026))
    ridge_parts.append(_ridge_quality(2026, 1, dt_2026))

    ridge_quality = pd.concat(ridge_parts, ignore_index=True)
    cache_save("ridge_quality", ridge_quality)
    print("  [CACHE SAVED] ridge_quality")

# Compare GLM vs Ridge quality distributions
print(f"  GLM  quality — mean={glm_quality['quality'].mean():.2f}  std={glm_quality['quality'].std():.2f}  "
      f"range=[{glm_quality['quality'].min():.1f}, {glm_quality['quality'].max():.1f}]")
print(f"  Ridge quality — mean={ridge_quality['quality'].mean():.2f}  std={ridge_quality['quality'].std():.2f}  "
      f"range=[{ridge_quality['quality'].min():.1f}, {ridge_quality['quality'].max():.1f}]")

# Build tourney_data variant with Ridge quality
print("  Building tourney_data with Ridge quality...")
td_ridge = tourney_data.drop(columns=["T1_quality","T2_quality","diff_quality"]).copy()
rq_T1 = ridge_quality.rename(columns={"TeamID":"T1_TeamID","quality":"T1_quality"})
rq_T2 = ridge_quality.rename(columns={"TeamID":"T2_TeamID","quality":"T2_quality"})
td_ridge = td_ridge.merge(rq_T1, on=["Season","T1_TeamID"], how="left")
td_ridge = td_ridge.merge(rq_T2, on=["Season","T2_TeamID"], how="left")
td_ridge["diff_quality"] = td_ridge["T1_quality"] - td_ridge["T2_quality"]

r_margins, r_targets, r_ss, r_mw, r_models = \
    run_loso(td_ridge, FEATURES_GOLDEN, "opt_ridge_quality")
r_spline, r_probs, r_labels = calibrate(r_margins, r_targets, CLIP_LO, CLIP_HI)
ridge_brier, r_brier_men, r_brier_women = stage1_brier_report(
    r_probs, r_labels, r_ss, r_mw, "Ridge Quality"
)
print(f"  vs Golden Config: {golden_brier:.5f}")
option_b_improves = ridge_brier < golden_brier
print(f"  Option B {'IMPROVES' if option_b_improves else 'does NOT improve'} over Golden Config "
      f"(delta={golden_brier - ridge_brier:+.5f})")

# Save Option B metrics
ridge_rows = []
for s in sorted(set(r_ss)):
    mask = r_ss == s
    if mask.sum() > 0:
        g_mask = g_ss == s
        ridge_rows.append({
            "season":      int(s),
            "ridge_brier": round(brier_score_loss(r_labels[mask], r_probs[mask]), 5),
            "glm_brier":   round(brier_score_loss(g_labels[g_mask], g_probs[g_mask]), 5)
                           if g_mask.sum() > 0 else None,
        })
ridge_df = pd.DataFrame(ridge_rows)
ridge_df["delta"] = (ridge_df["glm_brier"] - ridge_df["ridge_brier"]).round(5)
# Summary row
summary_row = pd.DataFrame([{
    "season": "STAGE1_2022-25",
    "ridge_brier": round(ridge_brier, 5),
    "glm_brier":   round(golden_brier, 5),
    "delta":        round(golden_brier - ridge_brier, 5),
}])
ridge_df = pd.concat([summary_row, ridge_df], ignore_index=True)
ridge_df.to_csv("Optimization/ridge_quality_vs_glm.csv", index=False)
print("  Saved Optimization/ridge_quality_vs_glm.csv")


# ─────────────────────────────────────────────────────────────────────────────
# DECISION: Which config wins?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DECISION: Best Configuration")
print("=" * 60)
print(f"  Golden Config:  {golden_brier:.5f}")
print(f"  Option A (Split): {gender_split_brier:.5f}  {'BETTER' if option_a_improves else 'worse'}")
print(f"  Option B (Ridge): {ridge_brier:.5f}  {'BETTER' if option_b_improves else 'worse'}")

if option_a_improves and option_b_improves:
    print("\n  Both improve -> running Combined (Gender-Split + Ridge Quality)...")
    # Split ridge tourney_data by gender
    tdr_men   = td_ridge[td_ridge["men_women"] == 1].copy().reset_index(drop=True)
    tdr_women = td_ridge[td_ridge["men_women"] == 0].copy().reset_index(drop=True)
    cmb_men_m, cmb_men_t, cmb_men_ss, cmb_men_mw, cmb_men_models = \
        run_loso(tdr_men,   FEATURES_GOLDEN, "opt_combined_men")
    cmb_wom_m, cmb_wom_t, cmb_wom_ss, cmb_wom_mw, cmb_wom_models = \
        run_loso(tdr_women, FEATURES_GOLDEN, "opt_combined_women")
    _, cmb_men_probs,   cmb_men_labels   = calibrate(cmb_men_m, cmb_men_t)
    _, cmb_women_probs, cmb_women_labels = calibrate(cmb_wom_m, cmb_wom_t)
    cmb_s1_probs  = np.concatenate([cmb_men_probs[cmb_men_ss >= 2022],
                                     cmb_women_probs[cmb_wom_ss >= 2022]])
    cmb_s1_labels = np.concatenate([cmb_men_labels[cmb_men_ss >= 2022],
                                     cmb_women_labels[cmb_wom_ss >= 2022]])
    combined_brier = brier_score_loss(cmb_s1_labels, cmb_s1_probs)
    print(f"  Combined (A+B): {combined_brier:.5f}")
    best_config  = "combined"
    final_brier  = combined_brier
    final_men_models   = cmb_men_models
    final_women_models = cmb_wom_models
    final_men_spline   = calibrate(cmb_men_m, cmb_men_t)[0]
    final_women_spline = calibrate(cmb_wom_m, cmb_wom_t)[0]
    use_ridge    = True
    use_split    = True
elif option_a_improves:
    best_config  = "gender_split"
    final_brier  = gender_split_brier
    final_men_models   = men_models
    final_women_models = women_models
    final_men_spline   = men_spline
    final_women_spline = women_spline
    use_ridge    = False
    use_split    = True
    print(f"  -> Using Option A (Gender-Split), Brier={final_brier:.5f}")
elif option_b_improves:
    best_config  = "ridge_quality"
    final_brier  = ridge_brier
    use_ridge    = True
    use_split    = False
    final_spline = r_spline
    final_models = r_models
    print(f"  -> Using Option B (Ridge Quality), Brier={final_brier:.5f}")
else:
    best_config  = "golden"
    final_brier  = golden_brier
    use_ridge    = False
    use_split    = False
    final_spline = g_spline
    final_models = g_models
    print(f"  -> Staying with Golden Config, Brier={final_brier:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: GENERATE submission_2026_advanced.csv  (Stage 1, 519,144 rows)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PHASE 3: Generating submission_2026_advanced.csv (Stage 1)")
print("=" * 60)

# Choose quality table and tourney_data for feature merging
_quality_T1 = rq_T1 if use_ridge else glm_T1
_quality_T2 = rq_T2 if use_ridge else glm_T2

sub1 = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage1.csv")
sub1["Season"]    = sub1["ID"].str.split("_").str[0].astype(int)
sub1["T1_TeamID"] = sub1["ID"].str.split("_").str[1].astype(int)
sub1["T2_TeamID"] = sub1["ID"].str.split("_").str[2].astype(int)
sub1["men_women"] = sub1["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)

# Merge all features
sub1 = sub1.merge(ss_T1,      on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(ss_T2,      on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(seeds_T1,   on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(seeds_T2,   on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(mom_T1,     on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(mom_T2,     on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(ord_T1,     on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(ord_T2,     on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(_quality_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(_quality_T2, on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(elos_T1,    on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(elos_T2,    on=["Season","T2_TeamID"], how="left")

sub1["Seed_diff"]    = sub1["T2_seed"]    - sub1["T1_seed"]
sub1["elo_diff"]     = sub1["T1_elo"]     - sub1["T2_elo"]
sub1["diff_quality"] = sub1["T1_quality"] - sub1["T2_quality"]
for sys in ORDINAL_SYSTEMS:
    sub1[f"ord_diff_{sys}"] = sub1[f"T2_ord_{sys}"] - sub1[f"T1_ord_{sys}"]
for log_col in LOG_FEATS:
    base_col = log_col.replace("log_", "")
    if base_col in sub1.columns:
        sub1[log_col] = np.log1p(sub1[base_col].clip(lower=0))
    else:
        sub1[log_col] = np.nan

sub1["Pred"] = np.nan

# ── LEVEL 1: Season-specific LOSO model ──────────────────────────────────────
for yr in [2022, 2023, 2024, 2025]:
    mask = sub1["Season"] == yr
    if mask.sum() == 0:
        continue
    if use_split:
        # Men
        m_mask = mask & (sub1["men_women"] == 1)
        if m_mask.sum() > 0 and yr in final_men_models:
            mg = final_men_models[yr].predict(xgb.DMatrix(sub1.loc[m_mask, FEATURES_GOLDEN].values))
            sub1.loc[m_mask, "Pred"] = np.clip(
                final_men_spline(np.clip(mg, -T_SPLINE, T_SPLINE)), CLIP_LO, CLIP_HI
            )
        # Women
        w_mask = mask & (sub1["men_women"] == 0)
        if w_mask.sum() > 0 and yr in final_women_models:
            wg = final_women_models[yr].predict(xgb.DMatrix(sub1.loc[w_mask, FEATURES_GOLDEN].values))
            sub1.loc[w_mask, "Pred"] = np.clip(
                final_women_spline(np.clip(wg, -T_SPLINE, T_SPLINE)), CLIP_LO, CLIP_HI
            )
    else:
        if yr in final_models:
            mg = final_models[yr].predict(xgb.DMatrix(sub1.loc[mask, FEATURES_GOLDEN].values))
            sub1.loc[mask, "Pred"] = np.clip(
                final_spline(np.clip(mg, -T_SPLINE, T_SPLINE)), CLIP_LO, CLIP_HI
            )

# ── LEVEL 2: Elo-diff sigmoid fallback ───────────────────────────────────────
lvl2 = sub1["Pred"].isna()
if lvl2.sum() > 0:
    ed = sub1.loc[lvl2, "elo_diff"].fillna(0)
    sub1.loc[lvl2, "Pred"] = (1 / (1 + np.exp(-ed / 150))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 2 fallback (Elo sigmoid): {lvl2.sum():,} rows")

# ── LEVEL 3: NET-rank difference sigmoid ─────────────────────────────────────
lvl3 = sub1["Pred"].isna()
if lvl3.sum() > 0:
    rd = sub1.loc[lvl3, "ord_diff_NET"].fillna(0)
    sub1.loc[lvl3, "Pred"] = (1 / (1 + np.exp(-rd / 50))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 3 fallback (NET rank):    {lvl3.sum():,} rows")

# ── LEVEL 4: Seed-difference heuristic ────────────────────────────────────────
lvl4 = sub1["Pred"].isna()
if lvl4.sum() > 0:
    sd = sub1.loc[lvl4, "Seed_diff"].fillna(0)
    sub1.loc[lvl4, "Pred"] = (1 / (1 + np.exp(-sd * 0.4))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 4 fallback (Seed diff):   {lvl4.sum():,} rows")

# ── LEVEL 5: Global mean ──────────────────────────────────────────────────────
remaining = sub1["Pred"].isna().sum()
if remaining > 0:
    sub1["Pred"] = sub1["Pred"].fillna(0.5)
    print(f"  Level 5 fallback (0.5):         {remaining:,} rows")

sub1["Pred"] = sub1["Pred"].clip(CLIP_LO, CLIP_HI).round(6)

# Distribution check
print(f"\n  Rows: {len(sub1):,}  (target: 519,144)")
print(f"  Mean: {sub1.Pred.mean():.4f}  Std: {sub1.Pred.std():.4f}  "
      f"Min: {sub1.Pred.min():.4f}  Max: {sub1.Pred.max():.4f}")
print(f"  NaN: {sub1.Pred.isna().sum()}")

sub1[["ID","Pred"]].to_csv("submission_2026_advanced.csv", index=False)
with open("submission_2026_advanced.csv") as f:
    line_count = sum(1 for _ in f)
print(f"  Saved submission_2026_advanced.csv  ({line_count:,} lines incl. header)")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE: Men vs Women (if gender split is used)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE: Men vs. Women Models")
print("=" * 60)

m_fi = fi_summary(
    final_men_models if use_split else g_models,
    FEATURES_GOLDEN,
    "MEN'S model"
)
w_fi = fi_summary(
    final_women_models if use_split else g_models,
    FEATURES_GOLDEN,
    "WOMEN'S model"
)

# Save gender FI comparison to Optimization/
feat_map = {f"f{i}": nm for i, nm in enumerate(FEATURES_GOLDEN)}
men_fi_dict   = {feat_map.get(k, k): v for k, v in m_fi}
women_fi_dict = {feat_map.get(k, k): v for k, v in w_fi}
all_feats = sorted(set(list(men_fi_dict.keys()) + list(women_fi_dict.keys())))
fi_compare = pd.DataFrame([{
    "feature":       f,
    "men_gain":      men_fi_dict.get(f, 0.0),
    "women_gain":    women_fi_dict.get(f, 0.0),
    "is_momentum":   "_mom_" in f,
    "is_ordinal":    "ord_" in f,
    "is_quality":    "quality" in f,
    "is_elo":        "elo" in f,
    "is_seed":       "seed" in f.lower(),
} for f in all_feats])
fi_compare["men_rank"]   = fi_compare["men_gain"].rank(ascending=False).astype(int)
fi_compare["women_rank"] = fi_compare["women_gain"].rank(ascending=False).astype(int)
fi_compare = fi_compare.sort_values("men_gain", ascending=False).reset_index(drop=True)
fi_compare.to_csv("Optimization/gender_feature_importance.csv", index=False)
print("\n  Saved Optimization/gender_feature_importance.csv")

# Momentum-specific comparison
mom_fi = fi_compare[fi_compare["is_momentum"]]
if len(mom_fi) > 0:
    print("\n  Momentum features — men vs women gain:")
    for _, row in mom_fi.iterrows():
        ratio = row["men_gain"] / row["women_gain"] if row["women_gain"] > 0 else float("inf")
        print(f"    {row['feature']:<35} men={row['men_gain']:>8.1f}  "
              f"women={row['women_gain']:>8.1f}  ratio={ratio:.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Best configuration: {best_config.upper()}")
print(f"  Overall Brier Score (Stage1 2022-25): {final_brier:.5f}")
print(f"  vs original innovation baseline:       0.16238  (clip [0.01,0.99])")
print(f"  vs previous submission (0.025 clip):   0.16244")
print(f"  Use Ridge Quality: {use_ridge}")
print(f"  Use Gender Split:  {use_split}")
print(f"\n  Output files:")
print(f"    submission_2026_advanced.csv          (519,144 rows, Stage 1)")
print(f"    Optimization/gender_split_metrics.json")
print(f"    Optimization/ridge_quality_vs_glm.csv")
print(f"    Optimization/gender_feature_importance.csv")
print("\nDone.")
