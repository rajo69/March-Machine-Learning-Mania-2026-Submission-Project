"""
feature_engineering.py -- 5 new feature experiments on top of best Ridge Quality config
========================================================================================
Baseline: Ridge Quality + 68 feat + clip=[0.01,0.99], Stage1 Brier=0.16143

Experiments:
  FE1: Strength of Schedule (SOS)  -- avg opponent Ridge quality
  FE2: Late-season form delta      -- last-10 stats minus full-season stats
  FE3: Win% vs top-25 opponents    -- quality wins
  FE4: Neutral-site performance    -- win rate / scoring on neutral court
  FE5: Conference strength         -- avg Ridge quality of conference members

For each experiment: run LOSO, compare Brier.  If improvement -> keep.
Combine all winners -> regenerate submission_stage1.csv.

Run with: uv run python feature_engineering.py
Prerequisites: run advanced.py first (needs ridge_quality.pkl + opt_ridge_quality.pkl caches)
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
os.makedirs(".cache", exist_ok=True)

CACHE_DIR      = ".cache"
DATA_DIR       = "./Data"
SEASON_CUTOFF  = 2003
WOMEN_MIN      = 2010
T_SPLINE       = 25
RIDGE_ALPHA    = 1.0
CLIP_LO, CLIP_HI = 0.01, 0.99

BASELINE_BRIER = 0.16143   # opt_ridge_quality result

XGB_PARAM = {
    "objective":         "reg:squarederror",
    "booster":           "gbtree",
    "eta":               0.0093,
    "subsample":         0.6,
    "colsample_bynode":  0.8,
    "num_parallel_tree": 2,
    "min_child_weight":  4,
    "max_depth":         4,
    "tree_method":       "hist",
    "grow_policy":       "lossguide",
    "max_bin":           38,
}
NUM_ROUNDS = 704


# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def cache_load(name):
    p = os.path.join(CACHE_DIR, name + ".pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def cache_save(name, obj):
    with open(os.path.join(CACHE_DIR, name + ".pkl"), "wb") as f:
        pickle.dump(obj, f)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (mirrors advanced.py)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("FEATURE ENGINEERING EXPERIMENTS")
print("Baseline: Ridge Quality, Stage1 Brier=0.16143")
print("=" * 60)
print("\nLoading data...")

M_regular   = pd.read_csv(f"{DATA_DIR}/MRegularSeasonDetailedResults.csv")
M_tourney   = pd.read_csv(f"{DATA_DIR}/MNCAATourneyDetailedResults.csv")
M_seeds     = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
M_ordinals  = pd.read_csv(f"{DATA_DIR}/MMasseyOrdinals.csv")
M_compact   = pd.read_csv(f"{DATA_DIR}/MRegularSeasonCompactResults.csv")   # has WLoc
W_regular   = pd.read_csv(f"{DATA_DIR}/WRegularSeasonDetailedResults.csv")
W_tourney   = pd.read_csv(f"{DATA_DIR}/WNCAATourneyDetailedResults.csv")
W_seeds     = pd.read_csv(f"{DATA_DIR}/WNCAATourneySeeds.csv")
W_compact   = pd.read_csv(f"{DATA_DIR}/WRegularSeasonCompactResults.csv")   # has WLoc
M_conf      = pd.read_csv(f"{DATA_DIR}/MTeamConferences.csv")
W_conf      = pd.read_csv(f"{DATA_DIR}/WTeamConferences.csv")

regular_results = pd.concat([M_regular, W_regular], ignore_index=True)
tourney_results = pd.concat([M_tourney, W_tourney], ignore_index=True)
seeds_df        = pd.concat([M_seeds,   W_seeds],   ignore_index=True)
compact_all     = pd.concat([M_compact, W_compact],  ignore_index=True)
conf_all        = pd.concat([M_conf,    W_conf],     ignore_index=True)

regular_results = regular_results[regular_results["Season"] >= SEASON_CUTOFF]
tourney_results = tourney_results[tourney_results["Season"] >= SEASON_CUTOFF]
seeds_df        = seeds_df[seeds_df["Season"] >= SEASON_CUTOFF]
compact_all     = compact_all[compact_all["Season"] >= SEASON_CUTOFF]

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

# Ridge Quality (load from cache — computed by advanced.py)
ridge_quality = cache_load("ridge_quality")
assert ridge_quality is not None, "ridge_quality cache missing. Run advanced.py first."
rq_T1 = ridge_quality.rename(columns={"TeamID":"T1_TeamID","quality":"T1_quality"})
rq_T2 = ridge_quality.rename(columns={"TeamID":"T2_TeamID","quality":"T2_quality"})
tourney_data = tourney_data.merge(rq_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(rq_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

# Momentum (last 20)
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

# Log-ordinal features
LOG_FEATS = []
for sys in ORDINAL_SYSTEMS:
    for prefix in ["T1_ord_", "T2_ord_"]:
        col     = f"{prefix}{sys}"
        log_col = f"log_{prefix}{sys}"
        if col in tourney_data.columns:
            tourney_data[log_col] = np.log1p(tourney_data[col].clip(lower=0))
            LOG_FEATS.append(log_col)

# FEATURE SETS (Golden Config = current best)
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
FEATURES_GOLDEN = FEATURES_BASE + FEATURES_MOM + FEATURES_ORD + LOG_FEATS
print(f"\n  Golden Config feature count: {len(FEATURES_GOLDEN)}")


# ─────────────────────────────────────────────────────────────────────────────
# LOSO + CALIBRATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def run_loso(td, features, cache_key):
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


def calibrate(margins, targets):
    labels = (targets > 0).astype(int)
    dat    = sorted(zip(margins, labels), key=lambda x: x[0])
    p, l   = zip(*dat)
    spline = UnivariateSpline(np.clip(p, -T_SPLINE, T_SPLINE), l, k=5)
    probs  = np.clip(spline(np.clip(margins, -T_SPLINE, T_SPLINE)), CLIP_LO, CLIP_HI)
    return spline, probs, (np.array(targets) > 0).astype(int)


def stage1_brier(probs, labels, ss):
    mask = ss >= 2022
    return brier_score_loss(labels[mask], probs[mask])


# ─────────────────────────────────────────────────────────────────────────────
# FE1: STRENGTH OF SCHEDULE
# avg Ridge quality of all opponents faced during the regular season
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FE1: Strength of Schedule (avg opponent Ridge quality)")
print("=" * 60)

# Merge Ridge quality as opponent quality in regular_data
rd_with_opp = regular_data[["Season","T1_TeamID","T2_TeamID"]].copy()
rd_with_opp = rd_with_opp.merge(
    ridge_quality[["Season","TeamID","quality"]].rename(
        columns={"TeamID":"T2_TeamID","quality":"opp_quality"}
    ),
    on=["Season","T2_TeamID"], how="left"
)
sos = rd_with_opp.groupby(["Season","T1_TeamID"])["opp_quality"].mean().reset_index()
sos = sos.rename(columns={"opp_quality":"sos"})
sos_T1 = sos.rename(columns={"T1_TeamID":"T1_TeamID","sos":"T1_sos"})
sos_T2 = sos.rename(columns={"T1_TeamID":"T2_TeamID","sos":"T2_sos"})

td_fe1 = tourney_data.copy()
td_fe1 = td_fe1.merge(sos_T1, on=["Season","T1_TeamID"], how="left")
td_fe1 = td_fe1.merge(sos_T2, on=["Season","T2_TeamID"], how="left")
td_fe1["sos_diff"] = td_fe1["T1_sos"] - td_fe1["T2_sos"]
FE1_COLS = ["T1_sos","T2_sos","sos_diff"]
nan_pct = td_fe1[FE1_COLS].isna().mean().mean()
print(f"  FE1 NaN rate: {nan_pct:.3f}")

FEAT_FE1 = FEATURES_GOLDEN + FE1_COLS
margins_fe1, targets_fe1, ss_fe1, mw_fe1, models_fe1 = run_loso(td_fe1, FEAT_FE1, "opt_fe1_sos")
_, probs_fe1, labels_fe1 = calibrate(margins_fe1, targets_fe1)
brier_fe1 = stage1_brier(probs_fe1, labels_fe1, ss_fe1)
delta_fe1 = BASELINE_BRIER - brier_fe1
print(f"  FE1 Brier: {brier_fe1:.5f}  (delta={delta_fe1:+.5f})  "
      f"{'IMPROVEMENT' if delta_fe1 > 0 else 'no gain'}")


# ─────────────────────────────────────────────────────────────────────────────
# FE2: LATE-SEASON FORM DELTA
# last-10-game averages MINUS full-season averages (sign of peaking vs declining)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FE2: Late-Season Form Delta (last-10 minus full-season avg)")
print("=" * 60)

DELTA_STATS = ["T1_Score","T1_FGA","T1_OR","T1_DR","PointDiff"]
rd_s10 = rd_sorted[rd_sorted["_rank"] <= 10].groupby(["Season","T1_TeamID"])[DELTA_STATS].mean().reset_index()

# Season average for same stats
ss_base = regular_data.groupby(["Season","T1_TeamID"])[DELTA_STATS].mean().reset_index()

# Delta = last10 - season_avg
for col in DELTA_STATS:
    rd_s10[f"delta_{col}"] = rd_s10[col]   # will subtract after merge

delta_cols = [f"delta_{c}" for c in DELTA_STATS]
rd_s10_d = rd_s10[["Season","T1_TeamID"] + DELTA_STATS].merge(
    ss_base[["Season","T1_TeamID"] + DELTA_STATS],
    on=["Season","T1_TeamID"], suffixes=("_l10","_full")
)
for col in DELTA_STATS:
    rd_s10_d[f"delta_{col}"] = rd_s10_d[f"{col}_l10"] - rd_s10_d[f"{col}_full"]
rd_s10_d = rd_s10_d[["Season","T1_TeamID"] + delta_cols]

delta_T1 = rd_s10_d.rename(columns={
    "T1_TeamID":"T1_TeamID",
    **{f"delta_{c}": f"T1_delta_{c.replace('T1_','').replace('PointDiff','PD')}"
       for c in DELTA_STATS}
})
delta_T2 = rd_s10_d.rename(columns={
    "T1_TeamID":"T2_TeamID",
    **{f"delta_{c}": f"T2_delta_{c.replace('T1_','').replace('PointDiff','PD')}"
       for c in DELTA_STATS}
})
FE2_T1_COLS = [f"T1_delta_{c.replace('T1_','').replace('PointDiff','PD')}" for c in DELTA_STATS]
FE2_T2_COLS = [f"T2_delta_{c.replace('T1_','').replace('PointDiff','PD')}" for c in DELTA_STATS]
FE2_COLS    = FE2_T1_COLS + FE2_T2_COLS

td_fe2 = tourney_data.copy()
td_fe2 = td_fe2.merge(delta_T1, on=["Season","T1_TeamID"], how="left")
td_fe2 = td_fe2.merge(delta_T2, on=["Season","T2_TeamID"], how="left")
nan_pct2 = td_fe2[FE2_COLS].isna().mean().mean()
print(f"  FE2 NaN rate: {nan_pct2:.3f}")

FEAT_FE2 = FEATURES_GOLDEN + FE2_COLS
margins_fe2, targets_fe2, ss_fe2, mw_fe2, models_fe2 = run_loso(td_fe2, FEAT_FE2, "opt_fe2_delta")
_, probs_fe2, labels_fe2 = calibrate(margins_fe2, targets_fe2)
brier_fe2 = stage1_brier(probs_fe2, labels_fe2, ss_fe2)
delta_fe2 = BASELINE_BRIER - brier_fe2
print(f"  FE2 Brier: {brier_fe2:.5f}  (delta={delta_fe2:+.5f})  "
      f"{'IMPROVEMENT' if delta_fe2 > 0 else 'no gain'}")


# ─────────────────────────────────────────────────────────────────────────────
# FE3: WIN% vs TOP-25 OPPONENTS
# win rate in regular season games where opponent quality is in top 25
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FE3: Win% vs Top-25 Quality Opponents")
print("=" * 60)

# Top-25 teams per season by Ridge quality rank
rq_ranked = ridge_quality.copy()
rq_ranked["q_rank"] = rq_ranked.groupby("Season")["quality"].rank(ascending=False, method="first")
top25_set = set(
    (str(row.Season) + "/" + str(row.TeamID))
    for _, row in rq_ranked[rq_ranked["q_rank"] <= 25].iterrows()
)

rd_top25 = regular_data[["Season","T1_TeamID","T2_TeamID","win"]].copy()
rd_top25["key2"] = rd_top25["Season"].astype(str) + "/" + rd_top25["T2_TeamID"].astype(str)
rd_top25["vs_top25"] = rd_top25["key2"].isin(top25_set).astype(int)

# Win rate vs top-25 for each team
rd_vs25 = rd_top25[rd_top25["vs_top25"] == 1].groupby(["Season","T1_TeamID"])["win"].agg(
    ["mean","count"]
).reset_index()
rd_vs25.columns = ["Season","TeamID","win_pct_top25","n_top25_games"]

winpct_T1 = rd_vs25.rename(columns={"TeamID":"T1_TeamID",
                                     "win_pct_top25":"T1_win_pct_top25",
                                     "n_top25_games":"T1_n_top25"})
winpct_T2 = rd_vs25.rename(columns={"TeamID":"T2_TeamID",
                                     "win_pct_top25":"T2_win_pct_top25",
                                     "n_top25_games":"T2_n_top25"})

td_fe3 = tourney_data.copy()
td_fe3 = td_fe3.merge(winpct_T1[["Season","T1_TeamID","T1_win_pct_top25"]], on=["Season","T1_TeamID"], how="left")
td_fe3 = td_fe3.merge(winpct_T2[["Season","T2_TeamID","T2_win_pct_top25"]], on=["Season","T2_TeamID"], how="left")
td_fe3["win_pct_top25_diff"] = td_fe3["T1_win_pct_top25"] - td_fe3["T2_win_pct_top25"]
# Fill NaN (teams that never played a top-25 team) with 0
td_fe3[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]] = \
    td_fe3[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]].fillna(0)

FE3_COLS = ["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]
nan_pct3 = td_fe3[FE3_COLS].isna().mean().mean()
print(f"  FE3 NaN rate after fillna: {nan_pct3:.3f}")

FEAT_FE3 = FEATURES_GOLDEN + FE3_COLS
margins_fe3, targets_fe3, ss_fe3, mw_fe3, models_fe3 = run_loso(td_fe3, FEAT_FE3, "opt_fe3_top25")
_, probs_fe3, labels_fe3 = calibrate(margins_fe3, targets_fe3)
brier_fe3 = stage1_brier(probs_fe3, labels_fe3, ss_fe3)
delta_fe3 = BASELINE_BRIER - brier_fe3
print(f"  FE3 Brier: {brier_fe3:.5f}  (delta={delta_fe3:+.5f})  "
      f"{'IMPROVEMENT' if delta_fe3 > 0 else 'no gain'}")


# ─────────────────────────────────────────────────────────────────────────────
# FE4: NEUTRAL-SITE PERFORMANCE
# win rate + avg scoring margin on neutral-court games (WLoc='N')
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FE4: Neutral-Site Performance (win rate + margin)")
print("=" * 60)

# compact_all has WLoc: H=home, A=away, N=neutral
# Winner perspective: if WLoc=N, it's a neutral game
compact_neutral = compact_all[compact_all["WLoc"] == "N"].copy()
compact_neutral["WMargin"] = compact_neutral["WScore"] - compact_neutral["LScore"]

# For winner
cn_w = compact_neutral[["Season","WTeamID","WMargin"]].rename(
    columns={"WTeamID":"TeamID"}
)
cn_w["won"] = 1
cn_w["margin"] = cn_w["WMargin"]

# For loser
cn_l = compact_neutral[["Season","LTeamID","WMargin"]].rename(
    columns={"LTeamID":"TeamID"}
)
cn_l["won"] = 0
cn_l["margin"] = -cn_l["WMargin"]

cn_all = pd.concat([cn_w[["Season","TeamID","won","margin"]],
                    cn_l[["Season","TeamID","won","margin"]]], ignore_index=True)

neutral_stats = cn_all.groupby(["Season","TeamID"]).agg(
    neutral_winpct=("won","mean"),
    neutral_margin=("margin","mean"),
    neutral_n_games=("won","count")
).reset_index()

neut_T1 = neutral_stats.rename(columns={
    "TeamID":"T1_TeamID",
    "neutral_winpct":"T1_neutral_winpct",
    "neutral_margin":"T1_neutral_margin",
})
neut_T2 = neutral_stats.rename(columns={
    "TeamID":"T2_TeamID",
    "neutral_winpct":"T2_neutral_winpct",
    "neutral_margin":"T2_neutral_margin",
})

td_fe4 = tourney_data.copy()
td_fe4 = td_fe4.merge(neut_T1[["Season","T1_TeamID","T1_neutral_winpct","T1_neutral_margin"]],
                      on=["Season","T1_TeamID"], how="left")
td_fe4 = td_fe4.merge(neut_T2[["Season","T2_TeamID","T2_neutral_winpct","T2_neutral_margin"]],
                      on=["Season","T2_TeamID"], how="left")
td_fe4["neutral_winpct_diff"] = td_fe4["T1_neutral_winpct"] - td_fe4["T2_neutral_winpct"]
td_fe4["neutral_margin_diff"] = td_fe4["T1_neutral_margin"] - td_fe4["T2_neutral_margin"]
FE4_COLS = ["T1_neutral_winpct","T2_neutral_winpct","neutral_winpct_diff",
            "T1_neutral_margin","T2_neutral_margin","neutral_margin_diff"]
nan_pct4 = td_fe4[FE4_COLS].isna().mean().mean()
print(f"  FE4 NaN rate: {nan_pct4:.3f}")
# Do NOT fill NaN -- let XGBoost handle missing (some teams have no neutral games)

FEAT_FE4 = FEATURES_GOLDEN + FE4_COLS
margins_fe4, targets_fe4, ss_fe4, mw_fe4, models_fe4 = run_loso(td_fe4, FEAT_FE4, "opt_fe4_neutral")
_, probs_fe4, labels_fe4 = calibrate(margins_fe4, targets_fe4)
brier_fe4 = stage1_brier(probs_fe4, labels_fe4, ss_fe4)
delta_fe4 = BASELINE_BRIER - brier_fe4
print(f"  FE4 Brier: {brier_fe4:.5f}  (delta={delta_fe4:+.5f})  "
      f"{'IMPROVEMENT' if delta_fe4 > 0 else 'no gain'}")


# ─────────────────────────────────────────────────────────────────────────────
# FE5: CONFERENCE STRENGTH
# average Ridge quality of all conference members (per season)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FE5: Conference Strength (avg Ridge quality of conference)")
print("=" * 60)

conf_all = conf_all[conf_all["Season"] >= SEASON_CUTOFF].copy()
# Merge Ridge quality per team
conf_with_q = conf_all.merge(
    ridge_quality[["Season","TeamID","quality"]],
    on=["Season","TeamID"], how="left"
)
# Avg quality per conference per season (exclude the team itself? No -- avg incl. team is fine)
conf_strength = conf_with_q.groupby(["Season","ConfAbbrev"])["quality"].mean().reset_index()
conf_strength = conf_strength.rename(columns={"quality":"conf_avg_quality"})

# Merge conf label per team
conf_T1_label = conf_all.rename(columns={"TeamID":"T1_TeamID","ConfAbbrev":"T1_conf"})
conf_T2_label = conf_all.rename(columns={"TeamID":"T2_TeamID","ConfAbbrev":"T2_conf"})
td_fe5 = tourney_data.copy()
td_fe5 = td_fe5.merge(conf_T1_label[["Season","T1_TeamID","T1_conf"]], on=["Season","T1_TeamID"], how="left")
td_fe5 = td_fe5.merge(conf_T2_label[["Season","T2_TeamID","T2_conf"]], on=["Season","T2_TeamID"], how="left")
# Merge conference strength
td_fe5 = td_fe5.merge(
    conf_strength.rename(columns={"ConfAbbrev":"T1_conf","conf_avg_quality":"T1_conf_strength"}),
    on=["Season","T1_conf"], how="left"
)
td_fe5 = td_fe5.merge(
    conf_strength.rename(columns={"ConfAbbrev":"T2_conf","conf_avg_quality":"T2_conf_strength"}),
    on=["Season","T2_conf"], how="left"
)
td_fe5["conf_strength_diff"] = td_fe5["T1_conf_strength"] - td_fe5["T2_conf_strength"]
FE5_COLS = ["T1_conf_strength","T2_conf_strength","conf_strength_diff"]
nan_pct5 = td_fe5[FE5_COLS].isna().mean().mean()
print(f"  FE5 NaN rate: {nan_pct5:.3f}")

FEAT_FE5 = FEATURES_GOLDEN + FE5_COLS
margins_fe5, targets_fe5, ss_fe5, mw_fe5, models_fe5 = run_loso(td_fe5, FEAT_FE5, "opt_fe5_conf")
_, probs_fe5, labels_fe5 = calibrate(margins_fe5, targets_fe5)
brier_fe5 = stage1_brier(probs_fe5, labels_fe5, ss_fe5)
delta_fe5 = BASELINE_BRIER - brier_fe5
print(f"  FE5 Brier: {brier_fe5:.5f}  (delta={delta_fe5:+.5f})  "
      f"{'IMPROVEMENT' if delta_fe5 > 0 else 'no gain'}")


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY + DECIDE WHICH TO KEEP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results = {
    "baseline_ridge_quality": BASELINE_BRIER,
    "FE1_sos":          {"brier": round(brier_fe1, 5), "delta": round(delta_fe1, 5), "n_features": len(FE1_COLS)},
    "FE2_late_delta":   {"brier": round(brier_fe2, 5), "delta": round(delta_fe2, 5), "n_features": len(FE2_COLS)},
    "FE3_top25_winpct": {"brier": round(brier_fe3, 5), "delta": round(delta_fe3, 5), "n_features": len(FE3_COLS)},
    "FE4_neutral_site": {"brier": round(brier_fe4, 5), "delta": round(delta_fe4, 5), "n_features": len(FE4_COLS)},
    "FE5_conf_strength":{"brier": round(brier_fe5, 5), "delta": round(delta_fe5, 5), "n_features": len(FE5_COLS)},
}

print(f"\n  {'Experiment':<28} {'Brier':>8}  {'Delta':>8}  {'Keep?':>6}")
print(f"  {'-'*55}")
print(f"  {'Baseline (Ridge Quality)':<28} {BASELINE_BRIER:>8.5f}  {'':>8}  {'[base]':>6}")
exp_names = ["FE1_sos","FE2_late_delta","FE3_top25_winpct","FE4_neutral_site","FE5_conf_strength"]
exp_labels = ["FE1 SOS","FE2 Late Delta","FE3 Top-25 Win%","FE4 Neutral Site","FE5 Conf Strength"]
briers_list = [brier_fe1, brier_fe2, brier_fe3, brier_fe4, brier_fe5]
deltas_list = [delta_fe1, delta_fe2, delta_fe3, delta_fe4, delta_fe5]
keeps       = [d > 0 for d in deltas_list]
for nm, lbl, b, d, k in zip(exp_names, exp_labels, briers_list, deltas_list, keeps):
    print(f"  {lbl:<28} {b:>8.5f}  {d:>+8.5f}  {'YES' if k else 'no':>6}")

winners = [nm for nm, k in zip(exp_names, keeps) if k]
print(f"\n  Improvements: {len(winners)}")

# Save to Optimization/
with open("Optimization/feature_engineering_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved Optimization/feature_engineering_results.json")


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED EXPERIMENT (if multiple winners)
# ─────────────────────────────────────────────────────────────────────────────
combined_cols = []
combined_td   = tourney_data.copy()
combined_tag  = "opt_fe_combined"
combined_brier = BASELINE_BRIER

if len(winners) > 0:
    print("\n" + "=" * 60)
    print(f"COMBINED: merging {len(winners)} winning feature set(s)")
    print("=" * 60)

    if delta_fe1 > 0:
        combined_td = combined_td.merge(sos_T1, on=["Season","T1_TeamID"], how="left")
        combined_td = combined_td.merge(sos_T2, on=["Season","T2_TeamID"], how="left")
        combined_td["sos_diff"] = combined_td["T1_sos"] - combined_td["T2_sos"]
        combined_cols += FE1_COLS

    if delta_fe2 > 0:
        combined_td = combined_td.merge(delta_T1, on=["Season","T1_TeamID"], how="left")
        combined_td = combined_td.merge(delta_T2, on=["Season","T2_TeamID"], how="left")
        combined_cols += FE2_COLS

    if delta_fe3 > 0:
        combined_td = combined_td.merge(
            winpct_T1[["Season","T1_TeamID","T1_win_pct_top25"]], on=["Season","T1_TeamID"], how="left"
        )
        combined_td = combined_td.merge(
            winpct_T2[["Season","T2_TeamID","T2_win_pct_top25"]], on=["Season","T2_TeamID"], how="left"
        )
        combined_td["win_pct_top25_diff"] = combined_td["T1_win_pct_top25"] - combined_td["T2_win_pct_top25"]
        combined_td[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]] = \
            combined_td[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]].fillna(0)
        combined_cols += FE3_COLS

    if delta_fe4 > 0:
        combined_td = combined_td.merge(
            neut_T1[["Season","T1_TeamID","T1_neutral_winpct","T1_neutral_margin"]],
            on=["Season","T1_TeamID"], how="left"
        )
        combined_td = combined_td.merge(
            neut_T2[["Season","T2_TeamID","T2_neutral_winpct","T2_neutral_margin"]],
            on=["Season","T2_TeamID"], how="left"
        )
        combined_td["neutral_winpct_diff"] = combined_td["T1_neutral_winpct"] - combined_td["T2_neutral_winpct"]
        combined_td["neutral_margin_diff"] = combined_td["T1_neutral_margin"] - combined_td["T2_neutral_margin"]
        combined_cols += FE4_COLS

    if delta_fe5 > 0:
        combined_td = combined_td.merge(
            conf_T1_label[["Season","T1_TeamID","T1_conf"]], on=["Season","T1_TeamID"], how="left"
        )
        combined_td = combined_td.merge(
            conf_T2_label[["Season","T2_TeamID","T2_conf"]], on=["Season","T2_TeamID"], how="left"
        )
        combined_td = combined_td.merge(
            conf_strength.rename(columns={"ConfAbbrev":"T1_conf","conf_avg_quality":"T1_conf_strength"}),
            on=["Season","T1_conf"], how="left"
        )
        combined_td = combined_td.merge(
            conf_strength.rename(columns={"ConfAbbrev":"T2_conf","conf_avg_quality":"T2_conf_strength"}),
            on=["Season","T2_conf"], how="left"
        )
        combined_td["conf_strength_diff"] = combined_td["T1_conf_strength"] - combined_td["T2_conf_strength"]
        combined_cols += FE5_COLS

    FEAT_COMBINED = FEATURES_GOLDEN + combined_cols
    print(f"  Combined feature count: {len(FEAT_COMBINED)}")

    margins_cmb, targets_cmb, ss_cmb, mw_cmb, models_cmb = \
        run_loso(combined_td, FEAT_COMBINED, combined_tag)
    _, probs_cmb, labels_cmb = calibrate(margins_cmb, targets_cmb)
    combined_brier = stage1_brier(probs_cmb, labels_cmb, ss_cmb)
    combined_delta = BASELINE_BRIER - combined_brier
    print(f"  Combined Brier: {combined_brier:.5f}  (delta={combined_delta:+.5f})  "
          f"{'IMPROVEMENT over baseline' if combined_delta > 0 else 'no improvement'}")
    results["combined"] = {"brier": round(combined_brier, 5), "delta": round(combined_delta, 5),
                           "features_added": combined_cols}
    with open("Optimization/feature_engineering_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Use combined only if it beats baseline
    use_combined = combined_delta > 0
else:
    FEAT_COMBINED  = FEATURES_GOLDEN
    models_cmb     = None
    combined_cols  = []
    use_combined   = False


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE FINAL SUBMISSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING submission_stage1.csv")
print("=" * 60)

# Determine final model to use
if use_combined:
    final_models  = models_cmb
    final_spline  = calibrate(margins_cmb, targets_cmb)[0]
    final_feats   = FEAT_COMBINED
    final_desc    = f"Ridge + {len(combined_cols)} new FE cols"
    final_brier   = combined_brier
else:
    # Fall back to best single experiment if any won, else use baseline
    if len(winners) == 1:
        winner_name = winners[0]
        idx = exp_names.index(winner_name)
        final_brier = briers_list[idx]
        if winner_name == "FE1_sos":
            final_models = models_fe1
            final_spline = calibrate(margins_fe1, targets_fe1)[0]
            final_feats  = FEAT_FE1
            final_desc   = "Ridge + SOS"
        elif winner_name == "FE2_late_delta":
            final_models = models_fe2
            final_spline = calibrate(margins_fe2, targets_fe2)[0]
            final_feats  = FEAT_FE2
            final_desc   = "Ridge + Late Delta"
        elif winner_name == "FE3_top25_winpct":
            final_models = models_fe3
            final_spline = calibrate(margins_fe3, targets_fe3)[0]
            final_feats  = FEAT_FE3
            final_desc   = "Ridge + Top25 Win%"
        elif winner_name == "FE4_neutral_site":
            final_models = models_fe4
            final_spline = calibrate(margins_fe4, targets_fe4)[0]
            final_feats  = FEAT_FE4
            final_desc   = "Ridge + Neutral Site"
        else:  # FE5
            final_models = models_fe5
            final_spline = calibrate(margins_fe5, targets_fe5)[0]
            final_feats  = FEAT_FE5
            final_desc   = "Ridge + Conf Strength"
    else:
        # No single experiment helped; keep baseline Ridge Quality
        # Reload baseline cache
        print("  No improvements found -- keeping baseline Ridge Quality config")
        base_cache = cache_load("opt_ridge_quality")
        b_margins, b_targets, b_ss, b_mw, base_models = base_cache
        final_models = base_models
        final_spline = calibrate(b_margins, b_targets)[0]
        final_feats  = FEATURES_GOLDEN
        final_desc   = "Ridge Quality (no FE improvement)"
        final_brier  = BASELINE_BRIER

print(f"\n  Final config: {final_desc}")
print(f"  Final Brier:  {final_brier:.5f}")

# ─── Build submission rows ───────────────────────────────────────────────────
sub1 = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage1.csv")
sub1["Season"]    = sub1["ID"].str.split("_").str[0].astype(int)
sub1["T1_TeamID"] = sub1["ID"].str.split("_").str[1].astype(int)
sub1["T2_TeamID"] = sub1["ID"].str.split("_").str[2].astype(int)
sub1["men_women"] = sub1["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)

# Merge all base features
sub1 = sub1.merge(ss_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(ss_T2, on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(seeds_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(seeds_T2, on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(mom_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(mom_T2, on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(ord_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(ord_T2, on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(rq_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(rq_T2, on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(elos_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(elos_T2, on=["Season","T2_TeamID"], how="left")
sub1["Seed_diff"]    = sub1["T2_seed"]    - sub1["T1_seed"]
sub1["elo_diff"]     = sub1["T1_elo"]     - sub1["T2_elo"]
sub1["diff_quality"] = sub1["T1_quality"] - sub1["T2_quality"]
for sys in ORDINAL_SYSTEMS:
    sub1[f"ord_diff_{sys}"] = sub1[f"T2_ord_{sys}"] - sub1[f"T1_ord_{sys}"]
for log_col in LOG_FEATS:
    base_col = log_col.replace("log_", "")
    sub1[log_col] = np.log1p(sub1[base_col].clip(lower=0)) if base_col in sub1.columns else np.nan

# Merge winning FE features if any
if use_combined or len(winners) == 1:
    if delta_fe1 > 0 and "FE1_sos" in (winners if not use_combined else winners):
        sub1 = sub1.merge(sos_T1, on=["Season","T1_TeamID"], how="left")
        sub1 = sub1.merge(sos_T2, on=["Season","T2_TeamID"], how="left")
        sub1["sos_diff"] = sub1["T1_sos"] - sub1["T2_sos"]
    if delta_fe2 > 0 and "FE2_late_delta" in (winners if not use_combined else winners):
        sub1 = sub1.merge(delta_T1, on=["Season","T1_TeamID"], how="left")
        sub1 = sub1.merge(delta_T2, on=["Season","T2_TeamID"], how="left")
    if delta_fe3 > 0 and "FE3_top25_winpct" in (winners if not use_combined else winners):
        sub1 = sub1.merge(winpct_T1[["Season","T1_TeamID","T1_win_pct_top25"]], on=["Season","T1_TeamID"], how="left")
        sub1 = sub1.merge(winpct_T2[["Season","T2_TeamID","T2_win_pct_top25"]], on=["Season","T2_TeamID"], how="left")
        sub1["win_pct_top25_diff"] = sub1["T1_win_pct_top25"] - sub1["T2_win_pct_top25"]
        sub1[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]] = \
            sub1[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]].fillna(0)
    if delta_fe4 > 0 and "FE4_neutral_site" in (winners if not use_combined else winners):
        sub1 = sub1.merge(neut_T1[["Season","T1_TeamID","T1_neutral_winpct","T1_neutral_margin"]],
                          on=["Season","T1_TeamID"], how="left")
        sub1 = sub1.merge(neut_T2[["Season","T2_TeamID","T2_neutral_winpct","T2_neutral_margin"]],
                          on=["Season","T2_TeamID"], how="left")
        sub1["neutral_winpct_diff"] = sub1["T1_neutral_winpct"] - sub1["T2_neutral_winpct"]
        sub1["neutral_margin_diff"] = sub1["T1_neutral_margin"] - sub1["T2_neutral_margin"]
    if delta_fe5 > 0 and "FE5_conf_strength" in (winners if not use_combined else winners):
        sub1 = sub1.merge(conf_T1_label[["Season","T1_TeamID","T1_conf"]], on=["Season","T1_TeamID"], how="left")
        sub1 = sub1.merge(conf_T2_label[["Season","T2_TeamID","T2_conf"]], on=["Season","T2_TeamID"], how="left")
        sub1 = sub1.merge(
            conf_strength.rename(columns={"ConfAbbrev":"T1_conf","conf_avg_quality":"T1_conf_strength"}),
            on=["Season","T1_conf"], how="left"
        )
        sub1 = sub1.merge(
            conf_strength.rename(columns={"ConfAbbrev":"T2_conf","conf_avg_quality":"T2_conf_strength"}),
            on=["Season","T2_conf"], how="left"
        )
        sub1["conf_strength_diff"] = sub1["T1_conf_strength"] - sub1["T2_conf_strength"]

sub1["Pred"] = np.nan

# Level 1: Season-specific LOSO models
for yr in [2022, 2023, 2024, 2025]:
    mask = sub1["Season"] == yr
    if mask.sum() == 0 or yr not in final_models:
        continue
    raw = final_models[yr].predict(xgb.DMatrix(sub1.loc[mask, final_feats].values))
    sub1.loc[mask, "Pred"] = np.clip(
        final_spline(np.clip(raw, -T_SPLINE, T_SPLINE)), CLIP_LO, CLIP_HI
    )

# Level 2: Elo sigmoid
lvl2 = sub1["Pred"].isna()
if lvl2.sum() > 0:
    ed = sub1.loc[lvl2, "elo_diff"].fillna(0)
    sub1.loc[lvl2, "Pred"] = (1 / (1 + np.exp(-ed / 150))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 2 fallback (Elo): {lvl2.sum():,} rows")

# Level 3: NET rank
lvl3 = sub1["Pred"].isna()
if lvl3.sum() > 0:
    rd_ = sub1.loc[lvl3, "ord_diff_NET"].fillna(0)
    sub1.loc[lvl3, "Pred"] = (1 / (1 + np.exp(-rd_ / 50))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 3 fallback (NET): {lvl3.sum():,} rows")

# Level 4: Seed diff
lvl4 = sub1["Pred"].isna()
if lvl4.sum() > 0:
    sd = sub1.loc[lvl4, "Seed_diff"].fillna(0)
    sub1.loc[lvl4, "Pred"] = (1 / (1 + np.exp(-sd * 0.4))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 4 fallback (Seed): {lvl4.sum():,} rows")

# Level 5: 0.5
remaining = sub1["Pred"].isna().sum()
if remaining > 0:
    sub1["Pred"] = sub1["Pred"].fillna(0.5)
    print(f"  Level 5 fallback (0.5): {remaining:,} rows")

sub1["Pred"] = sub1["Pred"].clip(CLIP_LO, CLIP_HI).round(6)
print(f"\n  Rows: {len(sub1):,}  NaN: {sub1.Pred.isna().sum()}")
print(f"  Mean: {sub1.Pred.mean():.4f}  Std: {sub1.Pred.std():.4f}  "
      f"Min: {sub1.Pred.min():.4f}  Max: {sub1.Pred.max():.4f}")

sub1[["ID","Pred"]].to_csv("submission_stage1.csv", index=False)
print(f"\n  Saved submission_stage1.csv")
print(f"\n  Final OOF Stage1 Brier: {final_brier:.5f}")
print(f"  vs Baseline:            {BASELINE_BRIER:.5f}")
if final_brier < BASELINE_BRIER:
    print(f"  Improvement:           +{BASELINE_BRIER - final_brier:.5f}")
else:
    print("  No improvement -- submission unchanged from Ridge Quality baseline")
