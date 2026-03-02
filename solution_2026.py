"""
March Machine Learning Mania 2026 - Solution
=============================================
Ported from 2025 1st-place winner with:
  [BUG FIX]  men_women encoding consistent between training AND inference
  [BUG FIX]  Elo and GLM quality computed for Season 2026 (not just seeded seasons)
  [BUG FIX]  2026 seed proxy from NET ranking (men) and Elo rank (women)
  [PATH]     data_dir updated to ./Data
  [REMOVED]  2025-specific manual overrides
  [NEW]      Massey Ordinals features (POM, NET, MAS, SAG, MOR)
  [NEW]      Late-season momentum features (last-20-game rolling averages)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import tqdm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(name):
    return os.path.join(CACHE_DIR, name + ".pkl")

def cache_save(name, obj):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f)

def cache_load(name):
    p = _cache_path(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

DATA_DIR = "./Data"
SEASON_CUTOFF = 2003   # men's detailed data starts here
WOMEN_MIN     = 2010   # women's detailed data starts here

print("=" * 60)
print("MARCH MACHINE LEARNING MANIA 2026")
print("=" * 60)

# ────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────────────────────
print("\n[1/9] Loading data...")

M_regular  = pd.read_csv(f"{DATA_DIR}/MRegularSeasonDetailedResults.csv")
M_tourney  = pd.read_csv(f"{DATA_DIR}/MNCAATourneyDetailedResults.csv")
M_seeds    = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
M_ordinals = pd.read_csv(f"{DATA_DIR}/MMasseyOrdinals.csv")

W_regular = pd.read_csv(f"{DATA_DIR}/WRegularSeasonDetailedResults.csv")
W_tourney  = pd.read_csv(f"{DATA_DIR}/WNCAATourneyDetailedResults.csv")
W_seeds    = pd.read_csv(f"{DATA_DIR}/WNCAATourneySeeds.csv")

regular_results = pd.concat([M_regular, W_regular], ignore_index=True)
tourney_results  = pd.concat([M_tourney, W_tourney],  ignore_index=True)
seeds            = pd.concat([M_seeds,   W_seeds],    ignore_index=True)

regular_results = regular_results[regular_results["Season"] >= SEASON_CUTOFF]
tourney_results  = tourney_results[tourney_results["Season"]  >= SEASON_CUTOFF]
seeds            = seeds[seeds["Season"] >= SEASON_CUTOFF]

# ────────────────────────────────────────────────────────────
# 2. PREPARE DATA  (OT-normalise + double rows)
# ────────────────────────────────────────────────────────────
print("[2/9] Preparing data (OT normalisation + row doubling)...")

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
    # FIX: men=1 (TeamID 1xxx), women=0 (TeamID 3xxx) — consistent everywhere
    out["men_women"] = out["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)
    return out

regular_data = prepare_data(regular_results)
tourney_data  = prepare_data(tourney_results)

# ────────────────────────────────────────────────────────────
# 3. SEED FEATURES
# ────────────────────────────────────────────────────────────
print("[3/9] Building seed features...")

seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds[["Season","TeamID","seed"]].rename(columns={"TeamID":"T1_TeamID","seed":"T1_seed"})
seeds_T2 = seeds[["Season","TeamID","seed"]].rename(columns={"TeamID":"T2_TeamID","seed":"T2_seed"})

tourney_data = tourney_data[["Season","T1_TeamID","T2_TeamID","DayNum","PointDiff","win","men_women"]]
tourney_data = tourney_data.merge(seeds_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(seeds_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["Seed_diff"] = tourney_data["T2_seed"] - tourney_data["T1_seed"]

# ────────────────────────────────────────────────────────────
# 4. SEASON-AVERAGE FEATURES
# ────────────────────────────────────────────────────────────
print("[4/9] Computing season averages...")

BOXCOLS = [
    "T1_Score","T1_FGM","T1_FGA","T1_FGM3","T1_FGA3","T1_FTM","T1_FTA",
    "T1_OR","T1_DR","T1_Ast","T1_TO","T1_Stl","T1_Blk","T1_PF",
    "T2_Score","T2_FGM","T2_FGA","T2_FGM3","T2_FGA3","T2_FTM","T2_FTA",
    "T2_OR","T2_DR","T2_Ast","T2_TO","T2_Stl","T2_Blk","T2_PF",
    "PointDiff",
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

# ────────────────────────────────────────────────────────────
# 4b. LATE-SEASON MOMENTUM  [Innovation 5]
# ────────────────────────────────────────────────────────────
print("[4b/9] Computing late-season momentum (last 20 games)...")

MOM_COLS = ["T1_Score","T1_FGA","T1_OR","T1_DR","T1_Blk","T1_PF",
            "T2_Score","T2_FGA","T2_OR","T2_DR","T2_Blk","T2_PF","PointDiff"]

rd_sorted = regular_data.sort_values(["Season","T1_TeamID","DayNum"]).copy()
rd_sorted["_rank"] = rd_sorted.groupby(["Season","T1_TeamID"])["DayNum"].rank(
    ascending=False, method="first"
)
mom = rd_sorted[rd_sorted["_rank"] <= 20].groupby(["Season","T1_TeamID"])[MOM_COLS].mean().reset_index()

mom_T1 = mom.copy()
mom_T1.columns = ["T1_mom_" + c.replace("T1_","").replace("T2_","opp_") for c in mom_T1.columns]
mom_T1 = mom_T1.rename(columns={"T1_mom_Season":"Season","T1_mom_TeamID":"T1_TeamID"})

mom_T2 = mom.copy()
mom_T2.columns = ["T2_mom_" + c.replace("T1_","").replace("T2_","opp_") for c in mom_T2.columns]
mom_T2 = mom_T2.rename(columns={"T2_mom_Season":"Season","T2_mom_TeamID":"T2_TeamID"})

tourney_data = tourney_data.merge(mom_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(mom_T2, on=["Season","T2_TeamID"], how="left")

# ────────────────────────────────────────────────────────────
# 5. ELO RATINGS
#    FIX: iterate over ALL seasons with data (includes 2026)
# ────────────────────────────────────────────────────────────
print("[5/9] Computing Elo ratings (all seasons including 2026)...")

def _update_elo(w, l, k=100, width=400):
    exp = 1.0 / (1 + 10 ** ((l - w) / width))
    c = k * (1 - exp)
    return w + c, l - c

# Use ALL seasons from regular data (not just seeded seasons) → includes 2026
_all_reg_seasons = sorted(regular_data["Season"].unique())

elos_list = []
for season in _all_reg_seasons:
    ss_s = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)].reset_index(drop=True)
    teams = set(ss_s["T1_TeamID"]) | set(ss_s["T2_TeamID"])
    elo = {t: 1000 for t in teams}
    for i in range(len(ss_s)):
        w_id, l_id = ss_s.loc[i,"T1_TeamID"], ss_s.loc[i,"T2_TeamID"]
        elo[w_id], elo[l_id] = _update_elo(elo[w_id], elo[l_id])
    edf = pd.DataFrame.from_dict(elo, orient="index").reset_index()
    edf.columns = ["TeamID","elo"]
    edf["Season"] = season
    elos_list.append(edf)

elos = pd.concat(elos_list, ignore_index=True)
elos_T1 = elos.rename(columns={"TeamID":"T1_TeamID","elo":"T1_elo"})
elos_T2 = elos.rename(columns={"TeamID":"T2_TeamID","elo":"T2_elo"})
tourney_data = tourney_data.merge(elos_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(elos_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["elo_diff"] = tourney_data["T1_elo"] - tourney_data["T2_elo"]

# ────────────────────────────────────────────────────────────
# 6. MASSEY ORDINALS  [Innovation 1]
#    Men-only: POM, NET, MAS, SAG, MOR (latest ranking per season)
# ────────────────────────────────────────────────────────────
print("[6/9] Building Massey Ordinals features...")

ORDINAL_SYSTEMS = ["POM","NET","MAS","SAG","MOR"]
ORDINAL_SYSTEMS = [s for s in ORDINAL_SYSTEMS if s in M_ordinals["SystemName"].unique()]
print(f"  Systems: {ORDINAL_SYSTEMS}")

ord_latest = (
    M_ordinals[M_ordinals["SystemName"].isin(ORDINAL_SYSTEMS)]
    .sort_values("RankingDayNum")
    .groupby(["Season","SystemName","TeamID"]).last().reset_index()
    [["Season","SystemName","TeamID","OrdinalRank"]]
)
ord_pivot = ord_latest.pivot_table(
    index=["Season","TeamID"], columns="SystemName", values="OrdinalRank"
).reset_index()
# Name columns ord_XXX
ord_pivot.columns = (
    ["Season","TeamID"] +
    [f"ord_{c}" for c in ord_pivot.columns[2:]]
)
# Rename to T1/T2 prefixed versions
_ord_rename_t1 = {"TeamID":"T1_TeamID"}
_ord_rename_t2 = {"TeamID":"T2_TeamID"}
for sys in ORDINAL_SYSTEMS:
    _ord_rename_t1[f"ord_{sys}"] = f"T1_ord_{sys}"
    _ord_rename_t2[f"ord_{sys}"] = f"T2_ord_{sys}"

ord_T1 = ord_pivot.rename(columns=_ord_rename_t1)
ord_T2 = ord_pivot.rename(columns=_ord_rename_t2)

tourney_data = tourney_data.merge(ord_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(ord_T2, on=["Season","T2_TeamID"], how="left")
for sys in ORDINAL_SYSTEMS:
    # positive diff = T2 has higher (worse) rank = T1 is better
    tourney_data[f"ord_diff_{sys}"] = tourney_data[f"T2_ord_{sys}"] - tourney_data[f"T1_ord_{sys}"]

# ────────────────────────────────────────────────────────────
# 7. GLM QUALITY SCORE
#    FIX: separate 2026 pass uses all teams (no tournament filter)
# ────────────────────────────────────────────────────────────
print("[7/9] Computing GLM quality scores...")

# Build tournament-connected set from historical seeds (through 2025)
regular_data["ST1"] = regular_data["Season"].astype(str) + "/" + regular_data["T1_TeamID"].astype(str)
regular_data["ST2"] = regular_data["Season"].astype(str) + "/" + regular_data["T2_TeamID"].astype(str)
seeds_T1["ST1"]     = seeds_T1["Season"].astype(str) + "/" + seeds_T1["T1_TeamID"].astype(str)
seeds_T2["ST2"]     = seeds_T2["Season"].astype(str) + "/" + seeds_T2["T2_TeamID"].astype(str)

st = set(seeds_T1["ST1"]) | set(seeds_T2["ST2"])
st = st | set(regular_data.loc[
    (regular_data["T1_Score"] > regular_data["T2_Score"]) & regular_data["ST2"].isin(st),
    "ST1"])

def _glm_quality(season, men_women, data_df):
    """OLS fixed-effects quality from PointDiff ~ -1 + T1 + T2."""
    sub = data_df[(data_df["Season"] == season) & (data_df["men_women"] == men_women)]
    if len(sub) < 20:
        return pd.DataFrame(columns=["TeamID","quality","Season"])
    try:
        glm = sm.GLM.from_formula(
            "PointDiff ~ -1 + T1_TeamID + T2_TeamID",
            data=sub,
            family=sm.families.Gaussian(),
        ).fit(disp=False)
        q = pd.DataFrame({"pname": glm.params.index, "quality": glm.params.values})
        q = q[q["pname"].str.startswith("T1_TeamID")]
        q["TeamID"] = q["pname"].str.extract(r"\[([^\]]+)\]")[0].astype(int)
        q = q[["TeamID","quality"]].copy()
        q["Season"] = season
        return q
    except Exception:
        return pd.DataFrame(columns=["TeamID","quality","Season"])

# --- Historical GLM (tournament-connected filter, 2003-2025) ---
_cached_glm = cache_load("glm_quality")
if _cached_glm is not None:
    print("  [CACHE HIT] Loading GLM quality from cache...")
    glm_quality = _cached_glm
else:
    dt_hist = regular_data[
        (regular_data["ST1"].isin(st) | regular_data["ST2"].isin(st)) &
        (regular_data["Season"] <= 2025)
    ].copy()
    dt_hist["T1_TeamID"] = dt_hist["T1_TeamID"].astype(str)
    dt_hist["T2_TeamID"] = dt_hist["T2_TeamID"].astype(str)
    dt_hist.loc[~dt_hist["ST1"].isin(st), "T1_TeamID"] = "0000"
    dt_hist.loc[~dt_hist["ST2"].isin(st), "T2_TeamID"] = "0000"

    glm_parts = []
    hist_seasons = sorted(seeds["Season"].unique())
    for s in tqdm.tqdm(hist_seasons, desc="  GLM hist"):
        if s >= WOMEN_MIN:
            glm_parts.append(_glm_quality(s, 0, dt_hist))
        if s >= SEASON_CUTOFF:
            glm_parts.append(_glm_quality(s, 1, dt_hist))

    # --- 2026 GLM (all active teams, no tournament filter) ---
    print("  Computing 2026 GLM quality (all teams)...")
    dt_2026 = regular_data[regular_data["Season"] == 2026].copy()
    dt_2026["T1_TeamID"] = dt_2026["T1_TeamID"].astype(str)
    dt_2026["T2_TeamID"] = dt_2026["T2_TeamID"].astype(str)
    game_counts = dt_2026.groupby("T1_TeamID").size()
    active_teams = set(game_counts[game_counts >= 10].index)
    dt_2026 = dt_2026[dt_2026["T1_TeamID"].isin(active_teams) & dt_2026["T2_TeamID"].isin(active_teams)]
    glm_parts.append(_glm_quality(2026, 0, dt_2026))
    glm_parts.append(_glm_quality(2026, 1, dt_2026))

    glm_quality = pd.concat(glm_parts, ignore_index=True)
    cache_save("glm_quality", glm_quality)
    print("  [CACHE SAVED] GLM quality")
glm_T1 = glm_quality.rename(columns={"TeamID":"T1_TeamID","quality":"T1_quality"})
glm_T2 = glm_quality.rename(columns={"TeamID":"T2_TeamID","quality":"T2_quality"})

tourney_data = tourney_data.merge(glm_T1, on=["Season","T1_TeamID"], how="left")
tourney_data = tourney_data.merge(glm_T2, on=["Season","T2_TeamID"], how="left")
tourney_data["diff_quality"] = tourney_data["T1_quality"] - tourney_data["T2_quality"]

print(f"  2026 GLM men teams:   {len(glm_quality[(glm_quality.Season==2026) & (glm_quality.TeamID < 2000)])}")
print(f"  2026 GLM women teams: {len(glm_quality[(glm_quality.Season==2026) & (glm_quality.TeamID >= 3000)])}")
print(f"  2026 Elo men teams:   {len(elos[(elos.Season==2026) & (elos.TeamID < 2000)])}")
print(f"  2026 Elo women teams: {len(elos[(elos.Season==2026) & (elos.TeamID >= 3000)])}")

# ────────────────────────────────────────────────────────────
# 8. FEATURE SETS
# ────────────────────────────────────────────────────────────
FEATURES_BASE = [
    "men_women","T1_seed","T2_seed","Seed_diff",
    "T1_avg_Score","T1_avg_FGA","T1_avg_OR","T1_avg_DR","T1_avg_Blk","T1_avg_PF",
    "T1_avg_opponent_FGA","T1_avg_opponent_Blk","T1_avg_opponent_PF","T1_avg_PointDiff",
    "T2_avg_Score","T2_avg_FGA","T2_avg_OR","T2_avg_DR","T2_avg_Blk","T2_avg_PF",
    "T2_avg_opponent_FGA","T2_avg_opponent_Blk","T2_avg_opponent_PF","T2_avg_PointDiff",
    "T1_elo","T2_elo","elo_diff",
    "T1_quality","T2_quality",
]
FEATURES_MOM = (
    [f"T1_mom_{c}" for c in ["Score","FGA","OR","DR","Blk","PF","PointDiff"]] +
    [f"T2_mom_{c}" for c in ["Score","FGA","OR","DR","Blk","PF","PointDiff"]]
)
FEATURES_ORD = []
for sys in ORDINAL_SYSTEMS:
    FEATURES_ORD += [f"T1_ord_{sys}", f"T2_ord_{sys}", f"ord_diff_{sys}"]

FEATURES_INNOV = FEATURES_BASE + FEATURES_MOM + FEATURES_ORD

print(f"\n  Baseline features:   {len(FEATURES_BASE)}")
print(f"  Innovation features: {len(FEATURES_INNOV)}")
print(f"  tourney_data rows:   {len(tourney_data):,}")

# ────────────────────────────────────────────────────────────
# 9. XGBOOST LOSO + SPLINE CALIBRATION
# ────────────────────────────────────────────────────────────
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

def run_loso(df, features, tag):
    cache_key = f"loso_{tag}"
    cached = cache_load(cache_key)
    if cached is not None:
        print(f"  [CACHE HIT] Loading LOSO [{tag}] from cache...")
        return cached

    seasons = sorted(df["Season"].unique())
    models_, margins_, targets_, ss_, mw_ = {}, [], [], [], []
    for s in tqdm.tqdm(seasons, desc=f"  LOSO [{tag}]"):
        tr = df["Season"] != s
        vl = df["Season"] == s
        dtrain = xgb.DMatrix(df.loc[tr, features].values, label=df.loc[tr,"PointDiff"].values)
        models_[s] = xgb.train(XGB_PARAM, dtrain, num_boost_round=NUM_ROUNDS)
        preds = models_[s].predict(xgb.DMatrix(df.loc[vl, features].values))
        margins_ += list(preds)
        targets_ += list(df.loc[vl, "PointDiff"].values)
        ss_      += [s] * vl.sum()
        mw_      += list(df.loc[vl, "men_women"].values)

    T = 25
    dat = sorted(zip(margins_, np.array(targets_) > 0), key=lambda x: x[0])
    p, l = zip(*dat)
    spline = UnivariateSpline(np.clip(p, -T, T), l, k=5)
    probs  = np.clip(spline(np.clip(np.array(margins_), -T, T)), 0.01, 0.99)
    result = (probs, (np.array(targets_) > 0).astype(int), np.array(ss_), np.array(mw_), models_, spline, T)
    cache_save(cache_key, result)
    print(f"  [CACHE SAVED] LOSO [{tag}]")
    return result

def print_report(probs, labels, ss, mw, title):
    s1 = ss >= 2022
    m  = mw == 1
    w  = mw == 0
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Brier (all seasons 2003-25): {brier_score_loss(labels, probs):.5f}")
    if s1.sum() > 0:
        print(f"  Brier Stage1 (2022-25):      {brier_score_loss(labels[s1], probs[s1]):.5f}")
    if m.sum() > 0:
        print(f"  Brier Men  (all):            {brier_score_loss(labels[m], probs[m]):.5f}")
    if w.sum() > 0:
        print(f"  Brier Women (all):           {brier_score_loss(labels[w], probs[w]):.5f}")
    print(f"\n  Per-season breakdown (Stage 1):")
    for yr in [2022, 2023, 2024, 2025]:
        mask = ss == yr
        if mask.sum() == 0:
            continue
        b  = brier_score_loss(labels[mask], probs[mask])
        bm = brier_score_loss(labels[mask&m], probs[mask&m]) if (mask&m).sum()>0 else float("nan")
        bw = brier_score_loss(labels[mask&w], probs[mask&w]) if (mask&w).sum()>0 else float("nan")
        print(f"    {yr}: overall={b:.5f}  men={bm:.5f}  women={bw:.5f}")
    print("=" * 60)

# --- Baseline run ---
print("\n[8/9] BASELINE LOSO (29 features)...")
b_probs, b_labels, b_ss, b_mw, b_models, b_spline, b_t = run_loso(tourney_data, FEATURES_BASE, "baseline")
print_report(b_probs, b_labels, b_ss, b_mw, "BASELINE REPORT (29 features)")

# --- Innovation run ---
print(f"\n[9/9] INNOVATION LOSO ({len(FEATURES_INNOV)} features)...")
i_probs, i_labels, i_ss, i_mw, i_models, i_spline, i_t = run_loso(tourney_data, FEATURES_INNOV, "innovation")
print_report(i_probs, i_labels, i_ss, i_mw, f"INNOVATION REPORT ({len(FEATURES_INNOV)} features)")

# Delta
s1 = b_ss >= 2022
delta = brier_score_loss(b_labels[s1], b_probs[s1]) - brier_score_loss(i_labels[s1], i_probs[s1])
print(f"\n  Delta Brier Stage1 (baseline - innovation): {delta:+.5f}  "
      f"({'IMPROVEMENT' if delta > 0 else 'no gain - keeping baseline'})")

USE_INNOV   = delta > 0
use_models  = i_models  if USE_INNOV else b_models
use_spline  = i_spline  if USE_INNOV else b_spline
use_features = FEATURES_INNOV if USE_INNOV else FEATURES_BASE
use_t       = i_t

print(f"  Using {'INNOVATION' if USE_INNOV else 'BASELINE'} model for submission.")

# ────────────────────────────────────────────────────────────
# 10. STAGE 2 SUBMISSION
# ────────────────────────────────────────────────────────────
print("\n[10/10] Generating Stage 2 submission...")

sub = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage2.csv")
sub["Season"]    = sub["ID"].str.split("_").str[0].astype(int)
sub["T1_TeamID"] = sub["ID"].str.split("_").str[1].astype(int)
sub["T2_TeamID"] = sub["ID"].str.split("_").str[2].astype(int)
# FIX: consistent men_women encoding (men=1)
sub["men_women"] = sub["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)

# Merge all features
sub = sub.merge(ss_T1,   on=["Season","T1_TeamID"], how="left")
sub = sub.merge(ss_T2,   on=["Season","T2_TeamID"], how="left")
sub = sub.merge(seeds_T1, on=["Season","T1_TeamID"], how="left")
sub = sub.merge(seeds_T2, on=["Season","T2_TeamID"], how="left")
sub = sub.merge(mom_T1,  on=["Season","T1_TeamID"], how="left")
sub = sub.merge(mom_T2,  on=["Season","T2_TeamID"], how="left")
sub = sub.merge(ord_T1,  on=["Season","T1_TeamID"], how="left")
sub = sub.merge(ord_T2,  on=["Season","T2_TeamID"], how="left")
sub = sub.merge(glm_T1,  on=["Season","T1_TeamID"], how="left")
sub = sub.merge(glm_T2,  on=["Season","T2_TeamID"], how="left")
sub = sub.merge(elos_T1, on=["Season","T1_TeamID"], how="left")
sub = sub.merge(elos_T2, on=["Season","T2_TeamID"], how="left")

sub["elo_diff"]     = sub["T1_elo"]     - sub["T2_elo"]
sub["diff_quality"] = sub["T1_quality"] - sub["T2_quality"]
for sys in ORDINAL_SYSTEMS:
    sub[f"ord_diff_{sys}"] = sub[f"T2_ord_{sys}"] - sub[f"T1_ord_{sys}"]

# ── SEED PROXY for 2026  ────────────────────────────────────────────────────
# Seeds won't be released until March 15 (Selection Sunday).
# Proxy for men: use NET rank (ordinal).  Proxy for women: use Elo rank.
print("  Building 2026 seed proxies...")

# Men: map NET ordinal rank to seed 1-16  (top 4 = seed 1, next 4 = seed 2, ...)
net_2026 = ord_pivot[(ord_pivot["Season"] == 2026)][["TeamID","ord_NET"]].copy()
net_2026["TeamID"] = net_2026["TeamID"].astype(int)
n_men_teams = len(net_2026)
net_2026["est_seed"] = np.ceil((net_2026["ord_NET"] / n_men_teams) * 16).clip(1, 16)

est_seed_T1 = net_2026[["TeamID","est_seed"]].rename(columns={"TeamID":"T1_TeamID","est_seed":"T1_seed_est"})
est_seed_T2 = net_2026[["TeamID","est_seed"]].rename(columns={"TeamID":"T2_TeamID","est_seed":"T2_seed_est"})
est_seed_T1["Season"] = 2026
est_seed_T2["Season"] = 2026

sub = sub.merge(est_seed_T1, on=["Season","T1_TeamID"], how="left")
sub = sub.merge(est_seed_T2, on=["Season","T2_TeamID"], how="left")

# Women: use Elo rank as seed proxy
elo_w_2026 = elos[(elos["Season"] == 2026) & (elos["TeamID"] >= 3000)].copy()
elo_w_2026 = elo_w_2026.sort_values("elo", ascending=False).reset_index(drop=True)
n_w = len(elo_w_2026)
elo_w_2026["est_seed"] = np.clip(np.ceil((np.arange(1, n_w + 1) / n_w) * 16), 1, 16)
est_seed_w_T1 = elo_w_2026[["TeamID","est_seed"]].rename(columns={"TeamID":"T1_TeamID","est_seed":"T1_seed_est_w"})
est_seed_w_T2 = elo_w_2026[["TeamID","est_seed"]].rename(columns={"TeamID":"T2_TeamID","est_seed":"T2_seed_est_w"})
est_seed_w_T1["Season"] = 2026
est_seed_w_T2["Season"] = 2026
sub = sub.merge(est_seed_w_T1, on=["Season","T1_TeamID"], how="left")
sub = sub.merge(est_seed_w_T2, on=["Season","T2_TeamID"], how="left")

# Fill NaN seeds: use est_seed for 2026, else neutral 8.5
sub["T1_seed"] = sub["T1_seed"].fillna(sub["T1_seed_est"]).fillna(sub["T1_seed_est_w"]).fillna(8.5)
sub["T2_seed"] = sub["T2_seed"].fillna(sub["T2_seed_est"]).fillna(sub["T2_seed_est_w"]).fillna(8.5)
sub["Seed_diff"] = sub["T2_seed"] - sub["T1_seed"]

# ── GENERATE PREDICTIONS ────────────────────────────────────────────────────
preds_all = []
for season_key in use_models:
    dtest   = xgb.DMatrix(sub[use_features].values)
    margins = use_models[season_key].predict(dtest)
    probs   = np.clip(use_spline(np.clip(margins, -use_t, use_t)), 0.025, 0.975)
    preds_all.append(probs)

sub["Pred"] = np.mean(preds_all, axis=0).round(6)

# ── SANITY CHECK ────────────────────────────────────────────────────────────
print("\n  Submission prediction summary:")
print(f"    Rows:   {len(sub):,}")
print(f"    Mean:   {sub.Pred.mean():.4f}  (should be ~0.50 for all matchups)")
print(f"    Std:    {sub.Pred.std():.4f}")
print(f"    Min:    {sub.Pred.min():.4f}")
print(f"    Max:    {sub.Pred.max():.4f}")
near_half = ((sub.Pred > 0.40) & (sub.Pred < 0.60)).sum()
print(f"    In [0.40, 0.60]: {near_half:,} ({100*near_half/len(sub):.1f}%)")

out_path = "submission_2026.csv"
sub[["ID","Pred"]].to_csv(out_path, index=False)
print(f"\n  Saved {len(sub):,} predictions -> {out_path}")

# ────────────────────────────────────────────────────────────
# 11. STAGE 1 SUBMISSION  (2022-2025 — current leaderboard)
#     Left-join model predictions onto the Stage 1 template.
#     Hierarchical fallback ensures every row gets a value.
# ────────────────────────────────────────────────────────────
print("\n[11/11] Generating Stage 1 submission (2022-2025 leaderboard)...")

sub1 = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage1.csv")
sub1["Season"]    = sub1["ID"].str.split("_").str[0].astype(int)
sub1["T1_TeamID"] = sub1["ID"].str.split("_").str[1].astype(int)
sub1["T2_TeamID"] = sub1["ID"].str.split("_").str[2].astype(int)
sub1["men_women"] = sub1["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)

# Merge all computed feature lookups (same pattern as Stage 2)
sub1 = sub1.merge(ss_T1,    on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(ss_T2,    on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(seeds_T1, on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(seeds_T2, on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(mom_T1,   on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(mom_T2,   on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(ord_T1,   on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(ord_T2,   on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(glm_T1,   on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(glm_T2,   on=["Season","T2_TeamID"], how="left")
sub1 = sub1.merge(elos_T1,  on=["Season","T1_TeamID"], how="left")
sub1 = sub1.merge(elos_T2,  on=["Season","T2_TeamID"], how="left")

sub1["Seed_diff"]    = sub1["T2_seed"]    - sub1["T1_seed"]
sub1["elo_diff"]     = sub1["T1_elo"]     - sub1["T2_elo"]
sub1["diff_quality"] = sub1["T1_quality"] - sub1["T2_quality"]
for sys in ORDINAL_SYSTEMS:
    sub1[f"ord_diff_{sys}"] = sub1[f"T2_ord_{sys}"] - sub1[f"T1_ord_{sys}"]

# ── LEVEL 1: Season-specific LOSO model (proper LOO, no leakage) ─────────────
# For each historical season, use the model trained on all OTHER seasons.
sub1["Pred"] = np.nan
for yr in [2022, 2023, 2024, 2025]:
    mask = sub1["Season"] == yr
    if mask.sum() == 0 or yr not in use_models:
        continue
    X_yr = sub1.loc[mask, use_features].values
    margins_yr = use_models[yr].predict(xgb.DMatrix(X_yr))
    probs_yr   = np.clip(use_spline(np.clip(margins_yr, -use_t, use_t)), 0.025, 0.975)
    sub1.loc[mask, "Pred"] = probs_yr

# ── LEVEL 2: Elo-diff sigmoid fallback ────────────────────────────────────────
# Handles rows where the LOSO model returned NaN (e.g., seasons not in use_models).
lvl2_mask = sub1["Pred"].isna()
if lvl2_mask.sum() > 0:
    ed = sub1.loc[lvl2_mask, "elo_diff"].fillna(0)
    sub1.loc[lvl2_mask, "Pred"] = (1 / (1 + np.exp(-ed / 150))).clip(0.025, 0.975)
    print(f"  Level 2 fallback (Elo sigmoid): {lvl2_mask.sum():,} rows")

# ── LEVEL 3: NET-rank difference sigmoid ─────────────────────────────────────
lvl3_mask = sub1["Pred"].isna()
if lvl3_mask.sum() > 0:
    rd = sub1.loc[lvl3_mask, "ord_diff_NET"].fillna(0)   # positive = T2 is ranked worse
    sub1.loc[lvl3_mask, "Pred"] = (1 / (1 + np.exp(-rd / 50))).clip(0.025, 0.975)
    print(f"  Level 3 fallback (NET rank):    {lvl3_mask.sum():,} rows")

# ── LEVEL 4: Seed-difference heuristic ───────────────────────────────────────
lvl4_mask = sub1["Pred"].isna()
if lvl4_mask.sum() > 0:
    sd = sub1.loc[lvl4_mask, "Seed_diff"].fillna(0)       # positive = T2 has higher (worse) seed
    sub1.loc[lvl4_mask, "Pred"] = (1 / (1 + np.exp(-sd * 0.4))).clip(0.025, 0.975)
    print(f"  Level 4 fallback (Seed diff):   {lvl4_mask.sum():,} rows")

# ── LEVEL 5: Global mean ──────────────────────────────────────────────────────
remaining = sub1["Pred"].isna().sum()
if remaining > 0:
    sub1["Pred"] = sub1["Pred"].fillna(0.5)
    print(f"  Level 5 fallback (0.5 global):  {remaining:,} rows")

sub1["Pred"] = sub1["Pred"].clip(0.025, 0.975).round(6)

# ── DISTRIBUTION VALIDATION ───────────────────────────────────────────────────
print("\n  Stage 1 prediction distribution:")
print(f"    Total rows : {len(sub1):,}  (target 519,144)")
print(f"    Mean       : {sub1.Pred.mean():.4f}")
print(f"    Std        : {sub1.Pred.std():.4f}")
print(f"    Min        : {sub1.Pred.min():.4f}")
print(f"    Max        : {sub1.Pred.max():.4f}")

# Verify top-10 Elo men's teams of 2025 show high win probability vs median
print("\n  Top-10 Elo men teams (2025) vs median opponent:")
top_elo_2025 = (
    elos[(elos["Season"] == 2025) & (elos["TeamID"] < 2000)]
    .nlargest(10, "elo")[["TeamID","elo"]]
)
M_teams = pd.read_csv(f"{DATA_DIR}/MTeams.csv")[["TeamID","TeamName"]]
top_elo_2025 = top_elo_2025.merge(M_teams, on="TeamID", how="left")
med_elo = elos[(elos["Season"] == 2025) & (elos["TeamID"] < 2000)]["elo"].median()
for _, row in top_elo_2025.iterrows():
    # sigmoid of elo gap — approximate win% vs median team
    approx_win = 1 / (1 + 10 ** ((med_elo - row["elo"]) / 400))
    print(f"    {row['TeamName']:<20}  Elo={row['elo']:.0f}  vs-median={approx_win:.1%}")

# Line count check
out1_path = "submission_stage1.csv"
sub1[["ID","Pred"]].to_csv(out1_path, index=False)
with open(out1_path, "r") as f:
    line_count = sum(1 for _ in f)
print(f"\n  Saved {len(sub1):,} predictions -> {out1_path}  ({line_count:,} lines incl. header)")
print(f"  NaN remaining: {sub1.Pred.isna().sum()}")

print("\nDone.")
