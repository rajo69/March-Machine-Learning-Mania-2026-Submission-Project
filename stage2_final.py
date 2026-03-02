"""
stage2_final.py -- Regenerate Stage 2 submission with best model + form overrides
==================================================================================
Step 1: Rebuild 2026 features using best config (Ridge Quality + 93 FE features)
Step 2: Apply domain knowledge form overrides for known injuries/form changes
Step 3: Save submission_2026_final.csv (132,133 rows)

Overrides (logit-space adjustments, applied to 2026 rows only):
  Michigan  (1276) : L.J. Cason ACL tear Feb 27  -- key contributor out for season
  Tennessee (1397) : Nate Ament doubtful Feb 28   -- 2nd leading scorer (17.9 ppg)
  Gonzaga   (1211) : Braden Huff out since Jan 8  -- star (17.8 ppg, partial in model)

Run with: uv run python stage2_final.py
Prerequisites: run feature_engineering.py first (needs opt_fe_combined cache)
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

CACHE_DIR    = ".cache"
DATA_DIR     = "./Data"
SEASON_CUTOFF = 2003
WOMEN_MIN    = 2010
T_SPLINE     = 25
CLIP_LO, CLIP_HI = 0.01, 0.99

# ─── Domain knowledge overrides ──────────────────────────────────────────────
# logit-space delta applied when the affected team is a significant factor
# delta < 0 = team is weaker than model thinks
# magnitude derived from: ~1 star player (~18 ppg) ~ 8-10 pt margin impact
#   in logit space: -0.5 ~ 10% weaker from neutral, scaled by confidence
OVERRIDES = {
    1276: {
        "name":   "Michigan",
        "reason": "L.J. Cason ACL tear Feb 27 -- key contributor out for season",
        "delta":  -0.30,   # moderate: key contributor, not primary star
        "confidence": "medium",
    },
    1397: {
        "name":   "Tennessee",
        "reason": "Nate Ament doubtful Mar 2 -- 2nd leading scorer 17.9ppg, lottery pick",
        "delta":  -0.45,   # significant: top-2 scorer, status uncertain
        "confidence": "medium-high",
    },
    1211: {
        "name":   "Gonzaga",
        "reason": "Braden Huff knee injury since Jan 8 -- 17.8ppg 66%FG, may miss tourney",
        "delta":  -0.20,   # small: partially captured in model (injury predates DayNum 93)
        "confidence": "low-medium",
    },
}


def cache_load(name):
    p = os.path.join(CACHE_DIR, name + ".pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


print("=" * 60)
print("STAGE 2 FINAL SUBMISSION")
print("Step 1: Best model (93 feat) | Step 2: Form overrides")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA (same as feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading data...")

M_regular   = pd.read_csv(f"{DATA_DIR}/MRegularSeasonDetailedResults.csv")
M_tourney   = pd.read_csv(f"{DATA_DIR}/MNCAATourneyDetailedResults.csv")
M_seeds     = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
M_ordinals  = pd.read_csv(f"{DATA_DIR}/MMasseyOrdinals.csv")
M_compact   = pd.read_csv(f"{DATA_DIR}/MRegularSeasonCompactResults.csv")
W_regular   = pd.read_csv(f"{DATA_DIR}/WRegularSeasonDetailedResults.csv")
W_tourney   = pd.read_csv(f"{DATA_DIR}/WNCAATourneyDetailedResults.csv")
W_seeds     = pd.read_csv(f"{DATA_DIR}/WNCAATourneySeeds.csv")
W_compact   = pd.read_csv(f"{DATA_DIR}/WRegularSeasonCompactResults.csv")
M_conf      = pd.read_csv(f"{DATA_DIR}/MTeamConferences.csv")
W_conf      = pd.read_csv(f"{DATA_DIR}/WTeamConferences.csv")

regular_results = pd.concat([M_regular, W_regular], ignore_index=True)
tourney_results = pd.concat([M_tourney, W_tourney], ignore_index=True)
seeds_df        = pd.concat([M_seeds,   W_seeds],   ignore_index=True)
compact_all     = pd.concat([M_compact, W_compact],  ignore_index=True)
conf_all        = pd.concat([M_conf,    W_conf],     ignore_index=True)

regular_results = regular_results[regular_results["Season"] >= SEASON_CUTOFF]
seeds_df        = seeds_df[seeds_df["Season"] >= SEASON_CUTOFF]
compact_all     = compact_all[compact_all["Season"] >= SEASON_CUTOFF]

seeds_df["seed"] = seeds_df["Seed"].apply(lambda x: int(x[1:3]))
seeds_T1 = seeds_df[["Season","TeamID","seed"]].rename(columns={"TeamID":"T1_TeamID","seed":"T1_seed"})
seeds_T2 = seeds_df[["Season","TeamID","seed"]].rename(columns={"TeamID":"T2_TeamID","seed":"T2_seed"})

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
print(f"  Regular season rows: {len(regular_data):,}")


# ─────────────────────────────────────────────────────────────────────────────
# BUILD ALL FEATURES (same pipeline as feature_engineering.py)
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding features...")

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

# Elo
def _update_elo(w, l, k=100, width=400):
    exp = 1.0 / (1 + 10 ** ((l - w) / width))
    c = k * (1 - exp)
    return w + c, l - c

elos_list = []
for season in tqdm.tqdm(sorted(regular_data["Season"].unique()), desc="  Elo"):
    ss_s  = regular_data[(regular_data["Season"] == season) & (regular_data["win"] == 1)].reset_index(drop=True)
    teams = set(ss_s["T1_TeamID"]) | set(ss_s["T2_TeamID"])
    elo   = {t: 1000 for t in teams}
    for i in range(len(ss_s)):
        w_id, l_id = ss_s.loc[i,"T1_TeamID"], ss_s.loc[i,"T2_TeamID"]
        elo[w_id], elo[l_id] = _update_elo(elo[w_id], elo[l_id])
    edf = pd.DataFrame.from_dict(elo, orient="index").reset_index()
    edf.columns = ["TeamID","elo"]
    edf["Season"] = season
    elos_list.append(edf)

elos    = pd.concat(elos_list, ignore_index=True)
elos_T1 = elos.rename(columns={"TeamID":"T1_TeamID","elo":"T1_elo"})
elos_T2 = elos.rename(columns={"TeamID":"T2_TeamID","elo":"T2_elo"})

# Ordinals
ORDINAL_SYSTEMS = [s for s in ["POM","NET","MAS","SAG","MOR"]
                   if s in M_ordinals["SystemName"].unique()]
ord_latest = (
    M_ordinals[M_ordinals["SystemName"].isin(ORDINAL_SYSTEMS)]
    .sort_values("RankingDayNum")
    .groupby(["Season","SystemName","TeamID"]).last()
    .reset_index()[["Season","SystemName","TeamID","OrdinalRank"]]
)
ord_pivot  = ord_latest.pivot_table(
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

# Ridge quality
ridge_quality = cache_load("ridge_quality")
assert ridge_quality is not None, "ridge_quality cache missing. Run advanced.py first."
rq_T1 = ridge_quality.rename(columns={"TeamID":"T1_TeamID","quality":"T1_quality"})
rq_T2 = ridge_quality.rename(columns={"TeamID":"T2_TeamID","quality":"T2_quality"})

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

# FE1: Strength of Schedule
rd_with_opp = regular_data[["Season","T1_TeamID","T2_TeamID"]].copy()
rd_with_opp = rd_with_opp.merge(
    ridge_quality[["Season","TeamID","quality"]].rename(
        columns={"TeamID":"T2_TeamID","quality":"opp_quality"}
    ), on=["Season","T2_TeamID"], how="left"
)
sos = rd_with_opp.groupby(["Season","T1_TeamID"])["opp_quality"].mean().reset_index()
sos = sos.rename(columns={"opp_quality":"sos"})
sos_T1 = sos.rename(columns={"T1_TeamID":"T1_TeamID","sos":"T1_sos"})
sos_T2 = sos.rename(columns={"T1_TeamID":"T2_TeamID","sos":"T2_sos"})

# FE2: Late-season delta
DELTA_STATS = ["T1_Score","T1_FGA","T1_OR","T1_DR","PointDiff"]
rd_s10 = rd_sorted[rd_sorted["_rank"] <= 10].groupby(["Season","T1_TeamID"])[DELTA_STATS].mean().reset_index()
ss_base = regular_data.groupby(["Season","T1_TeamID"])[DELTA_STATS].mean().reset_index()
rd_s10_d = rd_s10[["Season","T1_TeamID"] + DELTA_STATS].merge(
    ss_base[["Season","T1_TeamID"] + DELTA_STATS],
    on=["Season","T1_TeamID"], suffixes=("_l10","_full")
)
for col in DELTA_STATS:
    rd_s10_d[f"delta_{col}"] = rd_s10_d[f"{col}_l10"] - rd_s10_d[f"{col}_full"]
delta_cols = [f"delta_{c}" for c in DELTA_STATS]
rd_s10_d = rd_s10_d[["Season","T1_TeamID"] + delta_cols]
delta_T1 = rd_s10_d.rename(columns={
    "T1_TeamID":"T1_TeamID",
    **{f"delta_{c}": f"T1_delta_{c.replace('T1_','').replace('PointDiff','PD')}" for c in DELTA_STATS}
})
delta_T2 = rd_s10_d.rename(columns={
    "T1_TeamID":"T2_TeamID",
    **{f"delta_{c}": f"T2_delta_{c.replace('T1_','').replace('PointDiff','PD')}" for c in DELTA_STATS}
})
FE2_T1_COLS = [f"T1_delta_{c.replace('T1_','').replace('PointDiff','PD')}" for c in DELTA_STATS]
FE2_T2_COLS = [f"T2_delta_{c.replace('T1_','').replace('PointDiff','PD')}" for c in DELTA_STATS]

# FE3: Win% vs top-25
rq_ranked = ridge_quality.copy()
rq_ranked["q_rank"] = rq_ranked.groupby("Season")["quality"].rank(ascending=False, method="first")
top25_set = set(
    (str(row.Season) + "/" + str(row.TeamID))
    for _, row in rq_ranked[rq_ranked["q_rank"] <= 25].iterrows()
)
rd_top25 = regular_data[["Season","T1_TeamID","T2_TeamID","win"]].copy()
rd_top25["key2"] = rd_top25["Season"].astype(str) + "/" + rd_top25["T2_TeamID"].astype(str)
rd_top25["vs_top25"] = rd_top25["key2"].isin(top25_set).astype(int)
rd_vs25 = rd_top25[rd_top25["vs_top25"] == 1].groupby(["Season","T1_TeamID"])["win"].agg(
    ["mean","count"]
).reset_index()
rd_vs25.columns = ["Season","TeamID","win_pct_top25","n_top25_games"]
winpct_T1 = rd_vs25.rename(columns={"TeamID":"T1_TeamID","win_pct_top25":"T1_win_pct_top25","n_top25_games":"T1_n_top25"})
winpct_T2 = rd_vs25.rename(columns={"TeamID":"T2_TeamID","win_pct_top25":"T2_win_pct_top25","n_top25_games":"T2_n_top25"})

# FE4: Neutral site
compact_neutral = compact_all[compact_all["WLoc"] == "N"].copy()
compact_neutral["WMargin"] = compact_neutral["WScore"] - compact_neutral["LScore"]
cn_w = compact_neutral[["Season","WTeamID","WMargin"]].rename(columns={"WTeamID":"TeamID"})
cn_w["won"] = 1; cn_w["margin"] = cn_w["WMargin"]
cn_l = compact_neutral[["Season","LTeamID","WMargin"]].rename(columns={"LTeamID":"TeamID"})
cn_l["won"] = 0; cn_l["margin"] = -cn_l["WMargin"]
cn_all = pd.concat([cn_w[["Season","TeamID","won","margin"]],
                    cn_l[["Season","TeamID","won","margin"]]], ignore_index=True)
neutral_stats = cn_all.groupby(["Season","TeamID"]).agg(
    neutral_winpct=("won","mean"), neutral_margin=("margin","mean")
).reset_index()
neut_T1 = neutral_stats.rename(columns={"TeamID":"T1_TeamID","neutral_winpct":"T1_neutral_winpct","neutral_margin":"T1_neutral_margin"})
neut_T2 = neutral_stats.rename(columns={"TeamID":"T2_TeamID","neutral_winpct":"T2_neutral_winpct","neutral_margin":"T2_neutral_margin"})

# FE5: Conference strength
conf_all_f = conf_all[conf_all["Season"] >= SEASON_CUTOFF].copy()
conf_with_q = conf_all_f.merge(ridge_quality[["Season","TeamID","quality"]], on=["Season","TeamID"], how="left")
conf_strength = conf_with_q.groupby(["Season","ConfAbbrev"])["quality"].mean().reset_index()
conf_strength = conf_strength.rename(columns={"quality":"conf_avg_quality"})
conf_T1_label = conf_all_f.rename(columns={"TeamID":"T1_TeamID","ConfAbbrev":"T1_conf"})
conf_T2_label = conf_all_f.rename(columns={"TeamID":"T2_TeamID","ConfAbbrev":"T2_conf"})

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
LOG_FEATS = [f"log_T{p}_ord_{sys}" for sys in ORDINAL_SYSTEMS for p in ["1","2"]]

FE1_COLS = ["T1_sos","T2_sos","sos_diff"]
FE2_COLS = FE2_T1_COLS + FE2_T2_COLS
FE3_COLS = ["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]
FE4_COLS = ["T1_neutral_winpct","T2_neutral_winpct","neutral_winpct_diff",
            "T1_neutral_margin","T2_neutral_margin","neutral_margin_diff"]
FE5_COLS = ["T1_conf_strength","T2_conf_strength","conf_strength_diff"]

FEAT_COMBINED = FEATURES_BASE + FEATURES_MOM + FEATURES_ORD + LOG_FEATS + FE1_COLS + FE2_COLS + FE3_COLS + FE4_COLS + FE5_COLS
print(f"  Total features: {len(FEAT_COMBINED)}")

# Load combined FE models + calibration spline
combined_cache = cache_load("opt_fe_combined")
assert combined_cache is not None, "opt_fe_combined cache missing. Run feature_engineering.py first."
margins, targets, ss_arr, mw_arr, models = combined_cache
labels = (targets > 0).astype(int)
dat    = sorted(zip(margins, labels), key=lambda x: x[0])
p_, l_ = zip(*dat)
spline = UnivariateSpline(np.clip(p_, -T_SPLINE, T_SPLINE), l_, k=5)
print(f"  Loaded {len(models)} LOSO season models (seasons {min(models)}-{max(models)})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: BUILD STAGE 2 SUBMISSION WITH BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Predicting Stage 2 (2026) with best 93-feature model")
print("=" * 60)

sub2 = pd.read_csv(f"{DATA_DIR}/SampleSubmissionStage2.csv")
sub2["Season"]    = sub2["ID"].str.split("_").str[0].astype(int)
sub2["T1_TeamID"] = sub2["ID"].str.split("_").str[1].astype(int)
sub2["T2_TeamID"] = sub2["ID"].str.split("_").str[2].astype(int)
sub2["men_women"] = sub2["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)
print(f"  Stage 2 rows: {len(sub2):,}")

# Merge all features
sub2 = sub2.merge(ss_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(ss_T2, on=["Season","T2_TeamID"], how="left")
sub2 = sub2.merge(seeds_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(seeds_T2, on=["Season","T2_TeamID"], how="left")
sub2 = sub2.merge(mom_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(mom_T2, on=["Season","T2_TeamID"], how="left")
sub2 = sub2.merge(ord_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(ord_T2, on=["Season","T2_TeamID"], how="left")
sub2 = sub2.merge(rq_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(rq_T2, on=["Season","T2_TeamID"], how="left")
sub2 = sub2.merge(elos_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(elos_T2, on=["Season","T2_TeamID"], how="left")

# FE1
sub2 = sub2.merge(sos_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(sos_T2, on=["Season","T2_TeamID"], how="left")
sub2["sos_diff"] = sub2["T1_sos"] - sub2["T2_sos"]

# FE2
sub2 = sub2.merge(delta_T1, on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(delta_T2, on=["Season","T2_TeamID"], how="left")

# FE3
sub2 = sub2.merge(winpct_T1[["Season","T1_TeamID","T1_win_pct_top25"]], on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(winpct_T2[["Season","T2_TeamID","T2_win_pct_top25"]], on=["Season","T2_TeamID"], how="left")
sub2["win_pct_top25_diff"] = sub2["T1_win_pct_top25"] - sub2["T2_win_pct_top25"]
sub2[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]] = \
    sub2[["T1_win_pct_top25","T2_win_pct_top25","win_pct_top25_diff"]].fillna(0)

# FE4
sub2 = sub2.merge(neut_T1[["Season","T1_TeamID","T1_neutral_winpct","T1_neutral_margin"]], on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(neut_T2[["Season","T2_TeamID","T2_neutral_winpct","T2_neutral_margin"]], on=["Season","T2_TeamID"], how="left")
sub2["neutral_winpct_diff"] = sub2["T1_neutral_winpct"] - sub2["T2_neutral_winpct"]
sub2["neutral_margin_diff"] = sub2["T1_neutral_margin"] - sub2["T2_neutral_margin"]

# FE5
sub2 = sub2.merge(conf_T1_label[["Season","T1_TeamID","T1_conf"]], on=["Season","T1_TeamID"], how="left")
sub2 = sub2.merge(conf_T2_label[["Season","T2_TeamID","T2_conf"]], on=["Season","T2_TeamID"], how="left")
sub2 = sub2.merge(conf_strength.rename(columns={"ConfAbbrev":"T1_conf","conf_avg_quality":"T1_conf_strength"}), on=["Season","T1_conf"], how="left")
sub2 = sub2.merge(conf_strength.rename(columns={"ConfAbbrev":"T2_conf","conf_avg_quality":"T2_conf_strength"}), on=["Season","T2_conf"], how="left")
sub2["conf_strength_diff"] = sub2["T1_conf_strength"] - sub2["T2_conf_strength"]

# Derived features
sub2["Seed_diff"]    = sub2["T2_seed"]    - sub2["T1_seed"]
sub2["elo_diff"]     = sub2["T1_elo"]     - sub2["T2_elo"]
sub2["diff_quality"] = sub2["T1_quality"] - sub2["T2_quality"]
for sys in ORDINAL_SYSTEMS:
    sub2[f"ord_diff_{sys}"] = sub2[f"T2_ord_{sys}"] - sub2[f"T1_ord_{sys}"]
for sys in ORDINAL_SYSTEMS:
    for prefix in ["T1_ord_","T2_ord_"]:
        col = f"{prefix}{sys}"
        sub2[f"log_{col}"] = np.log1p(sub2[col].clip(lower=0)) if col in sub2.columns else np.nan

sub2["Pred"] = np.nan

# Level 1: LOSO model for 2026 (use model trained on all other seasons)
# We use a "full" model — train on all 2003-2025, predict 2026
# Since 2026 is the actual submission year, use ALL season models
# Best proxy: average of the 2022/2023/2024/2025 models (most recent, most relevant)
RECENT_YRS = [2022, 2023, 2024, 2025]
recent_preds = []
for yr in RECENT_YRS:
    if yr in models:
        raw = models[yr].predict(xgb.DMatrix(sub2[FEAT_COMBINED].values))
        recent_preds.append(raw)
if recent_preds:
    avg_raw = np.mean(recent_preds, axis=0)
    sub2["Pred"] = np.clip(spline(np.clip(avg_raw, -T_SPLINE, T_SPLINE)), CLIP_LO, CLIP_HI)
    print(f"  Level 1: ensemble of {len(recent_preds)} recent-year models -> {sub2['Pred'].notna().sum():,} rows predicted")

# Level 2: Elo fallback
lvl2 = sub2["Pred"].isna()
if lvl2.sum() > 0:
    ed = sub2.loc[lvl2, "elo_diff"].fillna(0)
    sub2.loc[lvl2, "Pred"] = (1 / (1 + np.exp(-ed / 150))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 2 (Elo): {lvl2.sum():,} rows")

# Level 3: NET rank
lvl3 = sub2["Pred"].isna()
if lvl3.sum() > 0 and "ord_diff_NET" in sub2.columns:
    rd_ = sub2.loc[lvl3, "ord_diff_NET"].fillna(0)
    sub2.loc[lvl3, "Pred"] = (1 / (1 + np.exp(-rd_ / 50))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 3 (NET): {lvl3.sum():,} rows")

# Level 4: Seed diff
lvl4 = sub2["Pred"].isna()
if lvl4.sum() > 0:
    sd = sub2.loc[lvl4, "Seed_diff"].fillna(0)
    sub2.loc[lvl4, "Pred"] = (1 / (1 + np.exp(-sd * 0.4))).clip(CLIP_LO, CLIP_HI)
    print(f"  Level 4 (Seed): {lvl4.sum():,} rows")

# Level 5: 0.5
remaining = sub2["Pred"].isna().sum()
if remaining > 0:
    sub2["Pred"] = sub2["Pred"].fillna(0.5)
    print(f"  Level 5 (0.5): {remaining:,} rows")

sub2["Pred"] = sub2["Pred"].clip(CLIP_LO, CLIP_HI).round(6)
print(f"\n  Pre-override stats: mean={sub2.Pred.mean():.4f}  std={sub2.Pred.std():.4f}  "
      f"min={sub2.Pred.min():.4f}  max={sub2.Pred.max():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: APPLY FORM OVERRIDES (logit-space adjustment)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Applying form/injury overrides")
print("=" * 60)

override_log = []

for team_id, info in OVERRIDES.items():
    t1_mask = sub2["T1_TeamID"] == team_id
    t2_mask = sub2["T2_TeamID"] == team_id
    delta   = info["delta"]

    # When affected team is T1 (Pred = P(T1 wins)): delta < 0 lowers P(T1 wins)
    if t1_mask.sum() > 0:
        p_before = sub2.loc[t1_mask, "Pred"].values.copy()
        p_after  = np.clip(sigmoid(logit(p_before) + delta), CLIP_LO, CLIP_HI)
        sub2.loc[t1_mask, "Pred"] = p_after
        avg_shift_t1 = (p_after - p_before).mean()
        print(f"  {info['name']} as T1 ({t1_mask.sum()} rows): "
              f"mean pred {p_before.mean():.4f} -> {p_after.mean():.4f}  "
              f"(avg shift {avg_shift_t1:+.4f})")

    # When affected team is T2 (Pred = P(T1 wins)): delta < 0 raises P(T1 wins)
    if t2_mask.sum() > 0:
        p_before = sub2.loc[t2_mask, "Pred"].values.copy()
        p_after  = np.clip(sigmoid(logit(p_before) - delta), CLIP_LO, CLIP_HI)
        sub2.loc[t2_mask, "Pred"] = p_after
        avg_shift_t2 = (p_after - p_before).mean()
        print(f"  {info['name']} as T2 ({t2_mask.sum()} rows): "
              f"mean pred {p_before.mean():.4f} -> {p_after.mean():.4f}  "
              f"(avg shift {avg_shift_t2:+.4f})")

    override_log.append({
        "team_id": team_id,
        "team_name": info["name"],
        "reason": info["reason"],
        "delta_logit": delta,
        "confidence": info["confidence"],
        "rows_t1": int(t1_mask.sum()),
        "rows_t2": int(t2_mask.sum()),
    })

sub2["Pred"] = sub2["Pred"].clip(CLIP_LO, CLIP_HI).round(6)

print(f"\n  Post-override stats: mean={sub2.Pred.mean():.4f}  std={sub2.Pred.std():.4f}  "
      f"min={sub2.Pred.min():.4f}  max={sub2.Pred.max():.4f}")
print(f"  NaN: {sub2.Pred.isna().sum()}")
print(f"  Rows: {len(sub2):,}  (target: 132,133)")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
sub2[["ID","Pred"]].to_csv("submission_2026_final.csv", index=False)
print(f"\n  Saved submission_2026_final.csv")

# Also update submission_2026.csv (the active Stage 2 submission)
sub2[["ID","Pred"]].to_csv("submission_2026.csv", index=False)
print(f"  Updated submission_2026.csv")

# Save override log
with open("Optimization/form_overrides.json", "w") as f:
    json.dump({
        "date_applied": "2026-03-02",
        "base_model": "Ridge Quality + 93 FE features (opt_fe_combined)",
        "overrides": override_log,
        "note": "logit-space delta: negative = team weaker than model expects",
    }, f, indent=2)
print(f"  Saved Optimization/form_overrides.json")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Stage 2 submission: best model (93 feat) + {len(OVERRIDES)} form overrides")
print(f"  Teams adjusted: " + ", ".join(f"{v['name']} ({v['delta']:+.2f} logit)" for v in OVERRIDES.values()))
print(f"  Note: OOF Brier unchanged (Stage 1 is 2022-25 historical, no 2026 rows)")
print(f"  Submit: submission_2026.csv for Stage 2 when Kaggle opens it")
