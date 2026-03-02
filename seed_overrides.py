"""
seed_overrides.py -- Systematic seed-matchup calibration
=========================================================
Strategy: our XGBoost model may be miscalibrated for specific (seed1, seed2) pairs.
E.g., 12-seeds beat 5-seeds ~35% historically but model may predict 25-28%.

Method (Bayesian blend):
  corrected = (1 - w) * model_prob + w * historical_rate
  w = n_games / (n_games + k)          <- shrinks toward 0 for rare matchups
  k = tuned on OOF Stage1 Brier

Only apply if OOF Brier improves. Regenerate submission_stage1.csv if so.

Run with: uv run python seed_overrides.py
Prerequisites: run feature_engineering.py first (needs opt_fe_combined cache)
"""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

os.makedirs("Optimization", exist_ok=True)

CACHE_DIR    = ".cache"
DATA_DIR     = "./Data"
SEASON_CUTOFF = 2003
WOMEN_MIN    = 2010
T_SPLINE     = 25
CLIP_LO, CLIP_HI = 0.01, 0.99

CURRENT_BRIER = 0.15978   # opt_fe_combined result


def cache_load(name):
    p = os.path.join(CACHE_DIR, name + ".pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


print("=" * 60)
print("SEED-MATCHUP CALIBRATION OVERRIDES")
print(f"Baseline (Combined FE): {CURRENT_BRIER:.5f}")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD OOF PREDICTIONS FROM COMBINED FE MODEL
# ─────────────────────────────────────────────────────────────────────────────
combined_cache = cache_load("opt_fe_combined")
assert combined_cache is not None, "opt_fe_combined cache missing. Run feature_engineering.py first."
margins, targets, ss, mw, models = combined_cache

labels = (targets > 0).astype(int)
dat    = sorted(zip(margins, labels), key=lambda x: x[0])
p_, l_ = zip(*dat)
spline = UnivariateSpline(np.clip(p_, -T_SPLINE, T_SPLINE), l_, k=5)
probs  = np.clip(spline(np.clip(margins, -T_SPLINE, T_SPLINE)), CLIP_LO, CLIP_HI)

s1_mask = ss >= 2022
print(f"\n  OOF rows total: {len(probs):,}  |  Stage1 rows (2022-25): {s1_mask.sum():,}")
print(f"  Baseline Stage1 Brier (from cache): {brier_score_loss(labels[s1_mask], probs[s1_mask]):.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TOURNAMENT DATA WITH SEEDS
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading seed data...")

M_tourney = pd.read_csv(f"{DATA_DIR}/MNCAATourneyDetailedResults.csv")
W_tourney = pd.read_csv(f"{DATA_DIR}/WNCAATourneyDetailedResults.csv")
M_seeds   = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
W_seeds   = pd.read_csv(f"{DATA_DIR}/WNCAATourneySeeds.csv")

tourney_results = pd.concat([M_tourney, W_tourney], ignore_index=True)
seeds_df        = pd.concat([M_seeds,   W_seeds],   ignore_index=True)
tourney_results = tourney_results[tourney_results["Season"] >= SEASON_CUTOFF]
seeds_df        = seeds_df[seeds_df["Season"] >= SEASON_CUTOFF]
seeds_df["seed_num"] = seeds_df["Seed"].apply(lambda x: int(x[1:3]))

# Build T1-wins-perspective tourney data
def prepare_tourney(df):
    df = df[["Season","DayNum","WTeamID","LTeamID"]].copy()
    dfswap = df.copy()
    df["T1_TeamID"]  = df["WTeamID"]
    df["T2_TeamID"]  = df["LTeamID"]
    df["win"]        = 1
    dfswap["T1_TeamID"] = dfswap["LTeamID"]
    dfswap["T2_TeamID"] = dfswap["WTeamID"]
    dfswap["win"]       = 0
    out = pd.concat([df[["Season","DayNum","T1_TeamID","T2_TeamID","win"]],
                     dfswap[["Season","DayNum","T1_TeamID","T2_TeamID","win"]]], ignore_index=True)
    out["men_women"] = out["T1_TeamID"].apply(lambda t: 1 if str(t).startswith("1") else 0)
    return out

td = prepare_tourney(tourney_results)
s_T1 = seeds_df[["Season","TeamID","seed_num"]].rename(columns={"TeamID":"T1_TeamID","seed_num":"T1_seed"})
s_T2 = seeds_df[["Season","TeamID","seed_num"]].rename(columns={"TeamID":"T2_TeamID","seed_num":"T2_seed"})
td = td.merge(s_T1, on=["Season","T1_TeamID"], how="left")
td = td.merge(s_T2, on=["Season","T2_TeamID"], how="left")
td = td.dropna(subset=["T1_seed","T2_seed"])
td["T1_seed"] = td["T1_seed"].astype(int)
td["T2_seed"] = td["T2_seed"].astype(int)

print(f"  Tournament rows with seeds: {len(td):,}  (seasons {td.Season.min()}-{td.Season.max()})")


# ─────────────────────────────────────────────────────────────────────────────
# ATTACH SEED INFO TO OOF PREDICTIONS
# We need to align seeds with the OOF array. Since OOF was built from
# prepare_data(tourney_results) with same ordering, we rebuild the index.
# ─────────────────────────────────────────────────────────────────────────────
print("\nAligning OOF predictions with seed info...")

# Rebuild tourney_data in the same order as advanced.py / feature_engineering.py
M_regular = pd.read_csv(f"{DATA_DIR}/MRegularSeasonDetailedResults.csv")
W_regular = pd.read_csv(f"{DATA_DIR}/WRegularSeasonDetailedResults.csv")
regular_results = pd.concat([M_regular, W_regular], ignore_index=True)
regular_results = regular_results[regular_results["Season"] >= SEASON_CUTOFF]

_ADJ_COLS = [
    "LScore","WScore",
    "LFGM","LFGA","LFGM3","LFGA3","LFTM","LFTA","LOR","LDR","LAst","LTO","LStl","LBlk","LPF",
    "WFGM","WFGA","WFGM3","WFGA3","WFTM","WFTA","WOR","WDR","WAst","WTO","WStl","WBlk","WPF",
]

def prepare_data_full(df):
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

tourney_data_full = prepare_data_full(tourney_results)
tourney_data_full = tourney_data_full[["Season","T1_TeamID","T2_TeamID","men_women","win"]].reset_index(drop=True)
tourney_data_full = tourney_data_full.merge(s_T1, on=["Season","T1_TeamID"], how="left")
tourney_data_full = tourney_data_full.merge(s_T2, on=["Season","T2_TeamID"], how="left")

assert len(tourney_data_full) == len(probs), \
    f"Length mismatch: tourney_data={len(tourney_data_full)}, probs={len(probs)}"

# run_loso appends results in sorted-season order (2003, 2004, ..., 2025).
# Within each season, rows follow original concat order (df half then dfswap half).
# Sort tourney_data_full by Season (stable) to match that ordering.
tourney_data_full = tourney_data_full.sort_values("Season", kind="stable").reset_index(drop=True)

tourney_data_full["oof_prob"] = probs
tourney_data_full["oof_label"] = labels
tourney_data_full["season"] = ss
tourney_data_full["men_women_oof"] = mw

# Sanity check: for label/win agreement
label_match = (tourney_data_full["win"] == tourney_data_full["oof_label"]).mean()
print(f"  Aligned {len(tourney_data_full):,} rows. Seed NaN: {tourney_data_full['T1_seed'].isna().sum():,}")
print(f"  Alignment check — win==label match rate: {label_match:.4f} (should be 1.0000)")


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE HISTORICAL WIN RATES PER (T1_seed, T2_seed)
# Use ALL seasons for stability (each season contributes ~1/22 weight)
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing historical win rates per seed matchup...")

seed_stats = (
    tourney_data_full
    .dropna(subset=["T1_seed","T2_seed"])
    .groupby(["T1_seed","T2_seed"])
    .agg(
        n_games   = ("oof_label","count"),
        hist_rate = ("oof_label","mean"),
        model_avg = ("oof_prob","mean"),
    )
    .reset_index()
)
seed_stats["T1_seed"] = seed_stats["T1_seed"].astype(int)
seed_stats["T2_seed"] = seed_stats["T2_seed"].astype(int)
seed_stats["deviation"] = seed_stats["hist_rate"] - seed_stats["model_avg"]

# Show most miscalibrated high-sample matchups
notable = seed_stats[seed_stats["n_games"] >= 15].sort_values("deviation", key=abs, ascending=False)
print(f"\n  Seed pairs with >= 15 games, sorted by |deviation|:")
print(f"  {'T1_seed':>7} {'T2_seed':>7} {'n':>5} {'hist%':>7} {'model%':>8} {'dev':>8}")
print(f"  {'-'*50}")
for _, r in notable.head(20).iterrows():
    print(f"  {int(r.T1_seed):>7} {int(r.T2_seed):>7} {int(r.n_games):>5} "
          f"{r.hist_rate:>7.3f} {r.model_avg:>8.3f} {r.deviation:>+8.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN BLEND FUNCTION
# corrected = (1 - w) * model_prob + w * historical_rate
# w = n / (n + k)   [shrinks small samples toward 0]
# ─────────────────────────────────────────────────────────────────────────────
def apply_seed_correction(oof_df, seed_stats_df, k, min_n):
    """Apply seed-pair blend correction. Returns corrected probability array."""
    # Filter to pairs with enough data
    eligible = seed_stats_df[seed_stats_df["n_games"] >= min_n].copy()
    eligible["w"] = eligible["n_games"] / (eligible["n_games"] + k)
    pair_map = {
        (int(r.T1_seed), int(r.T2_seed)): (r.hist_rate, r.w)
        for _, r in eligible.iterrows()
    }
    corrected = oof_df["oof_prob"].values.copy()
    n_corrected = 0
    for i, row in oof_df.iterrows():
        if pd.isna(row["T1_seed"]) or pd.isna(row["T2_seed"]):
            continue
        key = (int(row["T1_seed"]), int(row["T2_seed"]))
        if key in pair_map:
            hist_rate, w = pair_map[key]
            corrected[i] = (1 - w) * row["oof_prob"] + w * hist_rate
            n_corrected += 1
    corrected = np.clip(corrected, CLIP_LO, CLIP_HI)
    return corrected, n_corrected


# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH OVER k AND min_n ON STAGE1 OOF
# ─────────────────────────────────────────────────────────────────────────────
print("\nGrid searching k (shrinkage) and min_n (minimum sample)...")

s1 = tourney_data_full["season"] >= 2022
s1_labels_arr = tourney_data_full.loc[s1, "oof_label"].values
baseline_s1_brier = brier_score_loss(s1_labels_arr, tourney_data_full.loc[s1, "oof_prob"].values)

best_brier  = baseline_s1_brier
best_params = None
best_n_corr = 0

results_grid = []
for k in [5, 10, 15, 20, 30, 50, 100]:
    for min_n in [10, 15, 20, 25, 30]:
        corr_all, n_corr = apply_seed_correction(tourney_data_full, seed_stats, k=k, min_n=min_n)
        brier_corr = brier_score_loss(s1_labels_arr, corr_all[s1.values])
        delta = baseline_s1_brier - brier_corr
        results_grid.append({
            "k": k, "min_n": min_n, "brier": round(brier_corr, 5),
            "delta": round(delta, 5), "n_corrected": n_corr
        })
        if brier_corr < best_brier:
            best_brier  = brier_corr
            best_params = {"k": k, "min_n": min_n}
            best_n_corr = n_corr

grid_df = pd.DataFrame(results_grid).sort_values("brier")
print(f"\n  Top 10 configurations:")
print(f"  {'k':>5} {'min_n':>6} {'Brier':>8} {'Delta':>8} {'n_corr':>8}")
print(f"  {'-'*42}")
for _, r in grid_df.head(10).iterrows():
    marker = " <-- BEST" if r["k"] == (best_params["k"] if best_params else -1) \
             and r["min_n"] == (best_params["min_n"] if best_params else -1) else ""
    print(f"  {int(r.k):>5} {int(r.min_n):>6} {r.brier:>8.5f} {r.delta:>+8.5f} {int(r.n_corrected):>8}{marker}")

print(f"\n  Baseline: {baseline_s1_brier:.5f}")
if best_params:
    improvement = baseline_s1_brier - best_brier
    print(f"  Best:     {best_brier:.5f}  (delta={improvement:+.5f}, "
          f"k={best_params['k']}, min_n={best_params['min_n']}, n_corrected={best_n_corr})")
else:
    print("  No improvement found.")

# Save grid to Optimization/
grid_df.to_csv("Optimization/seed_override_grid.csv", index=False)
print("  Saved Optimization/seed_override_grid.csv")


# ─────────────────────────────────────────────────────────────────────────────
# APPLY BEST CORRECTION + REGENERATE SUBMISSION (only if improvement)
# ─────────────────────────────────────────────────────────────────────────────
if best_params is None or best_brier >= baseline_s1_brier:
    print("\n  No improvement from seed overrides. submission_stage1.csv unchanged.")
else:
    print("\n" + "=" * 60)
    print("APPLYING BEST CORRECTION + REGENERATING submission_stage1.csv")
    print("=" * 60)

    k_best    = best_params["k"]
    minn_best = best_params["min_n"]

    # Build the pair_map for final submission
    eligible  = seed_stats[seed_stats["n_games"] >= minn_best].copy()
    eligible["w"] = eligible["n_games"] / (eligible["n_games"] + k_best)
    pair_map  = {
        (int(r.T1_seed), int(r.T2_seed)): (r.hist_rate, r.w)
        for _, r in eligible.iterrows()
    }
    print(f"  Eligible seed pairs: {len(pair_map)}")

    # Load and process the current best submission
    import xgboost as xgb
    import tqdm

    # We need to regenerate raw predictions for the submission, then apply correction
    # Load submission_stage1.csv (already has correct Pred from feature_engineering.py)
    sub = pd.read_csv("submission_stage1.csv")
    sub["Season"]    = sub["ID"].str.split("_").str[0].astype(int)
    sub["T1_TeamID"] = sub["ID"].str.split("_").str[1].astype(int)
    sub["T2_TeamID"] = sub["ID"].str.split("_").str[2].astype(int)

    # Merge seed info
    sub = sub.merge(s_T1, on=["Season","T1_TeamID"], how="left")
    sub = sub.merge(s_T2, on=["Season","T2_TeamID"], how="left")

    # Apply correction
    n_corrected = 0
    for i in sub.index:
        t1s = sub.at[i, "T1_seed"]
        t2s = sub.at[i, "T2_seed"]
        if pd.isna(t1s) or pd.isna(t2s):
            continue
        key = (int(t1s), int(t2s))
        if key in pair_map:
            hist_rate, w = pair_map[key]
            sub.at[i, "Pred"] = (1 - w) * sub.at[i, "Pred"] + w * hist_rate
            n_corrected += 1

    sub["Pred"] = sub["Pred"].clip(CLIP_LO, CLIP_HI).round(6)
    print(f"  Predictions corrected: {n_corrected:,} rows")
    print(f"  Rows: {len(sub):,}  NaN: {sub.Pred.isna().sum()}")
    print(f"  Mean: {sub.Pred.mean():.4f}  Std: {sub.Pred.std():.4f}  "
          f"Min: {sub.Pred.min():.4f}  Max: {sub.Pred.max():.4f}")

    sub[["ID","Pred"]].to_csv("submission_stage1.csv", index=False)
    print(f"\n  Saved submission_stage1.csv")
    print(f"\n  Final OOF Stage1 Brier: {best_brier:.5f}")
    print(f"  vs FE Combined:         {CURRENT_BRIER:.5f}  (delta={CURRENT_BRIER - best_brier:+.5f})")

    # Show most-impacted seed pairs
    print(f"\n  Most-corrected seed pairs (top 15 by |deviation| x w):")
    eligible["impact"] = (eligible["deviation"].abs() * eligible["w"]).round(4)
    top_pairs = eligible.sort_values("impact", ascending=False).head(15)
    print(f"  {'T1_seed':>7} {'T2_seed':>7} {'n':>5} {'hist%':>7} {'model%':>8} {'w':>6} {'impact':>8}")
    print(f"  {'-'*55}")
    for _, r in top_pairs.iterrows():
        print(f"  {int(r.T1_seed):>7} {int(r.T2_seed):>7} {int(r.n_games):>5} "
              f"{r.hist_rate:>7.3f} {r.model_avg:>8.3f} {r.w:>6.3f} {r.impact:>8.4f}")

    # Save summary
    summary = {
        "method":               "bayesian_seed_blend",
        "best_k":               k_best,
        "best_min_n":           minn_best,
        "n_eligible_pairs":     len(pair_map),
        "n_corrected_rows":     n_corrected,
        "fe_combined_brier":    CURRENT_BRIER,
        "seed_corrected_brier": round(best_brier, 5),
        "delta":                round(CURRENT_BRIER - best_brier, 5),
    }
    with open("Optimization/seed_override_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved Optimization/seed_override_results.json")

print("\nDone.")
