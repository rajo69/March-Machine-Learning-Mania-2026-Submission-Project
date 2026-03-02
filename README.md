# March Machine Learning Mania 2026

**Kaggle Competition:** [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
**Current Leaderboard:** Rank ~763 | Public Score: 0.15583 (Stage 1)
**Best OOF Brier Score:** 0.15978 (Stage 1, 2022–2025 historical CV)
**Author:** rajo1909

---

## Competition Overview

Predict the outcome probabilities for every possible NCAA Men's and Women's basketball tournament matchup. Submissions are evaluated using **Brier Score** (lower is better).

- **Stage 1:** Predict all possible matchups for seasons 2022–2025 (historical validation)
- **Stage 2:** Predict all possible matchups for the 2026 NCAA Tournament (live scoring)
- **Selection Sunday:** March 15, 2026 — seeds and bracket released

---

## Pipeline Architecture

The full pipeline is an **XGBoost regression ensemble** predicting point differential (margin), calibrated to win probabilities via a spline.

```
Raw game data
    │
    ├── Season averages (box score)
    ├── Elo ratings (updated game-by-game)
    ├── Ridge Quality scores (L2-regularized team strength)
    ├── Massey Ordinals (POM, NET, MAS, SAG, MOR)
    ├── Last-20-game momentum features
    ├── Log-transformed ordinal features
    ├── Strength of Schedule
    ├── Late-season form delta (last-10 vs season avg)
    ├── Win% vs top-25 quality opponents
    ├── Neutral-site performance (win rate + margin)
    └── Conference strength (avg Ridge quality)
          │
          ▼
    XGBoost (reg:squarederror on PointDiff)
    Leave-One-Season-Out CV (one model per held-out season)
          │
          ▼
    UnivariateSpline calibration (k=5, margins → probabilities)
          │
          ▼
    Clip [0.01, 0.99]
          │
          ▼
    Hierarchical fallback for non-tournament pairs:
      L1: Season LOSO model
      L2: Elo sigmoid
      L3: NET rank sigmoid
      L4: Seed diff heuristic
      L5: 0.5
```

---

## Brier Score Progression

| Config | Stage1 Brier | Delta |
|--------|-------------|-------|
| Baseline (29 feat, GLM quality, clip=0.025) | 0.16259 | — |
| Innovation (58 feat: +momentum +ordinals) | 0.16238 | −0.00021 |
| Golden Config (68 feat: +log-ordinals, clip=0.01, no boost) | 0.16213 | −0.00025 |
| + Ridge Quality (sklearn Ridge alpha=1.0 vs OLS) | 0.16143 | −0.00070 |
| **+ Feature Engineering (93 feat, all 5 additions)** | **0.15978** | **−0.00165** |

Total improvement over baseline: **−0.00281**

---

## Key Innovations

### 1. Ridge Quality Scores (biggest single gain: −0.00070)
Replaced the original OLS/GLM team quality scores with **sklearn Ridge(alpha=1.0)**. The L2 penalty prevents extreme quality scores for teams with sparse schedules, producing better-regularized strength estimates.

```python
ridge = Ridge(alpha=1.0, fit_intercept=False)
ridge.fit(X_sparse, PointDiff)   # sparse design: T1 block + T2 block
team_quality = ridge.coef_[:n_teams]
```

### 2. Feature Engineering (combined gain: −0.00165)
Five new feature sets, all validated via LOSO CV:

| Feature | Brier Delta | Description |
|---------|-------------|-------------|
| Neutral-site performance | −0.00117 | Win rate + margin on neutral court |
| Conference strength | −0.00108 | Avg Ridge quality of conference |
| Strength of Schedule | −0.00033 | Avg opponent quality |
| Late-season form delta | −0.00025 | Last-10 stats minus season avg |
| Win% vs top-25 | −0.00006 | Quality wins metric |

### 3. Log-Ordinal Features (−0.00025)
Log-transforming Massey Ordinal ranks `np.log1p(rank)` compresses the large-rank tail and improves feature utility for lower-ranked teams.

### 4. Momentum Window (−0.00021 vs default)
Last-20-game rolling averages outperform last-5, last-10, and weighted-decay windows.

### 5. Calibration Improvements
- Remove the original +10% underdog boost (it *hurts* Brier by +0.00161)
- Use clip=[0.01, 0.99] instead of [0.025, 0.975]

---

## Experiments Conducted

All experiment results are saved in the `Optimization/` folder.

| File | Contents |
|------|----------|
| `feature_importance.csv` | XGBoost gain/weight for all 68 features |
| `momentum_test_results.json` | Brier scores for 4 momentum window configs |
| `ablation_study.txt` | Underdog boost vs raw spline comparison |
| `calibration_tuning.csv` | Grid search over 7 clip configurations |
| `fallback_audit.csv` | Sample of 10 non-tournament predictions |
| `winning_config.json` | Recommended config from optimize.py |
| `gender_split_metrics.json` | Option A: gender-split model (rejected) |
| `ridge_quality_vs_glm.csv` | Per-season Ridge vs GLM comparison |
| `feature_engineering_results.json` | All 5 FE experiments + combined result |
| `seed_override_grid.csv` | Seed-matchup calibration grid search (no improvement) |
| `form_overrides.json` | 2026 domain knowledge overrides applied to Stage 2 |

### Experiments That Did NOT Help
- **Gender-split model** (separate men/women models): Brier 0.16596 vs 0.16213 — smaller training sets hurt more than the gender-specific signal helps
- **Seed-matchup calibration** (Bayesian blend of historical seed win rates): All 35 configurations worsened Brier — the 93-feature model already captures seed matchup patterns better than historical averages alone

---

## Stage 2: Domain Knowledge Overrides

The Stage 2 submission (`submission_2026.csv`) incorporates current injury/form information unavailable in the training data (cutoff: ~Feb 4, 2026):

| Team | Reason | Adjustment |
|------|--------|-----------|
| **Michigan** (1276) | L.J. Cason ACL tear Feb 27 — key contributor out for season | −0.30 logit |
| **Tennessee** (1397) | Nate Ament doubtful Mar 2 — 2nd leading scorer (17.9 ppg), lottery pick | −0.45 logit |
| **Gonzaga** (1211) | Braden Huff knee injury since Jan 8 — star (17.8 ppg, 66% FG) | −0.20 logit |

Adjustments applied in logit space: `new_prob = sigmoid(logit(base_prob) + delta)`

---

## Submission Files

| File | Description | Rows |
|------|-------------|------|
| `submission_stage1.csv` | Stage 1 — best model (93 feat + Ridge Quality) | 519,144 |
| `submission_2026.csv` | Stage 2 — best model + 2026 form overrides | 132,133 |
| `submission_2026_final.csv` | Stage 2 backup copy | 132,133 |

> **Note:** `submission_2026_advanced.csv` is an older Stage 1 file using only the Ridge Quality config (no FE). Superseded by `submission_stage1.csv`.

---

## Scripts Reference

| Script | Purpose | Run Order |
|--------|---------|-----------|
| `solution_2026.py` | Original baseline + innovation pipeline | 1st |
| `optimize.py` | 5 optimization experiments (momentum, calibration, etc.) | 2nd |
| `advanced.py` | Golden Config + Gender Split + Ridge Quality | 3rd |
| `feature_engineering.py` | 5 FE experiments + regenerate Stage 1 submission | 4th |
| `seed_overrides.py` | Seed-matchup calibration (validation — no improvement) | Optional |
| `stage2_final.py` | Stage 2 submission: best model + form overrides | 5th |
| `regen_best_submission.py` | Regenerate Stage 1 from best config | Utility |
| `update_seeds.py` | Refresh seed features when seeds drop March 15 | Post-Mar 15 |
| `solution_2026.ipynb` | Notebook version of solution_2026.py | Reference |

---

## Reproducing Results

### Prerequisites
```bash
# Install uv (package manager)
pip install uv

# Create virtual environment and install dependencies
uv venv
uv pip install xgboost scikit-learn scipy pandas numpy tqdm statsmodels
```

### Data Setup
Download competition data from Kaggle and place CSVs in `Data/`:
```
Data/
├── MRegularSeasonDetailedResults.csv
├── WRegularSeasonDetailedResults.csv
├── MNCAATourneyDetailedResults.csv
├── WNCAATourneyDetailedResults.csv
├── MNCAATourneySeeds.csv
├── WNCAATourneySeeds.csv
├── MMasseyOrdinals.csv
├── MTeamConferences.csv
├── WTeamConferences.csv
├── MRegularSeasonCompactResults.csv
├── WRegularSeasonCompactResults.csv
├── SampleSubmissionStage1.csv
└── SampleSubmissionStage2.csv
```

### Run Pipeline (in order)
```bash
uv run python solution_2026.py        # ~10 min (builds base caches)
uv run python optimize.py             # ~15 min (5 optimization experiments)
uv run python advanced.py             # ~20 min (Ridge Quality)
uv run python feature_engineering.py # ~25 min (5 FE experiments)
uv run python stage2_final.py        # ~3 min  (Stage 2 submission)
```

Results are cached in `.cache/` — subsequent runs are near-instant.

### After March 15 (seeds released)
```bash
uv run python update_seeds.py         # refresh seed features
uv run python feature_engineering.py # regenerate Stage 1 with real seeds
uv run python stage2_final.py        # regenerate Stage 2 with real seeds + overrides
```

---

## XGBoost Configuration

```python
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
```

---

## Top Feature Importances (by XGBoost gain)

| Rank | Feature | Gain | Category |
|------|---------|------|----------|
| 1 | Seed_diff | 30,475 | Seed |
| 2 | T2_seed | 11,298 | Seed |
| 3 | T1_seed | 11,204 | Seed |
| 4 | elo_diff | 5,476 | Elo |
| 5 | T1_elo | ~3,800 | Elo |
| 6 | T2_quality | ~3,200 | Quality |
| 7 | T1_quality | ~3,100 | Quality |
| 8–12 | Ordinal systems | ~1,672 avg | Ordinals |

> Seeds are unavailable until March 15. The model falls back to Elo + ordinals for 2026 predictions until then.

---

## Project Structure

```
.
├── solution_2026.py              # Core pipeline
├── optimize.py                   # Optimization experiments
├── advanced.py                   # Ridge Quality + architecture experiments
├── feature_engineering.py        # 5 FE experiments (BEST MODEL)
├── seed_overrides.py             # Seed calibration validation
├── stage2_final.py               # Stage 2 final submission
├── regen_best_submission.py      # Stage 1 regeneration utility
├── update_seeds.py               # Post-March-15 seed refresh
├── solution_2026.ipynb           # Notebook version
├── CLAUDE.md                     # Claude Code guidance
├── REPLICATION_PLAN.md           # Detailed analysis of 2025 winner approach
├── Optimization/                 # All experiment results (JSON/CSV)
│   ├── feature_engineering_results.json
│   ├── ridge_quality_vs_glm.csv
│   ├── momentum_test_results.json
│   ├── form_overrides.json
│   └── ... (10 files total)
├── submission_stage1.csv         # Stage 1 submission (best)
├── submission_2026.csv           # Stage 2 submission (best + overrides)
└── Data/                         # Kaggle competition CSVs (not tracked)
```

---

## Next Steps (Post-March 15)

1. **Refresh seeds** — run `update_seeds.py` after Selection Sunday
2. **Re-run Stage 1** — regenerate `submission_stage1.csv` with real 2026 seeds (expected large improvement — seed_diff is the #1 feature with 30,475 gain)
3. **Re-run Stage 2** — regenerate `submission_2026.csv` with real seeds + updated form overrides
4. **Targeted bracket overrides** — with known matchups, apply expert knowledge for 5–8 specific upset picks
5. **Monitor injuries** — especially Tennessee (Ament status) and Gonzaga (Huff recovery)

---

## Key Lessons Learned

- **Ridge vs OLS for team quality** is the single biggest algorithmic improvement (+0.00070)
- **More data beats gender-specific models** — splitting by gender reduces training set size too much to overcome the marginal benefit of gender-specific patterns
- **Seed-matchup historical rates are already captured** by the model — trying to override with raw historical rates hurts performance
- **Neutral-site and conference strength** are surprisingly strong features (+0.00117, +0.00108) — tournament games are played on neutral courts, so neutral-site history is directly relevant
- **The underdog boost hurts** — counter-intuitive but the raw spline calibration is already optimal; post-processing adjustments based on intuition (not data) degrade performance
