# REPLICATION_PLAN.md — March Machine Learning Mania 2026

---

## Phase 1: Winning DNA Summary (2025 1st Place)

### Architecture at a Glance
Single XGBoost model (no ensemble), trained to predict **point margin** (PointDiff), calibrated to probabilities via a fitted spline. Leave-one-season-out CV for validation.

### Feature Engineering (29 features, 4 tiers)

| Tier | Features | Method |
|------|----------|--------|
| Easy | `men_women`, `T1_seed`, `T2_seed`, `Seed_diff` | Parse seed string `"W01"→1` |
| Medium | `T1/T2_avg_{Score,FGA,OR,DR,Blk,PF,opponent_FGA,opponent_Blk,opponent_PF,PointDiff}` | Season-level regular-season means |
| Hard | `T1_elo`, `T2_elo`, `elo_diff` | Season-reset Elo (k=100, width=400) |
| Hardest | `T1_quality`, `T2_quality` | GLM fixed-effects on PointDiff (OLS, statsmodels) |

**Non-obvious transformations:**
- **OT adjustment** (Cell 12): all box-score stats divided by `(40 + 5*NumOT)/40` to remove overtime inflation before averaging.
- **Dataset doubling** (Cell 12): every game stored twice — once as (Winner→T1, Loser→T2) and once swapped. This removes ordering bias; training target is signed PointDiff.
- **Opponent quality capture** (Cell 24): season averages are computed from the perspective of T1 *in all T1's games*, so `T1_avg_opponent_FGA` = average FGA of everyone T1 faced — a strength-of-schedule proxy.
- **GLM quality score** (Cell 31): OLS regression of `PointDiff ~ T1_TeamID + T2_TeamID - 1` across all regular-season games involving tourney-connected teams. Each team's coefficient is its "quality" controlling for opponent strength. Non-relevant teams are collapsed to dummy "0000". Separate regressions per `(season, gender)`.

### Model Config
```python
param = {
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
num_rounds = 704
```

### Validation
Leave-one-season-out: train on all seasons except `S`, validate on `S`. Produces N separate models (one per historical season). Final submission = mean of all N model predictions.

### Calibration
`UnivariateSpline(k=5)` fit on (OOF predicted margin, actual win binary). Converts raw XGBoost margin predictions to probabilities. Clipped to `[0.01, 0.99]`.

### Post-processing
Step 1: `pred += pred * 0.1` if pred < 0.85 (boosts underdog confidence).
Step 2: Six manual overrides for clear seed-strength mismatches in early rounds.

---

## Phase 2: Gap Analysis (2025 → 2026)

### What is the Same
- **Schema**: All 35 CSV files have identical column names and data types.
- **Evaluation metric**: Brier score (unchanged).
- **Submission format**: `ID,Pred` with IDs `SEASON_T1_T2` (lower TeamID first).
- **Men's TeamIDs**: 1000–1999 range. **Women's TeamIDs**: 3000–3999 range.
- **Data loading files**: Same 8 core files (M/W RegularSeasonDetailedResults, NCAATourneyDetailedResults, NCAATourneySeeds).

### What Has Changed / New Constraints

| Item | 2025 State | 2026 State | Impact |
|------|-----------|-----------|--------|
| **2026 seeds** | Known at submission time | **Unknown until Selection Sunday March 15** | Critical — seed features cannot be used for Stage 2 without workaround |
| **Massey Ordinals** | Arrived late, winner had to pivot away | Available through DayNum=93, **56 systems** (POM, NET, MAS, RPI, AP, etc.) | Opportunity to add as features |
| **Regular season data** | Ended ~Feb 2025 | Through DayNum=93 (~Feb 4, 2026); **update expected before March 19** | Wait for Kaggle's data refresh |
| **Stage 2 submission size** | ~similar | 132,133 rows (66,430 men + 65,703 women — all possible matchups) | Must predict ALL team pairs, not just seeded teams |
| **Season regions** | Known | `RegionW/X/Y/Z = TBD1-4` | No impact on modeling |
| `MSecondaryTourneyTeams.csv` | Not mentioned | Present (shape 1895×3) | Minor; available if needed |

### Critical Bug in 2025 Code (MUST FIX)
`men_women` is **inverted** between training (Cell 12) and inference (Cell 45):

```python
# Cell 12 (TRAINING) — correct: men=1, women=0
output["men_women"] = (output["T1_TeamID"].apply(lambda t: str(t).startswith("1"))) * 1

# Cell 45 (INFERENCE) — BUG: men=0, women=1
X['men_women'] = X['T1_TeamID'].apply(lambda t: 0 if str(t)[0]=='1' else 1)
```
Fix inference to: `X['men_women'] = (X['T1_TeamID'].apply(lambda t: str(t).startswith("1"))).astype(int)`

### The Seeds Problem for 2026
Seeds are not released until Selection Sunday (March 15, 2026 = DayNum 132). The Stage 2 submission predicts **all possible team pairs**, most of which will never play. The winning 2025 code left seeds as `NaN` for non-tournament teams and relied on XGBoost's native NaN handling. For 2026, the same approach works — do a left-join on seeds; non-seeded teams get `NaN` for seed features and XGBoost will handle them.

---

## Phase 3: Actionable Blueprint

### A. Porting Guide — What Transfers Directly

| Component | Source Cell | Status | Notes |
|-----------|-------------|--------|-------|
| `prepare_data()` | Cell 12 | ✅ Use as-is | The OT-adjustment + doubling trick is critical; keep it |
| Season filter (≥2003) | Cell 3 | ✅ Use as-is | Women's detailed data starts 2010; GLM loop already handles this |
| Seed extraction `int(x[1:3])` | Cell 17 | ✅ Use as-is | |
| Season-average feature computation | Cell 24 | ✅ Use as-is | `ss_T1` / `ss_T2` naming pattern |
| `boxcols` list | Cell 22 | ✅ Use as-is | All 29 box-score columns |
| `update_elo()`, `expected_result()` | Cell 26 | ✅ Use as-is | Reset every season; k=100, width=400 |
| `team_quality()` GLM function | Cell 31 | ✅ Use as-is | The `st` filter for tourney-connected teams is key |
| XGBoost `param` dict | Cell 38 | ✅ Use as-is | Tune only if adding new features |
| Leave-one-season-out CV loop | Cell 39 | ✅ Use as-is | |
| Spline calibration | Cell 41 | ✅ Use as-is | k=5, t=25 clip |
| Ensemble mean of per-season models | Cell 46 | ✅ Use as-is | |
| `data_dir` path | Cell 1 | ⚠️ Update | Change to `"./Data"` |
| `men_women` in inference | Cell 45 | 🐛 Fix bug | See bug description above |
| Manual overrides | Cell 47 | ❌ Remove | 2025-specific game IDs; replace with 2026 equivalents post-Selection Sunday |

### B. Adaptation Strategy for 2026

**1. Data path**
```python
data_dir = "./Data"
```

**2. Fix men_women encoding bug**
```python
# In prepare_data() — already correct (men=1)
# In submission construction — fix to match:
X['men_women'] = X['T1_TeamID'].apply(lambda t: 1 if str(t).startswith('1') else 0)
```

**3. Handle missing 2026 seeds gracefully**
Seeds will be `NaN` for all teams until March 15. XGBoost handles `NaN` natively — no change needed to the merge logic. However, `Seed_diff` will also be `NaN`. Consider adding a fallback:
```python
# After seed merge — estimate seed from Elo rank for teams without a seed
# (optional quality-of-life improvement, not strictly required)
X['T1_seed'] = X['T1_seed'].fillna(X['T1_elo'].rank(ascending=False) / X['T1_elo'].rank(ascending=False).max() * 16)
```

**4. Massey Ordinals integration (new feature block)**
The 2025 winner planned this but couldn't execute due to late data arrival. In 2026, ordinals are available through DayNum=93 for 56 systems. Priority systems:
```python
priority_systems = ['POM', 'NET', 'MAS', 'SAG', 'KPK', 'AP', 'RPI', 'MOR']
# Get final ranking for each season per system (use max RankingDayNum available)
ordinals = pd.read_csv(f"{data_dir}/MMasseyOrdinals.csv")
ordinals_final = (
    ordinals[ordinals['SystemName'].isin(priority_systems)]
    .sort_values('RankingDayNum')
    .groupby(['Season', 'SystemName', 'TeamID'])
    .last()
    .reset_index()
)
# Pivot: one column per system per team-season
ordinals_pivot = ordinals_final.pivot_table(
    index=['Season', 'TeamID'], columns='SystemName', values='OrdinalRank'
).reset_index()
# Merge for T1 and T2
# (ordinals are men-only; women need a placeholder or skip)
```
Note: Massey Ordinals only cover men's teams. Add `0` or `NaN` for women.

**5. Add late-season weight**
The current model weights all regular-season games equally. Recent games are more predictive:
```python
# In regular_data after prepare_data():
regular_data['weight'] = 1 + (regular_data['DayNum'] / 132)  # linear recency weight
# Then use weighted groupby:
ss = regular_data.groupby(['Season','T1_TeamID'])[boxcols].apply(
    lambda g: np.average(g, weights=regular_data.loc[g.index, 'weight'], axis=0)
)
```

**6. Update data refresh strategy**
Kaggle will release a data update before March 19 with games through ~March 15. Re-run the full pipeline after each refresh to capture late-season form.

### C. New Innovations for 2026

**Innovation 1: Massey Ordinals as direct features (HIGH impact)**
- Use `POM`, `NET`, `SAG`, `MOR`, `KPK` rank differences as features
- These encode external expert knowledge the box-score averages can't capture
- Add `T1_POM_rank`, `T2_POM_rank`, `POM_rank_diff`, etc.
- Expected: 0.002–0.005 Brier improvement based on past research

**Innovation 2: NET ranking as seed proxy (MEDIUM impact)**
- NCAA uses NET (NCAA Evaluation Tool) as a primary seeding input
- `NET` ordinal for 2026 is available in MMasseyOrdinals
- Use NET rank as a seed proxy before seeds are released:
  ```python
  X['T1_seed_est'] = X['T1_NET_rank'].rank() / 68 * 16  # scale to 1-16 range
  ```

**Innovation 3: Conference-adjusted quality (MEDIUM impact)**
- Current GLM quality uses all regular-season games equally
- Add conference strength weights via `MTeamConferences.csv` + `Conferences.csv`
- Power conferences (B10, B12, SEC, ACC, BE) games should get higher weight in quality computation

**Innovation 4: Coach experience feature (LOW-MEDIUM impact)**
- `MTeamCoaches.csv` contains head coach history
- "Tournament experience" = count of previous tournament appearances under current coach
- Rookie coaches at tournament-caliber teams tend to underperform vs. expectations

**Innovation 5: Late-season momentum (MEDIUM impact)**
- Compute separate averages for last 10 games vs. full season
- `T1_last10_PointDiff`, `T1_last10_Score`, etc.
- Teams peaking into March perform better than their season average suggests

**Innovation 6: Consistent isotonic calibration (LOW impact but correct)**
- Replace UnivariateSpline with `sklearn.isotonic.IsotonicRegression`
- Isotonic regression is monotone-constrained and requires no hyperparameter (vs. spline degree k)
- Or stack both: use spline for initial calibration, isotonic for final

**Innovation 7: Remove the "+10% for <85%" global boost (INVESTIGATE)**
- This was a 2025-specific hack that happened to work
- For 2026, evaluate whether the raw spline calibration Brier score is better or worse with this boost
- If the model is well-calibrated, the boost may hurt by pushing probabilities away from the true distribution

---

## Execution Order

```
1. Set up environment (done: uv venv .venv)
2. Port the notebook as solution_2026.ipynb
   a. Update data_dir = "./Data"
   b. Fix men_women bug in inference section
   c. Verify prepare_data() runs on 2026 data
3. Replicate baseline (29 features, XGBoost as-is)
   → Target: reproduce OOF Brier score from 2025 solution on 2022-2025 data
4. Add Massey Ordinals block (men-only features)
5. Add late-season momentum features
6. Re-tune XGBoost (eta/rounds) if feature count changes significantly
7. Generate Stage 1 predictions — validate against SampleSubmissionStage1.csv ground truth
8. Await Kaggle data refresh (post March 4 expected), re-run pipeline
9. After Selection Sunday (March 15): add actual seeds, replace seed estimates
10. Apply manual overrides for extreme seed mismatches (post-bracket release)
11. Generate Stage 2 predictions and submit before March 19 4PM UTC
```

---

## Environment

```
venv: .venv (uv, Python 3.12.8)
activate: source .venv/Scripts/activate   # Windows/bash
packages: numpy, pandas, xgboost==3.2.0, scikit-learn, scipy, statsmodels, matplotlib, seaborn, tqdm
```
