# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Context

**March Machine Learning Mania 2026** — Kaggle competition to predict NCAA Men's and Women's basketball tournament game outcomes.

- **Evaluation metric**: Brier score (MSE between predicted probabilities and actual outcomes)
- **Submission format**: `ID,Pred` where ID is `SEASON_TEAMID1_TEAMID2` (lower TeamID first), Pred is probability that Team1 wins
- **Stage 1**: Historical predictions (2022–2025 tournaments) — for sanity-checking models
- **Stage 2**: 2026 tournament predictions — actual competition submission
- **Final deadline**: March 19, 2026 at 4PM UTC

## Data Layout

All CSVs live in `Data/`. Key files:

| Prefix | Meaning |
|--------|---------|
| `M` | Men's division |
| `W` | Women's division |
| `*CompactResults` | Game results with score only |
| `*DetailedResults` | Game results + full box scores (FG, 3P, FT, rebounds, assists, TO, etc.) |

**Coverage gaps**: Detailed box scores available from 2003 (men) / 2010 (women). Massey Ordinals (external rankings) from 2003.

Key files to load for modeling:
- `MRegularSeasonDetailedResults.csv` / `WRegularSeasonDetailedResults.csv` — box-score features
- `MNCAATourneySeeds.csv` / `WNCAATourneySeeds.csv` — seed numbers
- `MMasseyOrdinals.csv` — external ranking systems (Pomeroy, Sagarin, RPI, ESPN, etc.)
- `SampleSubmissionStage2.csv` — defines which game pairs to predict for 2026

## Reference Solution (2025 1st Place)

See `Last_Year_Winner/final-solution-ncaa-2025.ipynb` for a working approach:

- **Model**: XGBoost regression (`reg:squarederror` objective), eta=0.0093, max_depth=4, ~700 rounds
- **Features (29 total)**: seed, avg score, FGA, rebounds, blocks, fouls, point differential, Elo ratings, opponent metrics, M/W indicator
- **Training data**: Historical tournament games + regular season averages
- **Row format**: One row per game, from Team1's perspective (Team1 = lower ID); target = 1 if Team1 wins, 0 otherwise
- **Post-processing**: Manual probability adjustments for strong seed mismatches

## Development Workflow

Work primarily in Jupyter notebooks. Typical pipeline:

1. Load and merge CSVs from `Data/` using pandas
2. Engineer team-season aggregate features from regular season results
3. Create matchup rows (one per potential game pair) with feature differences
4. Train on historical tournament results; evaluate with Brier score
5. Generate predictions for all pairs in `SampleSubmissionStage2.csv`
6. Clip probabilities to avoid log-loss extremes (e.g., `np.clip(pred, 0.025, 0.975)`)
7. Submit CSV in format: `ID,Pred`

## Submission ID Construction

```python
# Always put lower TeamID first
if team1_id < team2_id:
    game_id = f"{season}_{team1_id}_{team2_id}"
    prob = model_prob  # prob that team1 wins
else:
    game_id = f"{season}_{team2_id}_{team1_id}"
    prob = 1 - model_prob  # flip since ID order is reversed
```

## Massey Ordinals Usage

`MMasseyOrdinals.csv` has columns: `Season, RankingDayNum, SystemName, TeamID, OrdinalRank`. To use rankings as of a specific date:

```python
# Get rankings close to tournament (e.g., day 128)
ordinals = ordinals[ordinals['RankingDayNum'] <= 128]
latest = ordinals.groupby(['Season', 'SystemName', 'TeamID'])['RankingDayNum'].max()
```

Common systems: `POM` (Pomeroy/KenPom), `SAG` (Sagarin), `RPI`, `ESPN`, `MOR`, `WLK`.
