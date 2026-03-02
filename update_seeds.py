"""
update_seeds.py — 2026 NCAA Tournament Seed Manager
====================================================
Run on or after Selection Sunday (March 15, 2026) to inject the
official tournament seeds into the competition data files, then
clear relevant pipeline caches so solution_2026.py recomputes
2026 features using real seeds instead of the NET/Elo proxy.

Usage
-----
  # From a CSV file (Season,Seed,TeamID):
  python update_seeds.py --input bracket_2026.csv

  # Interactive entry at the terminal:
  python update_seeds.py --interactive

  # Preview changes without writing:
  python update_seeds.py --input bracket_2026.csv --dry-run

  # After confirming correct, re-run the main pipeline:
  python solution_2026.py

Seed format
-----------
  Region letter: W, X, Y, Z
  Seed number  : 01 – 16
  Play-in tag  : a or b (only for seeds 11, 12, 16 with two play-in teams)
  Examples     : W01, X08, Z16a, W11b
  Men's TeamIDs are 1000–1999; Women's are 3000–3999.

CSV file example
----------------
  Season,Seed,TeamID
  2026,W01,1181
  2026,W02,1314
  2026,X01,1163
  ...
  2026,W01,3380
  2026,W02,3107
  ...

Cache files cleared after update
---------------------------------
  .cache/glm_quality.pkl   — rebuilt to include new 2026 quality pass
  .cache/loso_baseline.pkl — rebuilt with new seed features
  .cache/loso_innovation.pkl
"""

import argparse
import os
import re
import sys
import pandas as pd

DATA_DIR  = "./Data"
CACHE_DIR = ".cache"
SEED_RE   = re.compile(r"^[WXYZwxyz]\d{2}[ab]?$")

# ── helpers ───────────────────────────────────────────────────────────────────

def _load(gender: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{gender}NCAATourneySeeds.csv")
    return pd.read_csv(path)

def _save(df: pd.DataFrame, gender: str, dry_run: bool) -> None:
    path = os.path.join(DATA_DIR, f"{gender}NCAATourneySeeds.csv")
    if dry_run:
        print(f"  [DRY RUN] Would write {len(df)} rows to {path}")
    else:
        df.to_csv(path, index=False)
        print(f"  Saved {path}  ({len(df)} total rows)")

def _id_range(gender: str):
    return (1000, 1999) if gender == "M" else (3000, 3999)

def _validate(df: pd.DataFrame, gender: str) -> bool:
    lo, hi = _id_range(gender)
    ok = True
    bad_seed = df[~df["Seed"].str.match(r"^[WXYZwxyz]\d{2}[ab]?$")]
    if len(bad_seed):
        print(f"  WARNING: {len(bad_seed)} invalid seed strings: {bad_seed['Seed'].tolist()}")
        ok = False
    bad_id = df[~df["TeamID"].between(lo, hi)]
    if len(bad_id):
        print(f"  WARNING: {len(bad_id)} TeamIDs outside {lo}–{hi}: {bad_id['TeamID'].tolist()}")
        ok = False
    # Normalise to uppercase
    df["Seed"] = df["Seed"].str.upper()
    counts = df.groupby("Season")["Seed"].count()
    print(f"  {'Men' if gender=='M' else 'Women'} seeds per season: {counts.to_dict()}")
    return ok

def update_gender(new_rows: pd.DataFrame, gender: str, dry_run: bool) -> int:
    lo, hi = _id_range(gender)
    subset = new_rows[new_rows["TeamID"].between(lo, hi)].copy()
    if len(subset) == 0:
        print(f"  No {gender} seeds found in input — skipping.")
        return 0
    subset["Seed"] = subset["Seed"].str.upper()
    _validate(subset, gender)
    existing = _load(gender)
    non2026  = existing[existing["Season"] != 2026]
    combined = pd.concat([non2026, subset[["Season","Seed","TeamID"]]], ignore_index=True)
    combined = combined.sort_values(["Season","Seed"]).reset_index(drop=True)
    _save(combined, gender, dry_run)
    return len(subset)

def clear_cache(dry_run: bool) -> None:
    targets = ["glm_quality.pkl", "loso_baseline.pkl", "loso_innovation.pkl"]
    for fname in targets:
        path = os.path.join(CACHE_DIR, fname)
        if os.path.exists(path):
            if dry_run:
                print(f"  [DRY RUN] Would delete {path}")
            else:
                os.remove(path)
                print(f"  Deleted cache: {path}")
        else:
            print(f"  (not found, skipping) {path}")

# ── interactive entry ─────────────────────────────────────────────────────────

def _interactive() -> pd.DataFrame:
    rows = []
    print("\n=== 2026 NCAA Tournament Seed Entry (interactive) ===")
    print("Enter one seed per line: SEED TEAMID  (e.g. W01 1181)")
    print("Regions: W X Y Z  |  Seed 01-16  |  Play-in: e.g. W11a W11b")
    print("Type 'done' to finish, 'show' to list entries so far.\n")

    for label, lo, hi in [("Men   (TeamID 1000-1999)", 1000, 1999),
                           ("Women (TeamID 3000-3999)", 3000, 3999)]:
        print(f"--- {label} ---")
        while True:
            try:
                raw = input("  Seed TeamID > ").strip()
            except EOFError:
                break
            if raw.lower() == "done":
                break
            if raw.lower() == "show":
                print(f"  Collected so far: {rows}")
                continue
            parts = raw.split()
            if len(parts) != 2:
                print("  Format: SEED TEAMID  (e.g.  W01  1181)")
                continue
            seed_str, tid_str = parts
            if not SEED_RE.match(seed_str.upper()):
                print(f"  Invalid seed '{seed_str}'. Use W01 … Z16 (with optional a/b)")
                continue
            try:
                tid = int(tid_str)
            except ValueError:
                print("  TeamID must be an integer.")
                continue
            if not (lo <= tid <= hi):
                print(f"  TeamID {tid} out of range for {label}. Try again.")
                continue
            rows.append({"Season": 2026, "Seed": seed_str.upper(), "TeamID": tid})
            print(f"    OK: {seed_str.upper()} -> {tid}")

    return pd.DataFrame(rows, columns=["Season","Seed","TeamID"])

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inject 2026 NCAA tournament seeds and clear model caches."
    )
    parser.add_argument("--input", metavar="FILE",
                        help="CSV with columns Season,Seed,TeamID")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter seeds manually at the terminal")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    parser.add_argument("--no-cache-clear", action="store_true",
                        help="Skip deleting cache files")
    args = parser.parse_args()

    if not args.input and not args.interactive:
        parser.print_help()
        print("\nERROR: supply --input FILE or --interactive")
        sys.exit(1)

    # Load new seeds
    if args.input:
        new_seeds = pd.read_csv(args.input)
        # Allow omitting Season column (default 2026)
        if "Season" not in new_seeds.columns:
            new_seeds["Season"] = 2026
        new_seeds["Season"] = new_seeds["Season"].astype(int)
        print(f"Loaded {len(new_seeds)} seeds from '{args.input}'")
    else:
        new_seeds = _interactive()
        print(f"\nCollected {len(new_seeds)} seeds via interactive mode.")

    if len(new_seeds) == 0:
        print("No seeds collected. Exiting.")
        sys.exit(0)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Updating seed files...")
    n_m = update_gender(new_seeds, "M", args.dry_run)
    n_w = update_gender(new_seeds, "W", args.dry_run)
    print(f"  Men: {n_m} seeds added  |  Women: {n_w} seeds added")

    if not args.no_cache_clear:
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Clearing pipeline caches...")
        clear_cache(args.dry_run)

    print("""
Next steps:
  1. Verify seed counts above (expect 68 men + 68 women for 2026).
  2. Run:  python solution_2026.py
     The pipeline will recompute 2026 GLM quality and LOSO models
     with actual seeds replacing the NET/Elo proxy.
  3. Resubmit submission_2026.csv to Kaggle before the March 19 deadline.
""")

if __name__ == "__main__":
    main()
