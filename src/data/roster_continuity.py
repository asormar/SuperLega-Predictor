"""
roster_continuity.py — Compute T-1 roster continuity for pre-season churn signal.

Roster continuity measures the fraction of a team's T-1 total points that
were scored by players who remain on the team in season T. It is a pure
pre-season feature: all inputs are known before the season starts.

Concepts:
  - A roster with continuity = 1.0 returned every point-scoring player.
  - A roster with continuity = 0.0 lost every point-scoring player to
    transfer or retirement.
  - A team appearing for the first time (no T-1 data) receives the
    league-median continuity across all team-seasons with T-1 data.

Important corrections applied:
  - Cisterna split: LT (historical, "Cisterna Top Volley") and CIS-VOLLEY
    (modern, "Cisterna") are treated as separate entities. Players from
    LT do NOT carry forward to CIS-VOLLEY.
  - Piacenza split: PC (historical, "Piacenza Copra") and PIACENZAYOU
    (modern, "Piacenza") are treated as separate entities.
  - Player join: exact string match on ``jugador`` column (no fuzzy).
    Name variations (e.g. "M. Giannelli" vs "Simone Giannelli") are
    accepted as <3% noise per the design (R-DATA-2).

Usage:
    from src.data.roster_continuity import compute_roster_continuity
    continuity = compute_roster_continuity(player_stats_df)
    # continuity[temporada][equipo] -> float or NaN (imputed)

CLI:
    python -m src.data.roster_continuity --dry-run
        Prints the full continuity table without writing to disk.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.team_id_mapper import ID_EQUIPO_MAP


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def compute_roster_continuity(
    player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Compute T-1 roster continuity for every (equipo_id, temporada).

    Args:
        player_stats: DataFrame with columns ``equipo_id``, ``jugador``,
            ``temporada``, ``puntos``. Typically the output of
            ``data_pipeline.load_player_stats()``.

    Returns:
        DataFrame indexed by ``(equipo_id, temporada)`` with columns:
        - ``continuity``: raw fraction of T-1 points from returning players.
        - ``imputed``: True if this season has no T-1 data (league median used).
        - ``t1_total_pts``: total points in T-1 season.
        - ``returning_pts``: points in T-1 from players present in T.
    """
    # Validate required columns
    required = {"equipo_id", "jugador", "temporada", "puntos"}
    missing = required - set(player_stats.columns)
    if missing:
        raise ValueError(f"player_stats missing columns: {missing}")

    # Build per-player per-season points
    # Some players appear in multiple teams within a season; aggregate
    # at (equipo_id, temporada, jugador) level first.
    pp = (
        player_stats.groupby(["equipo_id", "temporada", "jugador"], as_index=False)["puntos"]
        .sum()
        .sort_values(["equipo_id", "temporada", "jugador"])
    )

    # Build team-season totals
    ts_pts = pp.groupby(["equipo_id", "temporada"])["puntos"].sum().rename("t1_total_pts")

    # Build set of (equipo_id, jugador) pairs per season — who played where
    roster_by_season = pp.groupby(["equipo_id", "temporada"])["jugador"].apply(set)

    results = []
    flat = ts_pts.reset_index()
    for _, row in flat.iterrows():
        eid = row["equipo_id"]
        temp = row["temporada"]
        total_t = row["t1_total_pts"]
        temp_num = _parse_season(temp)

        # Find T-1 season
        t1 = _season_str(temp_num - 1)

        # If the team didn't exist in T-1, it's the first season — impute later
        if (eid, t1) not in roster_by_season.index:
            results.append(
                {
                    "equipo_id": eid,
                    "temporada": temp,
                    "continuity": np.nan,
                    "imputed": np.nan,  # will be filled by league median
                    "t1_total_pts": 0,
                    "returning_pts": 0,
                }
            )
            continue

        # Roster in T-1
        t1_players = roster_by_season.loc[(eid, t1)]
        t1_points_lookup = pp[
            (pp["equipo_id"] == eid) & (pp["temporada"] == t1)
        ].set_index("jugador")["puntos"]

        # Roster in current season T
        t_players = roster_by_season.loc[(eid, temp)]

        # Players present in BOTH T-1 and T
        returning = t1_players & t_players
        returning_pts = t1_points_lookup[t1_points_lookup.index.isin(returning)].sum()
        total_t1 = t1_points_lookup.sum()

        continuity = returning_pts / total_t1 if total_t1 > 0 else np.nan

        results.append(
            {
                "equipo_id": eid,
                "temporada": temp,
                "continuity": continuity,
                "imputed": False,
                "t1_total_pts": int(total_t1),
                "returning_pts": int(returning_pts),
            }
        )

    result_df = pd.DataFrame(results).set_index(["equipo_id", "temporada"])

    # ── League-median imputation for first-season teams (REQ-005) ──
    non_imputed = result_df[result_df["imputed"] == False]["continuity"].dropna()
    if len(non_imputed) > 0:
        league_median = float(non_imputed.median())
    else:
        league_median = 1.0

    imputed_mask = result_df["imputed"].isna()
    result_df.loc[imputed_mask, "continuity"] = league_median
    result_df.loc[imputed_mask, "imputed"] = True

    return result_df


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _parse_season(season_str: str) -> int:
    """Convert '2024/2025' → 2025 (temporada_inicio style)."""
    return int(season_str.split("/")[0])


def _season_str(year: int) -> str:
    """Convert 2024 → '2024/2025'."""
    return f"{year}/{year + 1}"


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compute T-1 roster continuity for all team-seasons.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print full table but do NOT write any artifact.",
    )
    args = parser.parse_args()

    from src.data.data_pipeline import load_player_stats

    print("=" * 60)
    print("  T-1 ROSTER CONTINUITY — pre-season churn signal")
    print("=" * 60)

    ps = load_player_stats()
    print(f"  Loaded {len(ps)} player-season rows")

    result = compute_roster_continuity(ps)

    print(f"\n  Computed continuity for {len(result)} team-seasons")
    print(f"  League-median continuity: {result['continuity'].median():.4f}")
    n_imputed = result["imputed"].sum()
    print(f"  Imputed (first-season) rows: {n_imputed} ({n_imputed / len(result) * 100:.1f}%)")

    # Print the full table
    print("\n  Continuity by team-season:")
    print(f"  {'Equipo ID':<20} {'Temporada':<14} {'Continuity':<12} {'Imputed':<10}")
    print("  " + "-" * 56)
    for (eid, temp), row in result.sort_index().iterrows():
        imputed_str = "YES" if row["imputed"] else "no"
        canonical = ID_EQUIPO_MAP.get(eid, eid)
        print(
            f"  {eid:<8} ({canonical:<20}) {temp:<14} "
            f"{row['continuity']:<12.4f} {imputed_str:<10}"
        )

    if args.dry_run:
        print("\n  --dry-run: no files written.")
    else:
        # Future: write to feature cache or DB/features
        print("\n  (File output not yet implemented — use --dry-run to preview.)")


if __name__ == "__main__":
    main()
