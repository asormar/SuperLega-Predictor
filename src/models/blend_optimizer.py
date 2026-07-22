"""
blend_optimizer.py — LOFO-CV blend weight optimizer for B4 match predictor.

Derives the optimal blend weight `w` between the margin-Elo match probability
(P_elo) and the derived best-of-5 match probability (P_derived from SetPredictor).

    P_final = w * P_elo + (1 - w) * P_derived

The optimizer uses Leave-One-Year-Out cross-validation over rolling years,
minimises log-loss per fold via grid search, and optionally refines the
best grid weight with golden-section search.  Deterministic by design
(NFR-003): same DataFrame + same kwargs produces bit-identical results.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

logger = logging.getLogger("b4_blend")


def blend_p_match(p_elo: float, p_derived: float, w: float) -> float:
    """Linear blend of Elo and derived match probabilities.

    Args:
        p_elo: Match probability from margin-Elo.
        p_derived: Match probability from best-of-5 formula on SetPredictor.
        w: Blend weight in [0, 1]; 1.0 = pure Elo, 0.0 = pure derived.

    Returns:
        Blended probability = w * p_elo + (1 - w) * p_derived.
    """
    return w * p_elo + (1.0 - w) * p_derived


def _logloss_per_fold(
    df: pd.DataFrame,
    w: float,
    val_year: int,
    elo_col: str,
    derived_col: str,
    y_col: str,
) -> float:
    """Compute log-loss on a single validation year for a given blend weight w."""
    va = df[df["temporada_inicio"] == val_year]
    if len(va) < 5:
        return float("nan")
    p_blend = w * va[elo_col].values + (1.0 - w) * va[derived_col].values
    p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)
    return float(log_loss(va[y_col].values, p_blend))


def _grid_search_w(
    df: pd.DataFrame,
    w_grid: List[float],
    val_year: int,
    elo_col: str,
    derived_col: str,
    y_col: str,
) -> float:
    """Find the w in the grid with lowest log-loss on val_year.

    Returns NaN if the fold has no valid rows.
    """
    best_w = float("nan")
    best_ll = float("inf")
    for w in w_grid:
        ll = _logloss_per_fold(df, w, val_year, elo_col, derived_col, y_col)
        if np.isnan(ll):
            continue
        if ll < best_ll:
            best_ll = ll
            best_w = w
    return best_w


def _golden_section_refine(
    df: pd.DataFrame,
    w_approx: float,
    val_year: int,
    elo_col: str,
    derived_col: str,
    y_col: str,
    tol: float = 1e-3,
    bracket: float = 0.15,
) -> float:
    """Golden-section search around *w_approx* within [w_approx-bracket, w_approx+bracket].

    Falls back to *w_approx* if the bracket leaves [0, 1].
    """
    lo = max(0.0, w_approx - bracket)
    hi = min(1.0, w_approx + bracket)
    if hi - lo < 1e-6:
        return w_approx

    phi = (np.sqrt(5) - 1) / 2  # golden ratio ~0.618
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _logloss_per_fold(df, c, val_year, elo_col, derived_col, y_col)
    fd = _logloss_per_fold(df, d, val_year, elo_col, derived_col, y_col)

    for _ in range(40):  # enough for double-precision convergence
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = _logloss_per_fold(df, c, val_year, elo_col, derived_col, y_col)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = _logloss_per_fold(df, d, val_year, elo_col, derived_col, y_col)

    best_w = (a + b) / 2
    logger.info(f"  golden-section [{lo:.3f}, {hi:.3f}] → w={best_w:.6f}")
    return best_w


def optimize_blend_w(
    df: pd.DataFrame,
    *,
    w_grid: Optional[List[float]] = None,
    val_years: Optional[List[int]] = None,
    elo_col: str = "p_elo",
    derived_col: str = "p_derived",
    y_col: str = "y",
    refine: str = "golden_section",
    refine_tol: float = 1e-3,
) -> Dict:
    """Leave-One-Year-Out cross-validation blend weight optimizer.

    For each year in *val_years*:
      1. Grid-search over *w_grid* on that year alone to find the argmin w.
      2. Optionally refine with golden-section search around the grid argmin.

    After all folds produce their per-fold optimal *w*, the global
    *w_global* is the mean across folds (REQ-006).

    Args:
        df: DataFrame with columns *elo_col*, *derived_col*, *y_col*, and
            ``temporada_inicio``.
        w_grid: Candidate blend weights. Default ``np.linspace(0.0, 1.0, 21)``.
        val_years: Years to use as validation folds. Default ``[2021, 2022,
            2023, 2024]`` (all years except the test held-out 2025).
        elo_col: Column name for Elo match probability.
        derived_col: Column name for best-of-5 derived match probability.
        y_col: Column name for binary match outcome (1 = home wins).
        refine: Refinement strategy. ``"golden_section"`` (default) or
            ``"none"`` to use the raw grid argmin.
        refine_tol: Tolerance for golden-section convergence.

    Returns:
        dict with keys:
            n_folds, w_global, w_per_fold_lofo, logloss_per_fold,
            logloss_elo_only_per_fold, logloss_derived_only_per_fold,
            logloss_mean, logloss_elo_only_mean, improvement_mean,
            sigma_lofo, w_grid_argmin_per_fold.
    """
    if w_grid is None:
        w_grid = list(np.linspace(0.0, 1.0, 21))
    if val_years is None:
        val_years = [2021, 2022, 2023, 2024]

    w_per_fold: List[float] = []
    ll_per_fold: List[float] = []
    ll_elo_only: List[float] = []
    ll_derived_only: List[float] = []
    w_grid_argmin_per_fold: List[float] = []

    for vy in val_years:
        # Grid search
        gw = _grid_search_w(df, w_grid, vy, elo_col, derived_col, y_col)
        if np.isnan(gw):
            logger.warning(f"  fold {vy}: no valid data, skipping")
            continue

        w_grid_argmin_per_fold.append(gw)

        # Refine
        if refine == "golden_section":
            w_star = _golden_section_refine(
                df,
                gw,
                vy,
                elo_col,
                derived_col,
                y_col,
                tol=refine_tol,
            )
            # Sanity: refined w should be within 0.05 of grid argmin
            if abs(w_star - gw) > 0.05:
                logger.warning(
                    f"  fold {vy}: refined w={w_star:.4f} drifted >0.05 from "
                    f"grid argmin {gw:.4f}; falling back to grid argmin"
                )
                w_star = gw
        else:
            w_star = gw

        w_per_fold.append(w_star)

        # Log-losses at the chosen w
        ll_w = _logloss_per_fold(df, w_star, vy, elo_col, derived_col, y_col)
        ll_elo = _logloss_per_fold(df, 1.0, vy, elo_col, derived_col, y_col)
        ll_derived = _logloss_per_fold(df, 0.0, vy, elo_col, derived_col, y_col)
        ll_per_fold.append(ll_w)
        ll_elo_only.append(ll_elo)
        ll_derived_only.append(ll_derived)

        improvement = ll_elo - ll_w
        logger.info(
            f"  fold {vy}: w={w_star:.4f}  logloss={ll_w:.4f}  "
            f"elo={ll_elo:.4f}  derived={ll_derived:.4f}  "
            f"improvement={improvement:+.4f}"
        )

    n_folds = len(w_per_fold)
    if n_folds == 0:
        logger.error("No valid folds — returning degenerate dict")
        return {
            "n_folds": 0,
            "w_global": 1.0,
            "w_per_fold_lofo": [],
            "logloss_per_fold": [],
            "logloss_elo_only_per_fold": [],
            "logloss_derived_only_per_fold": [],
            "logloss_mean": float("nan"),
            "logloss_elo_only_mean": float("nan"),
            "improvement_mean": float("nan"),
            "sigma_lofo": 0.0,
            "w_grid_argmin_per_fold": [],
        }

    w_global = float(np.mean(w_per_fold))
    improvements = [e - b for e, b in zip(ll_elo_only, ll_per_fold)]
    sigma = float(np.std(improvements)) if len(improvements) > 1 else 0.0
    # REQ-008: floor sigma at 0.005
    sigma = max(sigma, 0.005)

    result = {
        "n_folds": n_folds,
        "w_global": w_global,
        "w_per_fold_lofo": w_per_fold,
        "logloss_per_fold": ll_per_fold,
        "logloss_elo_only_per_fold": ll_elo_only,
        "logloss_derived_only_per_fold": ll_derived_only,
        "logloss_mean": float(np.mean(ll_per_fold)),
        "logloss_elo_only_mean": float(np.mean(ll_elo_only)),
        "improvement_mean": float(np.mean(improvements)),
        "sigma_lofo": sigma,
        "w_grid_argmin_per_fold": w_grid_argmin_per_fold,
    }

    logger.info(
        f"  GLOBAL: w={w_global:.4f}  improvement={result['improvement_mean']:+.4f}±{sigma:.4f}"
    )
    return result
