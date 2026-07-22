"""
train_improved.py — Entrenamiento mejorado (resultado del PLAN_MEJORA_PRECISION).

Consolida los hallazgos de las fases del plan en artefactos reproducibles,
SIN romper la pipeline de producción existente. Guarda modelos v2 y un
snapshot de precisión para comparar antes/después.

El snapshot sigue la misma forma que precision_baseline.json:
  - champion: nombre del modelo campeón
  - cv: métricas de validación rolling-origin (media ± std sobre folds)
  - test: métricas en el conjunto de test held-out
  - n_features: número de features usadas

Hallazgos clave (ver comparación al final del plan):
  1. Las features enriquecidas por temporada completa tenían LEAKAGE; medidas
     honestamente el AUC de match era 0.53, no 0.71.
  2. Con dataset pequeño y ruidoso, un modelo Elo con margen de victoria
     (sin entrenamiento) supera a los árboles: AUC 0.75, logloss 0.585.
  3. Las temporadas viejas (2016-2020, 34-55 partidos, home-win ~0.35)
     enseñan el signo equivocado. Recencia (half-life ~1.5) o entrenar solo
     2022+ lo corrige.
  4. Modelos lineales regularizados (LogReg) baten a los árboles profundos
     en este régimen de datos, tanto en set como en match.

Config final:
  - MATCH: probabilidad de Elo con margen (rolling, sin leakage).
  - SET: LogReg C=0.5 con pesos de recencia (half-life=2), features de set.
"""

import sys
import json
import logging
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    accuracy_score,
    brier_score_loss,
)

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import SET_FEATURE_COLS
from src.data.rolling_features import build_rolling_match_features
from src.data.set_feature_contract import SetContext, build_set_features
from src.simulation.set_math import p_match_from_p_set

b4_logger = logging.getLogger("b4_blend")

MODELS_DIR = BASE_DIR / "models"

# Config final (derivada de la experimentación de las fases)
MATCH_FEATURES = [
    "elo_diff",
    "diff_form_ewma",
    "h2h_win_rate_h",
    "diff_set_ratio",
    "diff_point_ratio",
]
RECENT_TRAIN_SEASONS = [2022, 2023, 2024]
TEST_SEASON = 2025
SET_RECENCY_HALFLIFE = 2.0


def _metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "acc": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "n": int(len(y)),
    }


def _set_rolling_cv(data: dict) -> dict:
    """Rolling-origin CV for the v2 LogReg (C=0.5, recency half-life=2).

    Folds (sliding window, train fijo 2022-2024):
      - train [2022, 2023] → val 2024
      - train [2023, 2024] → val 2025

    Returns dict with logloss_mean, logloss_std, auc_mean, auc_std,
    brier_mean, acc_mean, n_folds.
    """
    ds = data["set_features"].copy()
    cols = [c for c in SET_FEATURE_COLS if c in ds.columns]

    fold_configs = [
        ([2022, 2023], 2024),
        ([2023, 2024], 2025),
    ]

    per_fold = {"logloss": [], "auc": [], "brier": [], "acc": []}

    for train_years, val_year in fold_configs:
        tr = ds[ds.temporada_inicio.isin(train_years)]
        va = ds[ds.temporada_inicio == val_year]
        if len(va) == 0 or len(tr) == 0:
            continue
        X_tr, y_tr = tr[cols].fillna(0), tr["ganador_set_local"]
        X_va, y_va = va[cols].fillna(0), va["ganador_set_local"]
        if y_va.nunique() < 2:
            continue

        # Recency weights: closer seasons matter more
        sw = 0.5 ** ((val_year - tr.temporada_inicio.values) / SET_RECENCY_HALFLIFE)

        model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
        model.fit(X_tr, y_tr, sample_weight=sw)
        p = model.predict_proba(X_va)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)

        per_fold["logloss"].append(log_loss(y_va, p))
        per_fold["auc"].append(roc_auc_score(y_va, p))
        per_fold["brier"].append(brier_score_loss(y_va, p))
        per_fold["acc"].append(accuracy_score(y_va, (p >= 0.5).astype(int)))

    out = {"n_folds": len(per_fold["logloss"])}
    for k, v in per_fold.items():
        out[f"{k}_mean"] = float(np.mean(v)) if v else float("nan")
        out[f"{k}_std"] = float(np.std(v)) if v else float("nan")
    return out


def train_match(dfm: pd.DataFrame) -> dict:
    """Modelo de match: probabilidad de Elo con margen (sin entrenamiento).

    Returns dict con la misma forma que precision_baseline.json (match).
    """
    te = dfm[dfm.temporada_inicio == TEST_SEASON]
    p_elo = te["elo_win_prob_h"].values
    m = _metrics(te["gana_local"], p_elo)

    # El "modelo" es el sistema Elo (parámetros en rolling_features).
    joblib.dump(
        {
            "type": "margin_elo",
            "features": ["elo_win_prob_h"],
            "note": "Probabilidad de Elo con margen, rolling sin leakage.",
        },
        MODELS_DIR / "match_elo_v2.joblib",
    )

    return {
        "champion": "margin_elo",
        "cv": {
            "n_folds": 0,
            "note": "Elo es determinista (sin entrenamiento), no requiere CV.",
        },
        "test": {
            "test_season": TEST_SEASON,
            "n_test": m["n"],
            "logloss": m["logloss"],
            "auc": m["auc"],
            "brier": m["brier"],
            "acc": m["acc"],
        },
        "n_features": 1,  # solo elo_win_prob_h
    }


def train_set(data: dict) -> dict:
    """Modelo de set: LogReg regularizado con pesos de recencia.

    Returns dict con la misma forma que precision_baseline.json (set).
    Lee el CSV de features v2 (A3 contract) en lugar del legacy de la pipeline.
    """
    v2_path = BASE_DIR / "DB" / "features" / "set_features_v2.csv"
    ds = pd.read_csv(v2_path)
    data["set_features"] = ds  # so _set_rolling_cv uses the same v2 data
    cols = [c for c in SET_FEATURE_COLS if c in ds.columns]

    tr = ds[ds.temporada_inicio.isin(RECENT_TRAIN_SEASONS)]
    te = ds[ds.temporada_inicio == TEST_SEASON]
    sw = 0.5 ** ((TEST_SEASON - tr.temporada_inicio.values) / SET_RECENCY_HALFLIFE)

    model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    model.fit(tr[cols].fillna(0), tr["ganador_set_local"], sample_weight=sw)
    p = model.predict_proba(te[cols].fillna(0))[:, 1]
    test_m = _metrics(te["ganador_set_local"], p)

    # Guardar artefacto v2 (sin cambios)
    joblib.dump(
        {
            "type": "logreg_recency",
            "model": model,
            "features": cols,
            "recency_halflife": SET_RECENCY_HALFLIFE,
            "train_seasons": RECENT_TRAIN_SEASONS,
        },
        MODELS_DIR / "set_predictor_v2.joblib",
    )

    # Rolling-origin CV (independiente del test)
    cv = _set_rolling_cv(data)

    return {
        "champion": "LogisticRegression",
        "cv": cv,
        "test": {
            "test_season": TEST_SEASON,
            "n_test": test_m["n"],
            "logloss": test_m["logloss"],
            "auc": test_m["auc"],
            "brier": test_m["brier"],
            "acc": test_m["acc"],
        },
        "n_features": len(cols),
    }


def run_b4_route(
    *,
    test_year: int = 2025,
    val_years: tuple = (2021, 2022, 2023, 2024),
) -> str:
    """B4 experiment route: LOFO-CV blend of Elo + SetPredictor-derived match prob.

    Steps:
      1. Load sets_partidos.csv and build rolling match features (includes p_elo).
      2. Load set_predictor_v2.joblib (fallback to error).
      3. For each match, construct a SetContext (pre-match, 0-0 set state),
         compute build_set_features → predict_proba → p_set → p_match_from_p_set.
      4. Call optimize_blend_w with the enriched DataFrame.
      5. Apply REQ-017 hard-shortcut and REQ-016 AND-of-4 gate.
      6. Write models/b4_blend_results.json.
      7. Return one of 'adopted', 'negative', 'shortcut_negative'.

    Args:
        test_year: Held-out season (default 2025).
        val_years: Validation folds for LOFO-CV.

    Returns:
        Verdict string.
    """
    import logging

    logger = logging.getLogger("b4_blend")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    logger.info("=" * 60)
    logger.info("  B4 — Match predictor from SetPredictor (best-of-5)")
    logger.info("=" * 60)

    # ── Step 1: Load data ──────────────────────────────────────────────
    sp_path = BASE_DIR / "DB" / "sets_partidos.csv"
    if not sp_path.exists():
        raise FileNotFoundError(f"Missing source data: {sp_path}")
    sp = pd.read_csv(sp_path, encoding="utf-8")
    logger.info(f"Loaded {len(sp)} rows from sets_partidos.csv")

    dfm = build_rolling_match_features(sp)
    logger.info(f"Built {len(dfm)} rolling match features")

    # ── Step 2: Load SetPredictor v2 ───────────────────────────────────
    v2_path = MODELS_DIR / "set_predictor_v2.joblib"
    if not v2_path.exists():
        logger.warning("set_predictor_v2.joblib NOT FOUND — falling back to auto-NEGATIVE")
        return _write_negative_fallback(
            "SetPredictor v2 artifact missing from models/",
            test_year,
        )

    v2_artifact = joblib.load(v2_path)
    v2_model = v2_artifact.get("model")
    v2_features = v2_artifact.get("features", SET_FEATURE_COLS)
    if v2_model is None:
        logger.warning("SetPredictor v2 artifact has no 'model' key — auto-NEGATIVE")
        return _write_negative_fallback(
            "SetPredictor v2 model key missing",
            test_year,
        )

    logger.info(f"Loaded SetPredictor v2 ({len(v2_features)} features)")

    # ── Step 3: Compute p_derived per match ────────────────────────────
    rows = []
    for _, row in dfm.iterrows():
        ctx = SetContext(
            temporada_inicio=int(row.get("temporada_inicio", 0)),
            jornada_num=int(row.get("jornada_num", 0)),
            match_id=str(row.get("partido_id", "")),
            set_index=1,
            equipo_local=str(row.get("home", "")),
            equipo_visitante=str(row.get("away", "")),
            elo_local=float(row.get("elo_h", 1500.0)),
            elo_visitante=float(row.get("elo_a", 1500.0)),
            strength_local=float(row.get("strength_h", 0.5)),
            strength_visitante=float(row.get("strength_a", 0.5)),
            h_win_rate_global=float(row.get("h_win_rate_global", 0.5)),
            a_win_rate_global=float(row.get("a_win_rate_global", 0.5)),
            h_set_win_rate=float(row.get("h_set_win_rate", 0.5)),
            a_set_win_rate=float(row.get("a_set_win_rate", 0.5)),
            h_form_ewma=float(row.get("h_form_ewma", 0.5)),
            a_form_ewma=float(row.get("a_form_ewma", 0.5)),
            h_set_diff_exp=float(row.get("h_set_diff_exp", 0.0)),
            a_set_diff_exp=float(row.get("a_set_diff_exp", 0.0)),
            h_point_ratio=float(row.get("h_point_ratio", 0.5)),
            a_point_ratio=float(row.get("a_point_ratio", 0.5)),
            h2h_win_rate=float(row.get("h2h_win_rate_h", 0.5)),
            sets_h_antes=0,
            sets_a_antes=0,
            prev_home_won=-1,
            target_score=25,
        )
        feats = build_set_features(ctx)
        feat_df = pd.DataFrame([feats])[v2_features]
        p_set = float(v2_model.predict_proba(feat_df.fillna(0))[0, 1])
        p_derived = p_match_from_p_set(p_set)

        rows.append(
            {
                "p_elo": float(row.get("elo_win_prob_h", 0.5)),
                "p_derived": p_derived,
                "y": int(row.get("gana_local", 0)),
                "temporada_inicio": int(row.get("temporada_inicio", 0)),
            }
        )

    blend_df = pd.DataFrame(rows)
    logger.info(
        f"Built blend DataFrame: {len(blend_df)} rows, "
        f"{blend_df['temporada_inicio'].nunique()} seasons"
    )

    # ── Step 4: LOFO-CV optimisation ───────────────────────────────────
    from src.models.blend_optimizer import optimize_blend_w

    result = optimize_blend_w(
        blend_df,
        w_grid=list(np.linspace(0.0, 1.0, 21)),
        val_years=list(val_years),
        elo_col="p_elo",
        derived_col="p_derived",
        y_col="y",
        refine="golden_section",
        refine_tol=1e-3,
    )

    # ── Step 5: Apply gates ────────────────────────────────────────────
    failing_conditions = []

    # REQ-017: Hard shortcut — if w_global >= 0.95, the blend is
    # indistinguishable from pure Elo → shortcut_negative
    if result["w_global"] >= 0.95:
        verdict = "shortcut_negative"
        failing_conditions.append(f"w_global={result['w_global']:.4f} >= 0.95 (hard shortcut)")
    else:
        # REQ-016: AND-of-4 adoption gate
        conditions = {
            "improvement_mean > 0": result["improvement_mean"] > 0,
            "improvement_mean > sigma_lofo": result["improvement_mean"] > result["sigma_lofo"],
            "n_folds >= 2": result["n_folds"] >= 2,
            "w_global < 0.95": result["w_global"] < 0.95,
        }
        failed = [k for k, v in conditions.items() if not v]
        if failed:
            verdict = "negative"
            failing_conditions = failed
        else:
            verdict = "adopted"

    # ── Step 6: Test metrics IF adopted ────────────────────────────────
    test_metrics = None
    if verdict == "adopted":
        te = blend_df[blend_df["temporada_inicio"] == test_year]
        if len(te) > 10:
            p_blend = (
                result["w_global"] * te["p_elo"].values
                + (1.0 - result["w_global"]) * te["p_derived"].values
            )
            p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)
            test_metrics = {
                "n": int(len(te)),
                "logloss": float(log_loss(te["y"].values, p_blend)),
                "auc": float(roc_auc_score(te["y"].values, p_blend)),
                "brier": float(brier_score_loss(te["y"].values, p_blend)),
                "acc": float(accuracy_score(te["y"].values, (p_blend >= 0.5).astype(int))),
            }

    # ── Write results ──────────────────────────────────────────────────
    output = {
        "shortcut": result["w_global"] >= 0.95,
        "verdict": verdict,
        "w_global": result["w_global"],
        "w_per_fold_lofo": result["w_per_fold_lofo"],
        "logloss_per_fold": result["logloss_per_fold"],
        "logloss_elo_only_per_fold": result["logloss_elo_only_per_fold"],
        "logloss_mean": result["logloss_mean"],
        "logloss_elo_only_mean": result["logloss_elo_only_mean"],
        "improvement_mean": result["improvement_mean"],
        "sigma_lofo": result["sigma_lofo"],
        "n_folds": result["n_folds"],
        "test_year": test_year,
        "val_years": list(val_years),
        "test_metrics_if_computed": test_metrics,
        "failing_conditions": failing_conditions,
        "note": (
            "B4 blend experiment: P_final = w * P_elo + (1-w) * P_derived "
            "where P_derived = p_match_from_p_set(SetPredictor.predict_proba)"
        ),
    }

    out_path = MODELS_DIR / "b4_blend_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results written to {out_path}")

    # Summary banner (NFR-005)
    banner = f"  >>> B4 VERDICT: {verdict.upper()}  (w={result['w_global']:.4f}, Δ={result['improvement_mean']:+.4f})"
    logger.info("─" * 60)
    logger.info(banner)
    logger.info("─" * 60)

    return verdict


def _write_negative_fallback(reason: str, test_year: int) -> str:
    """Write a degenerate B4 result when SetPredictor v2 is unavailable."""
    out_path = MODELS_DIR / "b4_blend_results.json"
    output = {
        "shortcut": True,
        "verdict": "shortcut_negative",
        "w_global": 1.0,
        "w_per_fold_lofo": [],
        "logloss_per_fold": [],
        "logloss_elo_only_per_fold": [],
        "logloss_mean": None,
        "logloss_elo_only_mean": None,
        "improvement_mean": None,
        "sigma_lofo": 0.0,
        "n_folds": 0,
        "test_year": test_year,
        "val_years": [],
        "test_metrics_if_computed": None,
        "failing_conditions": [f"SetPredictor unavailable: {reason}"],
        "note": (
            "B4 fallback — SetPredictor v2 not loaded; blend degenerates " "to w=1.0 (pure Elo)."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    b4_logger.info(f"Fallback results written to {out_path}")
    b4_logger.info("  >>> B4 VERDICT: SHORTCUT_NEGATIVE  (w=1.0, SetPredictor missing)")
    return "shortcut_negative"


def main():
    print("=" * 70)
    print("  ENTRENAMIENTO MEJORADO — resultado del plan de precisión")
    print("=" * 70)
    data = run_pipeline()
    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    dfm = build_rolling_match_features(sp)

    print("\n[MATCH] Elo con margen (rolling, sin leakage)...")
    m_match = train_match(dfm)
    print(
        f"  TEST {TEST_SEASON}: AUC={m_match['test']['auc']:.4f} "
        f"logloss={m_match['test']['logloss']:.4f} brier={m_match['test']['brier']:.4f} "
        f"acc={m_match['test']['acc']:.4f} (n={m_match['test']['n_test']})"
    )
    print(f"  CV: {m_match['cv']['n_folds']} folds (sin entrenamiento)")

    print("\n[SET] LogReg con recencia...")
    m_set = train_set(data)
    print(
        f"  TEST {TEST_SEASON}: AUC={m_set['test']['auc']:.4f} "
        f"logloss={m_set['test']['logloss']:.4f} brier={m_set['test']['brier']:.4f} "
        f"acc={m_set['test']['acc']:.4f} (n={m_set['test']['n_test']})"
    )
    cv = m_set["cv"]
    print(
        f"  CV ({cv['n_folds']} folds): logloss={cv.get('logloss_mean', 'N/A'):.4f}±{cv.get('logloss_std', 0):.4f} "
        f"auc={cv.get('auc_mean', 'N/A'):.4f}±{cv.get('auc_std', 0):.4f}"
    )

    snapshot = {
        "match": m_match,
        "set": m_set,
        "config": {
            "match_features": MATCH_FEATURES,
            "train_seasons": RECENT_TRAIN_SEASONS,
            "test_season": TEST_SEASON,
        },
    }
    out = MODELS_DIR / "precision_improved.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\n  Snapshot guardado en {out}")
    print("  Artefactos: match_elo_v2.joblib, set_predictor_v2.joblib")


if __name__ == "__main__":
    main()
