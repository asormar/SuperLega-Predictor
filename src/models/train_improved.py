"""
train_improved.py — Entrenamiento mejorado (resultado del PLAN_MEJORA_PRECISION).

Consolida los hallazgos de las fases del plan en artefactos reproducibles,
SIN romper la pipeline de producción existente. Guarda modelos v2 y un
snapshot de precisión para comparar antes/después.

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
    roc_auc_score, log_loss, accuracy_score, brier_score_loss,
)

from src.data.data_pipeline import run_pipeline
from src.data.feature_store import SET_FEATURE_COLS
from src.data.rolling_features import build_rolling_match_features

MODELS_DIR = BASE_DIR / "models"

# Config final (derivada de la experimentación de las fases)
MATCH_FEATURES = ["elo_diff", "diff_form_ewma", "h2h_win_rate_h",
                  "diff_set_ratio", "diff_point_ratio"]
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


def train_match(dfm: pd.DataFrame) -> dict:
    """Modelo de match: probabilidad de Elo con margen (sin entrenamiento)."""
    te = dfm[dfm.temporada_inicio == TEST_SEASON]
    p_elo = te["elo_win_prob_h"].values
    m = _metrics(te["gana_local"], p_elo)
    # El "modelo" es el sistema Elo (parámetros en rolling_features).
    joblib.dump(
        {"type": "margin_elo", "features": ["elo_win_prob_h"],
         "note": "Probabilidad de Elo con margen, rolling sin leakage."},
        MODELS_DIR / "match_elo_v2.joblib",
    )
    return m


def train_set(data: dict) -> dict:
    """Modelo de set: LogReg regularizado con pesos de recencia."""
    ds = data["set_features"].copy()
    ds["temporada_inicio"] = ds["partido_id"].apply(
        lambda x: int(str(x).split("/")[0]) if "/" in str(x) else 0)
    cols = [c for c in SET_FEATURE_COLS if c in ds.columns]

    tr = ds[ds.temporada_inicio.isin(RECENT_TRAIN_SEASONS)]
    te = ds[ds.temporada_inicio == TEST_SEASON]
    sw = 0.5 ** ((TEST_SEASON - tr.temporada_inicio.values) / SET_RECENCY_HALFLIFE)

    model = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    model.fit(tr[cols].fillna(0), tr["ganador_set_local"], sample_weight=sw)
    p = model.predict_proba(te[cols].fillna(0))[:, 1]

    joblib.dump(
        {"type": "logreg_recency", "model": model, "features": cols,
         "recency_halflife": SET_RECENCY_HALFLIFE,
         "train_seasons": RECENT_TRAIN_SEASONS},
        MODELS_DIR / "set_predictor_v2.joblib",
    )
    return _metrics(te["ganador_set_local"], p)


def main():
    print("=" * 70)
    print("  ENTRENAMIENTO MEJORADO — resultado del plan de precisión")
    print("=" * 70)
    data = run_pipeline()
    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")
    dfm = build_rolling_match_features(sp)

    print("\n[MATCH] Elo con margen (rolling, sin leakage)...")
    m_match = train_match(dfm)
    print(f"  TEST {TEST_SEASON}: AUC={m_match['auc']:.4f} "
          f"logloss={m_match['logloss']:.4f} brier={m_match['brier']:.4f} "
          f"acc={m_match['acc']:.4f} (n={m_match['n']})")

    print("\n[SET] LogReg con recencia...")
    m_set = train_set(data)
    print(f"  TEST {TEST_SEASON}: AUC={m_set['auc']:.4f} "
          f"logloss={m_set['logloss']:.4f} brier={m_set['brier']:.4f} "
          f"acc={m_set['acc']:.4f} (n={m_set['n']})")

    snapshot = {"match": m_match, "set": m_set,
                "config": {"match_features": MATCH_FEATURES,
                           "train_seasons": RECENT_TRAIN_SEASONS,
                           "test_season": TEST_SEASON}}
    out = MODELS_DIR / "precision_improved.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"\n  Snapshot guardado en {out}")
    print(f"  Artefactos: match_elo_v2.joblib, set_predictor_v2.joblib")


if __name__ == "__main__":
    main()
