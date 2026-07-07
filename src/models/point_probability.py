"""
point_probability.py — Modelo de probabilidad de ganar un punto.

Estima P(equipo local gana un punto) dado que está sacando o recibiendo.
Esta probabilidad alimenta el simulador Markov Chain punto a punto.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.simulation.constants import DEFAULT_SIDEOUT_RATE, POINT_PROB_CLIP

# Feature keys esperados por el modelo entrenado
_FEATURE_KEYS = [
    "elo_diff", "diff_win_rate_global", "diff_set_win_rate",
    "diff_dominancia", "diff_set_ratio", "diff_forma_efectiva",
]


def build_features_from_strengths(home_strength: float, away_strength: float) -> dict:
    """
    Construye el dict de features minimo para PointProbabilityModel
    a partir de las fuerzas relativas de los equipos.

    Todas las features derivan de la misma diferencia de fuerza porque
    durante un partido individual no hay contexto adicional disponible
    (sin historial de sets, sin racha reciente, etc.).

    Args:
        home_strength: fuerza del local [0, 1]
        away_strength: fuerza del visitante [0, 1]

    Returns:
        dict con las 6 features que espera el modelo entrenado.
    """
    diff = home_strength - away_strength
    return {
        "elo_diff": diff * 200,
        "diff_win_rate_global": diff,
        "diff_set_win_rate": diff,
        "diff_dominancia": diff,
        "diff_set_ratio": diff,
        "diff_forma_efectiva": diff,
    }

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class PointProbabilityModel:
    """
    Modelo que estima la probabilidad de ganar un punto para cada equipo.

    En volleyball, la probabilidad de ganar un punto depende de:
    1. La fuerza relativa de los equipos (strength_diff, elo_diff, etc.)
    2. Quién está sacando (el equipo al saque tiene ventaja con el ace,
       pero el equipo en recepción tiene la iniciativa del ataque)

    Approach:
    - Usamos las ratios de puntos por set (point_ratio_h/a) de match_features
      para derivar la probabilidad base de ganar un punto.
    - Ajustamos por sideout rate (típicamente ~60-65% en volleyball masculino
      profesional — el equipo que recibe gana el punto más a menudo).
    """

    # DEFAULT_SIDEOUT_RATE y POINT_PROB_CLIP ahora viven en src.simulation.constants
    # (centralizados en Batch 3 para evitar duplicación).

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, match_features: pd.DataFrame):
        """
        Ajusta el modelo a partir de match_features.

        Calcula las probabilidades base de ganar un punto para cada
        partido usando las ratios de puntos.
        """
        df = match_features.copy()

        # Calcular probabilidad base de punto para el local
        # point_ratio_h = puntos ganados / puntos totales del local
        # Esto da una estimación directa de P(local gana un punto)
        if "point_ratio_h" in df.columns and "point_ratio_a" in df.columns:
            # La media de point_ratio da la probabilidad base
            self.base_home_point_prob = df["point_ratio_h"].mean()
            self.base_away_point_prob = df["point_ratio_a"].mean()
        else:
            self.base_home_point_prob = 0.50
            self.base_away_point_prob = 0.50

        # Entrenar un modelo simple para ajustar la probabilidad
        # según las features del partido
        feature_cols = []
        for col in ["elo_diff", "diff_win_rate_global", "diff_set_win_rate",
                     "diff_dominancia", "diff_set_ratio", "diff_forma_efectiva"]:
            if col in df.columns:
                feature_cols.append(col)

        if feature_cols and "point_ratio_h" in df.columns:
            X = df[feature_cols].fillna(0)
            # Target: la ratio real de puntos del local (continuous)
            y = df["point_ratio_h"].fillna(0.5)

            X_scaled = self.scaler.fit_transform(X)
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            # Binarizamos para logistic regression: > 0.5 = 1
            y_binary = (y > 0.5).astype(int)
            self.model.fit(X_scaled, y_binary)
            self.feature_cols = feature_cols
            self.is_fitted = True

        print(f"  [PointProbability] Base home point prob: {self.base_home_point_prob:.4f}")
        print(f"  [PointProbability] Base away point prob: {self.base_away_point_prob:.4f}")
        print(f"  [PointProbability] Default sideout rate: {DEFAULT_SIDEOUT_RATE:.2f}")

    def get_point_probabilities(
        self,
        match_features: Optional[dict] = None,
        home_strength: float = 0.5,
        away_strength: float = 0.5,
        home_sideout: float = DEFAULT_SIDEOUT_RATE,
        away_sideout: float = DEFAULT_SIDEOUT_RATE,
    ) -> dict:
        """
        Calcula las probabilidades de ganar un punto para ambos equipos,
        distinguiendo entre sacar y recibir.

        Args:
            match_features: dict con features del partido (elo_diff, etc.)
            home_strength: fuerza relativa del local [0, 1]
            away_strength: fuerza relativa del visitante [0, 1]
            home_sideout: per-team sideout rate of the local team [0, 1].
                Falls back to DEFAULT_SIDEOUT_RATE if unknown.
            away_sideout: per-team sideout rate of the visitor team [0, 1].
                Falls back to DEFAULT_SIDEOUT_RATE if unknown.

        Returns:
            dict con:
                p_home_serving: P(local gana punto | local saca)
                p_home_receiving: P(local gana punto | visitante saca)
                p_away_serving: P(visitante gana punto | visitante saca)
                p_away_receiving: P(visitante gana punto | local saca)
        """
        # Probabilidad base de punto
        if match_features and self.is_fitted:
            # Usar el modelo para ajustar según las features
            X = pd.DataFrame([match_features])[self.feature_cols].fillna(0)
            X_scaled = self.scaler.transform(X)
            # Probabilidad predicha de que el local sea "dominante" en puntos
            p_home_dominant = self.model.predict_proba(X_scaled)[0, 1]
            # Convertir a probabilidad de punto
            p_home_point = 0.45 + 0.10 * p_home_dominant  # Range: [0.45, 0.55]
        else:
            # Usar strength directamente
            total = home_strength + away_strength
            if total > 0:
                p_home_point = home_strength / total
            else:
                p_home_point = 0.5

        p_away_point = 1.0 - p_home_point

        # Ajustar por sideout rate PER-TEAM (Batch 3 mid-effort).
        # En volleyball, el equipo que recibe gana ~60-65% de los rallies
        # porque puede organizar un ataque combinado. Cuando un equipo
        # saca, su probabilidad de ganar el punto es menor.
        # - Cuando LOCAL saca, la probabilidad depende del AWAY sideout
        #   (qué tan bueno es el visitante recibiendo).
        # - Cuando VISITANTE saca, depende del HOME sideout.
        p_home_serving = p_home_point * (1 - away_sideout) / (
            p_home_point * (1 - away_sideout) + p_away_point * away_sideout
        )
        p_home_receiving = p_home_point * home_sideout / (
            p_home_point * home_sideout + p_away_point * (1 - home_sideout)
        )

        # Clamp para evitar probabilidades extremas
        p_home_serving = np.clip(p_home_serving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1])
        p_home_receiving = np.clip(p_home_receiving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1])

        return {
            "p_home_serving": p_home_serving,
            "p_home_receiving": p_home_receiving,
            "p_away_serving": 1.0 - p_home_receiving,
            "p_away_receiving": 1.0 - p_home_serving,
        }

    def save(self, path: Optional[Path] = None):
        """Guarda el modelo."""
        if path is None:
            path = MODELS_DIR / "point_probability.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": getattr(self, "feature_cols", []),
            "base_home_point_prob": self.base_home_point_prob,
            "base_away_point_prob": self.base_away_point_prob,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(save_data, path)
        print(f"  Modelo guardado en {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PointProbabilityModel":
        """Carga un modelo previamente guardado."""
        if path is None:
            path = MODELS_DIR / "point_probability.joblib"

        save_data = joblib.load(path)
        model = cls()
        model.model = save_data["model"]
        model.scaler = save_data["scaler"]
        model.feature_cols = save_data["feature_cols"]
        model.base_home_point_prob = save_data["base_home_point_prob"]
        model.base_away_point_prob = save_data["base_away_point_prob"]
        model.is_fitted = save_data["is_fitted"]
        return model
