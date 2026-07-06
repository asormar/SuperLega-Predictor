"""
feature_builder.py — Constructor de features en runtime para la simulacion.

Mantiene el estado dinamico de la temporada (Elo, forma, rachas, H2H) y
combina con features estaticas (roster, team stats) para generar las 87
features que necesita el MatchPredictor en cada partido.
"""

from pathlib import Path
from collections import defaultdict
from typing import Optional
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

from src.data.team_mapper import normalize_team_name
from src.data.feature_store import (
    MATCH_FEATURE_COLS, ENRICHED_MATCH_COLS, ROSTER_BASIC_COLS,
)


# ─────────────────────────────────────────────────────────────
# Constantes de Elo
# ─────────────────────────────────────────────────────────────

ELO_K = 32          # Factor K para actualizacion
ELO_BASE = 1500     # Elo inicial para equipos nuevos
ELO_HOME_ADV = 65   # Ventaja de campo en puntos Elo (~P(home)=0.59 con igualdad)


def _elo_expected(elo_a: float, elo_b: float) -> float:
    """Probabilidad esperada de que A gane a B segun Elo."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _elo_update(elo: float, expected: float, actual: float) -> float:
    """Actualiza Elo tras un resultado."""
    return elo + ELO_K * (actual - expected)


# ─────────────────────────────────────────────────────────────
# Feature Builder
# ─────────────────────────────────────────────────────────────

class RuntimeFeatureBuilder:
    """
    Construye features de partido en runtime durante la simulacion.

    Carga perfiles estaticos desde match_features.csv y mantiene
    estado dinamico (Elo, resultados, H2H) actualizado tras cada
    partido simulado.
    """

    def __init__(self, csv_path: Optional[Path] = None):
        if csv_path is None:
            csv_path = BASE_DIR / "DB" / "features" / "match_features.csv"

        self._load_static_profiles(csv_path)
        self._init_dynamic_state()

    def _load_static_profiles(self, csv_path: Path):
        """Carga perfiles estaticos de equipo desde el CSV historico."""
        df = pd.read_csv(csv_path, encoding="utf-8")

        # Normalizar nombres
        if "local" in df.columns:
            df["local"] = df["local"].apply(normalize_team_name)
            df["visitante"] = df["visitante"].apply(normalize_team_name)

        # Perfiles estaticos: media de cada feature por equipo (todas las temporadas)
        self.static_profiles = {}

        # Features estaticas: todas las que no dependen del momento de la temporada
        static_feature_names = (
            MATCH_FEATURE_COLS +
            [c for c in ENRICHED_MATCH_COLS if c in df.columns] +
            [c for c in ROSTER_BASIC_COLS if c in df.columns]
        )

        for team_col, prefix in [("local", "h"), ("visitante", "a")]:
            for col in static_feature_names:
                if not col.startswith(f"{prefix}_") and not col.startswith("diff_"):
                    continue
                # Solo features de equipo (no diffs)
                if not col.startswith(f"{prefix}_"):
                    continue

                base_name = col[2:]  # quitar prefijo h_ o a_
                if base_name not in self.static_profiles:
                    self.static_profiles[base_name] = {}

                team_means = df.groupby(team_col)[col].mean()
                for team, val in team_means.items():
                    if team not in self.static_profiles[base_name]:
                        self.static_profiles[base_name][team] = float(val)

        # H2H historico
        self.historical_h2h = {}
        for _, row in df.iterrows():
            key = (row["local"], row["visitante"])
            if key not in self.historical_h2h:
                self.historical_h2h[key] = {"wins_h": 0, "total": 0}
            self.historical_h2h[key]["total"] += 1
            if row.get("gana_local", 0) == 1:
                self.historical_h2h[key]["wins_h"] += 1

        self.all_teams = set(df["local"].unique()) | set(df["visitante"].unique())
        print(f"  [FeatureBuilder] {len(self.all_teams)} equipos cargados desde CSV")

    def _init_dynamic_state(self):
        """Inicializa estado dinamico para una nueva temporada."""
        self.elo = {team: ELO_BASE for team in self.all_teams}
        self.results = defaultdict(list)     # team → [(win_bool, sets_favor, sets_contra)]
        self.h2h = {}                         # (a,b) → (wins_a, total)
        self.streaks = defaultdict(int)       # team → racha (± consecutiva)
        self.standings_points = defaultdict(int)
        self.sets_won_total = defaultdict(int)
        self.sets_lost_total = defaultdict(int)
        self.current_jornada = 0

    def build_features(
        self,
        local: str,
        visitante: str,
        jornada: int,
    ) -> pd.DataFrame:
        """
        Construye las 87 features para el partido (local vs visitante).

        Returns:
            DataFrame de 1 fila listo para MatchPredictor.predict_proba()
        """
        self.current_jornada = jornada

        features = {}

        for prefix, team in [("h", local), ("a", visitante)]:
            # Features estaticas (media historica)
            for feat_name, team_vals in self.static_profiles.items():
                col = f"{prefix}_{feat_name}"
                features[col] = team_vals.get(team, self._default_for(feat_name))

            # Win rates dinamicos
            results = self.results.get(team, [])
            total = len(results)
            if total > 0:
                wins = sum(1 for w, _, _ in results if w)
                # Win rate global
                features[f"{prefix}_win_rate_global"] = wins / total
                # Win rate last 5
                last5 = results[-5:]
                features[f"{prefix}_win_rate_last5"] = (
                    sum(1 for w, _, _ in last5 if w) / len(last5)
                )
                # Win rate home/away
                home_results = [w for w, _, _ in results if True]  # all results count
                features[f"{prefix}_win_rate_home"] = wins / total
                features[f"{prefix}_win_rate_away"] = wins / total
                # Set win rate
                sf = sum(sf for _, sf, _ in results)
                sc = sum(sc for _, _, sc in results)
                total_sets = sf + sc
                features[f"{prefix}_set_win_rate"] = sf / max(total_sets, 1)
                # Set diff exp
                features[f"{prefix}_set_diff_exp"] = (sf - sc) / max(total, 1)
                # Puntos fav/contra
                features[f"{prefix}_pts_fav_exp"] = sf * 23.5  # aprox
                features[f"{prefix}_pts_con_exp"] = sc * 23.5
                # Forma
                features[f"{prefix}_forma_home"] = wins / total
                features[f"{prefix}_forma_away"] = wins / total
                # Racha
                features[f"{prefix}_racha"] = self.streaks.get(team, 0)
                features[f"{prefix}_ultimo_set_diff"] = (
                    results[-1][1] - results[-1][2] if results else 0
                )
                # Ranking (por puntos SuperLega)
                features[f"{prefix}_rank_season"] = (
                    self.standings_points.get(team, 0)
                )
            else:
                features[f"{prefix}_win_rate_global"] = 0.5
                features[f"{prefix}_win_rate_last5"] = 0.5
                features[f"{prefix}_win_rate_home"] = 0.5
                features[f"{prefix}_win_rate_away"] = 0.5
                features[f"{prefix}_set_win_rate"] = 0.5
                features[f"{prefix}_set_diff_exp"] = 0.0
                features[f"{prefix}_pts_fav_exp"] = 23.5
                features[f"{prefix}_pts_con_exp"] = 23.5
                features[f"{prefix}_forma_home"] = 0.5
                features[f"{prefix}_forma_away"] = 0.5
                features[f"{prefix}_racha"] = 0
                features[f"{prefix}_ultimo_set_diff"] = 0
                features[f"{prefix}_rank_season"] = 0

            # Descanso (simplificado: asumimos 7 dias entre jornadas)
            features[f"{prefix}_descanso"] = 7

            # Elo
            features[f"elo_{prefix}"] = self.elo.get(team, ELO_BASE)

        # Diffs dinamicos
        for feat in ["win_rate_global", "win_rate_last5", "set_win_rate",
                     "set_diff_exp", "pts_fav_exp", "pts_con_exp",
                     "racha", "ultimo_set_diff", "descanso", "rank_season",
                     "forma_efectiva"]:
            h_val = features.get(f"h_{feat}", 0)
            a_val = features.get(f"a_{feat}", 0)
            features[f"diff_{feat}"] = h_val - a_val

        # Elo diffs
        features["elo_diff"] = features["elo_h"] - features["elo_a"]
        p_home = _elo_expected(features["elo_h"] + ELO_HOME_ADV, features["elo_a"])
        features["elo_win_prob_h"] = p_home
        features["elo_h_home"] = features["elo_h"] + ELO_HOME_ADV
        features["elo_a_away"] = features["elo_a"]

        # H2H (combinar historico con simulado)
        h2h_key = (local, visitante)
        hist = self.historical_h2h.get(h2h_key, {"wins_h": 0, "total": 0})
        sim = self.h2h.get(h2h_key, {"wins_h": 0, "total": 0})
        total_h2h = hist["total"] + sim["total"]
        wins_h2h = hist["wins_h"] + sim["wins_h"]
        h2h_rate = wins_h2h / max(total_h2h, 1)
        features["h_h2h_win_rate"] = h2h_rate if total_h2h > 0 else 0.5
        features["h_h2h_set_diff_exp"] = (h2h_rate - 0.5) * 2.0

        # Set ratios y dominancia (estaticos)
        features["set_ratio_h"] = features.get("h_set_win_rate", 0.5)
        features["set_ratio_a"] = features.get("a_set_win_rate", 0.5)
        features["diff_set_ratio"] = features["set_ratio_h"] - features["set_ratio_a"]

        # point_ratio (estatico desde perfil)
        features["point_ratio_h"] = features.get("h_point_ratio_h", 0.53)
        features["point_ratio_a"] = features.get("a_point_ratio_a", 0.52)

        features["dominancia_h"] = features.get("h_set_win_rate", 0.5) - 0.5
        features["dominancia_a"] = features.get("a_set_win_rate", 0.5) - 0.5
        features["diff_dominancia"] = features["dominancia_h"] - features["dominancia_a"]

        # SOS (estatico)
        features["sos_h"] = 0.5
        features["sos_a"] = 0.5
        features["diff_sos"] = 0.0

        # Jornada
        features["jornada_num"] = jornada

        # Rellenar cualquier feature faltante con 0
        all_needed = (
            [c for c in MATCH_FEATURE_COLS] +
            [c for c in ENRICHED_MATCH_COLS] +
            [c for c in ROSTER_BASIC_COLS]
        )
        for col in all_needed:
            if col not in features:
                features[col] = 0.0

        return pd.DataFrame([features])

    def update(
        self,
        local: str,
        visitante: str,
        sets_local: int,
        sets_visitante: int,
        winner: str,
    ):
        """
        Actualiza el estado dinamico tras un partido simulado.

        Args:
            local: nombre del equipo local
            visitante: nombre del equipo visitante
            sets_local: sets ganados por el local
            sets_visitante: sets ganados por el visitante
            winner: "home" o "away"
        """
        home_won = (winner == "home")

        # Elo
        elo_h = self.elo.get(local, ELO_BASE)
        elo_a = self.elo.get(visitante, ELO_BASE)
        expected_h = _elo_expected(elo_h + ELO_HOME_ADV, elo_a)
        self.elo[local] = _elo_update(elo_h, expected_h, 1.0 if home_won else 0.0)
        self.elo[visitante] = _elo_update(elo_a, 1 - expected_h, 0.0 if home_won else 1.0)

        # Resultados
        self.results[local].append((home_won, sets_local, sets_visitante))
        self.results[visitante].append((not home_won, sets_visitante, sets_local))

        # Rachas
        if home_won:
            if self.streaks[local] >= 0:
                self.streaks[local] += 1
            else:
                self.streaks[local] = 1
            if self.streaks[visitante] <= 0:
                self.streaks[visitante] -= 1
            else:
                self.streaks[visitante] = -1
        else:
            if self.streaks[visitante] >= 0:
                self.streaks[visitante] += 1
            else:
                self.streaks[visitante] = 1
            if self.streaks[local] <= 0:
                self.streaks[local] -= 1
            else:
                self.streaks[local] = -1

        # H2H
        h2h_key = (local, visitante)
        if h2h_key not in self.h2h:
            self.h2h[h2h_key] = {"wins_h": 0, "total": 0}
        self.h2h[h2h_key]["total"] += 1
        if home_won:
            self.h2h[h2h_key]["wins_h"] += 1

        # Puntos SuperLega (para ranking)
        if sets_local == 3 and sets_visitante <= 1:
            self.standings_points[local] += 3
        elif sets_local == 3 and sets_visitante == 2:
            self.standings_points[local] += 2
            self.standings_points[visitante] += 1
        elif sets_visitante == 3 and sets_local <= 1:
            self.standings_points[visitante] += 3
        else:
            self.standings_points[visitante] += 2
            self.standings_points[local] += 1

        # Sets totales
        self.sets_won_total[local] += sets_local
        self.sets_lost_total[local] += sets_visitante
        self.sets_won_total[visitante] += sets_visitante
        self.sets_lost_total[visitante] += sets_local

    @staticmethod
    def _default_for(feat_name: str) -> float:
        """Valor por defecto para una feature estatica desconocida."""
        defaults = {
            "point_ratio": 0.53,
            "top_scorer_avg": 3.5,
            "roster_depth": 1.5,
            "ace_threat": 0.35,
            "rec_quality": 0.35,
            "atq_pct": 0.45,
            "atq_eff": 0.28,
            "pts_set": 23.5,
            "aces_set": 1.5,
            "bloq_set": 2.0,
            "ace_ratio": 0.06,
            "rec_eff": 0.35,
        }
        for key, val in defaults.items():
            if key in feat_name:
                return val
        return 0.5
