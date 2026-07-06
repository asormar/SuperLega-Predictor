"""
simulator.py — Motor de simulacion de partidos de volleyball.

Simula partidos punto a punto usando Cadenas de Markov + Monte Carlo.
Cada punto se decide probabilisticamente, considerando:
- Fuerza relativa de los equipos
- Quien esta sacando (sideout rate)
- Momentum (rachas de puntos)
- Estado actual del set/partido
"""

import random
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

from src.simulation.constants import (
    DEFAULT_CLAMP_RANGE, CLAMP_MARGIN,
    POINT_PROB_CLIP_ADAPTIVE_HARD,
    DEFAULT_SIDEOUT_RATE, POINT_PROB_CLIP,
    GLOBAL_MOMENTUM_FACTOR,
    MOMENTUM_BONUS as _MOMENTUM_BONUS,
    MOMENTUM_MAX_STREAK as _MOMENTUM_MAX_STREAK,
    MOMENTUM_DECAY as _MOMENTUM_DECAY,
)


# ─────────────────────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────────────────────

@dataclass
class PointResult:
    """Resultado de un punto individual."""
    point_number: int
    score_home: int
    score_away: int
    winner: str  # "home" o "away"
    server: str  # "home" o "away"


@dataclass
class SetResult:
    """Resultado de un set completo."""
    set_number: int
    score_home: int
    score_away: int
    winner: str  # "home" o "away"
    points: list = field(default_factory=list)
    home_player_stats: list = field(default_factory=list)
    away_player_stats: list = field(default_factory=list)


@dataclass
class MatchResult:
    """Resultado de un partido completo."""
    home_team: str
    away_team: str
    sets_home: int
    sets_away: int
    winner: str
    resultado: str  # e.g., "3-1"
    sets: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Motor de simulacion
# ─────────────────────────────────────────────────────────────

class MatchSimulator:
    """
    Simulador de partidos de volleyball usando Cadenas de Markov.

    El estado del sistema es (score_home, score_away, who_serves).
    Las transiciones estan determinadas por P(home gana el punto).
    """

    # Parametros de momentum (desde constants.py)
    MOMENTUM_BONUS = _MOMENTUM_BONUS
    MOMENTUM_MAX_STREAK = _MOMENTUM_MAX_STREAK
    MOMENTUM_DECAY = _MOMENTUM_DECAY

    def __init__(
        self,
        set_predictor=None,
        point_model=None,
        player_stats_gen=None,
    ):
        self.set_predictor = set_predictor
        self.point_model = point_model
        self.player_stats_gen = player_stats_gen

    def simulate_match(
        self,
        home_team: str,
        away_team: str,
        home_strength: float = 0.5,
        away_strength: float = 0.5,
        match_features: Optional[dict] = None,
        generate_points: bool = True,
        generate_player_stats: bool = True,
        seed: Optional[int] = None,
        set_predictor: Optional[object] = None,
        team_features: Optional[dict] = None,
    ) -> MatchResult:
        """
        Simula un partido completo de volleyball.

        Args:
            home_team: nombre del equipo local
            away_team: nombre del equipo visitante
            home_strength: fuerza relativa del local [0, 1]
            away_strength: fuerza relativa del visitante [0, 1]
            match_features: dict con features para el modelo de punto
            generate_points: si True, genera punto a punto
            generate_player_stats: si True, genera stats de jugadores
            seed: semilla para reproducibilidad

        Returns:
            MatchResult con todos los detalles
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Obtener probabilidades de punto
        if self.point_model and match_features:
            point_probs = self.point_model.get_point_probabilities(
                match_features=match_features,
                home_strength=home_strength,
                away_strength=away_strength,
            )
        else:
            point_probs = self._default_point_probs(home_strength, away_strength)

        sets_home = 0
        sets_away = 0
        sets_results = []
        momentum_home = 0.0
        momentum_away = 0.0

        # Sorteo inicial: quien saca primero
        home_serves_first = random.random() < 0.5

        set_number = 0
        while sets_home < 3 and sets_away < 3:
            set_number += 1
            is_fifth_set = (sets_home == 2 and sets_away == 2)
            target_score = 15 if is_fifth_set else 25

            # Build set context for SetPredictor calibration
            set_context = None
            if set_predictor is not None and team_features is not None:
                set_context = self._build_set_context_base(
                    set_number, home_team, away_team,
                    sets_home, sets_away, team_features,
                )

            # Simular set
            set_result = self._simulate_set(
                set_number=set_number,
                home_team=home_team,
                away_team=away_team,
                point_probs=point_probs,
                target_score=target_score,
                home_serves_first=home_serves_first,
                momentum_home=momentum_home,
                momentum_away=momentum_away,
                generate_points=generate_points,
                sets_home_antes=sets_home,
                sets_away_antes=sets_away,
                set_predictor=set_predictor,
                set_context_base=set_context,
            )

            # Generar stats de jugadores
            if generate_player_stats and self.player_stats_gen:
                set_result.home_player_stats = self.player_stats_gen.generate_set_stats(
                    home_team, set_result.score_home, set_result.score_away,
                )
                set_result.away_player_stats = self.player_stats_gen.generate_set_stats(
                    away_team, set_result.score_away, set_result.score_home,
                )

            sets_results.append(set_result)

            if set_result.winner == "home":
                sets_home += 1
                momentum_home = momentum_home * self.MOMENTUM_DECAY + 0.5
                momentum_away = momentum_away * self.MOMENTUM_DECAY - 0.3
            else:
                sets_away += 1
                momentum_away = momentum_away * self.MOMENTUM_DECAY + 0.5
                momentum_home = momentum_home * self.MOMENTUM_DECAY - 0.3

            # Alternar quien saca primero en el siguiente set
            home_serves_first = not home_serves_first

        winner = "home" if sets_home > sets_away else "away"
        resultado = f"{sets_home}-{sets_away}"

        return MatchResult(
            home_team=home_team,
            away_team=away_team,
            sets_home=sets_home,
            sets_away=sets_away,
            winner=winner,
            resultado=resultado,
            sets=sets_results,
        )

    def _simulate_set(
        self,
        set_number: int,
        home_team: str,
        away_team: str,
        point_probs: dict,
        target_score: int,
        home_serves_first: bool,
        momentum_home: float,
        momentum_away: float,
        generate_points: bool,
        sets_home_antes: int = 0,
        sets_away_antes: int = 0,
        set_predictor: Optional[object] = None,
        set_context_base: Optional[dict] = None,
    ) -> SetResult:
        """Simula un set completo punto a punto."""
        score_home = 0
        score_away = 0
        home_serving = home_serves_first
        streak_home = 0
        streak_away = 0
        points = []
        point_num = 0

        # Clamp adaptativo via SetPredictor: se evalua una vez al inicio del set
        clamp_low, clamp_high = DEFAULT_CLAMP_RANGE

        # Evaluar SetPredictor una vez al inicio para ajustar el clamp
        if set_predictor is not None and set_context_base is not None:
            p_set_home = self._eval_set_predictor(
                set_predictor, set_context_base,
                0, 0, target_score,
                sets_home_antes, sets_away_antes,
            )
            if p_set_home is not None:
                clamp_low = max(POINT_PROB_CLIP_ADAPTIVE_HARD[0], p_set_home - CLAMP_MARGIN)
                clamp_high = min(POINT_PROB_CLIP_ADAPTIVE_HARD[1], p_set_home + CLAMP_MARGIN)

        while True:
            point_num += 1

            # Calcular probabilidad ajustada por momentum
            if home_serving:
                base_p = point_probs["p_home_serving"]
            else:
                base_p = point_probs["p_home_receiving"]

            # Ajuste por momentum
            momentum_adj = (
                min(streak_home, self.MOMENTUM_MAX_STREAK) * self.MOMENTUM_BONUS
                - min(streak_away, self.MOMENTUM_MAX_STREAK) * self.MOMENTUM_BONUS
            )
            # Ajuste adicional por momentum global del partido
            momentum_adj += (momentum_home - momentum_away) * GLOBAL_MOMENTUM_FACTOR

            p_home_wins = np.clip(base_p + momentum_adj, clamp_low, clamp_high)

            # Decidir quien gana el punto
            if random.random() < p_home_wins:
                score_home += 1
                streak_home += 1
                streak_away = 0
                winner = "home"
                # Sideout: si el visitante sacaba, ahora saca el local
                if not home_serving:
                    home_serving = True
            else:
                score_away += 1
                streak_away += 1
                streak_home = 0
                winner = "away"
                # Sideout
                if home_serving:
                    home_serving = False

            if generate_points:
                points.append(PointResult(
                    point_number=point_num,
                    score_home=score_home,
                    score_away=score_away,
                    winner=winner,
                    server="home" if (home_serving if winner == "home" else not home_serving) else "away",
                ))

            # Comprobar fin del set
            if self._set_finished(score_home, score_away, target_score):
                break

        set_winner = "home" if score_home > score_away else "away"

        return SetResult(
            set_number=set_number,
            score_home=score_home,
            score_away=score_away,
            winner=set_winner,
            points=points,
        )

    def _set_finished(self, score_h: int, score_a: int, target: int) -> bool:
        """Comprueba si un set ha terminado."""
        if score_h >= target and score_h - score_a >= 2:
            return True
        if score_a >= target and score_a - score_h >= 2:
            return True
        return False

    def _build_set_context_base(
        self,
        set_number: int,
        home_team: str,
        away_team: str,
        sets_home_antes: int,
        sets_away_antes: int,
        team_features: dict,
    ) -> dict:
        """Construye la parte estatica del contexto para SetPredictor."""
        import pandas as pd

        is_tiebreak = (sets_home_antes == 2 and sets_away_antes == 2)

        feats = {
            "set_num_norm": (set_number - 1) / 4.0,
            "sets_h_antes": sets_home_antes,
            "sets_a_antes": sets_away_antes,
            "diff_sets_antes": sets_home_antes - sets_away_antes,
            "es_desempate": 1 if is_tiebreak else 0,
        }

        # Team-level features from feature_builder
        for k, v in (team_features or {}).items():
            if k not in feats:
                feats[k] = v

        return feats

    def _eval_set_predictor(
        self,
        set_predictor,
        set_context_base: dict,
        score_home: int,
        score_away: int,
        target_score: int,
        sets_home_antes: int,
        sets_away_antes: int,
    ) -> Optional[float]:
        """Evalua el SetPredictor con el estado actual del set."""
        import pandas as pd

        if set_context_base is None or set_predictor is None:
            return None

        try:
            feats = dict(set_context_base)

            # Actualizar features que dependen del marcador actual
            total_points = max(score_home + score_away, 1)
            momentum = (score_home - score_away) / total_points

            feats["momentum_h"] = momentum
            feats["pts_fav_h"] = score_home
            feats["pts_fav_a"] = score_away

            # Build DataFrame con las columnas que espera el modelo
            if set_predictor.feature_names:
                row = {f: feats.get(f, 0.0) for f in set_predictor.feature_names}
                df = pd.DataFrame([row])
            else:
                return None

            proba = set_predictor.predict_proba(df)
            return float(proba[0, 1])
        except Exception:
            return None

    def _default_point_probs(
        self, home_strength: float, away_strength: float,
    ) -> dict:
        """Probabilidades por defecto basadas en la fuerza relativa."""
        total = home_strength + away_strength
        if total <= 0:
            p_base = 0.5
        else:
            p_base = home_strength / total

        # Ajustar por sideout (equipo recibiendo tiene ventaja)
        p_serving = p_base * (1 - DEFAULT_SIDEOUT_RATE) / (
            p_base * (1 - DEFAULT_SIDEOUT_RATE) + (1 - p_base) * DEFAULT_SIDEOUT_RATE
        )
        p_receiving = p_base * DEFAULT_SIDEOUT_RATE / (
            p_base * DEFAULT_SIDEOUT_RATE + (1 - p_base) * (1 - DEFAULT_SIDEOUT_RATE)
        )

        return {
            "p_home_serving": np.clip(p_serving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1]),
            "p_home_receiving": np.clip(p_receiving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1]),
            "p_away_serving": np.clip(1 - p_receiving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1]),
            "p_away_receiving": np.clip(1 - p_serving, POINT_PROB_CLIP[0], POINT_PROB_CLIP[1]),
        }

    def monte_carlo_simulate(
        self,
        home_team: str,
        away_team: str,
        home_strength: float = 0.5,
        away_strength: float = 0.5,
        match_features: Optional[dict] = None,
        n_simulations: int = 1000,
        seed: Optional[int] = None,
        set_predictor: Optional[object] = None,
        team_features: Optional[dict] = None,
    ) -> dict:
        """
        Ejecuta N simulaciones Monte Carlo y devuelve estadisticas agregadas.

        Args:
            seed: semilla maestra para reproducibilidad. Se aplica UNA SOLA VEZ
                  antes del loop para que cada iteracion sea independiente pero
                  la secuencia completa sea reproducible.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        results = {
            "home_wins": 0,
            "away_wins": 0,
            "score_distribution": {},
            "avg_sets_home": 0,
            "avg_sets_away": 0,
        }

        for i in range(n_simulations):
            match = self.simulate_match(
                home_team=home_team,
                away_team=away_team,
                home_strength=home_strength,
                away_strength=away_strength,
                match_features=match_features,
                generate_points=False,
                generate_player_stats=False,
                set_predictor=set_predictor,
                team_features=team_features,
            )

            if match.winner == "home":
                results["home_wins"] += 1
            else:
                results["away_wins"] += 1

            results["avg_sets_home"] += match.sets_home
            results["avg_sets_away"] += match.sets_away

            score_key = match.resultado
            results["score_distribution"][score_key] = (
                results["score_distribution"].get(score_key, 0) + 1
            )

        # Promedios
        results["avg_sets_home"] /= n_simulations
        results["avg_sets_away"] /= n_simulations
        results["home_win_prob"] = results["home_wins"] / n_simulations
        results["away_win_prob"] = results["away_wins"] / n_simulations

        # Normalizar distribuciones
        for key in results["score_distribution"]:
            results["score_distribution"][key] /= n_simulations

        return results


# ─────────────────────────────────────────────────────────────
# Test rapido
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = MatchSimulator()

    print("=" * 60)
    print("  TEST: Simulacion de partido Trento vs Perugia")
    print("=" * 60)

    match = sim.simulate_match(
        home_team="Trento",
        away_team="Perugia",
        home_strength=0.55,
        away_strength=0.52,
        seed=42,
    )

    print(f"\n  Resultado: {match.home_team} {match.resultado} {match.away_team}")
    print(f"  Ganador: {match.home_team if match.winner == 'home' else match.away_team}")

    for s in match.sets:
        print(f"    Set {s.set_number}: {s.score_home}-{s.score_away} "
              f"({'Local' if s.winner == 'home' else 'Visitante'})")

    print(f"\n  Monte Carlo (1000 simulaciones):")
    mc = sim.monte_carlo_simulate(
        "Trento", "Perugia",
        home_strength=0.55, away_strength=0.52,
        n_simulations=1000,
    )
    print(f"    P(Trento gana):  {mc['home_win_prob']:.1%}")
    print(f"    P(Perugia gana): {mc['away_win_prob']:.1%}")
    print(f"    Distribucion de resultados:")
    for score, pct in sorted(mc["score_distribution"].items()):
        print(f"      {score}: {pct:.1%}")
