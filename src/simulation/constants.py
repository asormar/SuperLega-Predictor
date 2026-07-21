"""Constantes del simulador. Centralizadas para facilitar experimentacion."""

# Ventaja de local
HOME_ADVANTAGE_STRENGTH_BONUS = 0.03  # Anadido a la fuerza del local

# Clamps de probabilidad de punto
POINT_PROB_CLIP = (0.25, 0.75)  # Rango final de P(gana punto)
POINT_PROB_CLIP_ADAPTIVE_HARD = (0.10, 0.90)  # Limites duros del clamp adaptativo

# Clamp por defecto (sin SetPredictor)
DEFAULT_CLAMP_RANGE = (0.20, 0.80)
CLAMP_MARGIN = 0.20  # Margen del SetPredictor adaptativo (LEGACY, escala de SET)

# Margen del clamp adaptativo en escala de PUNTO (A2).
# CLAMP_MARGIN (0.20) se construia alrededor de p_set, que vive en escala de
# SET: error de escala. A2 centra el clamp en el p_punto implicito
# (set_math.p_point_from_p_set) y usa este margen, mas estrecho porque la
# banda util de probabilidad de punto es ~[0.49, 0.55].
# NOTA: el momentum aporta hasta +-MOMENTUM_BONUS*MOMENTUM_MAX_STREAK (0.06)
# mas el momentum global; un margen por debajo de eso anula el momentum.
CLAMP_MARGIN_POINT = 0.10

# Probabilidad de sideout (PointProbabilityModel fallback)
DEFAULT_SIDEOUT_RATE = 0.62

# Damping del MatchPredictor
MATCH_PREDICTOR_DAMPING = 0.5

# Damping adaptativo (Batch 3 mid-effort #3).
# El damping es el exponente en `k_damped = k ** damping` dentro de
# _calibrate_strengths. Convención: damping=0 → no se aplica la
# corrección del MatchPredictor (full shrinkage hacia 0.5); damping=1
# → se aplica la corrección completa (sin shrinkage). El modelo base
# usa MATCH_PREDICTOR_DAMPING=0.5 (mitad de cada uno).
#
# Modo adaptativo: cuando las features están frías (inicio de temporada),
# queremos MÁS shrinkage hacia 0.5 → damping BAJO. Cuando las features
# están cálidas (final), queremos MÁS confianza en el modelo → damping
# ALTO. Por eso START (inicio) < END (final).
ADAPTIVE_DAMPING_START = 0.3  # early season: low damping = strong shrinkage
ADAPTIVE_DAMPING_END = 0.7    # late season: high damping = trust the model
SUPERLEGA_TOTAL_JORNADAS = 26  # ~13 equipos × 2 vueltas = 26 jornadas


def adaptive_damping(
    jornada: int,
    total_jornadas: int = SUPERLEGA_TOTAL_JORNADAS,
    damping_start: float = ADAPTIVE_DAMPING_START,
    damping_end: float = ADAPTIVE_DAMPING_END,
) -> float:
    """
    Linearly interpolate damping from `damping_start` (jornada 0) to
    `damping_end` (jornada >= total_jornadas). Clamps outside the range.

    Args:
        jornada: current match day (1-indexed in the season loop).
        total_jornadas: total jornadas in the season (default 26 for SuperLega).
        damping_start: damping for early season (default 0.3, strong shrinkage
                       when features are cold and the MatchPredictor is unreliable).
        damping_end: damping for late season (default 0.7, more trust in the
                      MatchPredictor once the FeatureBuilder has warmed up).

    Note on the math: the damping is the exponent in `k ** damping` where
    `k = odds_target / odds_base`. damping=0 means k^0 = 1 (no change to
    base strengths); damping=1 means k^1 = k (apply the model's correction
    in full). The intermediate values interpolate. So LOW damping = more
    shrinkage toward the base strength, HIGH damping = trust the model.

    Returns:
        damping value in [damping_start, damping_end] (linear interpolation).
    """
    if total_jornadas <= 0:
        return damping_start
    if jornada <= 0:
        return damping_start
    if jornada >= total_jornadas:
        return damping_end
    progress = jornada / total_jornadas
    return damping_start + (damping_end - damping_start) * progress


# Promedio puntos por set (para features pts_fav_exp)
AVG_POINTS_PER_SET = 23.5

# Momentum (no cambiar sin revisar AGENTS.md)
MOMENTUM_BONUS = 0.015
MOMENTUM_MAX_STREAK = 4
MOMENTUM_DECAY = 0.5
GLOBAL_MOMENTUM_FACTOR = 0.01

# Strength clamps
STRENGTH_CLAMP_RANGE = (0.05, 0.95)

# Monte Carlo cap
MAX_MC_ITERATIONS = 5000
