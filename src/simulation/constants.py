"""Constantes del simulador. Centralizadas para facilitar experimentacion."""

# Ventaja de local
HOME_ADVANTAGE_STRENGTH_BONUS = 0.03  # Anadido a la fuerza del local

# Clamps de probabilidad de punto
POINT_PROB_CLIP = (0.25, 0.75)  # Rango final de P(gana punto)
POINT_PROB_CLIP_ADAPTIVE_HARD = (0.10, 0.90)  # Limites duros del clamp adaptativo

# Clamp por defecto (sin SetPredictor)
DEFAULT_CLAMP_RANGE = (0.20, 0.80)
CLAMP_MARGIN = 0.20  # Margen del SetPredictor adaptativo

# Probabilidad de sideout (PointProbabilityModel fallback)
DEFAULT_SIDEOUT_RATE = 0.62

# Damping del MatchPredictor
MATCH_PREDICTOR_DAMPING = 0.5

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
