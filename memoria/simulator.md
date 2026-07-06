# Motor de Simulación (MatchSimulator)

## Descripción

El `MatchSimulator` es el núcleo del sistema. Implementa un motor de simulación punto a punto basado en **Cadenas de Markov** que modela partidos de voleibol de la SuperLega. Cada punto se decide probabilísticamente en función de la fuerza relativa de los equipos, quién está sacando (sideout), el momentum acumulado (rachas), y el estado actual del set. El motor también soporta un modo **Monte Carlo** que ejecuta N simulaciones completas para obtener distribuciones de probabilidad.

*Código: `src/simulation/simulator.py` (504 líneas)*

---

## 1. Arquitectura del Motor

```
┌─────────────────────────────────────────────────────────────┐
│                    MatchSimulator                            │
│                                                             │
│  Atributos:                                                  │
│  ├─ set_predictor (opcional) → clamp adaptativo             │
│  ├─ point_model (opcional) → PointProbabilityModel          │
│  └─ player_stats_gen (opcional) → PlayerStatsGenerator      │
│                                                             │
│  Métodos públicos:                                          │
│  ├─ simulate_match() → MatchResult                          │
│  │     (un partido completo, punto a punto)                 │
│  └─ monte_carlo_simulate() → dict (agregado)                │
│        (N simulaciones, distribuciones de probabilidad)     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
    ┌─────────────────────┐  ┌──────────────────────────┐
    │   _simulate_set()   │  │  _default_point_probs()  │
    │                     │  │                          │
    │  Bucle punto a      │  │  Fuerza relativa →       │
    │  punto con:         │  │  P(base) → ajuste por    │
    │  - Sideout          │  │  sideout (0.62) →        │
    │  - Momentum intra   │  │  clamp [0.25, 0.75]      │
    │  - Clamp adaptativo │  │                          │
    │    (si set_predictor)│  └──────────────────────────┘
    └─────────────────────┘
```

### 1.1. Estructuras de Datos

Tres dataclasses modelan el estado del partido:

| Clase | Campos | Descripción |
|---|---|---|
| `PointResult` | `point_number`, `score_home`, `score_away`, `winner`, `server` | Un punto individual con marcador |
| `SetResult` | `set_number`, `score_home`, `score_away`, `winner`, `points[]`, `home_player_stats[]`, `away_player_stats[]` | Un set completo con su secuencia de puntos y stats |
| `MatchResult` | `home_team`, `away_team`, `sets_home`, `sets_away`, `winner`, `resultado`, `sets[]` | Partido completo con todos los sets |

---

## 2. Algoritmo de Simulación (`simulate_match()`)

### 2.1. Flujo General

```
simulate_match(home, away, h_str, a_str)
│
├─ 1. Fijar semilla (si se provee)
│     random.seed(seed), np.random.seed(seed)
│
├─ 2. Obtener probabilidades de punto
│     ├─ Si point_model y match_features:
│     │   point_model.get_point_probabilities(match_features, h_str, a_str)
│     │   → dict con p_home_serving, p_home_receiving, p_away_serving, p_away_receiving
│     └─ Si no: _default_point_probs(h_str, a_str)
│         → fórmula basada en fuerza relativa + sideout (0.62)
│
├─ 3. Sorteo inicial: quién saca primero (50% cada equipo)
│
├─ 4. Bucle PRINCIPAL: mientras sets_home < 3 y sets_away < 3
│   │
│   ├─ Determinar si es 5º set (target=15) o normal (target=25)
│   │
│   ├─ Si set_predictor: construir contexto base para el set
│   │
│   ├─ _simulate_set() → SetResult
│   │
│   ├─ Si player_stats_gen: generar stats para ambos equipos
│   │
│   ├─ Actualizar sets (home/away)
│   │
│   ├─ Actualizar momentum entre sets:
│   │     Ganador:  momentum = momentum * 0.5 + 0.5
│   │     Perdedor: momentum = momentum * 0.5 - 0.3
│   │
│   └─ Alternar quién saca primero en el siguiente set
│
└─ 5. Devolver MatchResult(winner, resultado, sets[])
```

### 2.2. Bucle de Punto (`_simulate_set()`)

```
_simulate_set(set_number, point_probs, target_score, home_serves_first, ...)
│
├─ 1. Inicializar score_home=0, score_away=0
│      home_serving = home_serves_first
│      streak_home = 0, streak_away = 0
│
├─ 2. Si set_predictor: evaluar clamp adaptativo
│      clamp_low = max(0.10, p_set_home - 0.20)
│      clamp_high = min(0.90, p_set_home + 0.20)
│      (si no hay predictor: clamp_low=0.20, clamp_high=0.80)
│
├─ 3. BUCLE: hasta que _set_finished()
│   │
│   ├─ Seleccionar probabilidad base según sacador
│   │     home_serving → p_home_serving
│   │     away_serving → p_home_receiving
│   │
│   ├─ Calcular ajuste por momentum (rachas):
│   │     adj = min(streak_home, 4) * 0.015
│   │         - min(streak_away, 4) * 0.015
│   │     adj += (momentum_home - momentum_away) * 0.01
│   │
│   ├─ Clampear probabilidad:
│   │     p_home_wins = clip(base_p + adj, clamp_low, clamp_high)
│   │
│   ├─ Decidir ganador del punto:
│   │     random() < p_home_wins → local; si no → visitante
│   │
│   ├─ Sideout: si el receptor gana, toma el saque
│   │
│   ├─ Actualizar rachas (streaks)
│   │
│   └─ Comprobar fin de set:
│         (score ≥ target) AND (diff ≥ 2)
│
└─ 4. Devolver SetResult
```

---

## 3. Parámetros Clave

| Parámetro | Valor | Constante | Efecto |
|---|---|---|---|
| `MOMENTUM_BONUS` | 0.015 | `simulator.py:72` | +1.5% por punto consecutivo |
| `MOMENTUM_MAX_STREAK` | 4 | `simulator.py:73` | Máximo de puntos que acumulan bonus (+6% total) |
| `MOMENTUM_DECAY` | 0.5 | `simulator.py:74` | Decay del momentum entre sets |
| `sideout` | 0.62 | `_default_point_probs()` | P(receptor gana el rally) |
| Clamp por defecto | [0.20, 0.80] | `_simulate_set()` | Límites de p_home_wins |
| Clamp con SetPredictor | [0.10, 0.90] | con ajuste dinámico | Se relaja según contexto |
| Target score normal | 25 | — | Sets 1-4 |
| Target score tie-break | 15 | — | 5º set |
| Win margin | 2 | `_set_finished()` | Diferencia mínima para ganar |

---

## 4. Modelado del Momentum

### 4.1. Momentum Intra-Set (Rachas)

El ajuste por rachas es la diferencia entre las rachas de ambos equipos, cada una limitada a 4 puntos:

```python
momentum_adj = (
    min(streak_home, MOMENTUM_MAX_STREAK) * MOMENTUM_BONUS
    - min(streak_away, MOMENTUM_MAX_STREAK) * MOMENTUM_BONUS
)
```

| Rachas | Ajuste |
|---|---|
| 0-0 | 0% |
| 1-0 | +1.5% |
| 3-0 | +4.5% |
| 4-0 | +6.0% (máximo) |
| 2-1 | +1.5% |
| 4-2 | +3.0% |

Además, se añade un ajuste por **momentum global del partido**:

```python
momentum_adj += (momentum_home - momentum_away) * 0.01
```

### 4.2. Momentum Entre Sets

Al terminar un set, el momentum se actualiza según el ganador:

```python
# Ganador del set
momentum_ganador = momentum_anterior * 0.5 + 0.5
# Perdedor del set
momentum_perdedor = momentum_anterior * 0.5 - 0.3
```

Esto modela:
- **Impulso psicológico** de ganar un set (suma +0.5)
- **Desmoralización** de perderlo (resta -0.3)
- **Decay del 50%** entre sets: el momentum pasado se diluye

### 4.3. Clamp Adaptativo (con SetPredictor)

Cuando se activa el `SetPredictor` (típicamente en simulación de temporada), al inicio de cada set se evalúa el modelo y se ajustan los límites del clamp:

```python
# Si p_set_home = 0.75 (local muy favorito):
margin = 0.20
clamp_low = max(0.10, 0.75 - 0.20) = 0.55
clamp_high = min(0.90, 0.75 + 0.20) = 0.90

# El clamp se desplaza hacia arriba: [0.55, 0.90]
# El local nunca tiene menos de 55% de ganar un punto
```

Sin calibración, el clamp es fijo [0.20, 0.80]. Con el SetPredictor, se relaja para reflejar mejor las diferencias reales entre equipos cuando el modelo tiene alta confianza.

---

## 5. Modelo de Probabilidad de Punto por Defecto

Cuando no se provee un `PointProbabilityModel`, se usa `_default_point_probs()` que calcula cuatro probabilidades a partir de la fuerza relativa:

```python
p_base = home_strength / (home_strength + away_strength)

# Ajuste por sideout: el equipo que recibe tiene ~62% de ganar el rally
p_serving = p_base * (1-0.62) / (p_base * (1-0.62) + (1-p_base) * 0.62)
p_receiving = p_base * 0.62 / (p_base * 0.62 + (1-p_base) * (1-0.62))

# Clamp a [0.25, 0.75] para evitar eventos deterministas
```

El sideout del 62% refleja la ventaja del equipo receptor en voleibol masculino de alto nivel, que puede organizar un ataque combinado tras recibir el saque.

Ver `point_probability.md` para más detalles sobre el modelo alternativo (LogisticRegression + features de partido).

---

## 6. Modo Monte Carlo (`monte_carlo_simulate()`)

### 6.1. Algoritmo

```python
def monte_carlo_simulate(home, away, h_str, a_str, n_simulations=1000):
    results = {"home_wins": 0, "away_wins": 0,
               "score_distribution": {},
               "avg_sets_home": 0, "avg_sets_away": 0}

    for i in range(n_simulations):
        match = self.simulate_match(
            ..., generate_points=False, generate_player_stats=False)
        # Acumular resultados
        results["home_wins"] += 1 if match.winner == "home" else 0
        results["score_distribution"][match.resultado] += 1
        ...

    # Normalizar
    results["home_win_prob"] = home_wins / n_simulations
    for key in score_distribution:
        score_distribution[key] /= n_simulations

    return results
```

### 6.2. Salida

```json
{
  "home_win_prob": 0.623,
  "away_win_prob": 0.377,
  "score_distribution": {
    "3-0": 0.18, "3-1": 0.29, "3-2": 0.15,
    "2-3": 0.14, "1-3": 0.16, "0-3": 0.08
  },
  "avg_sets_home": 2.25,
  "avg_sets_away": 1.35,
  "n_simulations": 1000
}
```

### 6.3. Rendimiento

Cada iteración con `generate_points=False` y `generate_player_stats=False` cuesta ~0.5ms. Una simulación Monte Carlo de 2000 iteraciones se completa en ~1 segundo.

---

## 7. Integración con Modelos ML

El `MatchSimulator` acepta tres componentes opcionales que se conectan en distintos niveles:

| Componente | Clase | Cuándo se usa | Efecto |
|---|---|---|---|
| **PointProbabilityModel** | `point_probability.py` | Siempre que se provea | Probabilidades de punto basadas en features en lugar de solo fuerza |
| **SetPredictor** | `set_predictor.py` | En temporada (ver `prediccion_temporadas.md`) | Ajusta el clamp de probabilidad al inicio de cada set |
| **PlayerStatsGenerator** | `player_stats_generator.py` | Cuando se solicitan stats de jugadores | Genera stats sintéticas por set |

### 7.1. Flujo con Todos los Modelos

```
                    ┌────────────────────┐
                    │  team_strengths     │
                    │  (input del usuario)│
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ MatchPredictor      │ ← solo en temporada
                    │ (calibra fuerzas)   │
                    └─────────┬──────────┘
                              │ h_str_adj
                              ▼
                    ┌────────────────────┐
                    │ PointProbabilityModel│
                    │ (features → 4 probs) │
                    └─────────┬──────────┘
                              │ p_home_serving/receiving
                              ▼
                    ┌────────────────────┐
                    │  MatchSimulator     │
                    │  (Markov chain)     │
                    │                     │
                    │  Por set:           │
                    │  ┌───────────────┐  │
                    │  │ SetPredictor  │  │ ← clamp adaptativo
                    │  │ (evalúa una   │  │
                    │  │  vez/set)     │  │
                    │  └───────────────┘  │
                    │  ┌───────────────┐  │
                    │  │ PlayerStats   │  │ ← stats por jugador
                    │  │ Generator     │  │
                    │  └───────────────┘  │
                    └─────────┬──────────┘
                              │ MatchResult
                              ▼
                    ┌────────────────────┐
                    │   JSON response     │
                    └────────────────────┘
```

---

## 8. Limitaciones

1. **Independencia de puntos (i.i.d.)**: el modelo asume que los puntos son independientes condicionados al estado. No se modelan lesiones, fatiga acumulada, cambios tácticos, ni timeouts.

2. **Sideout rate constante**: 62% es un promedio de liga. En realidad varía significativamente por equipo (mejores receptores tienen sideout más alto).

3. **Sin adaptación in-match**: los equipos no cambian estrategia. Un equipo que va perdiendo 20-15 no se arriesga más en el saque, como ocurriría en un partido real.

4. **Momentum simplificado**: el modelo de rachas lineales (1.5% por punto) es una simplificación. En voleibol real, el momentum no es lineal ni simétrico.

5. **Clamp fijo sin calibración**: el clamp por defecto [0.20, 0.80] evita puntos deterministas pero también limita la expresividad del modelo cuando hay diferencias extremas de calidad.

6. **Four-point check**: el simulador no implementa la regla del four-point check (saque repetido del mismo jugador tras sideout). Esto es una simplificación aceptable para simulación pero diffiere de las reglas reales.

7. **Stats de jugadores post-hoc**: las estadísticas por jugador se generan al final de cada set, no como resultado de la simulación de cada acción individual (ver `player_stats_generator.md`).

---

## 9. Conclusión

El `MatchSimulator` implementa un motor de Cadenas de Markov con dos innovaciones clave para un TFG: (a) modelado de momentum a dos niveles (rachas intra-set y momentum entre sets), y (b) clamp adaptativo vía SetPredictor que ajusta dinámicamente el rango de probabilidad punto a punto. El modo Monte Carlo permite obtener distribuciones de probabilidad completas con ~2000 iteraciones en ~1 segundo. El motor es el orquestador central que integra los tres modelos ML (PointProbability, SetPredictor, PlayerStatsGenerator), cada uno operando a un nivel diferente de la simulación.
