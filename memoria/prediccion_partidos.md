# Predicción de Partidos Individuales

## Descripción

La sección de predicción de partidos individuales permite simular un partido concreto entre dos equipos (local y visitante) de la SuperLega. Ofrece dos modos de operación: **simulación detallada** (un solo partido punto a punto) y **simulación Monte Carlo** (N iteraciones para obtener distribución de probabilidades y marcadores).

*Endpoints: `POST /api/simular/partido` · Código: `src/api/main.py` (líneas 228-304), `src/simulation/simulator.py`*

---

## 1. Punto de Entrada: `POST /api/simular/partido`

El endpoint recibe los nombres normalizados de los dos equipos, opcionalmente fuerzas personalizadas, semilla para reproducibilidad, y flags para activar la generación de puntos detallados o de estadísticas de jugadores.

```json
{
  "local": "Trento",
  "visitante": "Perugia",
  "fuerza_local": 0.68,
  "fuerza_visitante": 0.65,
  "semilla": 42,
  "generar_puntos": true,
  "generar_stats_jugadores": true,
  "n_simulaciones_mc": 0
}
```

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `local` / `visitante` | string | requerido | Nombre del equipo (se normaliza con `normalize_team_name`) |
| `fuerza_local` / `fuerza_visitante` | float | calculado | Fuerza relativa en [0, 1]. Si no se envía, se obtiene de `TEAM_STRENGTHS` |
| `semilla` | int | `None` | Semilla para `random` y `np.random` (reproducibilidad) |
| `generar_puntos` | bool | `true` | Registrar secuencia de puntos con marcador, ganador y sacador |
| `generar_stats_jugadores` | bool | `true` | Generar stats por jugador para cada set |
| `n_simulaciones_mc` | int | `0` | Si >0, ejecutar Monte Carlo en lugar de simulación única |

Cuando `n_simulaciones_mc > 0`, el endpoint delega en `MatchSimulator.monte_carlo_simulate()` y devuelve distribución agregada. En caso contrario, delega en `MatchSimulator.simulate_match()` y devuelve el resultado punto a punto.

---

## 2. Cálculo de Fuerzas por Defecto

Cuando el usuario no envía `fuerza_local` o `fuerza_visitante`, se obtienen de la variable global `TEAM_STRENGTHS` calculada al arrancar la API (`src/api/main.py:84-118`):

1. Se lee `DB/features/match_features.csv` y se agrupa por equipo.
2. Para cada equipo con al menos 10 partidos, se calcula `wins / total_matches`.
3. Los equipos con menos de 10 partidos reciben un valor de fallback hardcodeado (`_STRENGTH_DEFAULTS`).

Esto produce una tabla de fuerzas en el rango [0.35, 0.68] para la temporada 2024/2025:

| Equipo | Fuerza | Equipo | Fuerza |
|---|---|---|---|
| Trento | 0.68 | Modena | 0.52 |
| Perugia | 0.65 | Monza | 0.48 |
| Verona | 0.60 | Padova | 0.47 |
| Piacenza | 0.58 | Cisterna | 0.45 |
| Lube | 0.56 | Taranto | 0.40 |
| Milano | 0.53 | Grottazzolina | 0.35 |

---

## 3. Arquitectura del Motor de Simulación

El simulador de partidos individuales se compone de tres capas encadenadas:

```
team_strengths  →  PointProbabilityModel  →  MatchSimulator (Markov Chain)
       ↓                    ↓                          ↓
  P(base)             P(gana punto)            Simula punto a punto
                                                  con momentum + sideout
```

### 3.1. PointProbabilityModel (`src/models/point_probability.py`)

Convierte las fuerzas relativas de los equipos en **cuatro probabilidades de punto**, según quién esté sacando:

| Probabilidad | Significado |
|---|---|
| `p_home_serving` | P(local gana el punto \| local saca) |
| `p_home_receiving` | P(local gana el punto \| visitante saca) |
| `p_away_serving` | P(visitante gana el punto \| visitante saca) |
| `p_away_receiving` | P(visitante gana el punto \| local saca) |

**Cálculo:**
1. Probabilidad base `p_base = home_strength / (home + away)`.
2. Ajuste por **sideout rate** (0.62): en volleyball masculino profesional, el equipo que recibe el saque gana ~62% de los rallies.
3. Clamp final al rango **[0.25, 0.75]** para evitar eventos deterministas.

### 3.2. MatchSimulator — Cadenas de Markov (`src/simulation/simulator.py`)

Simula un partido completo punto a punto con un modelo de Markov cuyo estado es:

```
Estado = (puntos_local, puntos_visitante, quién_saca, rachas, momentum)
```

**Algoritmo de simulación de un partido:**

```
1. Sortear quién saca primero (50% cada equipo)
2. Obtener probabilidades de punto del PointProbabilityModel
3. REPETIR hasta que un equipo gane 3 sets:
   a. Inicializar marcador del set (0-0)
   b. REPETIR hasta que el set termine:
      i.   Seleccionar p_home_wins según sacador actual
      ii.  Aplicar ajuste por momentum (rachas + global)
      iii. Clamp p_home_wins al rango [0.20, 0.80]
      iv.  random() < p_home_wins → punto local; si no → punto visitante
      v.   Sideout: si el receptor gana el rally, toma el saque
      vi.  Actualizar rachas (streak_home / streak_away)
      vii. Comprobar fin de set: ≥25 con ≥2 ventaja (≥15 en 5º set)
   c. Actualizar sets ganados y momentum entre sets
   d. Alternar quién saca primero en el siguiente set
4. Devolver MatchResult (ganador, marcador, sets, puntos)
```

### 3.3. Parámetros del Simulador

| Parámetro | Valor | Constante | Descripción |
|---|---|---|---|
| `MOMENTUM_BONUS` | 0.015 | `simulator.py:72` | Bonus por cada punto consecutivo |
| `MOMENTUM_MAX_STREAK` | 4 | `simulator.py:73` | Máximo de puntos que acumulan bonus |
| `MOMENTUM_DECAY` | 0.5 | `simulator.py:74` | Decay del momentum al cambiar de set |
| `sideout_rate` | 0.62 | `point_probability.py:38` | Probabilidad de que el receptor gane el rally |
| `target_score` | 25 / 15 | — | Normal / tie-break (5º set) |
| `win_margin` | 2 | `simulator.py:272` | Diferencia mínima para ganar un set |
| Clamp `p_home_wins` | [0.20, 0.80] | `simulator.py:229` | Evita probabilidades deterministas |

---

## 4. Modelado del Momentum

El simulador implementa momentum a dos niveles:

### 4.1. Momentum intra-set (rachas)

```python
momentum_adj = min(streak_home, 4) * 0.015  -  min(streak_away, 4) * 0.015
```

Una racha de 4 puntos consecutivos otorga un bonus de +6% a la probabilidad de ganar el siguiente punto. Si el rival marca, la racha se resetea a 0.

### 4.2. Momentum entre sets

```python
# Equipo que ganó el set
momentum_ganador = momentum_anterior * 0.5 + 0.5
# Equipo que perdió el set
momentum_perdedor = momentum_anterior * 0.5 - 0.3
```

Modela el impulso psicológico de ganar un set y la desmoralización de perderlo, con un decay del 50% entre sets.

---

## 5. Generación de Estadísticas de Jugadores

Si `generar_stats_jugadores=true`, al final de cada set se invocan dos llamadas a `PlayerStatsGenerator.generate_set_stats(team, pts_favor, pts_contra)` (`src/models/player_stats_generator.py`). El generador:

1. Para cada jugador del roster del equipo, tiene distribuciones estadísticas pre-ajustadas (media y desviación estándar de `puntos`, `aces`, `ataques_ganados`, `bloqueos`, `recepciones_exc`, `errores_saque`).
2. Dado el marcador del set (ej. 25-20), muestrea de esas distribuciones y normaliza al total de puntos del set.

Esto produce para cada set una lista de diccionarios con las stats de cada jugador, tanto del local como del visitante, que se incluyen en la respuesta del endpoint.

---

## 6. Modo Monte Carlo

Cuando `n_simulaciones_mc > 0`, el endpoint delega en `MatchSimulator.monte_carlo_simulate()` que ejecuta el algoritmo de simulación de partido **N veces** con `generate_points=False` y `generate_player_stats=False` para minimizar el coste.

### 6.1. Parámetros del Monte Carlo

| Parámetro | Default API | Default frontend | Descripción |
|---|---|---|---|
| `n_simulations` | 1000 | 2000 | Iteraciones del bucle de simulación |
| `generate_points` | `False` | `False` | No se registran puntos individuales |
| `generate_player_stats` | `False` | `False` | No se generan stats de jugadores |

Con esta configuración, cada iteración cuesta ~0.5ms, permitiendo 2000 simulaciones en ~1 segundo.

### 6.2. Salida del Monte Carlo

```json
{
  "tipo": "monte_carlo",
  "local": "Trento",
  "visitante": "Perugia",
  "prob_local": 0.623,
  "prob_visitante": 0.377,
  "distribucion": {
    "3-0": 0.18,
    "3-1": 0.29,
    "3-2": 0.15,
    "2-3": 0.14,
    "1-3": 0.16,
    "0-3": 0.08
  },
  "n_simulaciones": 2000
}
```

### 6.3. Error Estadístico

Con N=2000 iteraciones, el error estándar de la probabilidad estimada es `√(p·(1−p)/N) ≈ 1.1%`. Para una probabilidad real del 60%, el intervalo de confianza al 95% es [57.8%, 62.2%].

---

## 7. Salida de la Simulación Individual (sin Monte Carlo)

```json
{
  "tipo": "simulacion",
  "local": "Trento",
  "visitante": "Perugia",
  "sets_local": 3,
  "sets_visitante": 1,
  "resultado": "3-1",
  "ganador": "Trento",
  "sets": [
    {
      "numero": 1,
      "puntos_local": 25,
      "puntos_visitante": 22,
      "ganador": "Trento",
      "puntos": [
        {"num": 1, "marcador_local": 1, "marcador_visitante": 0, "ganador": "local", "sacador": "local"},
        "..."
      ],
      "stats_local": [{"jugador": "...", "puntos": 13, "aces": 1, "ataques_ganados": 9, "bloqueos": 2, "..."}],
      "stats_visitante": ["..."]
    },
    "..."
  ],
  "colores_local": {"primary": "#FFD700", "secondary": "#1B3A5C"},
  "colores_visitante": {"primary": "#000000", "secondary": "#D4AF37"}
}
```

---

## 8. Flujo Completo (Diagrama de Secuencia)

```
Usuario (frontend)
       │
       │ POST /api/simular/partido
       ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI handler (src/api/main.py:228)                      │
│  1. Normalizar nombres (normalize_team_name)                 │
│  2. Obtener fuerzas (TEAM_STRENGTHS o input)                │
│  3. ¿n_simulaciones_mc > 0?                                 │
│     ├── SÍ → MatchSimulator.monte_carlo_simulate()          │
│     │           │                                            │
│     │           └── N x MatchSimulator.simulate_match()      │
│     │                  │                                     │
│     │                  ├── PointProbabilityModel             │
│     │                  ├── Markov chain punto a punto        │
│     │                  └── (PlayerStatsGenerator si activo)  │
│     │                                                         │
│     └── NO → MatchSimulator.simulate_match()                │
│                │                                              │
│                ├── (1) si MatchPredictor: calibrar fuerzas   │
│                ├── (2) PointProbabilityModel                  │
│                ├── (3) Markov chain punto a punto            │
│                │     └── clamp adaptativo con SetPredictor   │
│                └── (4) PlayerStatsGenerator por set          │
└──────────────────────────────────────────────────────────────┘
       │
       │ JSON response
       ▼
Frontend muestra resultado
```

**Nota sobre (1) y el clamp de (3):** el MatchPredictor y el SetPredictor están integrados en el `MatchSimulator` y el `SeasonSimulator` (ver `prediccion_temporadas.md`), pero su uso es opcional. En el modo partido suelto (`/api/simular/partido`) se usan los `team_strengths` directamente sin calibración, ya que el partido suelto no tiene el contexto de temporada necesario para construir las 87 features del MatchPredictor.

---

## 9. Limitaciones

1. **Independencia de puntos**: El modelo asume puntos i.i.d. condicionado al estado. Factores reales como lesiones, fatiga acumulada, o cambios tácticos no se modelan.

2. **Sideout rate constante**: El 62% es un promedio de la liga; en realidad varía por equipo (mejores receptores tienen sideout más alto).

3. **Sin adaptación in-match**: Los equipos no cambian estrategia según el desarrollo del partido.

4. **Probabilidad de punto fija por partido**: No evoluciona a lo largo del set, solo se ajusta con momentum. En realidad los equipos modifican el plan de saque/recepción según el marcador.

5. **Stats de jugadores sintéticas**: Las estadísticas por jugador se muestrean de distribuciones pre-ajustadas, no son resultado de una simulación play-by-play.

---

## 10. Conclusión

La simulación de partidos individuales implementa un motor de **Cadenas de Markov con momentum y sideout** alimentado por un `PointProbabilityModel` que convierte fuerzas relativas en probabilidades de punto. El modo detallado genera la secuencia completa de puntos y stats por jugador; el modo Monte Carlo ejecuta N iteraciones para obtener distribuciones de probabilidad. La calidad de la predicción depende de la calidad de las fuerzas de equipo (calculadas desde win rates históricos con fallback hardcodeado) y del modelo de probabilidad de punto subyacente.
