# Predicción de Temporadas

## Descripción

La sección de predicción de temporadas genera un calendario round-robin (ida o ida+vuelta) para los equipos de la SuperLega, simula cada partido individualmente usando el motor de Cadenas de Markov con calibración ML, y acumula clasificaciones según el sistema de puntuación oficial. A diferencia de la predicción de partidos individuales, en este modo se activan dos integraciones ML que no se usan en el partido suelto: el **MatchPredictor** calibra las fuerzas de equipo antes de cada partido y el **SetPredictor** ajusta el clamp de probabilidad punto a punto al inicio de cada set.

*Endpoint: `POST /api/simular/temporada` · Código: `src/api/main.py` (líneas 307-391), `src/simulation/season_simulator.py`, `src/simulation/feature_builder.py`*

---

## 1. Punto de Entrada: `POST /api/simular/temporada`

```json
{
  "equipos": ["Trento", "Perugia", "Verona", "Piacenza", "Lube",
              "Milano", "Modena", "Monza", "Cisterna", "Padova",
              "Taranto", "Grottazzolina"],
  "doble_vuelta": true,
  "semilla": 42,
  "fuerzas": {"Trento": 0.70},
  "half": null,
  "first_half_state": null,
  "use_match_predictor": true,
  "use_set_calibration": true
}
```

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `equipos` | list[string] | requerido | 2-12 equipos de la SuperLega |
| `doble_vuelta` | bool | `true` | `true` = ida y vuelta, `false` = solo ida |
| `semilla` | int | `None` | Reproducibilidad de la simulación |
| `fuerzas` | dict | `None` | Fuerzas personalizadas `{equipo: fuerza}` |
| `half` | string | `null` | `"first"` = solo ida · `"second"` = solo vuelta (con `first_half_state`) |
| `first_half_state` | dict | `null` | Resultado de la primera vuelta (para continuar segunda) |
| `use_match_predictor` | bool | `true` | Calibrar fuerzas con MatchPredictor antes de cada partido |
| `use_set_calibration` | bool | `true` | Ajustar clamp punto a punto con SetPredictor al inicio de cada set |

---

## 2. Sistema de Puntuación SuperLega

La SuperLega italiana utiliza el siguiente sistema de puntos por partido:

| Resultado | Puntos Ganador | Puntos Perdedor |
|---|---|---|
| **3-0** o **3-1** | 3 | 0 |
| **3-2** | 2 | 1 |

Ganar por la vía rápida (3-0 o 3-1) otorga los 3 puntos completos. Forzar el tie-break (3-2) recompensa al perdedor con 1 punto, lo que incentiva competir hasta el final.

```python
# src/simulation/season_simulator.py:19
def match_points(sets_winner: int, sets_loser: int) -> tuple[int, int]:
    if sets_winner == 3 and sets_loser <= 1:
        return (3, 0)   # 3-0 o 3-1
    elif sets_winner == 3 and sets_loser == 2:
        return (2, 1)   # 3-2
```

---

## 3. Arquitectura General

La predicción de temporadas integra tres modelos ML con el motor de simulación:

```
                        ┌─────────────────┐
                        │ RuntimeFeature- │
                        │    Builder      │
                        │  (Elo, forma,   │
                        │   rachas, H2H)  │
                        └────────┬────────┘
                                 │ 87 features
                                 ▼
┌──────────────┐        ┌─────────────────┐        ┌──────────────────┐
│team_strengths│──damp──│ MatchPredictor   │──h_adj─│ MatchSimulator   │
│  (input)     │        │ (XGBoost+isot.)  │        │ (Markov + punto) │
└──────────────┘        │  AUC=0.707       │        └────────┬─────────┘
                       └─────────────────┘                 │
                                                           │ set_context
                                                           ▼
                                                  ┌──────────────────┐
                                                  │ SetPredictor     │
                                                  │ (ExtraTrees+isot)│
                                                  │  clamp [0.20,    │
                                                  │  0.80] adapt.    │
                                                  └──────────────────┘
```

**Comparación clave con la predicción de partidos sueltos:**

| Aspecto | Partido suelto | Temporada |
|---|---|---|
| MatchPredictor | ❌ No se usa (faltan features) | ✅ Se usa para calibrar fuerzas |
| SetPredictor | ❌ No se usa | ✅ Se usa para clamp adaptativo |
| RuntimeFeatureBuilder | ❌ No se usa | ✅ Mantiene estado dinámico |
| Estadísticas jugadores | Opcional | Siempre activas |
| Calendario | N/A | Round-robin generado |

---

## 4. Generador del Calendario

El calendario se genera con permutaciones de los equipos. Para N equipos:

| Modo | Nº partidos | Fórmula |
|---|---|---|
| Ida simple | N×(N−1)/2 | 12 equipos → **66** partidos |
| Ida y vuelta | N×(N−1) | 12 equipos → **132** partidos |

Los partidos se barajan aleatoriamente con `random.shuffle()` para simular un orden de jornadas realista. El simulador también soporta simulación en dos mitades (`half="first"` y `half="second"`) para permitir al usuario continuar desde un estado parcial sin perder progreso.

---

## 5. RuntimeFeatureBuilder — Estado Dinámico de la Temporada

El `RuntimeFeatureBuilder` (`src/simulation/feature_builder.py`) es la pieza clave que permite al MatchPredictor funcionar en tiempo de simulación. Mantiene el estado dinámico de la temporada y construye las 87 features que el modelo necesita en cada partido.

### 5.1. Estado Mantenido

| Atributo | Tipo | Descripción |
|---|---|---|
| `elo` | `dict[team, float]` | Rating Elo (K=28) actualizado tras cada partido |
| `results` | `dict[team, list]` | Historial de resultados `(win, sets_fav, sets_contra)` |
| `h2h` | `dict[(a,b), dict]` | Enfrentamientos directos simulados |
| `streaks` | `dict[team, int]` | Racha actual (+/- consecutiva) |
| `standings_points` | `dict[team, int]` | Puntos SuperLega (para ranking) |
| `elo_h_home` | derivado | Elo local con ventaja de campo (+65) |

### 5.2. Carga de Perfiles Estáticos

Al inicializarse, el `RuntimeFeatureBuilder` lee `DB/features/match_features.csv` y extrae para cada equipo la media histórica de:
- **Features estáticas** (no cambian durante la simulación): roster (`top_scorer_avg`, `roster_depth`, `ace_threat`), team stats (`pts_set`, `aces_set`, `atq_pct`, `rec_eff`, `bloq_set`, `ace_ratio`).
- **H2H histórico** entre pares de equipos (win rate y nº total de enfrentamientos).

### 5.3. Features Dinámicas

Para cada partido simulado, se construyen en tiempo real:

| Feature | Fuente |
|---|---|
| `h_win_rate_global`, `h_win_rate_last5` | Histórico de `results[team]` |
| `h_set_win_rate`, `h_set_diff_exp` | Suma de sets a favor / contra |
| `h_forma_home`, `h_forma_away` | Win rate desglosado |
| `h_racha`, `diff_racha` | `streaks[team]` |
| `elo_h`, `elo_a`, `elo_diff`, `elo_win_prob_h` | `elo[team]` con K=28 y home adv=60 (canónicos en `src/data/rolling_features.py`) |
| `h_h2h_win_rate` | Histórico + simulado |
| `jornada_num` | Contador del partido en el calendario |

### 5.4. Actualización tras cada Partido

```python
# src/simulation/feature_builder.py
def update(self, local, visitante, sets_local, sets_visitante, winner):
    # 1. Actualizar Elo (K=28, canónico de src.data.rolling_features)
    # 2. Añadir resultado a results[team]
    # 3. Actualizar rachas (streaks)
    # 4. Registrar H2H
    # 5. Sumar puntos SuperLega
    # 6. Acumular sets totales
```

---

## 6. MatchPredictor — Calibración de Fuerzas

El MatchPredictor (`src/models/match_predictor.py`) es un clasificador binario que predice `P(local gana partido)` a partir de 87 features. Fue entrenado con split temporal (train: 2016-2022, val: 2023, test: 2024):

| Métrica Test | Valor |
|---|---|
| AUC-ROC | 0.707 |
| Accuracy | 0.514 |
| Brier Score | 0.245 |

### 6.1. Integración en el Flujo de Temporada

Antes de cada partido, el `SeasonSimulator` ejecuta:

```python
# src/simulation/season_simulator.py
match_features_df = self.feature_builder.build_features(home, away, jornada_num)
match_features_df = match_features_df.reindex(columns=self.match_predictor.feature_names, fill_value=0.0)
p_match_home = self.match_predictor.predict_proba(match_features_df)[0, 1]
h_str_adj, a_str = self._calibrate_strengths(h_str_adj, a_str, float(p_match_home))
```

### 6.2. Función de Calibración con Damping

Para evitar sobrecorrección (el MatchPredictor tiene AUC=0.71, no es perfecto), se aplica damping exponencial sobre el odds ratio:

```python
# src/simulation/season_simulator.py:225
@staticmethod
def _calibrate_strengths(h_str, a_str, p_target, damping=0.5):
    p_base = h_str / (h_str + a_str)
    odds_target = p_target / (1 - p_target)
    odds_base = p_base / (1 - p_base)
    k = odds_target / odds_base
    k_damped = k ** damping      # damping=0.5: aplica raíz cuadrada al ajuste
    h_new = h_str * k_damped
    h_new = max(0.05, min(0.95, h_new))
    return h_new, a_str
```

Con `damping=0.5` se aplica `√k` al odds ratio: si el MatchPredictor sugiere el doble de probabilidad que el base, solo se aplica un ajuste de `√2 ≈ 1.41×`.

---

## 7. SetPredictor — Clamp Adaptativo

El SetPredictor (`src/models/set_predictor.py`) es un ExtraTreesClassifier que predice `P(local gana set)` a partir de 20 features (incluyendo estado in-match como `set_num_norm`, `sets_h_antes`, `momentum_h`). En el flujo de temporada, se usa para **relajar el clamp** de probabilidad punto a punto que la simulación de Markov aplica por defecto.

### 7.1. Clamp por Defecto vs. Adaptativo

```python
# src/simulation/simulator.py:229
p_home_wins = np.clip(base_p + momentum_adj, 0.20, 0.80)  # comportamiento por defecto
```

Sin calibración, el clamp es fijo en [0.20, 0.80]. Con el SetPredictor activado, al inicio de cada set se evalúa el modelo y se ajusta el clamp:

```python
# src/simulation/simulator.py:_eval_set_predictor
p_set_home = set_predictor.predict_proba(set_context_df)[0, 1]
margin = 0.20
clamp_low = max(0.10, p_set_home - margin)    # si p=0.75, clamp_low=0.55
clamp_high = min(0.90, p_set_home + margin)   # si p=0.75, clamp_high=0.90
```

### 7.2. Contexto de Set

El `_extract_set_team_features` en `season_simulator.py` mapea las 87 features del match a las 15 features de equipo que el SetPredictor espera (`strength_h`, `elo_diff`, `set_wr_h`, `forma_h`, `pts_fav_h`, `h2h_diff`, etc.). Las features in-match (`set_num_norm`, `sets_h_antes`, `momentum_h`) se calculan dentro del simulador a partir del estado actual.

### 7.3. Limitación Actual

En las primeras jornadas de la temporada, las features de equipo están en valores por defecto (los equipos aún no han jugado y no hay resultados para el `feature_builder`). Esto hace que el SetPredictor prediga ~0.5 para todos los partidos, por lo que el clamp no se desvía del rango por defecto. El efecto del SetPredictor aumenta a medida que avanza la temporada.

---

## 8. Flujo de Simulación de un Partido en Temporada

```
Para cada (home, away) en el calendario:
  │
  ├─ (1) Obtener fuerzas: h_str, a_str
  │       └─ h_str_adj = min(h_str + 0.03, 1.0)  # ventaja de campo
  │
  ├─ (2) Si use_match_predictor: calibrar fuerzas
  │       ├─ feature_builder.build_features(home, away, jornada)
  │       ├─ match_predictor.predict_proba(features) → p_match_home
  │       └─ _calibrate_strengths(h_str_adj, a_str, p_match_home) → h_new
  │
  ├─ (3) Si use_set_calibration: preparar team_features para SetPredictor
  │       └─ _extract_set_team_features(match_features_df) → dict
  │
  ├─ (4) MatchSimulator.simulate_match(
  │       home_team=home, away_team=away,
  │       home_strength=h_str_adj (calibrado si step 2),
  │       away_strength=a_str,
  │       set_predictor=sp (si step 3),
  │       team_features=team_feats,
  │       generate_points=False,
  │       generate_player_stats=True)
  │       │
  │       └─ Para cada set:
  │            ├─ Si set_predictor: ajustar clamp al inicio del set
  │            ├─ Loop punto a punto con momentum y sideout
  │            └─ PlayerStatsGenerator.generate_set_stats()
  │
  ├─ (5) _update_standings(standings, match)  # puntos SuperLega
  │
  ├─ (6) _accumulate_player_stats(player_season_stats, match)
  │
  └─ (7) feature_builder.update(home, away, sets_h, sets_a, winner)
         # actualiza Elo, forma, rachas, H2H para el próximo partido
```

---

## 9. Acumulación de Estadísticas

### 9.1. Clasificación por Equipos (`TeamStanding`)

Cada equipo acumula a lo largo de la simulación:

| Campo | Descripción |
|---|---|
| `points` | Puntos SuperLega (3 para 3-0/3-1, 2+1 para 3-2) |
| `matches_played`, `wins`, `losses` | Partidos y resultados |
| `sets_won`, `sets_lost` | Sets a favor / en contra |
| `points_scored`, `points_conceded` | Puntos de volleyball anotados / recibidos |
| `wins_3_0`, `wins_3_1`, `wins_3_2` | Desglose de victorias por marcador |
| `losses_0_3`, `losses_1_3`, `losses_2_3` | Desglose de derrotas por marcador |
| `set_ratio` (prop) | `sets_won / sets_lost` — desempate |
| `point_ratio` (prop) | `points_scored / points_conceded` — 2º desempate |

### 9.2. Estadísticas de Jugadores

Para cada jugador, se acumulan a lo largo de toda la temporada:

| Estadística | Descripción |
|---|---|
| `partidos` | Partidos disputados |
| `sets` | Sets disputados |
| `puntos` | Puntos totales anotados |
| `aces` | Aces (saques directos) |
| `ataques_ganados` | Ataques ganados |
| `bloqueos` | Bloqueos ganados |
| `recepciones_exc` | Recepciones excelentes |
| `errores_saque` | Errores de saque |

Las stats por set se generan con `PlayerStatsGenerator` (muestreo de distribuciones históricas normalizadas al marcador del set). El season_simulator las agrega partido a partido en `player_season_stats`.

---

## 10. Ordenación de la Clasificación

Los equipos se ordenan por los siguientes criterios en orden de prioridad:

1. **Puntos de clasificación** (descendente)
2. **Set ratio** (`sets_won / sets_lost`, descendente)
3. **Point ratio** (`points_scored / points_conceded`, descendente)

---

## 11. Salida de la Simulación

```json
{
  "clasificacion": [
    {
      "posicion": 1,
      "equipo": "Trento",
      "puntos": 55,
      "pj": 22, "pg": 20, "pp": 2,
      "sg": 62, "sp": 22, "sr": 2.82,
      "pts_favor": 1850, "pts_contra": 1500,
      "v3_0": 8, "v3_1": 10, "v3_2": 2,
      "d2_3": 1, "d1_3": 1, "d0_3": 0,
      "colores": {"primary": "#FFD700", "secondary": "#1B3A5C"}
    }
  ],
  "partidos": [
    {
      "local": "Trento", "visitante": "Perugia",
      "resultado": "3-1", "ganador": "Trento",
      "sets": [{"puntos_local": 25, "puntos_visitante": 22}, "..."]
    }
  ],
  "total_partidos": 132,
  "player_season_stats": [
    {
      "equipo": "Trento",
      "jugador": "Alessandro Michieletto",
      "partidos": 22, "sets": 75, "puntos": 342,
      "aces": 28, "ataques_ganados": 280, "bloqueos": 34,
      "recepciones_exc": 95, "errores_saque": 42
    }
  ],
  "half": null
}
```

---

## 12. Efecto de la Calibración ML en los Resultados

Comparativa de la distribución de marcadores con seed=42 (12 equipos, 132 partidos, stats activadas):

| Configuración | % 3-0 | % 3-1 | % 3-2 | Observación |
|---|---:|---:|---:|---|
| Baseline (sin ML) | 65.9% | 15.9% | 18.2% | Demasiados barridos, pocos partidos competitivos |
| MatchPredictor | 43.9% | 29.5% | 26.5% | Distribución más realista, liga más igualada |
| Match + Set | 45.5% | 30.3% | 24.2% | Similar a Match, con variabilidad extra |

**Interpretación:** la calibración con MatchPredictor reduce la proporción de 3-0 (del 66% al 44%) porque ajusta las fuerzas de los equipos débiles al alza cuando se enfrentan a rivales de su nivel, generando más sets competitivos. Sin calibración, los barridos están sobre-representados.

---

## 13. Rendimiento

| Configuración | Partidos | Tiempo |
|---|---|---|
| Baseline (sin ML) | 132 | ~10 s |
| MatchPredictor activado | 132 | ~70 s |
| Match + Set | 132 | ~72 s |

La calibración con MatchPredictor añade ~60s (de 10s a 70s) porque se evalúa el modelo 132 veces. La calibración con SetPredictor añade solo ~2s adicionales (se evalúa una vez por set, ~4 veces por partido × 132 partidos).

---

## 14. Limitaciones

1. **MatchPredictor con features frías**: en las primeras jornadas, todas las features dinámicas (Elo, forma, rachas) están en valores por defecto. El modelo predice ~0.5 para casi todos los partidos, por lo que la calibración tiene poco efecto hasta la jornada 5-6.

2. **SetPredictor no efectivo por ahora**: las features de equipo que recibe están en valores por defecto hasta que la temporada avance, así que su predicción es ~0.5 y el clamp no se desvía del rango por defecto.

3. **Damping fijo en 0.5**: el factor de damping es estático. Podría ajustarse dinámicamente (mayor damping al inicio de la temporada, menor al final cuando las features están más informadas).

4. **Sin Monte Carlo a nivel temporada**: cada partido se simula una sola vez. Para cuantificar la incertidumbre sobre la posición final, sería necesario ejecutar la temporada completa N veces.

5. **Sin lesiones ni mercado de fichajes**: el rendimiento de los equipos es constante durante toda la temporada.

6. **Elo simplificado**: se usa Elo clásico con K=28, sin ajustes por margen de victoria o importancia del partido.

---

## 15. Conclusión

La predicción de temporadas integra el motor de Cadenas de Markov con dos modelos ML de calibración: el MatchPredictor ajusta las fuerzas de equipo antes de cada partido en función del estado dinámico de la temporada (Elo, forma, rachas, H2H, roster), y el SetPredictor ajusta el clamp de probabilidad punto a punto al inicio de cada set. El RuntimeFeatureBuilder mantiene el estado dinámico y construye las 87 features que necesita el MatchPredictor en tiempo real. El resultado es una simulación con distribución de marcadores más realista (más 3-1 y 3-2, menos 3-0) que el baseline, manteniendo el tiempo de ejecución en ~70 segundos para una temporada completa de 132 partidos con stats de jugadores.
