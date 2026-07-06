# Índice de Documentación — TFG PREDICTOR(2)

Documentación técnica del simulador de partidos/temporadas de la SuperLega italiana. Cada documento cubre un subsistema del proyecto.

## Documentos por Subsistema

### Predicción (lo más maduro del proyecto)

| Documento | Cubre | Estado |
|---|---|---|
| [`prediccion_partidos.md`](prediccion_partidos.md) | `MatchSimulator` — simulación detallada de un partido + Monte Carlo | ✅ Completo |
| [`prediccion_temporadas.md`](prediccion_temporadas.md) | `SeasonSimulator` — simulación de temporada con calibración ML | ✅ Completo |

### Modelos ML

| Documento | Cubre | Estado |
|---|---|---|
| [`set_predictor.md`](set_predictor.md) | `SetPredictor` — ExtraTrees calibrado, AUC=0.654 | ✅ Completo |
| [`match_predictor.md`](match_predictor.md) | `MatchPredictor` — XGBoost calibrado, AUC=0.707, 87 features | ✅ Completo |
| [`point_probability.md`](point_probability.md) | `PointProbabilityModel` — LogisticRegression + sideout 0.62 | ✅ Completo |
| `benchmark.md` | `benchmark.py`, `benchmark_roster.py`, `benchmark_teams.py` — comparativas de modelos | ⏳ Pendiente |

### Capa de Datos

| Documento | Cubre | Estado |
|---|---|---|
| `data_layer.md` | `data_pipeline.py`, `feature_store.py`, `team_mapper.py` — pipeline de datos y features | ⏳ Pendiente |
| `player_stats_generator.md` | `player_stats_generator.py` — generación de stats sintéticas por jugador | ⏳ Pendiente |

### Simulación

| Documento | Cubre | Estado |
|---|---|---|
| `simulator.md` | `MatchSimulator` — motor de Markov chain con momentum y sideout | ⏳ Pendiente |

### Infra

| Documento | Cubre | Estado |
|---|---|---|
| [`../AGENTS.md`](../AGENTS.md) | Layout, comandos, convenciones y gotchas del proyecto | ✅ Completo |

---

## Mapa de Lectura Recomendado

### Para entender el proyecto de cero

1. [`../AGENTS.md`](../AGENTS.md) — layout y comandos
2. [`prediccion_partidos.md`](prediccion_partidos.md) — el corazón de la simulación
3. [`prediccion_temporadas.md`](prediccion_temporadas.md) — la integración ML
4. [`point_probability.md`](point_probability.md) — el modelo más simple (fácil de leer)
5. [`set_predictor.md`](set_predictor.md) — pipeline de un modelo con benchmark
6. [`match_predictor.md`](match_predictor.md) — el modelo más complejo

### Para la defensa del TFG

1. [`prediccion_partidos.md`](prediccion_partidos.md) secciones 1, 2, 3, 6 (lo más preguntable)
2. [`prediccion_temporadas.md`](prediccion_temporadas.md) secciones 3, 6, 7, 12 (calibración y resultados)
3. [`match_predictor.md`](match_predictor.md) sección 6 (uso en el simulador + comparativa con baseline)
4. [`set_predictor.md`](set_predictor.md) sección 5 (feature importance)
5. [`point_probability.md`](point_probability.md) sección 2 (fórmula)

### Para extender/mejorar el código

1. [`prediccion_temporadas.md`](prediccion_temporadas.md) sección 7.3 + sección 14 (limitaciones atacables)
2. [`match_predictor.md`](match_predictor.md) sección 8 (trabajo futuro)
3. `data_layer.md` (cuando exista) — para entender el pipeline de features
4. `simulator.md` (cuando exista) — para entender el Markov chain

---

## Resumen Ejecutivo del TFG

**Pregunta que responde el proyecto**: ¿se puede predecir el resultado de una temporada de la SuperLega italiana con buena calibración usando (a) cadenas de Markov para el detalle punto a punto, (b) tres modelos ML de scikit-learn/XGBoost para la calibración, y (c) datos públicos de los últimos 10 años?

**Respuesta corta**: sí, con limitaciones. El sistema predice el resultado de un partido suelto con error de ~10pp en probabilidad. Una temporada completa simulada genera clasificaciones realistas (44% de 3-0 con ML calibrado vs. 66% sin calibración) en ~70 segundos.

**Stack**:
- **Backend**: Python 3.12, FastAPI, scikit-learn 1.x, XGBoost, LightGBM, pandas, numpy, scipy, joblib
- **Frontend**: Vite + React 18, lucide-react icons, react-router-dom v7
- **Datos**: 22 CSVs de SuperLega, 10 años (2014-2024)
- **Sin tests, sin linter, sin CI** (deuda técnica a resolver)

**Componentes principales**:

```
   ┌─────────────┐
   │   Frontend  │ Vite + React (puerto 5173 dev / 8000 prod)
   │   (español) │
   └──────┬──────┘
          │ HTTP
   ┌──────▼──────┐
   │  FastAPI    │ 5 endpoints (src/api/main.py)
   │  (puerto    │ carga 4 modelos .joblib
   │   8000)     │
   └──────┬──────┘
          │
   ┌──────▼──────────────────────────────────────────┐
   │  MatchSimulator / SeasonSimulator               │
   │  (Markov chain + momentum + sideout)            │
   │                                                  │
   │  Calibrado por:                                  │
   │  ┌────────────────┐  ┌────────────────┐         │
   │  │ MatchPredictor │  │  SetPredictor  │         │
   │  │ (XGBoost AUC   │  │ (ExtraTrees    │         │
   │  │  0.71, damping)│  │  AUC 0.65,     │         │
   │  └────────────────┘  │  clamp adapt.) │         │
   │                      └────────────────┘         │
   │  Alimentado por:                                 │
   │  ┌────────────────┐  ┌────────────────┐         │
   │  │ Point          │  │ PlayerStats    │         │
   │  │ Probability    │  │ Generator      │         │
   │  │ (LogReg +      │  │ (muestreo de   │         │
   │  │  sideout 0.62) │  │  distrib.)     │         │
   │  └────────────────┘  └────────────────┘         │
   └──────────────────────────────────────────────────┘
          │
   ┌──────▼──────────────────────────────────────────┐
   │  Data Layer                                      │
   │  • data_pipeline.py — carga + limpia CSVs        │
   │  • feature_store.py — splits temporales          │
   │  • team_mapper.py — normalización de nombres     │
   │  • RuntimeFeatureBuilder — estado dinámico       │
   │    (Elo, forma, rachas, H2H)                     │
   └──────────────────────────────────────────────────┘
          │
   ┌──────▼──────────────────────────────────────────┐
   │  DB/ — 22 CSVs fuente (10 años de SuperLega)     │
   └──────────────────────────────────────────────────┘
```

**Métricas clave (test 2024, datos no vistos)**:

| Modelo | AUC | Accuracy | Brier |
|---|---:|---:|---:|
| SetPredictor (ExtraTrees + isotonic) | 0.654 | 0.622 | 0.229 |
| MatchPredictor (XGBoost + isotonic) | 0.707 | 0.514 | 0.245 |
| PointProbability (LogReg + sideout) | n/a (regresor) | n/a | n/a |

**Limitaciones documentadas**:
- MatchPredictor con features frías en las primeras jornadas
- SetPredictor inactivo hasta la jornada 5-6
- Damping fijo en 0.5 (no se adapta a la fase de la temporada)
- Stats de jugadores sintéticas (muestreadas, no simuladas)
- Sin Monte Carlo a nivel temporada (incertidumbre no cuantificada)
- Sin lesiones ni mercado de fichajes
- Elo simplificado (sin ajuste por margen de victoria)
- ~40 `print()` con caracteres Unicode que rompen en consola Windows (deuda técnica)
