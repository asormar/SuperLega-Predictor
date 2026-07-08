# Índice de Documentación — TFG PREDICTOR(2)

Documentación técnica del simulador de partidos/temporadas de la SuperLega italiana. Cada documento cubre un subsistema del proyecto.

> **⚠️ ACTUALIZACIÓN 2026-07-08 — Mejora de precisión.** Una auditoría con un
> protocolo de evaluación honesto (rolling-origin, sin leakage) reveló que el
> AUC=0.707 del MatchPredictor era **ficticio** (leakage temporal + un año de
> test afortunado); el valor real era ~0.53. Se reconstruyeron las features sin
> leakage y se integró un modelo Elo con margen de victoria, subiendo el AUC de
> partido a **0.75** y las fuerzas de equipo a la jerarquía real de la liga.
> Proceso completo en [`mejora_precision_2026-07.md`](mejora_precision_2026-07.md)
> y comparación de cifras en [`../COMPARACION_ANTES_DESPUES.md`](../COMPARACION_ANTES_DESPUES.md).
> Las secciones marcadas abajo con métricas antiguas se conservan como registro
> histórico; los valores vigentes están en el documento de proceso.

## Documentos por Subsistema

### Predicción (lo más maduro del proyecto)

| Documento | Cubre | Estado |
|---|---|---|
| [`prediccion_partidos.md`](prediccion_partidos.md) | `MatchSimulator` — simulación detallada de un partido + Monte Carlo | ✅ Completo |
| [`prediccion_temporadas.md`](prediccion_temporadas.md) | `SeasonSimulator` — simulación de temporada con calibración ML | ✅ Completo |

### Modelos ML

| Documento | Cubre | Estado |
|---|---|---|
| [`mejora_precision_2026-07.md`](mejora_precision_2026-07.md) | **Proceso completo de mejora de precisión** (auditoría de leakage, protocolo honesto, integración Elo) | ✅ Nuevo |
| [`set_predictor.md`](set_predictor.md) | `SetPredictor` — LogReg+recencia v2 en producción (test 2025 AUC 0.71*, CV 0.63 ± 0.08; legacy ExtraTrees como fallback) | ✅ Completo |
| [`match_predictor.md`](match_predictor.md) | `MatchPredictor` — el AUC=0.707 era leakage; sustituido por Elo con margen (AUC 0.75); el artefacto viejo queda como fallback | ✅ Completo |
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
   │  │ Elo con margen │  │  SetPredictor  │         │
   │  │  (rolling,     │  │ (LogReg v2     │         │
   │  │   AUC 0.75)    │  │  test 0.71*,   │         │
   │  │ [MatchPredictor│  │  CV 0.63,      │         │
   │  │ 87 feats:      │  │  clamp adapt.) │         │
   │  │ fallback]      │  │ [ExtraTrees:   │         │
   │  └────────────────┘  │  fallback]     │         │
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

**Métricas clave — protocolo honesto (rolling-origin, test held-out 2025/26)**:

| Modelo | AUC antes* | AUC después | Accuracy después |
|---|---:|---:|---:|
| MATCH (antes XGBoost / ahora Elo con margen) | 0.53 | **0.75** | 0.69 |
| SET (ExtraTrees → LogReg+recencia) | 0.65 | **0.71*** | 0.66 |

\* "Antes" = medido honestamente. El AUC=0.707 que reportaba el código para el
match era leakage; el valor real era 0.53. Detalle en
[`mejora_precision_2026-07.md`](mejora_precision_2026-07.md).

\*\* "0.71" del SET es el test sobre 2025 (853 sets, la temporada más grande
del dataset). El CV rolling-origin multi-temporada da **0.63 ± 0.08** y la
media per-year 2018-2025 da 0.61 — el detalle y el follow-up obligatorio
para 2026/27 están en
[`mejora_precision_2026-07.md` §7.2](mejora_precision_2026-07.md).

**Limitaciones documentadas**:
- ~~Elo simplificado (sin ajuste por margen de victoria)~~ → **RESUELTO**: Elo con margen integrado.
- Dataset pequeño a nivel de partido (725 en total; 34-59/temporada en las viejas) → régimen donde modelos lineales baten a árboles profundos.
- Stats de jugadores sintéticas (muestreadas, no simuladas)
- Sin Monte Carlo a nivel temporada por defecto (incertidumbre no cuantificada en un solo seed)
- Sin lesiones ni mercado de fichajes
- MatchPredictor de 87 features (leaky) sigue en disco como fallback; el camino de producción usa la probabilidad de Elo limpia.
- ~40 `print()` con caracteres Unicode que rompen en consola Windows (deuda técnica)
