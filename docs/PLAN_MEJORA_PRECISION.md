# Plan de Mejora de Precisión de los Modelos

> **STATUS: HISTÓRICO (2026-07-22)** — Items B0, B1, B2 (resultado negativo), B3 ejecutados y reflejados en el [Plan Consolidado](./PLAN_MEJORAS_CONSOLIDADO.md) §GRUPO B. Items B4, B5, B6, B7 pendientes y detallados allí. Se conserva este documento como registro del diagnóstico original (Problema 1-4) y de la expectativa honesta de ganancia.

Plan para ser ejecutado por un agente. Objetivo: subir la precisión **real** (no solo la medida) del SetPredictor, MatchPredictor y del pipeline de simulación completo. Ordenado por impacto esperado. Cada fase es independiente y commiteable por separado; ejecutar en orden porque la Fase 0 cambia cómo se mide todo lo demás.

## Contexto: estado actual y diagnóstico

Métricas actuales (test = temporada 2023/2024):
- SetPredictor: ExtraTrees calibrado, AUC test ≈ 0.659, Brier ≈ 0.229
- MatchPredictor: XGBoost enriquecido (87 features), AUC test ≈ 0.70, acc ≈ 0.61
- Batch 3 ya probó y descartó: Optuna tuning (sin mejora), selección de 30 features (degradó −0.088 AUC), sideout per-equipo (sin efecto), damping adaptativo (marginal).

Diagnóstico de por qué el tuning no movió nada — **cuatro problemas estructurales detectados**:

### Problema 1 — El 45% de los datos no se usa (CRÍTICO, mayor palanca)

`DB/features/match_features.csv` tiene **725 partidos** repartidos así:

| Temporada | Partidos | En split actual |
|---|---|---|
| 2016/17–2022/23 | 319 | train |
| 2023/24 | 81 | val |
| 2024/25 | 111 | test |
| **2025/26** | **214** | **SIN USAR** |

`TEMPORAL_SPLITS` en `src/data/feature_store.py:25-29` termina en test=[2024]. La temporada 2025/26 (la más grande del dataset, 214 partidos) no entra en ningún split. El train actual es de solo **319 partidos**. Lo mismo aplica a `set_features.csv` (5073 filas, 10 temporadas hasta 2025/26).

### Problema 2 — La selección de modelo se hace sobre 81 partidos (val=2023)

En `models/benchmark_results/match_benchmark.csv` los AUC de validación son 0.38–0.53 (¡peor que azar!) mientras los de test son 0.60–0.70. Con n=81, el error estándar del AUC es ~±0.06: **elegir el "campeón" por AUC en ese val es esencialmente aleatorio**, y explica por qué Optuna y el feature selection dieron resultados incoherentes entre val y test. Cualquier mejora futura es inmedible con este protocolo.

### Problema 3 — Leakage temporal en las features enriquecidas

`enrich_with_team_stats()` (feature_store.py:109) mergea las stats agregadas de `Comparacion_equipos_10_años.csv` por `(equipo, temporada_inicio)` — es decir, un partido de octubre 2022 recibe las stats del equipo **de la temporada 2022/23 completa**, que incluye partidos futuros. `compute_roster_features()` (feature_store.py:195) hace lo mismo con los totales de temporada de los jugadores. Esto infla las métricas medidas y degrada el uso real (el simulador en runtime no tiene esas stats futuras, usa medias históricas del `RuntimeFeatureBuilder` — hay un mismatch train/serve).

### Problema 4 — Calibración con CV no-temporal e isotónica con pocos datos

`CalibratedClassifierCV(cv=3, method="isotonic")` en set_predictor.py:183 y match_predictor.py:143:
- El `cv=3` interno usa StratifiedKFold que **mezcla temporadas** (leakage dentro de la calibración).
- Isotonic con ~319 muestras (match) sobreajusta; la literatura recomienda sigmoid/Platt por debajo de ~1000 muestras.
- Se calibra sobre train, no sobre predicciones out-of-fold del período de validación.

---

## Fase 0 — Protocolo de evaluación confiable (PRERREQUISITO)

Sin esto, ninguna mejora posterior se puede medir. No mejora los números por sí sola: hace que los números signifiquen algo.

**T0.1 — Validación rolling-origin (expanding window).**
Crear `src/models/evaluation.py` con un evaluador que reemplace el val único de 2023:
- Folds: train hasta T−1 → validar en T, para T ∈ {2021/22, 2022/23, 2023/24, 2024/25}. Test final intocable: 2025/26.
- La selección de modelo campeón y de hiperparámetros se hace por **media de log-loss (primario) y AUC (secundario) sobre los 4 folds**, no sobre un año suelto.
- Reportar desviación estándar entre folds junto a la media.

**T0.2 — Cambiar la métrica de selección de AUC a log-loss.**
En `SetPredictor.train()` y `MatchPredictor.train()`, el campeón se elige hoy por AUC. Las probabilidades alimentan un simulador Monte Carlo: lo que importa es calidad de probabilidad (log-loss/Brier), no ranking. Mantener AUC como métrica reportada.

**T0.3 — Actualizar `benchmark.py`** para usar el protocolo rolling-origin y regenerar los CSVs de `models/benchmark_results/` como nueva línea base. Documentar la línea base ANTES de aplicar cualquier otra fase, para poder atribuir mejoras.

Criterio de aceptación: `python -m src.models.benchmark` produce métricas por fold + media ± std; el test 2025/26 no se toca en ninguna decisión.

## Fase 1 — Datos: usar todo lo disponible y eliminar leakage (mayor ganancia esperada)

**T1.1 — Re-split temporal incluyendo 2024/25 y 2025/26.**
En `feature_store.py`:
```python
TEMPORAL_SPLITS = {
    "train": [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
    "val":   [2024],
    "test":  [2025],
}
```
El train pasa de 319 a ~511 partidos (+60%), val=111, test=214. Con árboles de gradiente y datasets pequeños, +60% de train suele valer más que cualquier tuning. Verificar que `team_mapper.py` cubre todos los equipos de 2025/26 (Grottazzolina, Cuneo, etc. ya están; comprobar contra el CSV). Actualizar los tests que pinean `TEMPORAL_SPLITS`.

**T1.2 — Eliminar el leakage de temporada en `enrich_with_team_stats` y `compute_roster_features`.**
Cambiar el merge de `(equipo, temporada_inicio)` a `(equipo, temporada_inicio - 1)`: cada partido recibe las stats de la **temporada anterior** del equipo (que sí son conocibles antes del partido). Para equipos recién ascendidos sin temporada anterior, imputar con la mediana de la liga (no 0, que hoy distorsiona los diffs). Nota honesta: esto puede BAJAR el AUC medido respecto a la versión con leakage — es esperado y correcto; lo que sube es la precisión real y la coherencia con el `RuntimeFeatureBuilder`. Documentar ambos números.

**T1.3 — Pesos de muestra por recencia.**
En el `fit` de los candidatos, pasar `sample_weight = 0.5 ** (años_de_antigüedad / half_life)` con half-life tuneado en {1.5, 2, 3, 5, ∞} temporadas vía el protocolo de Fase 0. El voleibol tiene rotación alta de plantillas; 2016 no debería pesar igual que 2024. Todos los candidatos de sklearn/XGB/LGBM aceptan `sample_weight`. Ganancia esperada: pequeña pero consistente.

**T1.4 (opcional, mayor esfuerzo) — Ampliar el dataset a nivel de partido.**
`match_features.csv` tiene solo 34–59 partidos en las temporadas viejas (una temporada completa de SuperLega tiene ~132–182). Si existe el generador de ese CSV (no está en el repo), regenerarlo cubriendo TODOS los partidos de cada temporada, no solo los de equipos "viables". `sets_partidos.csv` (5073 sets) sugiere que la cobertura de sets es mucho mejor que la de partidos; investigar la discrepancia. Duplicar el nº de partidos históricos sería la segunda palanca de datos más grande.

## Fase 2 — Features: mejorar la señal base

**T2.1 — Elo con margen de victoria y parámetros optimizados.**
El Elo actual (`feature_builder.py`: K=32, HOME_ADV=65, y el precomputado en match_features.csv) es win/loss puro con constantes nunca validadas. Implementar en un módulo `src/data/ratings.py`:
- Update de Elo escalado por margen: `K_eff = K * (1 + log2(1 + |diff_sets|))` o proporcional al ratio de puntos del partido (los ratings con margen predicen mejor en deportes de sets, cf. FiveThirtyEight).
- Optimizar `(K, HOME_ADV, factor de margen)` minimizando log-loss de `elo_win_prob` sobre los folds de Fase 0 (grid pequeño, es barato).
- Regenerar las columnas `elo_*` del dataset de entrenamiento con el rating mejorado, recorriendo los partidos en orden cronológico (usar `sets_partidos.csv` completo, no solo los 725 partidos con features).
- Mantener el mismo rating en `RuntimeFeatureBuilder` para coherencia train/serve.

**T2.2 — Forma con decaimiento exponencial (EWMA) en vez de last5.**
`win_rate_last5` es una ventana dura. Añadir features EWMA con half-life de ~5 partidos sobre: win rate, set ratio, ratio de puntos. Son las features de "forma" estándar y suelen superar a las ventanas fijas. Añadir tanto al dataset histórico como al `RuntimeFeatureBuilder` (mismo cálculo en ambos).

**T2.3 — H2H con decaimiento temporal.**
`h_h2h_win_rate` actual pesa igual un cruce de 2017 que uno de 2024. Ponderar por `0.5 ** (años / 2)`.

**T2.4 — Continuidad de plantilla (roster churn).**
Con `stats_por_equipo_completo/` calcular por equipo-temporada: % de puntos de la temporada anterior anotados por jugadores que siguen en el equipo. Captura el efecto fichajes/éxodos que ninguna feature actual ve. Es una feature pre-temporada, sin leakage. Añadir `h_roster_continuity`, `a_roster_continuity`, `diff_roster_continuity`.

**T2.5 — Auditar las features base de `match_features.csv` por leakage.**
Verificar (con el generador o empíricamente) que `h_win_rate_global`, `point_ratio_h`, `dominancia_h`, etc. son **rolling pre-partido** y no agregados de temporada completa. Test empírico: para las primeras jornadas de cada temporada esas features deberían estar cerca del prior, no correlacionar con el resultado final de temporada. Si hay leakage, regenerarlas rolling. (El sospechoso principal: `point_ratio_h/a` y `rank_season`.)

## Fase 3 — Modelado: ensemble, calibración y coherencia

**T3.1 — Calibración correcta.**
Reemplazar `CalibratedClassifierCV(cv=3)` por calibración sobre predicciones out-of-fold temporales: entrenar el campeón en cada fold de Fase 0, juntar las predicciones de validación de todos los folds, ajustar el calibrador (comparar `sigmoid` vs `isotonic` por Brier — con ~500 muestras es probable que gane sigmoid) y aplicarlo al modelo reentrenado con todo el train. Verificar con `reliability_curve.py`.

**T3.2 — Blend simple en lugar de campeón único.**
Con las predicciones out-of-fold ya disponibles (T3.1), evaluar el promedio (uniforme y con pesos optimizados por log-loss) de los 3 mejores candidatos (típicamente LogReg + XGBoost + ExtraTrees). El blending de modelos descorrelacionados (lineal + árboles) casi siempre gana 0.01–0.02 de AUC en datasets pequeños. El Stacking del benchmark actual fracasó (AUC test 0.38) porque su meta-learner se ajustó al val de 81 partidos; con OOF multi-fold tiene sentido reintentarlo, pero el promedio ponderado es más robusto y suficiente.

**T3.3 — Coherencia set↔partido: segundo predictor de partido derivado.**
El SetPredictor da p = P(local gana un set). Derivar P(local gana el partido) analíticamente con la fórmula de best-of-5 (suma de binomiales; usar p del primer set como aproximación o iterar con los features de estado de set). Esto da un **segundo estimador independiente** de P(partido) entrenado sobre ~4000 sets en vez de ~500 partidos (mucho más dato por parámetro). Ensamblar con el MatchPredictor directo (blend por log-loss en folds). Ganancia esperada: media-alta, es la forma estándar de explotar la estructura jerárquica del deporte.

**T3.4 — Re-ejecutar la búsqueda de hiperparámetros contra el protocolo nuevo.**
El Optuna de Batch 3 dio negativo, pero optimizaba contra el val de 81 partidos — el resultado no es concluyente. Repetir con la media de log-loss de los 4 folds como objetivo, 100–200 trials, espacio centrado en regularización (max_depth 3–6, min_child_weight/min_samples_leaf altos, learning_rate 0.01–0.1 con n_estimators por early stopping). Si vuelve a dar ≤ +0.005, cerrar definitivamente la vía del tuning y documentarlo.

## Fase 4 — Simulador: precisión end-to-end

La "precisión del proyecto" que ve el usuario final es la del simulador, no la del clasificador aislado.

**T4.1 — Backtest del simulador contra una temporada real.**
Script `src/models/backtest_simulator.py`: recorrer la temporada 2024/25 real partido a partido en orden cronológico; para cada partido, correr `monte_carlo_simulate` (n≥500) con el estado del `RuntimeFeatureBuilder` actualizado con los resultados REALES anteriores; registrar P(local) simulada vs resultado real. Métricas: Brier score del simulador, curva de fiabilidad, y distancia L1 entre la distribución simulada de marcadores (3-0/3-1/3-2) y la real. Esta es la métrica de cabecera del proyecto a partir de ahora.

**T4.2 — Ajustar los parámetros del simulador contra el backtest.**
`MOMENTUM_BONUS=0.015`, `GLOBAL_MOMENTUM_FACTOR`, `CLAMP_MARGIN=0.20`, `MATCH_PREDICTOR_DAMPING=0.5` son valores a priori nunca contrastados con datos. Grid search pequeño minimizando el Brier del backtest T4.1 sobre 2023/24 (tune) y validando en 2024/25. Respetar los clamps duros documentados en AGENTS.md; solo ajustar valores, no eliminar mecanismos.

**T4.3 — PointProbabilityModel: regresión continua.**
Hoy hace LogReg sobre `point_ratio_h > 0.5` binarizado y luego `p = 0.45 + 0.10 * p_dominante` (point_probability.py:161) — tira casi toda la información. Reemplazar por una regresión directa de `point_ratio_h` (GradientBoostingRegressor o Ridge sobre los mismos 6 features, clipped a [0.40, 0.60]). Validar que la P(match) implícita del Markov con ese ratio de punto se acerca a la del MatchPredictor (sanity check: con ratio de punto 0.52 constante, P(ganar set 25 pts) ≈ 0.66 — verificar la cadena).

## Guardrails para el agente ejecutor

1. **Nunca** mezclar temporadas entre train y val/test; el orden temporal es sagrado. El test final (2025/26 tras T1.1) se evalúa UNA vez por fase, nunca para decidir.
2. Todo nombre de equipo pasa por `normalize_team_name()`.
3. Cada cambio de features debe aplicarse **igual** en el dataset de entrenamiento y en `RuntimeFeatureBuilder` (coherencia train/serve). Es el error más fácil de cometer en este repo.
4. Correr `pytest` tras cada tarea; los tests pinean constantes (TEMPORAL_SPLITS, clamps, sideout) — actualizarlos conscientemente cuando la tarea lo requiera, no de pasada.
5. Reentrenar (`python -m src.models.train`) y regenerar benchmarks al cerrar cada fase; commitear los CSVs de resultados como evidencia.
6. Documentar resultados negativos igual que en Batch 3 (es un TFG: el registro honesto de experimentos vale tanto como la mejora).
7. Un commit por tarea, mensaje estilo `models(fase1): re-split temporal con 2025/26 — train 319→511`.

## Expectativa honesta de ganancia

| Fase | Ganancia esperada (AUC match, medida limpia) |
|---|---|
| F0 protocolo | 0 (pero hace medibles las demás) |
| F1 datos | +0.02 a +0.05 (la más segura) |
| F2 features | +0.01 a +0.03 (Elo con margen y EWMA son lo más probable) |
| F3 ensemble+calibración | +0.01 a +0.02 AUC; mejora clara de Brier |
| F4 simulador | no cambia el AUC del clasificador; baja el Brier end-to-end, que es lo que ve el usuario |

Techo realista: AUC ~0.72–0.75 y accuracy ~65–68% a nivel de partido. La literatura de predicción en voleibol profesional (y los mercados de apuestas) se mueve en ese rango; el deporte tiene varianza intrínseca alta por el formato de sets. Si tras F1–F3 el AUC limpio queda en ~0.70, eso ya es un resultado defendible: la mejora clave habrá sido pasar de métricas infladas-y-ruidosas a métricas reales.
