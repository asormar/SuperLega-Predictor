# Plan Consolidado de Mejoras — TFG PREDICTOR(2)

Fecha: 2026-07-15 (v2, ampliado a especificación de implementación).
Análisis de todos los `.md` del proyecto (`docs/` + `memoria/`) cruzado con el
código real. Cada mejora pendiente está especificada con referencias exactas
`archivo:línea`, pasos de implementación, criterio de verificación y gotchas,
para que un agente ejecutor pueda implementarla sin re-derivar el contexto.

> **Cómo usar este documento (para el agente ejecutor):** elige UN item, lee su
> sección completa incluyendo "Guardrails globales" (§final), implementa los
> pasos en orden, verifica con el criterio de aceptación, un commit por item.
> Las referencias `archivo:línea` son del estado actual del repo; verifica que
> siguen vigentes antes de editar (el código puede haber cambiado).

---

## 📊 Estado del plan al 2026-07-22

Leyenda: ✅ HECHO · 🚧 PENDIENTE · ⏸️ BLOQUEADO (por calendario / datos / decisión externa)

### GRUPO A — Clamp adaptativo del SetPredictor

✅ **CERRADO (2026-07-21)** — desenlace: `SET_BLEND_WEIGHT_ELO = 1.0` (SetPredictor cableado pero inactivo en runtime, ver `simulator.md` §4.3 y `set_predictor.md` §10.5). Items A1–A6 todos cerrados.

### GRUPO B — Precisión end-to-end del simulador

| Item | Estado | Notas |
|---|---|---|
| B1 — Backtest del simulador | ✅ HECHO (2026-07-15) | Cuantificó la sobreconfianza pre-B3 (ECE 0.242, 53% de 3-0) |
| B2 — Tuning de constantes del simulador | ✅ HECHO (2026-07-22) — RESULTADO NEGATIVO | Grid de constantes parametrizado y evaluado (36 combos, 12 distintos); resultado negativo, no se adoptan constantes nuevas. Detalle en §B2 más abajo. |
| **B3** — PointProbabilityModel: regresión continua | ✅ **HECHO (2026-07-22)** | Ridge(α=1.0) sobre target continuo + clip (0.40, 0.60). Brier 0.273→0.182, ECE 0.242→0.057, 3-0 53%→37.6% (real 38.7%). El simulador pasa de degradar al Elo a superarlo. |
| **B4** — Predictor de partido best-of-5 | ✅ **HECHO (2026-07-23) — RESULTADO NEGATIVO** | LOFO-CV w_global=0.669, improvement=−0.0004, per-fold wins 2/4. AND-of-4 gate rechazó. §7.5 de `memoria/mejora_precision_2026-07.md` documenta el experimento completo; artefacto en `models/b4_blend_results.json`. La señal del SetPredictor v2 vía best-of-5 no complementa al margin-Elo. |
| B5 — Roster churn | 🚧 PENDIENTE | Única señal pre-temporada de fichajes |
| B6 — Ampliar dataset | 🚧 PENDIENTE | Palanca grande y cara; mejor tras A+B hechos |
| B7 — Re-validación 2026/27 | ⏸️ BLOQUEADO | Depende de calendario; dejar script listo |

### GRUPO C — Infra y calidad

| Item | Estado | Notas |
|---|---|---|
| C1 — Ruff + Black + CI | ✅ HECHO (2026-07-22) | CI matrix Ubuntu+Windows verde; PR #2 en `asormar/SuperLega-Predictor`. Detalle en §C1 más abajo. |
| **C2** — Remote de GitHub | ✅ **HECHO** | `asormar/SuperLega-Predictor` activo, PR #1 mergeado, B3 mergeado |
| C3 — Migración de logging | 🚧 PENDIENTE | 260+ `print()` → `logging` |
| C4 — Hardening del API | 🚧 PENDIENTE | Pre-deploy |
| C5 — Deploy (Docker) | 🚧 PENDIENTE | Multi-stage build |
| C6 — Tests del frontend | 🚧 PENDIENTE | Vitest + MSW |

### GRUPO D — Backlog de "Limitaciones"

Todos 🚧 PENDIENTES (D1 arranque en frío features, D2 player stats realista, D3 saneamiento data layer, D4 sideout por forma).

### GRUPO E — Extras

Todos 🚧 PENDIENTES (E1 MC en UI, E2 explicabilidad, E3 `precision_report.py`, E4 predicción jornada real, E5 blindaje contra regresión).

### Resumen ejecutivo

- **Items cerrados**: B0 (colisión partido_id), B0b (regen set_features), B1, A1–A6, B2 (resultado negativo), B3, B4 (resultado negativo), C1, C2. Total: **12 items**.
- **Items pendientes no bloqueados**: B5, B6, C3, C4, C5, C6, D1–D4, E1–E5. Total: **14 items**.
- **Items bloqueados externamente**: B7. Total: **1 item**.

---

---

## ⚠️ HALLAZGO CRÍTICO (2026-07-15, al implementar B1) — BLOQUEA la validez del pipeline

**`partido_id` colisiona ida y vuelta en `DB/sets_partidos.csv`.** El id abrevia
los nombres a 5 chars y omite la columna `fase` (1st/2nd half), así que
"A vs B" y "B vs A" comparten id. `rolling_features._aggregate_matches` agrupa
por `partido_id` y SUMA los sets de los dos partidos → marcadores imposibles
(3-3, 6-1, 5-4) y target `gana_local` a menudo INVERTIDO.

**Alcance: 596/725 partido_ids (82%) colisionados.** Afecta a TODO lo que pasa
por `_aggregate_matches`: `get_historical_team_elo`, `get_historical_team_strengths`
(las fuerzas del API), `build_rolling_match_features`, y por tanto las AUC
"0.53→0.75" de la memoria.

Reconstrucción correcta = agrupar por `(partido_id, equipo_local)` → **1322
partidos válidos**. Con datos correctos:
- Home-win rate sano **0.48–0.61 en todas las temporadas** (el "0.32-0.35 de las
  temporadas viejas envenenadas" de la memoria era ARTEFACTO del bug).
- Elo AUC **2024=0.756, 2025=0.762** (vs roto: 2024=**0.545**, 2025=0.750).
- La narrativa "recencia para arreglar el signo invertido" se disuelve.

**B0 — Arreglar la colisión.** ✅ **OPCIÓN (a) HECHA (2026-07-15).**
`_aggregate_matches` ([rolling_features.py:53](../src/data/rolling_features.py))
ahora agrupa por `(partido_id, local)` y ordena por `(temporada, fase, jornada)`.
Efecto verificado: 725→**1322 partidos**, marcadores válidos, home-win **0.48-0.61**
todas las temporadas, Elo AUC **2024: 0.545→0.756**, **2025: 0.750→0.762**. Fuerzas
del API corregidas (Perugia 0.833 > Trento 0.734 > … > Grottazzolina 0.183; Taranto
0.35→0.526, resuelve la anomalía "Taranto 4º"). 138 tests verdes, API arranca OK,
`precision_improved.json` regenerado (MATCH 2025: n=314, AUC 0.762, Brier 0.193).

**PENDIENTE de B0 (follow-ups, requieren decisión/esfuerzo aparte):**
- **B0b — Regenerar los CSV pre-generados.** ✅ **set_features.csv HECHO
  (2026-07-15).** Nuevo generador
  [src/data/set_features_builder.py](../src/data/set_features_builder.py):
  reconstruye partidos reales, reutiliza `build_rolling_match_features` (pre-partido,
  sin leakage) y expande cada partido a sus sets con estado in-set correcto
  (`sets_h_antes` ya 0-2, no 0-5). `partido_id` único por partido → sin colisión.
  Backup del viejo en `DB/features/set_features_collided_backup.csv`.
  **Efecto en el SetPredictor v2 (retrain):** CV 2-fold **0.631±0.078 → 0.679±0.017**
  (más alto y MUCHO más estable), logloss CV 0.654→0.634; test 2025 0.709→0.697
  (el "0.71" era el número afortunado de 2025, ver §7.2). 142 tests verdes, API OK.
  - **match_features.csv — NO regenerar ingenuamente.** Sus 67 columnas incluyen
    stats de temporada completa (`enrich_with_team_stats`) que son EXACTAMENTE el
    leakage que el trabajo de precisión eliminó; regenerarlo con enriquecimiento
    lo re-añadiría. La ruta de producción ya NO lo usa para la señal (usa Elo
    rolling). Sus únicos consumidores vivos son: perfiles estáticos de
    `RuntimeFeatureBuilder` (fallback de arranque en frío → mejor atacarlo con
    **D1**, sembrar desde histórico) y el fallback de fuerzas del API (ya superado
    por margin-Elo). Conclusión: **`match_features.csv` es un artefacto legacy a
    retirar, no a regenerar.**
- **B0c — Revisar el hack de recencia.** La narrativa "temporadas viejas
  envenenadas / home-win 0.32 / signo invertido / AUC 0.42" de
  `mejora_precision_2026-07.md` §5.1 era ARTEFACTO de la colisión. Con datos limpios
  (home-win ~0.55 todas las temporadas) probablemente ya no haga falta entrenar solo
  2022-2024 (`RECENT_TRAIN_SEASONS` en train_improved.py:58); re-evaluar con folds.
- **B0d — A1/A2 mezclados.** El dataset incluye Serie A2 (Brescia, Pineto, …) que no
  se cruza con A1 → sus fuerzas quedan infladas fuera de contexto (islas Elo
  desconectadas). Dentro de A1 el orden es correcto. Decidir si separar por
  competición o filtrar A2.
- **B0e — Actualizar memoria + LaTeX.** `mejora_precision_2026-07.md` (§5.1, §5),
  `COMPARACION_ANTES_DESPUES.md`, `INDICE.md` y `latex/` citan la narrativa vieja.
  Añadidos banners de corrección; falta la reescritura editorial (decisión del autor).

---

## 0. Qué está YA hecho (no re-implementar)

Verificado contra código y commits:

- **Plan de precisión F0–F3**: protocolo rolling-origin ([evaluation.py](../src/models/evaluation.py)),
  features rolling sin leakage + Elo con margen ([rolling_features.py](../src/data/rolling_features.py)),
  recencia, LogReg set v2 ([set_predictor_v2.py](../src/models/set_predictor_v2.py)),
  señal de partido = Elo limpio en producción. AUC match 0.53→0.75.
- **Batch 2a** (13 fixes), **Batch 2b** (suite de ~142 tests en `tests/`),
  **Batch 3** (Optuna → negativo ×2, feature selection → −0.088 AUC, sideout
  per-equipo → `src/data/team_sideout.py`, damping adaptativo → opt-in sin efecto).
- **Bug `Optional`** en rolling_features (sembrado Elo roto en silencio) — arreglado.
- **El sembrado de Elo histórico YA es el camino por defecto del API**
  ([main.py:95-109](../src/api/main.py): `RuntimeFeatureBuilder(initial_elo=get_historical_team_elo())`
  con fallback a arranque plano si falla). Lo que sigue frío son las demás
  features dinámicas (ver D1).
- `pyproject.toml` con deps + pytest + ruff + black (line-length 100, target py310). `.gitignore` raíz existe.
- LaTeX: esqueleto de 10 capítulos + apéndices compilando (branch `mejora-precision`).

**Cerrado por evidencia negativa (NO reintentar sin motivo nuevo):** tuning de
hiperparámetros (2 intentos), selección de features, recalibración Platt sobre
histórico viejo (AUC 0.23), damping adaptativo como default.

---

## Mapa de código mínimo (leer antes de tocar nada)

```
src/simulation/simulator.py      MatchSimulator: Markov punto a punto.
  :246-258   clamp adaptativo (se evalúa UNA vez por set, contexto 0-0)
  :328-353   _build_set_context_base (features estáticas del set)
  :355-390   _eval_set_predictor (features in-match; AQUÍ vive el skew)
  :392-419   _default_point_probs (fórmula sideout)
  :421-489   monte_carlo_simulate
src/simulation/season_simulator.py  SeasonSimulator.
  :304-460   simulate_jornada (stateless, seed derivada seed*1000+jornada_index)
  :462-644   simulate_season
  :377-409 y :542-584  señal de partido: elo_win_prob_h del feature_builder;
             MatchPredictor legacy SOLO como fallback si no hay elo_win_prob_h
  :764-796   _calibrate_strengths (odds-ratio con damping k**damping)
  :798-854   _extract_set_team_features (mapeo match-features → set-features)
src/simulation/constants.py
  :7-12      POINT_PROB_CLIP=(0.25,0.75), POINT_PROB_CLIP_ADAPTIVE_HARD=(0.10,0.90),
             DEFAULT_CLAMP_RANGE=(0.20,0.80), CLAMP_MARGIN=0.20
  :15        DEFAULT_SIDEOUT_RATE=0.62
  :18        MATCH_PREDICTOR_DAMPING=0.5
  :77-80     MOMENTUM_BONUS=0.015, MOMENTUM_MAX_STREAK=4, MOMENTUM_DECAY=0.5,
             GLOBAL_MOMENTUM_FACTOR=0.01
src/simulation/feature_builder.py   RuntimeFeatureBuilder (estado dinámico).
  :133-143   _init_dynamic_state (Elo sembrado vía initial_elo; el RESTO arranca frío)
  :287-375   update() — Elo con margen (margin_mult = 1 + 0.15*(mov-1))
src/data/rolling_features.py
  :38-43     constantes canónicas: ELO_BASE=1500, ELO_K=28, ELO_HOME_ADV=60,
             ELO_SEASON_REGRESS=0.25, FORM_HALFLIFE=5, H2H_HALFLIFE_SEASONS=2
  :53-80     _aggregate_matches (sets_partidos.csv → 1 fila/partido, orden cronológico)
  :87-213    build_rolling_match_features
  :216-264   get_historical_team_elo / get_historical_team_strengths
src/data/feature_store.py
  :25-29     TEMPORAL_SPLITS legacy (train 2016-22/val 2023/test 2024) — solo legacy
  :81-100    SET_FEATURE_COLS (21 features del SetPredictor)
src/models/point_probability.py
  :20-23     _FEATURE_KEYS (6 features)
  :26-50     build_features_from_strengths (elo_diff = diff*200)
  :114-117   binarización y_binary = (point_ratio_h > 0.5)
  :161       mapping conservador p = 0.45 + 0.10*p_dominant
src/models/train_improved.py     entrenador de producción (v2).
  :56-60     MATCH_FEATURES, RECENT_TRAIN_SEASONS=[2022,2023,2024],
             TEST_SEASON=2025, SET_RECENCY_HALFLIFE=2.0
src/api/main.py
  :64-109    carga de modelos (v2 con fallback legacy) + Elo sembrado
  :118-164   TEAM_STRENGTHS desde margin-Elo (+_STRENGTH_DEFAULTS)
  :243,:328  use_set_calibration: bool = True (defaults Pydantic)
  :534,:692  set_predictor if req.use_set_calibration else None
Datos: DB/sets_partidos.csv (verdad set a set, 725 partidos, temporadas
  2016..2025 en formato "2016/2017"; temporada_inicio = primer año).
  2024→111 partidos, 2025→214 partidos.
```

**Tests que pinean constantes** (actualizar CONSCIENTEMENTE si un item cambia el valor):
- [test_team_mapper.py:87-142](../tests/test_team_mapper.py) — DEFAULT_SIDEOUT_RATE=0.62,
  DEFAULT_CLAMP_RANGE=(0.20,0.80), CLAMP_MARGIN=0.20, MOMENTUM_*.
- [test_simulator.py:66-107](../tests/test_simulator.py) — ambos clamp ranges y CLAMP_MARGIN,
  y verifica los bounds pasados a `np.clip` en `_simulate_set`.
- [test_models.py:176-177](../tests/test_models.py) — DEFAULT_SIDEOUT_RATE.

---

## GRUPO A — Clamp adaptativo del SetPredictor ✅ **CERRADO (2026-07-21)**

> **Desenlace del grupo: resultado negativo para el SetPredictor en el clamp.**
> A5 ✅, A3 ✅, A2 ✅, A4 ✅, A6 ✅ (A1 se saltó por acuerdo: solo aplicaba si no
> se hacía el resto). El tuneo de A4 elige `SET_BLEND_WEIGHT_ELO = 1.0`, es
> decir, **ignorar al SetPredictor**: `w=0.9` y `w=1.0` dan métricas idénticas
> y coincidentes con la config OFF.
>
> Lo que sí aporta valor es el **reescalado de A2**: centrar el clamp en el
> p_punto implícito (`src/simulation/set_math.py`) con ±0.10 en espacio de
> punto, en vez del rango fijo [0.20, 0.80]. Backtest A5 final:
>
> | Config | \|P_MC − p_elo\| | Spearman | Std pos | Std pts | T(s) |
> |---|---:|---:|---:|---:|---:|
> | OFF | 0.22470 | −0.9720 | 0.1667 | 0.4940 | 7.5 |
> | **NEW (A2+A4)** | **0.22470** | **−0.9720** | **0.0667** | 0.4006 | 9.3 |
>
> Los tres criterios de aceptación se cumplen por primera vez y el coste cae
> **14×** (con `w=1.0` la llamada al SetPredictor se cortocircuita).
>
> **Corrección metodológica (commit `fc8aa6b`):** A5 compartía un único
> `RuntimeFeatureBuilder` entre configs; como acumula estado Elo por temporada
> simulada, cada config contaminaba a la siguiente. Es el mismo gotcha que este
> plan documenta en E1. Todas las cifras de arriba son posteriores al arreglo.
>
> **Lo que el Grupo A NO arregla:** la sobreconfianza que midió B1 (ECE 0.242,
> 53% de 3-0 simulados vs 39% reales). Su origen es el modelo de punto, no el
> clamp. El MC de 20 temporadas post-A2/A4 da Spearman −1.0 con cuatro equipos
> a std 0.00: el simulador produce ligas casi deterministas. La palanca
> pendiente es **B3**.
>
> Artefactos: `models/backtest_clamp_results.json`,
> `models/tune_clamp_margin_results.json`, `models/tune_clamp_blend_results.json`,
> `models/mc_season_validation.json`. Scripts nuevos:
> `src/models/tune_clamp_margin.py`, `src/models/mc_season_validation.py`.
> Docs: `memoria/simulator.md` §4.3 y §10.3, `memoria/mejora_precision_2026-07.md` §7.1.

### Contexto original (histórico)

Diagnóstico ya cuantificado (no repetir): el clamp aporta ρ≈0 de señal
(p_set ∈ [0.537, 0.553] para los 132 pares, std 0.007), +22% de varianza de
posición, comprime favoritos, y cuesta 260 ms/predict (×60 por temporada).
Causas: (1) features runtime fuera de distribución, (2) p_set (escala SET)
usado como centro de clamp de PUNTO, (3) ExtraTrees legacy lento.

**Orden de ejecución del grupo: A5 → A3 → A2 → A4 → A6.** (Primero el metro,
luego las correcciones.) A1 solo si NO se va a hacer el resto pronto.

### A1 — Quick win: desactivar el clamp por defecto

- **Qué**: cambiar `use_set_calibration` de `True` a `False` en 4 sitios:
  [season_simulator.py:313](../src/simulation/season_simulator.py) (simulate_jornada),
  [season_simulator.py:470](../src/simulation/season_simulator.py) (simulate_season),
  [main.py:243](../src/api/main.py) (SimularTemporadaRequest),
  [main.py:328](../src/api/main.py) (SimularJornadaRequest).
- **Verificación**: `pytest -q` verde (ningún test pinea el default a True;
  [test_api_validation.py:174](../tests/test_api_validation.py) ya lo pasa explícito).
  Smoke: `python -m src.simulation.season_simulator` corre en ~1 s/temporada.
- **Documentar** en `memoria/simulator.md` (sección clamp) y en el docstring del request.
- **Esfuerzo**: 30 min. **Riesgo**: bajo (cambia el comportamiento por defecto del
  API; la UI que quiera el clamp debe pasarlo explícito). Reversible al cerrar A2–A5.

### A5 — Backtest reproducible del clamp (`src/models/backtest_clamp.py`) — HACER PRIMERO

- **Qué**: script nuevo que fija el veredicto con números, ejecutable antes y
  después de cada cambio del grupo.
- **Implementación** (módulo nuevo `src/models/backtest_clamp.py`, con el
  boilerplate `BASE_DIR`/`sys.path` estándar del repo):
  1. **Nivel partido**: para los 132 pares ordenados de los 12 equipos de
     `_STRENGTH_DEFAULTS` ([main.py:156-161](../src/api/main.py), los 12 primeros),
     calcular `p_elo = _elo_expected(elo_h + ELO_HOME_ADV, elo_a)` con
     `get_historical_team_elo()` y compararla con `P_MC(home)` de
     `MatchSimulator.monte_carlo_simulate(n_simulations=200, seed=fijo)` en tres
     configs: (a) clamp OFF (`set_predictor=None`), (b) clamp ON actual,
     (c) clamp NUEVO (cuando exista). Reportar media y p95 de `|P_MC − p_elo|`
     por config. Para el camino ON, replicar el wiring de producción: construir
     `team_feats` con `SeasonSimulator._extract_set_team_features(feature_builder.build_features(h, a, jornada=11))`.
  2. **Nivel temporada**: ≥10 temporadas de ida simple (12 equipos, seeds 0..9)
     por config vía `SeasonSimulator.simulate_season(half="first", seed=s, use_set_calibration=...)`.
     Métricas: Spearman(fuerza margin-Elo → posición media) con `scipy.stats.spearmanr`,
     y std de posición por equipo entre seeds (media de las 12 std).
  3. **Time-box** (lección 260ms): medir con `time.perf_counter` el coste de UN
     `predict` y de UNA temporada ANTES de lanzar el loop completo; abortar con
     mensaje si la proyección supera ~15 min.
  4. Salida: imprimir tabla + guardar `models/backtest_clamp_results.json`
     (dict con config → métricas) para citar en la memoria.
- **Criterio de aceptación del grupo** (medido con este script): NUEVO ≥ OFF en
  Spearman; std de posición NUEVO ≤ OFF + 5%; |P_MC − p_elo| de NUEVO ≤ OFF.
- **Esfuerzo**: 3–4 h. **Riesgo**: nulo (solo lectura). **Dependencias**: ninguna.

### A3 — Contrato de features runtime + SetPredictor v2 en el camino del clamp

- **Qué**: eliminar el train/serve skew alimentando al SetPredictor con las
  MISMAS features en entrenamiento y en simulación, y usar el v2 LogReg
  (~0.1 ms/predict) en lugar del ExtraTrees legacy en el clamp.
- **El skew actual, exacto** (entrenamiento = `DB/features/set_features.csv`
  vía `run_pipeline`; runtime = [_extract_set_team_features](../src/simulation/season_simulator.py) +
  [_build_set_context_base](../src/simulation/simulator.py) + [_eval_set_predictor](../src/simulation/simulator.py)):

  | Feature (SET_FEATURE_COLS) | Train | Runtime (código actual) |
  |---|---|---|
  | `pts_fav_h/a` | media ~3.6, rango [1, 5.1] (pts por rotación) | `score_home/away` del set: 0 al evaluar el clamp (simulator.py:377-378) |
  | `momentum_h` | [0, 1] | `(sh−sa)/total` ∈ [−1,1], 0.0 al inicio (simulator.py:374) |
  | `h2h_diff` | [−3, 3] (diff de sets) | `(h2h_wr−0.5)*2` ∈ [−1,1] (season_simulator.py:845) |
  | `strength_h/a` | media 0.53, std 0.09 | `elo/3000` ≈ [0.42, 0.57] (season_simulator.py:850-851) |
  | `set_wr_h/a` | [0.41, 0.58] | rolling [0, 1] |

- **Implementación**:
  1. Crear `src/data/set_feature_contract.py` con UNA función
     `build_set_features(...) -> dict` que produzca las 21 columnas de
     `SET_FEATURE_COLS` ([feature_store.py:81-100](../src/data/feature_store.py))
     con definiciones únicas. Documentar en el docstring la definición y el rango
     esperado de CADA feature.
  2. Regenerar el dataset de entrenamiento del set con ese builder (recorrer
     `sets_partidos.csv` cronológicamente igual que `build_rolling_match_features`,
     manteniendo estado por equipo y emitiendo una fila por set con el estado
     PRE-set). Guardar como `DB/features/set_features_v2.csv` (no pisar el viejo).
  3. Reentrenar el v2 en [train_improved.py](../src/models/train_improved.py)
     `train_set()` apuntando al CSV nuevo (misma config: LogReg C=0.5,
     `max_iter=2000`, recencia half-life 2.0, train 2022-2024). Comparar con
     `_set_rolling_cv` (los 2 folds de train_improved.py:87-90) contra el actual
     (CV 0.63 ± 0.08). Aceptar si CV logloss no empeora.
  4. Reescribir `_extract_set_team_features` (season_simulator.py:798) y
     `_eval_set_predictor` (simulator.py:355) para delegar en el contrato
     (mismo módulo, mismas escalas). En particular: `pts_fav_h/a` debe ser la
     media histórica de puntos por set del equipo (escala ~23) O la métrica que
     el contrato defina — nunca el marcador vivo del set, salvo que el contrato
     entrene con marcador vivo (opción válida: entrenar con snapshots del set;
     decidir en el contrato y aplicar EN AMBOS LADOS).
  5. El API ya carga v2 primero ([main.py:65-68](../src/api/main.py)); no tocar.
     Verificar que el objeto que llega al clamp es el v2 (`type_ == "logreg_recency"`).
- **Verificación**: `python -m src.models.train_improved` regenera artefactos;
  `python -m src.models.measure_precision` no empeora; A5 config (c) mejora vs (b);
  con el v2 en el clamp, p_set debe VARIAR entre pares (std > 0.05 sobre los 132
  pares — hoy es 0.007). Añadir esa asercion como test:
  `tests/test_set_contract.py::test_p_set_discriminates`.
- **Esfuerzo**: 4–6 h. **Riesgo**: medio (reentrenamiento). **Dependencias**: A5.

### A2 — Centro del clamp en p_punto implícito (no p_set)

- **Qué**: convertir p_set a la probabilidad de punto equivalente antes de
  construir el clamp. Corrige el error de escala: un favorito con P(set)=0.75
  necesita P(punto)≈0.55, no 0.75.
- **Implementación**:
  1. Nueva función en `src/simulation/constants.py` (o módulo
     `src/simulation/set_math.py`):
     ```python
     from functools import lru_cache
     from math import comb

     def p_set_from_p_point(p: float, target: int = 25) -> float:
         """P(ganar un set a `target` puntos | p = P(ganar cada punto), iid)."""
         # gana target-j a j, j <= target-2: los primeros target-1+j puntos
         # contienen target-1 ganados; el último punto lo gana el ganador.
         win = sum(comb(target - 1 + j, j) * p**target * (1 - p)**j
                   for j in range(target - 1))
         # deuce: llegar a (target-1)-(target-1) y ganar por 2 (geométrico)
         deuce_reach = comb(2 * (target - 1), target - 1) * (p * (1 - p))**(target - 1)
         p_deuce_win = p * p / (p * p + (1 - p) * (1 - p))
         return win + deuce_reach * p_deuce_win

     @lru_cache(maxsize=4096)
     def p_point_from_p_set(p_set: float, target: int = 25) -> float:
         """Inversa numérica por bisección (monótona creciente)."""
         p_set = min(max(round(p_set, 3), 0.001), 0.999)
         lo, hi = 0.01, 0.99
         for _ in range(60):
             mid = (lo + hi) / 2
             if p_set_from_p_point(mid, target) < p_set:
                 lo = mid
             else:
                 hi = mid
         return (lo + hi) / 2
     ```
     Sanity pineable en test: `p_set_from_p_point(0.52, 25) ≈ 0.66` (±0.02) y
     `p_point_from_p_set(0.5, t) == 0.5` para t ∈ {15, 25}.
  2. En [simulator.py:250-258](../src/simulation/simulator.py), sustituir:
     `p_center = p_point_from_p_set(p_set_home, target_score)` y
     `clamp = [p_center − CLAMP_MARGIN_POINT, p_center + CLAMP_MARGIN_POINT]`
     recortado a `POINT_PROB_CLIP_ADAPTIVE_HARD`. Nueva constante
     `CLAMP_MARGIN_POINT = 0.10` en constants.py (NO reutilizar `CLAMP_MARGIN=0.20`,
     que queda para el legacy hasta que A6 limpie). Nota: `target_score` ya está
     disponible en `_simulate_set` (15 en el 5º set — usar el target correcto).
  3. Tunear `CLAMP_MARGIN_POINT` ∈ {0.05, 0.08, 0.10} con A5 nivel-temporada.
     Recordar que el momentum añade hasta ±0.06 + global (simulator.py:270-275):
     un margen < 0.06 anula el momentum; documentar el elegido.
- **Tests afectados**: test_simulator.py:83-107 (inspecciona los bounds de
  `np.clip` y pinea `CLAMP_MARGIN == 0.20`) — actualizar conscientemente;
  test_team_mapper.py:113-114 (CLAMP_MARGIN) — mantener mientras la constante
  legacy exista.
- **Verificación**: A5 completo; tests nuevos para las 2 funciones matemáticas.
- **Esfuerzo**: 2–3 h. **Riesgo**: medio. **Dependencias**: A5 (y idealmente A3).

### A4 — Blend en espacio de punto en vez de clip duro

- **Qué**: el clip actual es un override del SetPredictor sobre el Elo. Cambiar a
  mezcla: `p_center = w·p_elo_punto + (1−w)·p_set_punto` con `w=0.7` inicial
  (el Elo demostró logloss 0.585; el set v2 CV 0.63). El clamp duro
  `POINT_PROB_CLIP_ADAPTIVE_HARD` queda solo de salvavidas.
- **Implementación**: en `_simulate_set`, `p_elo_punto = p_point_from_p_set(p_elo_match→p_set?)`
  — ojo: p_elo es probabilidad de PARTIDO. Convertir: p_elo(match) → p punto vía
  la cadena inversa match→set→punto, o más simple y robusto: usar como `p_elo_punto`
  la `base_p` que el simulador ya usa (point_probs, derivada de strengths ya
  calibradas por Elo en `_calibrate_strengths`). Es decir: blend entre la señal
  que YA gobierna el punto (base_p) y la del SetPredictor convertida (A2):
  `p_center = w·base_p_neutral + (1−w)·p_set_punto`, con
  `base_p_neutral = (point_probs["p_home_serving"] + point_probs["p_home_receiving"]) / 2`.
  Añadir `w` como constante `SET_BLEND_WEIGHT_ELO = 0.7` en constants.py y
  tunearlo ∈ {0.5, 0.7, 0.9, 1.0} con A5 (w=1.0 ≡ clamp OFF: si gana w=1.0,
  documentarlo y considerar retirar el mecanismo).
- **Esfuerzo**: 1–2 h sobre A2. **Riesgo**: bajo. **Dependencias**: A2 + A5.

### A6 — Tests, docs y corrección del §7.1 invalidado

1. Actualizar/añadir tests: los pineos de clamp (ver A2), tests de
   `p_set_from_p_point`/`p_point_from_p_set`, test de discriminación (A3).
2. Re-ejecutar el MC de 20 temporadas (12 equipos, ida y vuelta, seeds 0-19,
   clamp según config final) y **reemplazar la tabla invalidada** de
   [mejora_precision_2026-07.md §7.1](../memoria/mejora_precision_2026-07.md),
   dejando la vieja marcada como registro histórico. La memoria LaTeX cita esas
   cifras: sincronizar `latex/` si ya migró ese capítulo.
3. Documentar el mecanismo final en `memoria/simulator.md` (sección clamp) y
   el resultado del backtest en `mejora_precision_2026-07.md`.
- **Esfuerzo**: 2–3 h + CPU. **Riesgo**: nulo. **Dependencias**: A2–A5 cerrados.

---

## GRUPO B — Fase 4 (parcialmente completada): precisión end-to-end del simulador

> **Estado al 2026-07-23**: B1 ✅, B2 ✅ (negativo), B3 ✅ y B4 ✅ (negativo) cerrados. Pendientes: B5, B6. B7 bloqueado por calendario.

### B1 — Backtest del simulador contra la temporada real (`src/models/backtest_simulator.py`)  ✅ IMPLEMENTADO (2026-07-15)

- **Estado**: hecho. `src/models/backtest_simulator.py` corre
  `python -m src.models.backtest_simulator --season 2024 --n-sims 500`.
  Usa `load_real_matches` (reconstrucción correcta, evita el bug de B0). Salida:
  `models/backtest_simulator_2024.json` + `models/plots/backtest_simulator_2024.png`.
- **Resultado 2024 (222 partidos, n=500, clamp OFF)**: el SIMULADOR degrada la
  calidad de probabilidad respecto al Elo limpio — Brier 0.273 vs 0.194
  (+0.079), logloss 0.824 vs 0.569, ECE 0.242 vs 0.044 (mal calibrado,
  sobreconfiado), acc 0.649 vs 0.694. Distribución de márgenes: simula
  demasiados 3-0 (53% vs 39% real) y pocos 3-2 (17% vs 26%), L1=0.286.
  → Motiva directamente grupo A (clamp), B2 (momentum/damping) y B3 (point model).
- **Pendiente opcional**: (i) correr en 2025 UNA vez tras cerrar los ajustes;
  (ii) optimizar coste (1.38 s/partido: `get_point_probabilities` se llama
  n_sims veces por partido — cachear por partido, no por simulación);
  (iii) re-correr tras B0 con el pipeline de producción ya corregido para
  confirmar que coincide con `load_real_matches`.

- **Qué**: la métrica de cabecera que falta: ¿qué Brier tiene el SIMULADOR
  (no el clasificador) prediciendo partidos reales?
- **Implementación** (script nuevo):
  1. Cargar `DB/sets_partidos.csv`; `m = _aggregate_matches(sp)` (importar de
     rolling_features) da 1 fila/partido con `temporada_inicio`, `jornada_num`,
     `sets_h/a`, `pts_h/a`, ya en orden cronológico.
  2. Elegir temporada de backtest `T` (2024 para desarrollo; 2025 UNA sola vez
     al final — es el test held-out de los modelos, no quemarlo iterando).
  3. Sembrar estado con historia < T:
     `initial_elo = get_historical_team_elo(sp[sp_temporada < T])` (filtrar el
     DataFrame ANTES de pasarlo — la función replaya todo lo que recibe, ver
     [rolling_features.py:229-249](../src/data/rolling_features.py)) y
     `strengths = get_historical_team_strengths(sp_filtrado)`.
     `fb = RuntimeFeatureBuilder(initial_elo=initial_elo)`.
  4. Para cada partido real de T en orden `(jornada_num, partido_id)`:
     a. `df = fb.build_features(local, visitante, jornada_num)`;
        `p_elo = df.iloc[0]["elo_win_prob_h"]`.
     b. Replicar la calibración de producción:
        `h_adj = min(strengths.get(local,0.5) + HOME_ADVANTAGE_STRENGTH_BONUS, 1.0)`;
        `h_adj, a_str = SeasonSimulator._calibrate_strengths(h_adj, strengths.get(visitante,0.5), p_elo)`.
     c. `mc = MatchSimulator(point_model=…).monte_carlo_simulate(local, visitante,
        h_adj, a_str, match_features=<6 features desde df como en
        season_simulator.py:414-422>, n_simulations=500, seed=partido_idx)`.
        Registrar `p_sim = mc["home_win_prob"]`, la distribución 3-0/3-1/3-2 y `p_elo`.
     d. **Actualizar con el resultado REAL**: `fb.update(local, visitante,
        sets_h, sets_a, "home" si gana_local else "away", points_local=pts_h,
        points_visitante=pts_a)`.
  5. Métricas finales: Brier y logloss de `p_sim` vs `gana_local`; ídem para
     `p_elo` sola (referencia: Brier 0.200, logloss 0.585 en 2025); curva de
     fiabilidad (`sklearn.calibration.calibration_curve`, 8 bins, guardar PNG en
     `models/plots/` como hace reliability_curve.py); distancia L1 entre la
     distribución simulada de marcadores {3-0, 3-1, 3-2} (condicionada a victoria)
     y la real de la temporada.
  6. Guardar `models/backtest_simulator_<T>.json`. Time-box (medir 1 partido
     antes del loop; con clamp OFF debería ser < 1 s/partido con n=500).
- **Interpretación**: si Brier(p_sim) ≫ Brier(p_elo), el pipeline Markov
  DESTRUYE calidad de probabilidad (motiva B2/B3/grupo A). Si ≈, el simulador es
  fiel y solo añade el detalle de marcador.
- **Esfuerzo**: 4–6 h + CPU. **Riesgo**: nulo. **Dependencias**: ninguna.
  Ejecutarlo con `use_set_calibration` OFF y ON si el grupo A no está cerrado.

### B2 — Ajustar las constantes del simulador contra el backtest ✅ **HECHO (2026-07-22) — RESULTADO NEGATIVO**

> **No se adoptan valores nuevos.** El grid no encuentra ninguna mejora
> distinguible del ruido, así que `constants.py`, los pines de
> `test_team_mapper.py` y `AGENTS.md` quedan **sin tocar**.
>
> **1. El eje `damping` es degenerado.** De los 36 combos solo **12 son
> distintos**. `damping` solo mueve `_calibrate_strengths` → fuerzas, y
> `PointProbabilityModel.get_point_probabilities` ignora
> `home_strength`/`away_strength` cuando el modelo está fitted. Verificado:
> salida del modelo bit-idéntica entre fuerzas 0.50/0.50 y 0.80/0.20, y
> backtest 2024 (n=100) con damping 0.3 vs 0.7 → métricas idénticas
> (Brier 0.1850, ECE 0.0594). Implicación de fondo: toda la calibración
> Elo→fuerza es **código muerto** en producción. Anotado como trabajo aparte.
>
> **2. El grid entero cae bajo el ruido de Monte Carlo.** Pasada 2 (top-5,
> 2023+2024, n=500): el ganador es el propio baseline (0.015 / 0.01) con
> Brier ponderado 0.20889 y **delta +0.00000**. Rango completo del grid:
> **0.00157**. Suelo de ruido medido (misma config, 6 semillas base, 2024,
> n=500): Brier σ = **0.00127**, rango **0.00341** — o sea, el grid varía
> menos que cambiar solo la semilla. Y el ranking se invierte entre pasadas:
> el ganador de la pasada 2 era 9.º de 12 en la de n=100.
>
> Por eso el desenlace **no** es "los valores a priori estaban bien", sino
> "ningún valor es distinguible de otro con este poder estadístico".
> Adoptar el ganador sería sobreajustar ruido.
>
> **Lectura:** tras B3, el modelo de punto domina la calidad de probabilidad
> a nivel de partido; el momentum (máx. ±0.06 sobre `p_home_wins`) no mueve
> la aguja de forma medible. Resolver este grid exigiría más datos (B6) o una
> métrica sensible al detalle intra-set, no al resultado del partido.
>
> **3. Validación única sobre 2025 (held-out).** Con la config vigente y el
> modelo de punto entrenado solo con historia < 2025 (314 partidos, n=500):
> Brier **0.1878** vs Elo 0.1924, LogLoss **0.5544** vs 0.5645, Acc **0.7134**
> vs 0.6975, ECE 0.0626 vs 0.0494; márgenes 3-0 40.7 % (real 44.6 %), 3-1
> 34.0 % (30.9 %), 3-2 25.3 % (24.5 %), L1 0.0771. El patrón de 2024 se
> reproduce en held-out: **la mejora de B3 generaliza**.
>
> Artefactos: `models/tune_simulator_constants.json`,
> `models/backtest_noise_floor.json`,
> `models/point_probability_lt2025.joblib`. Scripts nuevos:
> `src/models/tune_simulator_constants.py`,
> `src/models/estimate_backtest_noise.py`.
> Docs: `memoria/mejora_precision_2026-07.md` §7.4, `memoria/simulator.md` §10.4.

### Spec original (histórico)

- **Qué**: primer contraste con datos de `MOMENTUM_BONUS=0.015`,
  `GLOBAL_MOMENTUM_FACTOR=0.01`, `MATCH_PREDICTOR_DAMPING=0.5` y el clamp
  (todas a priori, nunca validadas).
- **Implementación**: parametrizar B1 para aceptar overrides de constantes
  (pasarlas como argumentos, no mutar constants.py en runtime salvo con
  `monkeypatch`-style setattr documentado). Grid pequeño (~36 combos):
  `MOMENTUM_BONUS ∈ {0, 0.01, 0.015, 0.03}`, `GLOBAL_MOMENTUM_FACTOR ∈ {0, 0.01, 0.02}`,
  `damping ∈ {0.3, 0.5, 0.7}` (el parámetro `damping` de simulate_season ya
  acepta valores — season_simulator.py:471). Minimizar Brier del backtest sobre
  **2023 y 2024** (tune); validar el combo ganador UNA vez en 2025.
  Respetar los clamps duros (AGENTS.md): solo ajustar valores, no eliminar mecanismos.
- **Al adoptar valores nuevos**: actualizar constants.py + los pineos en
  test_team_mapper.py:121-130 y AGENTS.md (línea "Momentum params").
- **Esfuerzo**: 2–3 h + CPU (36 combos × ~132 partidos × 500 sims — time-box;
  si excede, bajar a n=200). **Dependencias**: B1.

### B3 — PointProbabilityModel: regresión continua  ✅ **HECHO (2026-07-22)**

> **Resultado POSITIVO y grande.** `Ridge(alpha=1.0)` sobre target continuo
> `point_ratio_h`, features rolling sin leakage, salida
> `clip(pred, POINT_RATIO_CLIP=(0.40, 0.60))`. Se elimina la binarización y el
> mapping `0.45 + 0.10·p_dominante`, cuyo sesgo (p = 0.5387 en neutro) la
> cadena amplificaba ~7× hasta P(local) = 0.845 entre iguales.
>
> Backtest B1 sobre 2024, con el modelo reentrenado **solo con historia < 2024**
> (medida sin leakage; 222 partidos, n=500, clamp OFF):
>
> | Métrica | Antes | **B3** | Elo (ref) |
> |---|---:|---:|---:|
> | Brier | 0.2731 | **0.1815** | 0.1941 |
> | LogLoss | 0.8241 | **0.5365** | 0.5690 |
> | Accuracy | 0.6486 | **0.7207** | 0.6892 |
> | ECE | 0.2419 | **0.0565** | 0.0454 |
> | 3-0 simulado | 53.0 % | **37.6 %** | real 38.7 % |
> | L1 (márgenes) | 0.2858 | **0.0315** | — |
>
> El simulador pasa de degradar la señal Elo a **superarla** en Brier, logloss y
> accuracy. Control de leakage: con el modelo de producción (2016-2025) sale
> Brier 0.1822 / ECE 0.0824 — la mejora es estructural.
>
> **Desviación consciente de la spec:** el plan mapeaba
> `diff_dominancia → diff_set_diff_exp`. Eso habría metido train/serve skew,
> porque en runtime `diff_dominancia`, `diff_set_win_rate` y `diff_set_ratio`
> son **algebraicamente idénticas** (`feature_builder.py:264-266`:
> `dominancia = set_win_rate − 0.5`). Se mapean las tres a `diff_set_ratio`
> para reproducir esa identidad; la L2 de Ridge absorbe la colinealidad.
>
> **Corrección al sanity del plan:** pedía pinear
> `p_set_from_p_point(0.52, 25) ≈ 0.66` → banda P(match) [0.71, 0.77]. Ese 0.66
> es incorrecto (mismo error que ya se corrigió en A2): el valor real es
> **0.6131** → **P(match) = 0.6967**. El test pinea el valor derivado de
> `set_math`. El MC da 0.6845 (Δ = 0.012): la cadena conserva.
>
> **Bonus — bug de instrumento encontrado:** `mc_season_validation.py` (A6)
> construía el simulador con `point_model=None`, midiendo el fallback en vez
> del modelo de producción. Corregido; la dispersión de temporada sube 2,7×
> (std 0.457 → 1.247) y el Spearman −1.0 pasa a −0.972. Parte de la
> "sub-dispersión" que A6 atribuyó al simulador era del instrumento.
>
> Artefactos: `models/point_probability.joblib` (regenerado),
> `models/point_probability_lt2024.joblib` (para backtest sin leakage),
> `models/backtest_simulator_2024.json`, `models/backtest_simulator_2024_pre_b3.json`,
> `models/mc_season_validation.json`.
> Docs: `memoria/mejora_precision_2026-07.md` §7.3 y §7.1,
> `memoria/simulator.md` §10.1-10.2.

### Spec original (histórico)

- **Qué**: hoy el modelo binariza el target ([point_probability.py:114-117](../src/models/point_probability.py))
  y aplasta la salida a `p = 0.45 + 0.10*p_dominant` (:161). Sustituir por
  regresión directa del point ratio.
- **Implementación**:
  1. En `fit()`: reemplazar LogReg binarizada por `Ridge(alpha=1.0)` (o
     `GradientBoostingRegressor` si Ridge no discrimina) con target continuo
     `y = point_ratio_h` (el ratio real del partido, es outcome — válido como
     target). **Entrenar sobre las features rolling sin leakage**: usar
     `build_rolling_match_features(sp)` y mapear las 6 `_FEATURE_KEYS` a sus
     equivalentes rolling (`elo_diff`→`elo_diff`, `diff_win_rate_global`→`diff_win_rate`,
     `diff_set_win_rate`→`diff_set_ratio`, `diff_dominancia`→`diff_set_diff_exp`,
     `diff_set_ratio`→`diff_set_ratio`, `diff_forma_efectiva`→`diff_form_ewma`) y
     target `pts_h/(pts_h+pts_a)` — o cambiar `_FEATURE_KEYS` a las rolling
     directamente. Si se cambian las keys: actualizar los 3 productores del dict
     (`build_features_from_strengths` :26-50, y los dos bloques
     `point_match_features` en season_simulator.py:414-422 y :589-597).
  2. En `get_point_probabilities()`: `p_home_point = clip(pred, 0.40, 0.60)`
     (nueva constante `POINT_RATIO_CLIP = (0.40, 0.60)` en constants.py).
     Eliminar el mapping 0.45+0.10·p.
  3. Reentrenar (`train.py` es quien llama a `PointProbabilityModel.fit` — localizar
     el call site con grep y adaptarlo al nuevo dataset) y regenerar
     `models/point_probability.joblib`. El formato del joblib cambia (guarda
     `model` + `scaler` + `feature_cols` — mantener claves para no romper `load()`).
  4. **Sanity check de la cadena de Markov** (test nuevo): con
     `p_home_point=0.52` constante y sin momentum, `p_set_from_p_point(0.52, 25) ≈ 0.66`;
     y `P_match` del MC (n=2000) con ese ratio debe caer en 0.66³ best-of-5 ≈ 0.74±0.03.
  5. Validación temporal: medir logloss/Brier de la P(match) implícita del MC
     con el modelo nuevo vs viejo usando B1 sobre 2024.
- **Tests afectados**: test_models.py (smoke del PointProbabilityModel con
  `.fit()` sintético — el fit nuevo debe seguir aceptando un DataFrame con esas
  columnas), test_simulator.py:193+ (conservación de Markov — no cambia).
- **Esfuerzo**: 3–4 h. **Riesgo**: medio (afecta a toda la distribución punto a
  punto). **Dependencias**: B1 para validar; A2 aporta `p_set_from_p_point`.

### B4 — Predictor de partido derivado del SetPredictor (best-of-5)  ✅ HECHO (2026-07-23) — resultado NEGATIVO documentado en §7.5

- **Qué**: segundo estimador independiente de P(match) desde p_set (entrenado
  sobre ~5000 sets, no ~500 partidos). Se pospuso por retorno marginal; es el
  candidato más plausible a superar el 0.75 del Elo.
- **Implementación**:
  1. `p_set = v2.predict_proba(contexto 0-0)` por partido (con el contrato de A3).
  2. `P(match) = q³·(1 + 3(1−q) + 6(1−q)²·q₅/q... )` — usar la forma exacta:
     `P = q³ + 3·q³·(1−q) + 6·q²·(1−q)²·q₅` donde `q` es p_set a 25 y
     `q₅` la p_set del tiebreak (usar `p_set_from_p_point(p_point_from_p_set(q,25),15)`
     si A2 existe; si no, `q₅ = q` como aproximación documentada).
  3. Evaluar blend `P_final = w·P_elo + (1−w)·P_derivada` con
     `evaluate_model_rolling`-style sobre los folds de [evaluation.py](../src/models/evaluation.py)
     (val years 2021-2024), optimizando `w` por logloss medio en folds.
     Test held-out 2025 UNA vez al final.
  4. **Criterio de adopción**: logloss medio en folds < logloss del Elo solo.
     Si no gana: documentar como experimento negativo (patrón Batch 3) en
     `memoria/mejora_precision_2026-07.md` y NO integrar.
  5. Si gana: integrar en `season_simulator` sustituyendo `p_match_home = p_elo`
     por el blend (líneas :388-391 y :558-563) y en `train_improved.py` como
     artefacto (guardar `w` y la config en el joblib).
- **Esfuerzo**: 3–4 h. **Riesgo**: bajo (gated por folds). **Dependencias**: A3 (contrato).

### B5 — Feature de continuidad de plantilla (roster churn)  ✅ HECHO (2026-07-23) — RESULTADO NEGATIVO

- **Qué**: % de los puntos de la temporada T−1 anotados por jugadores que siguen
  en el equipo en T. Única señal de fichajes/éxodos disponible; pre-temporada,
  sin leakage.
- **Implementación**:
  1. Explorar `DB/stats_por_equipo_completo/` (CSVs por equipo-temporada;
     verificar formato real antes de codificar — columnas de jugador y puntos
     totales). Construir tabla larga `(equipo_norm, temporada_inicio, jugador, puntos)`.
     Normalizar equipos con `normalize_team_name`; jugadores: matching exacto por
     nombre (documentar la limitación de homónimos/grafías).
  2. `churn(team, T) = Σ puntos_{T−1}(jugadores presentes en T) / Σ puntos_{T−1}(todos)`.
     Equipos sin T−1 (ascendidos): imputar la MEDIANA de la liga (no 0).
  3. Añadir a `build_rolling_match_features` como `h_roster_continuity`,
     `a_roster_continuity`, `diff_roster_continuity` (merge por
     (equipo, temporada_inicio); es constante dentro de la temporada).
  4. Evaluación: el match de producción es Elo puro (sin modelo entrenado), así
     que el churn solo puede entrar vía un modelo: evaluar
     `LogisticRegression(C=0.5)` sobre `[logit(elo_win_prob_h), diff_roster_continuity]`
     con pesos de recencia, contra el Elo solo, con el protocolo de folds
     (mismo gate que B4). También probarla en el SET v2 (añadir al contrato A3).
- **Esfuerzo**: 3–5 h (el join de jugadores es lo laborioso). **Riesgo**: bajo
  (gated). **Dependencias**: ninguna dura; sinergia con A3/B4.

### B6 — Ampliar el dataset de partidos (la palanca grande, la más cara)  🚧 PENDIENTE

- **Qué**: las temporadas viejas tienen 34-59 partidos de ~132-182 reales (solo
  se scrapeó a equipos "viables"). El generador de `sets_partidos.csv` NO está
  en el repo.
- **Implementación**: (a) identificar la fuente (los resultados oficiales de
  legavolley.it publican marcadores por set de todas las jornadas; verificar
  disponibilidad histórica); (b) scraper/loader nuevo en `src/data/` que emita
  filas con el MISMO esquema de `sets_partidos.csv` (columnas: `partido_id`,
  `temporada`, `jornada`, `equipo_local`, `equipo_visitante`, `puntos_local`,
  `puntos_visitante`, `ganador_set_local` — verificar contra el CSV real);
  (c) pasar todo nombre por `normalize_team_name` y AÑADIR alias nuevos a
  `TEAM_ALIASES` en team_mapper.py; (d) regenerar features y reentrenar
  (`train_improved`), re-medir con `measure_precision`.
- **Hipótesis verificable**: el % de victoria local ~0.32-0.36 de 2016-2020 es
  artefacto del muestreo sesgado (se scrapeó a equipos rastreados, que suelen
  ser fuertes y ganar fuera). Si con cobertura completa el home-win vuelve a
  ~0.55-0.60, las temporadas viejas dejan de "envenenar" y la recencia podría
  relajarse → más datos útiles.
- **Esfuerzo**: 6–12 h. **Riesgo**: medio (datos externos, nombres nuevos).
  **Dependencias**: ninguna, pero hacer DESPUÉS de A+B1 para poder medir su efecto.

### B7 — Re-validación con 2026/27 (bloqueada por calendario; dejar preparado)  ⏸️ BLOQUEADO

- **Qué**: el follow-up obligatorio W1 de mejora_precision §7.2 — decidir si el
  AUC 0.71 del set v2 en 2025 fue estructural o suerte.
- **Implementación ahora** (sin datos nuevos): script
  `src/models/revalidate_next_season.py` que (1) detecte la última temporada en
  `sets_partidos.csv`, (2) si es > 2025: reentrene con `train_improved` y mida
  en ella con el protocolo intacto, (3) compare contra la tabla per-year
  2018-2025 (hardcodearla del §7.2) e imprima el veredicto:
  AUC ≥ 0.65 → estructural; ~0.60 → el 0.71 era coincidencia.
  **No tocar protocolo ni modelo: solo añadir datos y medir igual.**
- **Esfuerzo**: 1 h ahora + 30 min cuando haya datos (temporada arranca ~octubre 2026).

---

## GRUPO C — Infra y calidad (Batch 2c/2d parcialmente avanzados)

> **Estado al 2026-07-22**: C1 ✅, C2 ✅ (remote activo en `asormar/SuperLega-Predictor`, PR #1 y B3 mergeados). Pendientes: C3, C4, C5, C6.

### C1 — Ruff + Black + CI  ✅ HECHO (2026-07-22)

Ruff + Black configurados en `pyproject.toml` (`[tool.ruff]` + `[tool.black]`, `line-length = 100`, `target-version = "py310"` para coincidir con `requires-python = ">=3.10"`, exclusiones: `src/web`, `latex`, `memoria`, `docs`). E402 ignorado en `[tool.ruff.lint]` por el boilerplate `sys.path.insert(0, str(BASE_DIR))` de los módulos de `src/` (Guardrail 7). 49 archivos reformateados por el autofix en un commit dedicado; 6 lints resueltos a mano en otro commit. `.github/workflows/ci.yml` con matrix `ubuntu-latest` + `windows-latest`, `fail-fast: false`, `concurrency` con `cancel-in-progress`, y `workflow_dispatch`. CI verde en ambos OS. PR #2 abierta en `asormar/SuperLega-Predictor`.

- **Esfuerzo**: ~2 h. **Dependencias**: C2 ✅ (hecho).
- **Gotcha**: `hyperparameter_search` (módulo de experimento, Batch 3 resultado negativo) requería `optuna`, que no es dep del proyecto; el test correspondiente usa `importorskip` para no romper CI.

### C2 — Remote de GitHub (seguro de vida del TFG)  ✅ **HECHO**

- Hoy el repo existe SOLO en el disco local. Requiere login del usuario:
  ```bash
  gh auth login
  cd "C:\Users\Alejandro\Desktop\Universidad\4toCarrera\TFG\PREDICTOR(2)"
  gh repo create asormar/superlega-predictor --private --source=. --remote=origin --push
  ```
  (privado por defecto para un TFG en curso; público cuando el usuario decida).
  Antes de pushear: verificar que `src/web/node_modules/` no está trackeado
  (AGENTS.md avisa que puede estarlo — `git ls-files src/web/node_modules | head`;
  si aparece: añadir a .gitignore y `git rm -r --cached`).
- **Esfuerzo**: 10-20 min. **Bloqueo**: interactivo (login del usuario).

**Implementación real (2026-07-15):** repo creado en `github.com/asormar/SuperLega-Predictor`
(privado), `git remote add origin` configurado, y dos PRs mergeados a `main`:
- PR #1 (merge 2026-07-21, commit `d26a9b0`): Grupo A del plan consolidado
  (A5, A3, A2, A4, A6 — `mejora-precision` branch).
- B3 (merge 2026-07-22, commit `26439cb`): PointProbabilityModel con regresión
  continua (`b3-point-prob-regression` branch).

`origin/main` y `main` local sincronizados en `07e6316`. C1 ya puede correr
workflows sobre el remote.

### C3 — Migración de logging  🚧 PENDIENTE

- 260+ `print()` en `src/` → `logging`. Crear `src/logging_config.py` con
  `logging.basicConfig`/dictConfig (formato `%(levelname)s %(name)s: %(message)s`,
  nivel INFO por defecto, DEBUG vía env var `PREDICTOR_LOG_LEVEL`). Llamarlo en
  los entrypoints (`api/main.py`, `train*.py`, `benchmark*.py`). Reemplazo
  mecánico módulo a módulo (un commit por módulo grande); mantener ASCII puro
  en los mensajes (consola Windows cp1252 — regla existente del repo).
  Los `print(..., file=sys.stderr)` de season_simulator.py:406,581 →
  `logger.warning`. **No tocar los prints de los `if __name__ == "__main__"`**
  (son CLIs de smoke test, salida legítima).
- **Gotcha**: tests capturan stdout en algunos smoke tests — correr `pytest -q`
  tras cada módulo migrado.
- **Esfuerzo**: ~2 h.

### C4 — Hardening del API  🚧 PENDIENTE

- CORS: `allow_origins` desde env var `PREDICTOR_CORS_ORIGINS` (default `*` en
  dev) en [main.py:53-59](../src/api/main.py). Rate limiting con `slowapi`
  sobre los 3 endpoints `/api/simular/*` (p.ej. 20/min por IP). Opcional API key
  por header `X-API-Key` con env var.
- **Esfuerzo**: 1–2 h. Solo tiene sentido antes de C5/deploy público.

### C5 — Deploy (Docker)  🚧 PENDIENTE

- Dockerfile multi-stage: (1) `node:20` → `npm ci && npm run build` en
  `src/web` → `dist/`; (2) `python:3.12-slim` → `pip install -e .` +
  copiar `dist/` → el API la sirve sola ([main.py:786-788](../src/api/main.py)).
  **Gotcha central**: `models/*.joblib` y `player_stats_params.json` están
  gitignored — el build debe ejecutar `python -m src.models.train_improved` y
  `python -m src.models.train` (para point_probability + player_stats) o
  copiar los artefactos como build context. CMD:
  `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` (gunicorn con workers
  requiere revisar el singleton `feature_builder` — con threading lock está
  bien EN un proceso; N workers = N estados independientes, aceptable pero
  documentarlo).
- **Esfuerzo**: 3–4 h.

### C6 — Tests del frontend  🚧 PENDIENTE

- Vitest + @testing-library/react para las 4 páginas de `src/web/src/pages/`
  (Dashboard, EquipoDetalle, SimularPartido, SimularTemporada). Prioridad: el
  flujo jornada-a-jornada de SimularTemporada (estado acumulado que el frontend
  reenvía — la lógica más frágil). Mockear fetch con MSW o stubs. Añadir
  `"test": "vitest run"` a package.json y al CI (C1).
- **Esfuerzo**: 3–4 h.

---

## GRUPO D — Backlog de las secciones "Limitaciones" de memoria/  🚧 PENDIENTE

> **Estado al 2026-07-22**: D1–D4 todos pendientes. Ningún avance aún.

### D1 — Arranque en frío de las features dinámicas no-Elo

- **Estado real**: el Elo YA arranca sembrado en el API (main.py:98-104). Pero
  `RuntimeFeatureBuilder._init_dynamic_state` ([feature_builder.py:133-143](../src/simulation/feature_builder.py))
  deja `results`, `streaks`, `h2h` (el simulado), `standings_points` vacíos →
  `win_rate_*`, `set_win_rate`, `forma_*`, `racha` valen 0.5/0 hasta la jornada
  ~5. La señal de partido (elo_win_prob_h) NO sufre; sufren las features del
  SetPredictor y del PointProbabilityModel.
- **Implementación**: nuevo parámetro `initial_form: Optional[dict]` (o ampliar
  `initial_elo` a un dict de estado) que siembre `results` con los últimos ~10
  partidos REALES de cada equipo (derivables de `_aggregate_matches(sp)` de la
  última temporada). En el API, construirlo junto al Elo (mismo try/except).
  Mantener el arranque vacío como default del constructor (los tests de schema
  actuales asumen eso — actualizar solo el call site del API).
- **Verificación**: en la jornada 1 de una simulación, `h_win_rate_global` de
  Perugia ≠ 0.5. **Esfuerzo**: 1–2 h. **Riesgo**: bajo. Sinergia con A3.

### D2 — PlayerStatsGenerator más realista

- De `player_stats_generator.md` §6: (a) muestrear conteos con Poisson/NegBin en
  vez de normal truncada; (b) correlación entre stats de un jugador (factor común
  "buen partido": muestrear un multiplicador por jugador-set y escalar todas sus
  stats); (c) distribuciones por posición. No afecta a resultados de partidos
  (las stats son post-hoc) — es credibilidad de la UI. Empezar por (b), que es
  lo más visible y barato (~1 h); (a) ~2 h con ajuste de parámetros por momentos;
  (c) requiere datos de posición (verificar si los CSVs de
  `stats_por_equipo_completo/` la traen antes de comprometerse).
- **Esfuerzo total**: 4–6 h. **Riesgo**: bajo (regenerar `player_stats_params.json`
  con `python -m src.models.train`; tests de player stats usan fit sintético).

### D3 — Saneamiento data layer: alias Cisterna + validación de normalización

- (a) Resolver la colisión `"Cisterna"` (2024/25) vs `"Cisterna Top Volley"`
  (ex-Latina, equipo históricamente distinto) en `TEAM_ALIASES`
  (`src/data/team_mapper.py`): decidir si son la misma franquicia para el modelo
  (probable: sí — misma plaza) y documentarlo, o separar alias.
  (b) Test nuevo `tests/test_normalization_coverage.py`: recorrer TODOS los
  nombres de equipo de TODOS los CSVs de `DB/` y asertar que
  `normalize_team_name(x)` devuelve un canónico conocido (lista blanca), para
  que un equipo nuevo/typo falle en tests en vez de caer en silencio al
  fallback de fuerza 0.5 (la clase de bug del incidente Optional).
- **Esfuerzo**: 2 h. **Riesgo**: nulo.

### D4 — Sideout por forma reciente

- `src/data/team_sideout.py` calcula el proxy con TODO el histórico. Cambiar a
  ventana rolling (últimas 2 temporadas o últimos 100 sets) para que refleje la
  plantilla actual. Mantener el filtro de mínimo (hoy <50 sets → fallback 0.62)
  y el fallback. Tests en test_team_sideout.py pinean el fallback — no cambia.
- **Esfuerzo**: 1–2 h. **Impacto**: bajo-medio.

---

## GRUPO E — Extras (propuestos, no están en ningún md)  🚧 PENDIENTE

> **Estado al 2026-07-22**: E1–E5 todos pendientes. Ningún avance aún.

### E1 — Monte Carlo de temporada como producto en la UI

- **Qué**: endpoint `POST /api/simular/temporada/montecarlo` que corra N
  temporadas (N=100-200) y devuelva por equipo la distribución de posición
  final: `P(campeón)`, `P(top-4)`, `P(últimos 2)`, posición media ± std.
- **Implementación**:
  1. Backend: request igual a `IniciarTemporadaRequest` + `n_temporadas: int`
    (cap 200; validar como los demás). Loop sobre seeds `semilla*1000+i`
    llamando a `simulate_season(half=None, use_set_calibration=False,
    use_match_predictor=True)`. **Obligatorio `use_set_calibration=False`**
    mientras el grupo A no cierre (60 s/temporada lo hace inviable ON).
    **Gotcha**: el `feature_builder` es un singleton compartido y acumula estado
    entre temporadas (`_init_dynamic_state` solo corre en el constructor) —
    para el MC crear un `RuntimeFeatureBuilder(initial_elo=…)` NUEVO por corrida
    o añadir un método `reset()`; si no, las últimas temporadas del MC ven Elo
    contaminado por las primeras.
  2. Agregación: matriz equipo×posición (12×12) normalizada; JSON
     `{equipo: {p_campeon, p_top4, p_bottom2, pos_media, pos_std, dist: [12]}}`.
  3. Frontend: nueva sección en SimularTemporada.jsx (o página nueva) con barras
     apiladas por equipo. Copy en español.
  4. Time-box: 100 temporadas ida y vuelta ≈ 100×2 s → correr con
     progreso/estimación y avisar en la UI (o empezar con ida simple N=100 ≈ 100 s... 
     medir ANTES de fijar N por defecto; si excede ~30 s, default N=50).
- **Esfuerzo**: 4–6 h. **Impacto**: alto en la demo del TFG. **Dependencias**:
  idealmente A1 o grupo A cerrado.

### E2 — Explicabilidad del partido suelto

- Ampliar la respuesta de `POST /api/simular/partido` (modo MC) con un bloque
  `"explicacion"`: Elo de cada equipo (de `get_historical_team_elo`), fuerzas
  usadas, sideouts per-team aplicados, y IC binomial 95% de `prob_local`
  (`±1.96*sqrt(p(1-p)/n)`). Frontend: tooltip/panel "por qué". Útil para
  preguntas del tribunal.
- **Esfuerzo**: 2–3 h. **Riesgo**: nulo (aditivo al JSON).

### E3 — `precision_report.py`: una sola fuente para todas las tablas de la memoria

- **Qué**: comando único que regenere TODAS las cifras que cita la memoria
  (per-year del set, CV, antes/después, backtests A5/B1 si existen) como CSV +
  tabla LaTeX (`\begin{tabular}`) en `models/report/`, para que `latex/` nunca
  cite números huérfanos (ya pasó: la tabla §7.1 quedó inválida tras el bug
  Optional).
- **Implementación**: orquestar `measure_precision.measure()`, el per-year del
  set (replicar el análisis de §7.2 como función), y leer los JSON de
  `models/precision_*.json` y `models/backtest_*.json`. Emitir con pandas
  `.to_csv` y `.to_latex`. Documentar en AGENTS.md (sección Commands).
- **Esfuerzo**: 3 h. **Impacto**: alto para la memoria (reproducibilidad).

### E4 — Predicción de jornada real (demo)

- Si la temporada 2026/27 está en curso durante la defensa: página/endpoint que
  prediga la próxima jornada real (equipos + P(victoria) + marcador más
  probable) y una tabla de aciertos acumulados vs resultados reales. Requiere
  el loader de datos frescos (solapa con B6/B7). **Esfuerzo**: 4–8 h.

### E5 — Blindaje contra regresión silenciosa a modelos leaky + desacople

- Dos arreglos pequeños de la misma familia que el bug Optional:
  1. **Gating acoplado**: en season_simulator.py:377 y :542, entrar al bloque de
     la señal Elo requiere `self.match_predictor` no-None — si
     `match_predictor.joblib` falta, se pierde TODA la señal (Elo incluido)
     aunque el feature_builder esté sano. Cambiar la condición a
     `use_match_predictor and self.feature_builder and hasattr(...)` y hacer el
     reindex/predict del match_predictor condicional a que exista.
  2. **Aviso de fallback**: en el startup del API, si `sp_source == "extra_trees_v1"`
     o el match cae al legacy de 87 features, emitir un WARNING explícito
     ("modelo legacy con métricas infladas — regenerar v2 con train_improved") y
     exponer `sp_source`/senal activa en `/api/modelo/info` (main.py:733-782).
- **Tests**: uno que construya SeasonSimulator con `match_predictor=None` +
  feature_builder válido y verifique que la señal Elo sigue aplicándose
  (las fuerzas calibradas difieren de las base).
- **Esfuerzo**: 1–2 h. **Riesgo**: bajo.

---

## Priorización recomendada (actualizada al 2026-07-22)

| Prioridad | Item(s) | Estado | Por qué |
|---|---|---|---|
| **0** | ~~**B0** arreglar colisión `partido_id`~~ | ✅ HECHO (2026-07-15) | Bloqueaba la validez del pipeline entero (Elo/fuerzas/AUC). Rescrito. |
| 1 | ~~**B1** backtest simulador~~ | ✅ HECHO (2026-07-15) | Reveló que el simulador estaba sobreconfiado (Brier +0.079, ECE +0.198 vs Elo). |
| 2 | ~~**Grupo A** (A5→A3→A2→A4→A6)~~ | ✅ HECHO (2026-07-21) | Diagnóstico hecho; elimina ruido cuantificado y ×60 de coste. Desenlace: `w=1.0` (SetPredictor cableado pero inactivo). |
| 3 | **B3** PointProb continuo | ✅ HECHO (2026-07-22) | Brier 0.273→0.182, ECE 0.242→0.057, 3-0 53%→37.6%. El simulador pasa de degradar al Elo a superarlo. |
| 4 | ~~**B2** tuning de constantes del simulador~~ | ✅ HECHO (2026-07-22) — RESULTADO NEGATIVO | Grid parametrizado y evaluado; no se adoptaron constantes nuevas. |
| 5 | ~~**C2** remote de GitHub~~ | ✅ HECHO | `asormar/SuperLega-Predictor` activo, PRs mergeados. |
| 6 | **C1** Ruff + Black + CI | ✅ HECHO | Ruff+Black+CI matrix Ubuntu+Windows en verde. PR #2. |
| 7 | **E1** MC de temporada en UI + **E3** reporte reproducible | 🚧 PENDIENTE | Máximo valor de defensa por hora. |
| 8 | **B5, D1, D3, E5** | 🚧 PENDIENTE | Backlog medio; cada uno se cierra en una tarde. |
| 9 | **B6** ampliar dataset | 🚧 PENDIENTE | La palanca de modelo más grande, pero la más cara; tras B2. |
| — | **B7** re-validación 2026/27 | ⏸️ BLOQUEADO | Bloqueado por calendario; dejar el script listo. |
| baja | C3–C6, D2, D4, E2, E4 | 🚧 PENDIENTE | Valor real, no crítico; según tiempo hasta la entrega. |

---

## Guardrails globales (para el agente ejecutor — leer SIEMPRE)

1. **Orden temporal sagrado**: nunca entrenar/decidir con datos de la temporada
   de test. El test held-out de los modelos es **2025/26** — se evalúa UNA vez
   por fase, jamás para tunear. Para tunear el simulador usar 2023/2024.
2. **Todo nombre de equipo pasa por `normalize_team_name()`** (también en
   scripts de análisis y tests nuevos).
3. **Coherencia train/serve**: cualquier feature nueva o redefinida se aplica
   IGUAL en el dataset de entrenamiento y en el runtime
   (RuntimeFeatureBuilder / contrato A3). Es el error más repetido del repo.
4. **Tests pineados**: la lista de la sección "Mapa de código" pinea constantes
   a propósito. Si un item cambia un valor: actualizar el pin EN EL MISMO
   commit, mencionándolo en el mensaje. Nunca "arreglar" un pin de pasada.
5. **Correr `pytest -q` tras cada tarea.** Con artefactos presentes, también
   los `-m slow`. Windows: `$env:PYTHONIOENCODING = "utf-8"` antes de los
   scripts de entrenamiento.
6. **Nada de emojis ni Unicode decorativo en código/prints** (consola Windows
   cp1252; hubo un batch entero de fixes por esto). UI y docstrings en español.
7. **Convención de módulos**: todo módulo nuevo bajo `src/` lleva el
   boilerplate `BASE_DIR = Path(__file__).resolve().parent.parent.parent;
   sys.path.insert(0, str(BASE_DIR))` y se ejecuta con `python -m`.
8. **Time-box de experimentos**: medir el coste de UNA unidad (predict,
   partido, temporada) ANTES de lanzar un loop grande; proyectar y abortar si
   excede el presupuesto (lección: 260 ms/predict → MC de >7 min).
9. **Resultados negativos se documentan igual** (patrón Batch 3): sección en el
   md correspondiente + JSON de resultados en `models/`. En un TFG el registro
   honesto vale tanto como la mejora.
10. **Un commit por tarea**, mensaje estilo
    `models(A2): clamp centrado en p_punto implicito — margen 0.08`.
    Commitear los JSON/CSV de resultados como evidencia.
11. **Rutas**: la carpeta del repo tiene espacios y paréntesis — citar SIEMPRE
    los paths en shell.
12. **Al cambiar cifras que la memoria cita** (métricas, constantes, tablas):
    revisar `memoria/*.md` y `latex/` y actualizar o marcar como histórico.
