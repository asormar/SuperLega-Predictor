# Siguiente Sesión — Plan de Trabajo

Esta guía resume el estado del TFG PREDICTOR(2) y los próximos pasos recomendados para cuando vuelvas a trabajar en él.

## Estado Actual (al cierre de la sesión del 2026-07-06)

- **Repo**: `C:\Users\Alejandro\Desktop\Universidad\4toCarrera\TFG\PREDICTOR(2)`. 18 commits en `main`. Sin remote (queda para cuando hagas `gh auth login`).
- **Documentación**: 10 archivos en `memoria/` (ver `memoria/INDICE.md` para el índice).
- **Modelos entrenados**: SetPredictor (ExtraTrees, AUC=0.654), MatchPredictor (XGBoost, AUC=0.707), PointProbabilityModel (LogReg), PlayerStatsGenerator (22 equipos).
- **API funcionando**: 5 endpoints, validación Pydantic, CORS dev-only, Monte Carlo determinista con seed.
- **Tests**: 0. Esto es la deuda más importante a resolver.
- **Linter/Format**: 0. Ruff + Black recomendados.
- **CI**: 0. GitHub Actions recomendado.
- **Memoria Engram**: sesión `predictor-tfg-2026-07-06` con 7 saves. Buscar con `mem_search(query: "TFG", project: "prueba")` o `mem_context(project: "prueba")`.

## Sub-agentes que trabajaron en esta sesión

| Task ID | Qué hizo |
|---|---|
| `ses_0c81b064cffeqcelOz3DJMXn9r` | Escribió 4 docs (data_layer, player_stats_generator, simulator, benchmark) |
| `ses_0c815248bffedi32YYTEMfZEsH` | Auditoría de 21 issues |
| `ses_0c802ec63ffeQ0oGrO9yySFj9L` | Batch 1: 7 fixes (colisión IDs, Unicode, emojis, PointProb integration, features dinámicas, validación, constants) |
| `ses_0c7f32622ffeqxozXBg2TuopBB` | Review fresh-context del Batch 1 |
| `ses_0c7d54739ffeMfJUYyg83AAK6X` | Batch 1.1: 6 fixes post-review (MC seed, n_solicitadas, elo_diff, doc, _resolve_team_key, mover función) |

## Próximos Pasos Recomendados (orden sugerido)

### Batch 2a — Issues de código pendientes (~1.5h)

Orden de menor a mayor impacto. Cada uno puede ser un commit independiente.

| Issue | Archivo | Esfuerzo | Riesgo | Estado |
|---|---|---|---|---|---|
| **N10** Imports no usados (`re`, `Optional`, `BASE_DIR`) | `team_mapper.py:8`, `benchmark.py:13`, `simulator.py:18` | 2 min | Nulo | [x] |
| **N13** Pandas re-imports dentro de funciones | `season_simulator.py:736+`, `simulator.py:323,353` | 2 min | Nulo | [x] |
| **N11** Mojibake `Cantù`→`Cant�` en CSV | `DB/features/match_features.csv` | 5 min | Nulo | [x] (ya resuelto previamente) |
| **N12** `set_predictor` redundante (seteado 2 veces) | `api/main.py:529-535` | 10 min | Bajo | [x] |
| **N3** `_generate_return_leg` confuso (3 iteraciones) | `season_simulator.py:576-596` | 5 min | Bajo | [x] |
| **N15** `userSelectedJornadaRef` posiblemente muerto | `web/src/pages/SimularTemporada.jsx:26` | 5 min | Nulo | [x] (vivo; se agregó comentario) |
| **N2** `_accumulate_player_stats` no cuenta rotaciones | `season_simulator.py:633-640` | 10 min | Bajo | [x] |
| **N14** `feature_names=None` desreferenciado | `season_simulator.py:374-377, 511-514` | 5 min | Bajo | [x] |
| **N4** `RuntimeFeatureBuilder` thread-safety | `api/main.py:86` | 15 min | Medio | [x] |
| **N7** CORS/auth/rate limit | `api/main.py:45-52` | 5 min docs | Nulo | [x] (docs-only) |

**Total Batch 2a: COMPLETADO. 10 commits, ~1.5h. Sin reentrenamiento.**

### Batch 2b — Tests (~3-4h)

El review del Batch 1 recomendó tests de regresión. Mínimo viable:

1. Crear `tests/` con `conftest.py` (fixtures con DataFrames sintéticos, modelos mock).
2. Tests de los 13 fixes aplicados (smoke tests):
   - `test_player_stats_collision.py` — 22 equipos, no 19
   - `test_encoding.py` — `print()` sin Unicode
   - `test_emojis.py` — sin emojis en código
   - `test_point_prob_integration.py` — `simulate_match` recibe `match_features`
   - `test_feature_builder.py` — `win_rate_home != win_rate_away`, `pts_fav_exp` correcto
   - `test_api_validation.py` — 13 casos 422 + 4 casos 200
   - `test_constants.py` — valores idénticos a hardcodeados
   - `test_monte_carlo_seed.py` — determinista
   - `test_elo_diff_scaling.py` — en [-200, +200]
3. Tests unitarios básicos de los 3 modelos + 2 simuladores (smoke tests con datos sintéticos).

**Total Batch 2b: ~3-4h.**

### Batch 2c — Linter + Format + CI (~2h)

1. Instalar `ruff` y `black` (`pip install ruff black`).
2. Crear `pyproject.toml` con configuración:
   ```toml
   [tool.ruff]
   line-length = 100
   target-version = "py312"
   
   [tool.black]
   line-length = 100
   target-version = ["py312"]
   ```
3. Correr `ruff check src/` y arreglar los issues.
4. Correr `black src/` (o `--check` primero).
5. Crear `.github/workflows/ci.yml` con:
   - `ruff check`
   - `black --check`
   - `pytest`
   - `python -m src.data.data_pipeline` (sanity check)
6. (Opcional) Configurar Codecov o similar para coverage.

**Total Batch 2c: ~2h.**

### Batch 2d — Tareas opcionales

- **N8 Logging migration** (2h): reemplazar 260+ `print()` por `logging.getLogger(__name__)`. Setup `dict-config` en `src/logging_config.py`.
- **Optimizaciones de performance**: vectorizar loops O(n²) en `simulator.py`/`season_simulator.py` con numpy.
- **Rate limiting real** con `slowapi` (1-2h, opcional).
- **Deploy a producción**: Docker, Gunicorn con N workers, nginx reverse proxy.

## Decisiones Pendientes

1. **Remote de GitHub**: cuando hagas `gh auth login`, correr:
   ```bash
   cd "C:\Users\Alejandro\Desktop\Universidad\4toCarrera\TFG\PREDICTOR(2)"
   gh repo create asormar/superlega-predictor --public --source=. --remote=origin --description "TFG PREDICTOR(2) - SuperLega volleyball match/season simulator with ML calibration" --push
   ```

2. **Tests prioritarios**: para la defensa del TFG, los 5 tests de smoke (modelos + simuladores) son los más valiosos. Cubren los componentes más preguntados.

3. **CI vs no CI**: si no querés GitHub Actions, podés correr `ruff` y `pytest` localmente antes de cada commit. Pero CI es valioso para mostrar buenas prácticas en el TFG.

4. **Logging**: si vas a defender el TFG con demo en vivo, los `print()` son ruidosos. Migrar a `logging` vale la pena.

5. **Tests E2E con el frontend**: hay 0 tests del frontend. Si querés, agregar tests con Vitest + Testing Library para las 4 páginas (Dashboard, EquipoDetalle, SimularPartido, SimularTemporada).

## Comandos Útiles

Activar Python y deps (venv, si tenés):
```bash
cd "C:\Users\Alejandro\Desktop\Universidad\4toCarrera\TFG\PREDICTOR(2)"
$env:PYTHONIOENCODING = "utf-8"
python -m src.models.train
python -m src.api.main
```

Frontend (en `src/web/`):
```bash
cd "C:\Users\Alejandro\Desktop\Universidad\4toCarrera\TFG\PREDICTOR(2)\src\web"
npm install
npm run dev   # localhost:5173
npm run build # producción → dist/
```

Smoke test API:
```bash
curl http://localhost:8000/api/equipos
curl -X POST http://localhost:8000/api/simular/partido -H "Content-Type: application/json" -d '{"local": "Trento", "visitante": "Perugia", "semilla": 42, "generar_puntos": false, "generar_stats_jugadores": false}'
```

## Tamaño del Proyecto (al cierre)

- **Código Python**: ~4000 líneas en `src/`
- **Frontend JSX**: ~2500 líneas en `src/web/src/`
- **Tests**: 0
- **Documentación**: 10 archivos, ~110 KB en `memoria/`
- **Modelos entrenados**: ~22 MB (gitignored, regenerables)
- **Datos**: 22 CSVs en `DB/`, ~5 MB
- **Commits**: 18 (3 docs + 13 fixes + 2 fixups)
- **Issues pendientes**: 11 (Batch 2a) + tests (Batch 2b) + linter/CI (Batch 2c)

## Batch 2a — Re-run correctivo

3 commits correctivos sobre primeros commits defectuosos del Batch 2a original.

| Commit | Tipo | Descripción |
|--------|------|-------------|
| `97f4f51` | revert | Elimina campo `rotaciones` de `_accumulate_player_stats` (siempre igual a `sets`, engañoso) |
| `48e5b9e` | refactor | Hoistea `import pandas as pd` a nivel de módulo en `main.py`, elimina inner import duplicado de `normalize_team_name` |
| `e6620d8` | docs | Restaura ~20 comentarios inline perdidos en el fix de thread-safety (5198f00). Extrae `7` a constante `ASSUMED_REST_DAYS` |

**Estado**: Batch 2a completo con 13 commits totales en main. Sin reentrenamiento.

## Batch 2b — Tests (en progreso, sesión 2026-07-06-2b)

**Fase actual**: explore + propose completos. Spec phase lanzado pero cancelado por el usuario a mitad de ejecución. **Punto de reanudación**: `sdd-spec` con el prompt ya preparado (decisiones autónomas baked in).

### Inventario final acordado

| Archivo | LoC | Scope |
|---|---|---|
| `pyproject.toml` | ~25 | pytest config minimal (`[project.optional-dependencies] test = ["pytest", "httpx"]`, `[tool.pytest.ini_options]`) |
| `tests/__init__.py` | 0 | vacío |
| `tests/conftest.py` | ~80 | fixtures: synthetic df + 4 modelos sintéticos + `feature_builder` override + autouse seed + tmp CSV |
| `tests/test_team_mapper.py` | ~140 | `normalize_team_name` 12+ dedup cases + **constantes pineadas (incluye DEFAULT_SIDEOUT_RATE×2, clamp ranges, MOMENTUM_*, ASSUMED_REST_DAYS, TEMPORAL_SPLITS)** |
| `tests/test_api_validation.py` | ~140 | 13 Pydantic 422 + 4 happy-path 200 |
| `tests/test_simulator.py` | ~150 | match shape, **AMBOS clamp ranges** `(0.20, 0.80)` y `(0.10, 0.90)`, MC determinismo, sideout math, feature_names=None guard (N14) |
| `tests/test_season_simulator.py` | ~90 | `_generate_return_leg` (N3), `_accumulate_player_stats` no-rotaciones (N2 revert), standings round-trip, two-pass half flow |
| `tests/test_models.py` | ~110 | 4 modelos smoke (SetPredictor / MatchPredictor / PointProbabilityModel / PlayerStatsGenerator) con `.fit()` sintético + Brier-score sanity |
| `tests/test_feature_builder.py` | ~80 | win_rate asymmetry, `pts_fav_exp`, `ASSUMED_REST_DAYS=7` (e6620d8), build_features schema, `elo_diff = diff * 200` (Batch 1.1) |
| `tests/test_data_pipeline.py` | ~60 | CSV loaders + normalize_team_name round-trip |

**Total: ~790 LoC** (debajo del budget de 800).

### Decisiones autónomas del orchestrator (auto mode)

1. **FOLD** `test_constants.py` → `test_team_mapper.py`. Ahorra ~40 líneas, total queda bajo 800.
2. **NO** arreglar la duplicación de `DEFAULT_SIDEOUT_RATE` (`point_probability.py:71` y `constants.py:15` ambos en 0.62). Defer a Batch 2c. En 2b solo se pinean ambos valores.

### Gotchas descubiertos (el spec los cubre)

1. `models/*.joblib` está en `.gitignore` — los tests deben correr en un clone fresco sin artefactos. Fixtures sintéticos son obligatorios.
2. Singletons a nivel de módulo en `src/api/main.py:64-97` — usar `app.dependency_overrides` exclusivamente.
3. `RuntimeFeatureBuilder` tiene `threading.Lock` pero es SINGLETON — el lock evita data races, no mezcla lógica de estado entre requests. Los tests NO deben depender de cross-request state.
4. **DOS clamp ranges**: `DEFAULT_CLAMP_RANGE = (0.20, 0.80)` (sin SetPredictor) Y `POINT_PROB_CLIP_ADAPTIVE_HARD = (0.10, 0.90)` con `CLAMP_MARGIN = 0.20` (con SetPredictor). Testear AMBOS.
5. `DEFAULT_SIDEOUT_RATE` duplicado: pinear AMBOS valores en `test_team_mapper.py`.
6. Inconsistencia Pydantic: `_val_team` devuelve el string original, `_val_diff_teams` compara normalizado. `"Diatec Trentino"` vs `"Trento"` es ACEPTADO. Testear explícitamente.
7. `player_stats_params.json` (~300KB, gitignored) — `PlayerStatsGenerator.load()` falla en clone fresco; tests usan `.fit()` con stats sintéticos.
8. `set_predictor.feature_names` guard (N14) — testear que con `feature_names=None` devuelve `None` y se skipea en `simulate_season`.
9. `build_features_from_strengths` escala `elo_diff * 200` (otros `diff_*` quedan sin escalar) — pinear.

### Guardrails para el sdd-apply (lección del Batch 2a)

El prompt del apply tiene que decir explícitamente:
- **"Do NOT modify any file under `src/`. Batch 2b adds `pyproject.toml` and `tests/` only."**
- **"No drive-by refactors. Test files are new files; there is nothing to delete."**
- **"One commit per test file. First commit: `pyproject.toml + conftest.py`. Last commit: `@pytest.mark.slow` integration test."**
- Si un test revela un bug real en `src/`, frenar y surfacear como cambio separado.

### Reanudación (próxima sesión)

1. Re-launch `sdd-spec` con el prompt que ya tengo en el historial de la sesión `predictor-tfg-2026-07-06-2b`. Topic key: `sdd/predictor-2b-batch/spec`.
2. `sdd-design` (diseño técnico): contenido exacto de `pyproject.toml`, estructura de `conftest.py`, contratos de fixtures, listas de parametrización.
3. `sdd-tasks` (un task por archivo de test + pyproject.toml/conftest + slow integration test).
4. `sdd-apply` (implementación con los guardrails de arriba).
5. `sdd-verify` (`pip install -e ".[test]" && pytest`, REQ-1..40 verde).
6. `sdd-archive`.

### Engram references (resumir con `mem_get_observation`)

- `#72` — `sdd-init/predictor(2)` contexto del proyecto
- `#79` — Session summary Batch 2a (lista los 13 fixes que el test suite debe cubrir)
- `#81` — Batch 2b explore (source map + risk surface)
- `#82` — Batch 2b propose (plan completo)
- Session id actual: `predictor-tfg-2026-07-06-2b`
