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
