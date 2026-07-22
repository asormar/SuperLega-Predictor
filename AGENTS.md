# AGENTS.md

Volleyball match/season simulator for the Italian SuperLega. Spanish-language web UI on top of a Python ML backend that loads pre-trained scikit-learn/XGBoost/LightGBM models and runs a Markov-chain Monte Carlo simulator.

## Layout

- `src/api/main.py` — FastAPI entry (`uvicorn src.api.main:app`). Also serves `src/web/dist/` as static SPA when present.
- `src/data/` — data pipeline, feature store, team name normalizer. All CSVs are read from `DB/`.
- `src/models/` — three persisted models (`SetPredictor` legacy + `set_predictor_v2.py` adapter, `PointProbabilityModel`, `PlayerStatsGenerator`) plus training, benchmark, and precision measurement scripts.
- `src/simulation/` — `MatchSimulator` (point-by-point Markov) and `SeasonSimulator` (round-robin + SuperLega points).
- `src/web/` — Vite + React (Spanish UI, `lucide-react` icons, `react-router-dom` v7).
- `models/` — generated artifacts (`*.joblib`, `player_stats_params.json`, feature cache CSVs, benchmark CSVs). Already populated; retrain only if data changes.
- `DB/` — source CSVs (`sets_partidos.csv`, `features/`, `enfrentamientos_directos/`, `stats_por_equipo_completo/`, `Comparacion_equipos_10_años.csv`).
- `papers/` — reference PDFs, not used at runtime.
- `memoria/` — TFG documentation in Spanish (match_predictor.md, set_predictor.md, etc.). Source for the unified `main.tex`; not used at runtime.
- `latex/` — `main.tex` is the TFG memoria work-in-progress with a 10-chapter skeleton + biblatex+biber. The chapters are still TODO placeholders to be migrated from `memoria/*.md`.
- `.opencode/skills/latex-document-skill/` — minimal LaTeX skill (clon recortado de ndpvt-web/latex-document-skill) para compilar la memoria unificada en PDF. SKILL.md define alcance y comandos. Requiere `pdflatex` en PATH (instalar MiKTeX con `winget install -e --id MiKTeX.MiKTeX`).

## Run order

The API can start without retraining, but the frontend will return empty rosters and zero team strengths until models are present. To rebuild from scratch:

1. `python -m src.data.data_pipeline` — sanity-checks CSVs in `DB/`.
2. `python -m src.models.train` — trains the legacy models (ExtraTrees set, XGBoost match) and writes to `models/`. **For production, prefer `train_improved` below** — see Conventions.
3. `python -m src.models.train_improved` — trains the production-grade models (LogReg set with recency + margin-Elo match) and writes `models/set_predictor_v2.joblib` + `models/match_elo_v2.joblib`. **The API loads v2 first; the legacy artifacts are only fallback.**
4. `python -m src.api.main` (or `uvicorn src.api.main:app --reload --port 8000`) — loads `models/*.joblib` + `models/player_stats_params.json`. Auto-mounts `src/web/dist/` if it exists.
5. `cd src/web && npm install && npm run dev` — Vite on `:5173`, proxies `/api` → `http://localhost:8000` (see `src/web/vite.config.js:9-13`).
6. Optional prod: `cd src/web && npm run build` then restart the API — the SPA is served from `/` on port 8000.

## Commands

From repo root unless noted.

Python (all run via `python -m`):
- `python -m src.models.train` — full retrain of legacy models (set predictor + point prob + player stats). Legacy set champion: ExtraTrees with calibration. Use `train_improved` for production.
- `python -m src.models.train_improved` — production trainer. Generates `set_predictor_v2.joblib` (LogReg + recency, AUC 0.71 en test 2025) and `match_elo_v2.joblib` (margin-Elo determinista, AUC 0.75). Refresh `models/precision_improved.json` with rolling-origin CV honesto.
- `python -m src.models.measure_precision` — mide set + match con el protocolo honesto (rolling-origin, test held-out 2025/26). `--save baseline|improved` para snapshot.
- `python -m src.models.benchmark` — 9-model comparison on SET and MATCH features. Output: `models/benchmark_results/{set,match,match_enriched}_benchmark.csv`.
- `python -m src.models.benchmark_teams` — 12-team vs 16-team feature ablation.
- `python -m src.models.benchmark_roster` — base vs +roster comparison.
- `python -m src.models.reliability_curve` — calibration/reliability plots → `models/plots/`.
- `python -m src.data.feature_store` — rebuilds `models/feature_cache/{match,set}_{X,y}_{train,val,test}.csv`.
- `python -m src.simulation.simulator` — quick CLI smoke test (Trento vs Perugia, 1000 MC sims).
- `python -m src.simulation.season_simulator` — quick CLI season smoke test.

Frontend (in `src/web/`):
- `npm run dev` / `npm run build` / `npm run preview` / `npm run lint` (ESLint flat config, ignores `dist`).

Quality (ruff + black + pytest, config in `pyproject.toml`):
- `ruff check src/ tests/` — lint (debe pasar antes de push).
- `black --check src/ tests/` — formato (debe pasar antes de push).
- `pytest -q -m "not slow"` — tests rápidos (los `slow` requieren `models/*.joblib` y `player_stats_params.json`, que están gitignored).

There is no root `.gitignore` (only `src/web/.gitignore`). Linter, formatter, tests and CI están configurados vía `pyproject.toml` y `.github/workflows/ci.yml` (ver la sección **Quality** arriba y la convención **Linting/formatting** abajo).

## Python dependencies (no requirements.txt)

Install into a venv manually: `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `xgboost`, `lightgbm`, `pandas`, `numpy`, `scipy`, `joblib`.

## API surface (`src/api/main.py`)

7 endpoints:
- `GET  /api/equipos` — list of viable teams with computed strength + colors.
- `GET  /api/equipos/{nombre}` — roster + per-set averages. Returns 404 if team not in `TEAM_STRENGTHS`.
- `POST /api/simular/partido` — body `{local, visitante, fuerza_local?, fuerza_visitante?, semilla?, generar_puntos?, generar_stats_jugadores?, n_simulaciones_mc?}`. `n_simulaciones_mc>0` returns aggregated MC distribution instead of a single match.
- `POST /api/simular/temporada` — body `{equipos, doble_vuelta, semilla?, fuerzas?, half?, first_half_state?}`. Use `half='first'` then `half='second'` + `first_half_state` for the two-pass double round-robin flow the UI relies on.
- `POST /api/simular/temporada/iniciar` — body `{equipos, doble_vuelta?, semilla?, fuerzas?}`. Initialises the calendar (generates round-robin jornadas) and returns `{schedule, total_jornadas, total_partidos, initial_standings, initial_player_stats}`. Does NOT simulate — the UI drives jornada-by-jornada via the next endpoint.
- `POST /api/simular/temporada/jornada` — body `{equipos, doble_vuelta?, schedule, jornada_index, current_standings?, current_player_stats?, semilla?, fuerzas?, use_match_predictor?, use_set_calibration?}`. Simulates a single jornada. Stateless: the frontend sends the accumulated state each call. Returns `{jornada_index, jornada_num, total_jornadas, matches, updated_standings, updated_player_stats, is_complete}`.
- `GET  /api/modelo/info` — selected set-predictor model name, features, validation metrics.
- CORS is wide-open (`allow_origins=["*"]`). Safe for dev only.

## Time-based data split

Defined in `src/data/feature_store.py:25-29`:
- **train**: 2016–2022
- **val**: 2023
- **test**: 2025

This is a strict temporal split — never shuffle or use future data in training.

## Conventions

- Every Python module under `src/` does `BASE_DIR = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(BASE_DIR))` to make `from src.X import Y` work when run as `python -m`. Preserve this when adding new modules; do not rely on `PYTHONPATH` or `pip install -e`.
- `src/__init__.py` exists and is empty; each subpackage has its own `__init__.py`.
- Team names are messy across sources (e.g. `MonzaMonza`, `Sir Safety Conad Perugia`, `Diatec Trentino`). Always pass user/CSV input through `src.data.team_mapper.normalize_team_name()` before using it as a key, including in any new code.
- The set of viable teams is defined in `src/data/team_mapper.py` (`TEAM_ALIASES`, `get_all_viable_teams()`). Update there when adding a new season.
- Team strength fallbacks live in `src/api/main.py:110-118` (`_STRENGTH_DEFAULTS`); the API also recomputes strengths from `DB/features/match_features.csv` at startup and overrides the defaults. A team missing from that CSV falls back silently.
- `DEFAULT_SIDEOUT_RATE = 0.62` in `src/simulation/constants.py:15` (centralizado en Batch 3 desde el atributo de clase `PointProbabilityModel`; los tests pinean que el atributo de clase NO existe). League-wide assumption feeding every point-by-point simulation. Changing it shifts every match outcome.
- `MatchSimulator` clamps `p_home_wins` to `[0.20, 0.80]` mid-rally (`src/simulation/simulator.py:247` clamp init, `simulator.py:277` clamp application). Momentum params in `src/simulation/constants.py:77-79`: `MOMENTUM_BONUS=0.015`, `MOMENTUM_MAX_STREAK=4`, `MOMENTUM_DECAY=0.5`. Don't silently remove these clamps or values.
- `SetPredictor` (legacy) selects the best model by validation AUC among 6 candidates (LR, RF, ET, GB, XGBoost, LightGBM), then isotonic-calibrates (`CalibratedClassifierCV` with `cv=3`, `method="isotonic"`). The legacy champion is ExtraTrees; it's still on disk as a fallback.
- `set_predictor_v2.py` (`LogRegSetPredictor` adapter) wraps the production model: LogReg C=0.5 con recency half-life=2 temporadas, entrenado en 2022-2024. Duck-typed al contrato del legacy (`feature_names` + `predict_proba(df)→[n,2]`). Lo carga el API con fallback al legacy si el v2 no está. Test AUC 2025 = 0.71, CV 2 folds = 0.63 ± 0.08 (el headline 0.71 está concentrado en 2025; ver `memoria/mejora_precision_2026-07.md`).
  - Guardrail: when renaming a `SET_FEATURE_COLS` entry, regenerate `set_features_v2.csv` AND retrain v2 in the same commit. The adapter will warn but not crash on drift.
- `MatchPredictor` ya NO se usa para señal de partido en producción (su AUC "0.71" era leakage — ver `memoria/mejora_precision_2026-07.md`). La señal es probabilidad de Elo con margen desde `src/data/rolling_features.py`. El artefacto `match_predictor.joblib` queda como fallback en disco.
- `PointProbabilityModel` (post-B3, 2026-07-22) entrena una `Ridge(alpha=1.0)` sobre el target **continuo** `point_ratio_h` con features rolling pre-partido, y la salida es `clip(pred, POINT_RATIO_CLIP)` con `POINT_RATIO_CLIP = (0.40, 0.60)` (`src/simulation/constants.py`) solo de salvavidas. **Ya no** binariza el target ni aplasta la salida a `0.45 + 0.10·p_dominante` (esa fórmula sesgaba la salida y la cadena de Markov amplificaba el sesgo ~7× hasta P(local) = 0.845 entre iguales). El joblib conserva las 6 claves (`model`, `scaler`, `feature_cols`, `base_home_point_prob`, `base_away_point_prob`, `is_fitted`) para no romper `load()`. Detalle y cifras en `memoria/mejora_precision_2026-07.md` §7.3 y `memoria/point_probability.md` §3.3.
- `benchmark.py` evaluates model performance including Brier score (calibration metric). It uses the same temporal split as the feature store.
- All user-facing copy, docstrings, and the `<title>` in `src/web/index.html` are in Spanish. Keep new UI strings Spanish unless told otherwise.
- **Linting/formatting**: ruff + black enforced via CI (`.github/workflows/ci.yml`, matrix Ubuntu+Windows, verde en ambos). Config en `pyproject.toml` (`[tool.ruff]`, `[tool.black]`, `line-length = 100`, `target-version = "py310"`, exclusiones: `src/web`, `latex`, `memoria`, `docs`). E402 ignorado en `[tool.ruff.lint]` por el boilerplate `sys.path.insert(0, str(BASE_DIR))` que los módulos de `src/` llevan antes de los `import src.*` (Guardrail 7). Antes de pushear: `ruff check src/ tests/ && black --check src/ tests/` y `pytest -q -m "not slow"`.

## Gotchas

- The model files in `models/` required for the API are: `set_predictor_v2.joblib` (preferred) or `set_predictor.joblib` (fallback), `match_elo_v2.joblib` or `match_predictor.joblib` (fallback), `point_probability.joblib`, `player_stats_params.json`. On a fresh clone, run `python -m src.models.train_improved` to generate the v2 artifacts; otherwise the API starts in degraded mode (uses legacy fallbacks if available, or returns empty rosters).
- `src/web/node_modules/` may be checked into git; if you change `package.json`, run `npm install` again.
- No `requirements.txt` exists; las dependencias viven en `pyproject.toml` (`[project]` + `[project.optional-dependencies]` con `test = ["pytest>=8", "httpx>=0.27"]`). Para instalar todo (runtime + tests) en un venv: `pip install -e ".[test]"`. La sección **Python dependencies** debajo lista los paquetes concretos a modo de referencia histórica.
- The repo folder name contains spaces and parentheses — quote all paths in shell commands.
- `get_all_viable_teams()` returns only teams meeting a minimum-data threshold. Teams below the threshold won't appear in the API even if they exist in `TEAM_ALIASES`.
