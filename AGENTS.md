# AGENTS.md

Volleyball match/season simulator for the Italian SuperLega. Spanish-language web UI on top of a Python ML backend that loads pre-trained scikit-learn/XGBoost/LightGBM models and runs a Markov-chain Monte Carlo simulator.

## Layout

- `src/api/main.py` — FastAPI entry (`uvicorn src.api.main:app`). Also serves `src/web/dist/` as static SPA when present.
- `src/data/` — data pipeline, feature store, team name normalizer. All CSVs are read from `DB/`.
- `src/models/` — three persisted models (`SetPredictor`, `PointProbabilityModel`, `PlayerStatsGenerator`) plus training and benchmark scripts.
- `src/simulation/` — `MatchSimulator` (point-by-point Markov) and `SeasonSimulator` (round-robin + SuperLega points).
- `src/web/` — Vite + React (Spanish UI, `lucide-react` icons, `react-router-dom` v7).
- `models/` — generated artifacts (`*.joblib`, `player_stats_params.json`, feature cache CSVs, benchmark CSVs). Already populated; retrain only if data changes.
- `DB/` — source CSVs (`sets_partidos.csv`, `features/`, `enfrentamientos_directos/`, `stats_por_equipo_completo/`, `Comparacion_equipos_10_años.csv`).
- `papers/` — reference PDFs, not used at runtime.
- `memoria/` — TFG documentation in Spanish (match_predictor.md, set_predictor.md, etc.).

## Run order

The API can start without retraining, but the frontend will return empty rosters and zero team strengths until models are present. To rebuild from scratch:

1. `python -m src.data.data_pipeline` — sanity-checks CSVs in `DB/`.
2. `python -m src.models.train` — trains all three models, writes to `models/`.
3. `python -m src.api.main` (or `uvicorn src.api.main:app --reload --port 8000`) — loads `models/*.joblib` + `models/player_stats_params.json`. Auto-mounts `src/web/dist/` if it exists.
4. `cd src/web && npm install && npm run dev` — Vite on `:5173`, proxies `/api` → `http://localhost:8000` (see `src/web/vite.config.js:9-13`).
5. Optional prod: `cd src/web && npm run build` then restart the API — the SPA is served from `/` on port 8000.

## Commands

From repo root unless noted.

Python (all run via `python -m`):
- `python -m src.models.train` — full retrain (set predictor + point prob + player stats). Champion: ExtraTrees with calibration.
- `python -m src.models.benchmark` — 9-model comparison on SET and MATCH features. Output: `models/benchmark_results/{set,match,match_enriched}_benchmark.csv`.
- `python -m src.models.benchmark_teams` — 12-team vs 16-team feature ablation.
- `python -m src.models.benchmark_roster` — base vs +roster comparison.
- `python -m src.models.reliability_curve` — calibration/reliability plots → `models/plots/`.
- `python -m src.data.feature_store` — rebuilds `models/feature_cache/{match,set}_{X,y}_{train,val,test}.csv`.
- `python -m src.simulation.simulator` — quick CLI smoke test (Trento vs Perugia, 1000 MC sims).
- `python -m src.simulation.season_simulator` — quick CLI season smoke test.

Frontend (in `src/web/`):
- `npm run dev` / `npm run build` / `npm run preview` / `npm run lint` (ESLint flat config, ignores `dist`).

There is no test suite, no Python linter, no formatter, no type checker, and no CI in this repo. There is no root `.gitignore` (only `src/web/.gitignore`).

## Python dependencies (no requirements.txt)

Install into a venv manually: `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `xgboost`, `lightgbm`, `pandas`, `numpy`, `scipy`, `joblib`.

## API surface (`src/api/main.py`)

5 endpoints:
- `GET  /api/equipos` — list of viable teams with computed strength + colors.
- `GET  /api/equipos/{nombre}` — roster + per-set averages. Returns 404 if team not in `TEAM_STRENGTHS`.
- `POST /api/simular/partido` — body `{local, visitante, fuerza_local?, fuerza_visitante?, semilla?, generar_puntos?, generar_stats_jugadores?, n_simulaciones_mc?}`. `n_simulaciones_mc>0` returns aggregated MC distribution instead of a single match.
- `POST /api/simular/temporada` — body `{equipos, doble_vuelta, semilla?, fuerzas?, half?, first_half_state?}`. Use `half='first'` then `half='second'` + `first_half_state` for the two-pass double round-robin flow the UI relies on.
- `GET  /api/modelo/info` — selected set-predictor model name, features, validation metrics.
- CORS is wide-open (`allow_origins=["*"]`). Safe for dev only.

## Time-based data split

Defined in `src/data/feature_store.py:25-29`:
- **train**: 2016–2022
- **val**: 2023
- **test**: 2024

This is a strict temporal split — never shuffle or use future data in training.

## Conventions

- Every Python module under `src/` does `BASE_DIR = Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(BASE_DIR))` to make `from src.X import Y` work when run as `python -m`. Preserve this when adding new modules; do not rely on `PYTHONPATH` or `pip install -e`.
- `src/__init__.py` exists and is empty; each subpackage has its own `__init__.py`.
- Team names are messy across sources (e.g. `MonzaMonza`, `Sir Safety Conad Perugia`, `Diatec Trentino`). Always pass user/CSV input through `src.data.team_mapper.normalize_team_name()` before using it as a key, including in any new code.
- The set of viable teams is defined in `src/data/team_mapper.py` (`TEAM_ALIASES`, `get_all_viable_teams()`). Update there when adding a new season.
- Team strength fallbacks live in `src/api/main.py:110-118` (`_STRENGTH_DEFAULTS`); the API also recomputes strengths from `DB/features/match_features.csv` at startup and overrides the defaults. A team missing from that CSV falls back silently.
- `PointProbabilityModel.DEFAULT_SIDEOUT_RATE = 0.62` in `src/models/point_probability.py:38` — league-wide assumption feeding every point-by-point simulation. Changing it shifts every match outcome.
- `MatchSimulator` clamps `p_home_wins` to `[0.20, 0.80]` mid-rally (`src/simulation/simulator.py:229`). Momentum params: `MOMENTUM_BONUS=0.015`, `MOMENTUM_MAX_STREAK=4`, `MOMENTUM_DECAY=0.5`. Don't silently remove these clamps or values.
- `SetPredictor` selects the best model by validation AUC among 6 candidates (LR, RF, ET, GB, XGBoost, LightGBM), then isotonic-calibrates (`CalibratedClassifierCV` with `cv=3`, `method="isotonic"`). The champion model is ExtraTrees in the trained artifacts.
- `benchmark.py` evaluates model performance including Brier score (calibration metric). It uses the same temporal split as the feature store.
- All user-facing copy, docstrings, and the `<title>` in `src/web/index.html` are in Spanish. Keep new UI strings Spanish unless told otherwise.

## Gotchas

- The three model files in `models/` (`set_predictor.joblib`, `point_probability.joblib`, `player_stats_params.json`) are required for the API. If missing, the API starts but returns degraded responses.
- `src/web/node_modules/` may be checked into git; if you change `package.json`, run `npm install` again.
- No `requirements.txt` or `pyproject.toml` exists. Dependencies must be installed manually into a venv.
- The repo folder name contains spaces and parentheses — quote all paths in shell commands.
- `get_all_viable_teams()` returns only teams meeting a minimum-data threshold. Teams below the threshold won't appear in the API even if they exist in `TEAM_ALIASES`.
