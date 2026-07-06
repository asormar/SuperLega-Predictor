"""
main.py — API backend FastAPI para el simulador de volleyball.

Endpoints:
- GET  /api/equipos              → Lista de equipos disponibles
- GET  /api/equipos/{nombre}     → Detalle y jugadores de un equipo
- POST /api/simular/partido      → Simular un partido individual
- POST /api/simular/temporada    → Simular una temporada completa
- GET  /api/modelo/info          → Info del modelo ML
"""

import sys
from pathlib import Path

# Asegurar imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import json

from src.data.team_mapper import get_superliga_teams, get_all_viable_teams, normalize_team_name, TEAM_ALIASES
from src.simulation.simulator import MatchSimulator
from src.simulation.season_simulator import SeasonSimulator, generate_jornadas
from src.simulation.feature_builder import RuntimeFeatureBuilder
from src.simulation.constants import MAX_MC_ITERATIONS
from src.models.set_predictor import SetPredictor
from src.models.match_predictor import MatchPredictor
from src.models.point_probability import PointProbabilityModel
from src.models.player_stats_generator import PlayerStatsGenerator

# ─────────────────────────────────────────────────────────────
# Inicialización
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="SuperLega Volleyball Simulator",
    description="API para simular partidos y temporadas de la SuperLega italiana",
    version="1.0.0",
)

# CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Cargar modelos ───
MODELS_DIR = BASE_DIR / "models"

try:
    set_predictor = SetPredictor.load(MODELS_DIR / "set_predictor.joblib")
    print("[API] Set predictor cargado OK")
except Exception as e:
    print(f"[API] WARN: No se pudo cargar set_predictor: {e}")
    set_predictor = None

try:
    point_model = PointProbabilityModel.load(MODELS_DIR / "point_probability.joblib")
    print("[API] Point probability cargado OK")
except Exception as e:
    print(f"[API] WARN: No se pudo cargar point_model: {e}")
    point_model = None

try:
    player_gen = PlayerStatsGenerator.load(MODELS_DIR / "player_stats_params.json")
    print("[API] Player stats generator cargado OK")
except Exception as e:
    print(f"[API] WARN: No se pudo cargar player_gen: {e}")
    player_gen = None

try:
    match_predictor = MatchPredictor.load(MODELS_DIR / "match_predictor.joblib")
    print(f"[API] Match predictor cargado OK ({match_predictor.best_model_name})")
except Exception as e:
    print(f"[API] WARN: No se pudo cargar match_predictor: {e}")
    match_predictor = None

try:
    feature_builder = RuntimeFeatureBuilder()
    print("[API] Feature builder cargado OK")
except Exception as e:
    print(f"[API] WARN: No se pudo cargar feature_builder: {e}")
    feature_builder = None

# ─── Simulador ───
simulator = MatchSimulator(
    set_predictor=set_predictor,
    point_model=point_model,
    player_stats_gen=player_gen,
)

# ─── Fuerzas de equipos (calculadas desde match_features) ───
def _compute_team_strengths() -> dict:
    """Calcula la fuerza de cada equipo desde sus win rates en match_features."""
    import pandas as pd
    try:
        mf = pd.read_csv(BASE_DIR / "DB" / "features" / "match_features.csv", encoding="utf-8")
        from src.data.team_mapper import normalize_team_name as _norm
        mf["local"] = mf["local"].apply(_norm)
        mf["visitante"] = mf["visitante"].apply(_norm)
        all_teams = set(mf["local"].unique()) | set(mf["visitante"].unique())
        strengths = {}
        for team in all_teams:
            home = mf[mf["local"] == team]
            away = mf[mf["visitante"] == team]
            total = len(home) + len(away)
            if total < 10:
                continue
            wins = home["gana_local"].sum() + (1 - away["gana_local"]).sum()
            strengths[team] = round(float(wins / total), 3)
        print(f"[API] Fuerzas calculadas para {len(strengths)} equipos")
        return strengths
    except Exception as e:
        print(f"[API] WARN: No se pudieron calcular fuerzas: {e}")
        return {}

TEAM_STRENGTHS = _compute_team_strengths()
# Fallbacks para equipos sin suficientes datos
_STRENGTH_DEFAULTS = {
    "Trento": 0.68, "Perugia": 0.65, "Verona": 0.60, "Piacenza": 0.58,
    "Lube": 0.56, "Milano": 0.53, "Modena": 0.52, "Monza": 0.48,
    "Padova": 0.47, "Cisterna": 0.45, "Taranto": 0.40, "Grottazzolina": 0.35,
    "Siena": 0.42, "Ravenna": 0.45, "Acicastello": 0.36, "Cuneo": 0.38,
}
for k, v in _STRENGTH_DEFAULTS.items():
    if k not in TEAM_STRENGTHS:
        TEAM_STRENGTHS[k] = v

# Colores de equipos para el frontend
TEAM_COLORS = {
    "Trento": {"primary": "#FFD700", "secondary": "#1B3A5C", "accent": "#FFFFFF"},
    "Perugia": {"primary": "#000000", "secondary": "#D4AF37", "accent": "#FFFFFF"},
    "Verona": {"primary": "#FFD700", "secondary": "#003DA5", "accent": "#FFFFFF"},
    "Piacenza": {"primary": "#E30613", "secondary": "#FFFFFF", "accent": "#000000"},
    "Lube": {"primary": "#D32F2F", "secondary": "#FFFFFF", "accent": "#1565C0"},
    "Milano": {"primary": "#E53935", "secondary": "#1E88E5", "accent": "#FFFFFF"},
    "Modena": {"primary": "#FFC107", "secondary": "#0D47A1", "accent": "#FFFFFF"},
    "Monza": {"primary": "#E91E63", "secondary": "#FFFFFF", "accent": "#37474F"},
    "Padova": {"primary": "#FF5722", "secondary": "#FFFFFF", "accent": "#212121"},
    "Cisterna": {"primary": "#4CAF50", "secondary": "#FFFFFF", "accent": "#1B5E20"},
    "Taranto": {"primary": "#E53935", "secondary": "#1565C0", "accent": "#FFFFFF"},
    "Grottazzolina": {"primary": "#FF9800", "secondary": "#2E7D32", "accent": "#FFFFFF"},
    "Ravenna": {"primary": "#B71C1C", "secondary": "#FFD600", "accent": "#FFFFFF"},
    "Vibo Valentia": {"primary": "#E53935", "secondary": "#FFD600", "accent": "#212121"},
    "Siena": {"primary": "#1B5E20", "secondary": "#FFFFFF", "accent": "#FFD700"},
    "Acicastello": {"primary": "#0D47A1", "secondary": "#F44336", "accent": "#FFFFFF"},
    "Cuneo": {"primary": "#283593", "secondary": "#FFFFFF", "accent": "#E53935"},
}


def _build_point_features(h_str: float, a_str: float) -> dict:
    """
    Construye dict de features minimo para PointProbabilityModel
    a partir de las fuerzas relativas de los equipos.
    """
    diff = h_str - a_str
    return {
        "elo_diff": diff * 3000,
        "diff_win_rate_global": diff,
        "diff_set_win_rate": diff,
        "diff_dominancia": diff,
        "diff_set_ratio": diff,
        "diff_forma_efectiva": diff,
    }


# ─────────────────────────────────────────────────────────────
# Schemas Pydantic
# ─────────────────────────────────────────────────────────────

class SimularPartidoRequest(BaseModel):
    local: str = Field(..., description="Nombre del equipo local")
    visitante: str = Field(..., description="Nombre del equipo visitante")
    fuerza_local: Optional[float] = Field(None, description="Fuerza relativa [0,1]")
    fuerza_visitante: Optional[float] = Field(None, description="Fuerza relativa [0,1]")
    semilla: Optional[int] = Field(None, description="Semilla para reproducibilidad")
    generar_puntos: bool = Field(True, description="Generar punto a punto")
    generar_stats_jugadores: bool = Field(True, description="Generar stats de jugadores")
    n_simulaciones_mc: Optional[int] = Field(None, description="Si >0, ejecutar Monte Carlo")

    @field_validator("local", "visitante")
    @classmethod
    def _val_team(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError(f"Nombre de equipo invalido: {v!r}")
        from src.data.team_mapper import normalize_team_name
        name = normalize_team_name(v)
        if name not in TEAM_STRENGTHS:
            raise ValueError(f"Equipo '{v}' no reconocido en la base de datos")
        return v

    @field_validator("fuerza_local", "fuerza_visitante")
    @classmethod
    def _val_strength(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"La fuerza debe estar en [0, 1]: {v}")
        return v

    @field_validator("semilla")
    @classmethod
    def _val_seed(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"La semilla debe ser >= 0: {v}")
        return v

    @model_validator(mode="after")
    def _val_diff_teams(self):
        from src.data.team_mapper import normalize_team_name
        if normalize_team_name(self.local) == normalize_team_name(self.visitante):
            raise ValueError("Los equipos local y visitante deben ser distintos")
        return self


class SimularTemporadaRequest(BaseModel):
    equipos: list[str] = Field(..., description="Lista de equipos")
    doble_vuelta: bool = Field(True, description="Ida y vuelta")
    semilla: Optional[int] = Field(None, description="Semilla para reproducibilidad")
    fuerzas: Optional[dict[str, float]] = Field(None, description="Fuerzas personalizadas")
    half: Optional[str] = Field(None, description="'first' para primera vuelta, 'second' para segunda")
    first_half_state: Optional[dict] = Field(None, description="Estado de la primera vuelta (para segunda)")
    use_match_predictor: bool = Field(True, description="Calibrar fuerzas con MatchPredictor ML")
    use_set_calibration: bool = Field(True, description="Calibrar clamp punto a punto con SetPredictor")

    @field_validator("semilla")
    @classmethod
    def _val_seed(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"La semilla debe ser >= 0: {v}")
        return v

    @field_validator("equipos")
    @classmethod
    def _val_equipos(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Se necesitan al menos 2 equipos")
        if len(v) > 12:
            raise ValueError("Maximo 12 equipos por temporada")
        from src.data.team_mapper import normalize_team_name
        normalized = [normalize_team_name(e) for e in v]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Equipos duplicados en la lista")
        for name in normalized:
            if name not in TEAM_STRENGTHS:
                raise ValueError(f"Equipo '{name}' no reconocido")
        return v

    @field_validator("half")
    @classmethod
    def _val_half(cls, v):
        if v is not None and v not in ("first", "second"):
            raise ValueError("half debe ser 'first' o 'second'")
        return v

    @model_validator(mode="after")
    def _val_half_state(self):
        if self.half == "second" and self.first_half_state is None:
            raise ValueError("half='second' requiere first_half_state")
        return self


class IniciarTemporadaRequest(BaseModel):
    equipos: list[str] = Field(..., description="Lista de equipos")
    doble_vuelta: bool = Field(True, description="Ida y vuelta")
    semilla: Optional[int] = Field(None, description="Semilla para reproducibilidad")
    fuerzas: Optional[dict[str, float]] = Field(None, description="Fuerzas personalizadas")

    @field_validator("semilla")
    @classmethod
    def _val_seed(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"La semilla debe ser >= 0: {v}")
        return v

    @field_validator("equipos")
    @classmethod
    def _val_equipos(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Se necesitan al menos 2 equipos")
        if len(v) > 12:
            raise ValueError("Maximo 12 equipos por temporada")
        from src.data.team_mapper import normalize_team_name
        normalized = [normalize_team_name(e) for e in v]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Equipos duplicados en la lista")
        for name in normalized:
            if name not in TEAM_STRENGTHS:
                raise ValueError(f"Equipo '{name}' no reconocido")
        return v


class SimularJornadaRequest(BaseModel):
    equipos: list[str] = Field(..., description="Lista de equipos")
    doble_vuelta: bool = Field(True, description="Ida y vuelta")
    schedule: list[list[list[str]]] = Field(
        ..., description="Calendario agrupado por jornada: [[[home, away], ...], ...]"
    )
    jornada_index: int = Field(..., description="Indice 0-based de la jornada a simular")
    current_standings: list[dict] = Field(
        default_factory=list, description="Clasificacion acumulada (lista serializada)"
    )
    current_player_stats: list[dict] = Field(
        default_factory=list, description="Player stats acumuladas (lista serializada)"
    )
    semilla: Optional[int] = Field(None, description="Semilla base para reproducibilidad")
    fuerzas: Optional[dict[str, float]] = Field(None, description="Fuerzas personalizadas")
    use_match_predictor: bool = Field(True, description="Calibrar fuerzas con MatchPredictor ML")
    use_set_calibration: bool = Field(True, description="Calibrar clamp con SetPredictor")

    @field_validator("semilla")
    @classmethod
    def _val_seed(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"La semilla debe ser >= 0: {v}")
        return v

    @field_validator("equipos")
    @classmethod
    def _val_equipos(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Se necesitan al menos 2 equipos")
        if len(v) > 12:
            raise ValueError("Maximo 12 equipos por temporada")
        from src.data.team_mapper import normalize_team_name
        normalized = [normalize_team_name(e) for e in v]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Equipos duplicados en la lista")
        for name in normalized:
            if name not in TEAM_STRENGTHS:
                raise ValueError(f"Equipo '{name}' no reconocido")
        return v

    @field_validator("jornada_index")
    @classmethod
    def _val_jornada_idx(cls, v):
        if v is not None and v < 0:
            raise ValueError(f"jornada_index debe ser >= 0: {v}")
        return v


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/api/equipos")
async def listar_equipos():
    """Lista todos los equipos viables (con datos completos)."""
    equipos = []
    for team_info in get_all_viable_teams():
        team = team_info["nombre"]
        colors = TEAM_COLORS.get(team, {"primary": "#607D8B", "secondary": "#FFFFFF", "accent": "#000"})
        strength = TEAM_STRENGTHS.get(team, 0.5)

        # Obtener roster si disponible
        roster = []
        if player_gen:
            roster = player_gen.get_roster(team)

        equipos.append({
            "nombre": team,
            "fuerza": strength,
            "colores": colors,
            "num_jugadores": len(roster),
            "categoria": team_info["categoria"],
        })

    return {"equipos": equipos}


@app.get("/api/equipos/{nombre}")
async def detalle_equipo(nombre: str):
    """Detalle de un equipo con su roster y stats."""
    team = normalize_team_name(nombre)
    strength = TEAM_STRENGTHS.get(team)
    if strength is None:
        raise HTTPException(404, f"Equipo '{nombre}' no encontrado")

    colors = TEAM_COLORS.get(team, {"primary": "#607D8B", "secondary": "#FFF", "accent": "#000"})

    # Roster con stats
    jugadores = []
    if player_gen:
        team_profile = player_gen.get_profile(team)
        for jugador, stats in team_profile.items():
            jugador_info = {"nombre": jugador}
            for stat_key in ["puntos", "aces", "ataques_ganados", "bloqueos"]:
                if stat_key in stats and isinstance(stats[stat_key], dict):
                    jugador_info[f"{stat_key}_por_set"] = round(stats[stat_key]["mean"], 2)
                    jugador_info[f"{stat_key}_total"] = stats[stat_key].get("total", 0)
            jugadores.append(jugador_info)

    # Ordenar por puntos por set
    jugadores.sort(key=lambda x: x.get("puntos_por_set", 0), reverse=True)

    return {
        "nombre": team,
        "fuerza": strength,
        "colores": colors,
        "jugadores": jugadores,
        "temporada": "2024/2025",
    }


@app.post("/api/simular/partido")
async def simular_partido(req: SimularPartidoRequest):
    """Simula un partido individual con resultado punto a punto."""
    local = normalize_team_name(req.local)
    visitante = normalize_team_name(req.visitante)

    h_str = req.fuerza_local or TEAM_STRENGTHS.get(local, 0.5)
    a_str = req.fuerza_visitante or TEAM_STRENGTHS.get(visitante, 0.5)

    # Construir match_features minimo para PointProbabilityModel
    _point_mf = _build_point_features(h_str, a_str)

    # Si piden Monte Carlo
    if req.n_simulaciones_mc and req.n_simulaciones_mc > 0:
        mc = simulator.monte_carlo_simulate(
            home_team=local,
            away_team=visitante,
            home_strength=h_str,
            away_strength=a_str,
            match_features=_point_mf,
            n_simulations=min(req.n_simulaciones_mc, MAX_MC_ITERATIONS),
        )
        return {
            "tipo": "monte_carlo",
            "local": local,
            "visitante": visitante,
            "prob_local": mc["home_win_prob"],
            "prob_visitante": mc["away_win_prob"],
            "distribucion": mc["score_distribution"],
            "n_simulaciones": req.n_simulaciones_mc,
        }

    # Simulacion individual
    match = simulator.simulate_match(
        home_team=local,
        away_team=visitante,
        home_strength=h_str,
        away_strength=a_str,
        match_features=_point_mf,
        generate_points=req.generar_puntos,
        generate_player_stats=req.generar_stats_jugadores,
        seed=req.semilla,
    )

    # Serializar resultado
    sets_data = []
    for s in match.sets:
        set_info = {
            "numero": s.set_number,
            "puntos_local": s.score_home,
            "puntos_visitante": s.score_away,
            "ganador": local if s.winner == "home" else visitante,
        }
        if req.generar_puntos and s.points:
            set_info["puntos"] = [
                {
                    "num": p.point_number,
                    "marcador_local": p.score_home,
                    "marcador_visitante": p.score_away,
                    "ganador": "local" if p.winner == "home" else "visitante",
                    "sacador": "local" if p.server == "home" else "visitante",
                }
                for p in s.points
            ]
        if req.generar_stats_jugadores:
            set_info["stats_local"] = s.home_player_stats
            set_info["stats_visitante"] = s.away_player_stats

        sets_data.append(set_info)

    return {
        "tipo": "simulacion",
        "local": local,
        "visitante": visitante,
        "sets_local": match.sets_home,
        "sets_visitante": match.sets_away,
        "resultado": match.resultado,
        "ganador": local if match.winner == "home" else visitante,
        "sets": sets_data,
        "colores_local": TEAM_COLORS.get(local, {}),
        "colores_visitante": TEAM_COLORS.get(visitante, {}),
    }


@app.post("/api/simular/temporada")
async def simular_temporada(req: SimularTemporadaRequest):
    """Simula una temporada completa o por mitades (primera/segunda vuelta)."""
    # Normalizar nombres
    equipos = [normalize_team_name(e) for e in req.equipos]

    if len(equipos) < 2:
        raise HTTPException(400, "Se necesitan al menos 2 equipos")
    if len(equipos) > 12:
        raise HTTPException(400, "Máximo 12 equipos por temporada")

    # Fuerzas
    fuerzas = {}
    if req.fuerzas:
        for k, v in req.fuerzas.items():
            fuerzas[normalize_team_name(k)] = v
    for e in equipos:
        if e not in fuerzas:
            fuerzas[e] = TEAM_STRENGTHS.get(e, 0.5)

    season_sim = SeasonSimulator(
        simulator=simulator,
        team_strengths=fuerzas,
        set_predictor=set_predictor if req.use_set_calibration else None,
        match_predictor=match_predictor if req.use_match_predictor else None,
        feature_builder=feature_builder if req.use_match_predictor else None,
    )

    result = season_sim.simulate_season(
        teams=equipos,
        double_round_robin=req.doble_vuelta,
        seed=req.semilla,
        half=req.half,
        first_half_state=req.first_half_state,
        use_match_predictor=req.use_match_predictor,
        use_set_calibration=req.use_set_calibration,
    )

    # Serializar clasificación
    clasificacion = []
    for i, s in enumerate(result["standings"], 1):
        clasificacion.append({
            "posicion": i,
            "equipo": s.team,
            "puntos": s.points,
            "pj": s.matches_played,
            "pg": s.wins,
            "pp": s.losses,
            "sg": s.sets_won,
            "sp": s.sets_lost,
            "sr": round(s.set_ratio, 2),
            "pts_favor": s.points_scored,
            "pts_contra": s.points_conceded,
            "v3_0": s.wins_3_0,
            "v3_1": s.wins_3_1,
            "v3_2": s.wins_3_2,
            "d2_3": s.losses_2_3,
            "d1_3": s.losses_1_3,
            "d0_3": s.losses_0_3,
            "colores": TEAM_COLORS.get(s.team, {}),
        })

    # Serializar resultados de partidos
    partidos = []
    for m in result["matches"]:
        partidos.append({
            "local": m.home_team,
            "visitante": m.away_team,
            "resultado": m.resultado,
            "ganador": m.home_team if m.winner == "home" else m.away_team,
            "sets": [{"puntos_local": s.score_home, "puntos_visitante": s.score_away}
                     for s in m.sets],
        })

    # Serializar stats de jugadores acumulados
    player_stats_list = sorted(
        result.get("player_season_stats", {}).values(),
        key=lambda p: p.get("puntos", 0),
        reverse=True,
    )

    response = {
        "clasificacion": clasificacion,
        "partidos": partidos,
        "total_partidos": result["total_matches"],
        "player_season_stats": player_stats_list,
        "half": result.get("half"),
    }

    return response


@app.post("/api/simular/temporada/iniciar")
async def iniciar_temporada(req: IniciarTemporadaRequest):
    """
    Inicializa una temporada: genera el calendario agrupado por jornadas
    (round-robin clasico) y devuelve el estado inicial.

    NO simula ningun partido; solo prepara la estructura. Pensado para
    alimentar la UI de simulacion jornada a jornada.
    """
    equipos = [normalize_team_name(e) for e in req.equipos]

    if len(equipos) < 2:
        raise HTTPException(400, "Se necesitan al menos 2 equipos")
    if len(equipos) > 12:
        raise HTTPException(400, "Máximo 12 equipos por temporada")

    schedule_jornadas = generate_jornadas(equipos, double=req.doble_vuelta, seed=req.semilla)

    # Estado inicial: standings vacios con todos los equipos
    initial_standings = [
        {
            "equipo": t,
            "puntos": 0,
            "pj": 0,
            "pg": 0,
            "pp": 0,
            "sg": 0,
            "sp": 0,
            "pts_favor": 0,
            "pts_contra": 0,
            "v3_0": 0,
            "v3_1": 0,
            "v3_2": 0,
            "d2_3": 0,
            "d1_3": 0,
            "d0_3": 0,
        }
        for t in equipos
    ]

    total_jornadas = len(schedule_jornadas)
    return {
        "schedule": schedule_jornadas,  # [[[home, away], ...], ...]
        "total_jornadas": total_jornadas,
        "total_partidos": sum(len(j) for j in schedule_jornadas),
        "initial_standings": initial_standings,
        "initial_player_stats": [],
        "doble_vuelta": req.doble_vuelta,
    }


@app.post("/api/simular/temporada/jornada")
async def simular_jornada(req: SimularJornadaRequest):
    """
    Simula UNA jornada del calendario enviado por el frontend.

    El backend es stateless: el frontend mantiene el estado acumulado y lo
    envia en cada llamada. Cada jornada usa una semilla derivada
    (semilla * 1000 + jornada_index) para que los resultados sean
    reproducibles jornada a jornada sin requerir replay de las previas.
    """
    equipos = [normalize_team_name(e) for e in req.equipos]

    if len(equipos) < 2:
        raise HTTPException(400, "Se necesitan al menos 2 equipos")
    if len(equipos) > 12:
        raise HTTPException(400, "Máximo 12 equipos por temporada")
    if not req.schedule:
        raise HTTPException(400, "Falta el schedule")
    if req.jornada_index < 0 or req.jornada_index >= len(req.schedule):
        raise HTTPException(400, f"jornada_index fuera de rango: {req.jornada_index}")

    # Reconstruir tuplas (home, away) a partir de la representacion JSON
    schedule_tuples: list[list[tuple[str, str]]] = []
    for jornada in req.schedule:
        schedule_tuples.append([(m[0], m[1]) for m in jornada])

    # Fuerzas
    fuerzas: dict[str, float] = {}
    if req.fuerzas:
        for k, v in req.fuerzas.items():
            fuerzas[normalize_team_name(k)] = v
    for e in equipos:
        if e not in fuerzas:
            fuerzas[e] = TEAM_STRENGTHS.get(e, 0.5)

    season_sim = SeasonSimulator(
        simulator=simulator,
        team_strengths=fuerzas,
        set_predictor=set_predictor if req.use_set_calibration else None,
        match_predictor=match_predictor if req.use_match_predictor else None,
        feature_builder=feature_builder if req.use_match_predictor else None,
    )

    result = season_sim.simulate_jornada(
        schedule=schedule_tuples,
        jornada_index=req.jornada_index,
        current_standings=req.current_standings,
        current_player_stats=req.current_player_stats,
        equipos=equipos,
        seed=req.semilla,
        use_match_predictor=req.use_match_predictor,
        use_set_calibration=req.use_set_calibration,
    )

    # Serializar los partidos de la jornada
    jornada_matches_serialized = []
    for m in result["jornada_matches"]:
        jornada_matches_serialized.append({
            "local": m.home_team,
            "visitante": m.away_team,
            "resultado": m.resultado,
            "ganador": m.home_team if m.winner == "home" else m.away_team,
            "sets": [
                {"puntos_local": s.score_home, "puntos_visitante": s.score_away}
                for s in m.sets
            ],
        })

    return {
        "jornada_index": result["jornada_index"],
        "jornada_num": result["jornada_index"] + 1,
        "total_jornadas": len(schedule_tuples),
        "matches": jornada_matches_serialized,
        "updated_standings": result["updated_standings"],
        "updated_player_stats": result["updated_player_stats"],
        "is_complete": result["is_complete"],
    }


@app.get("/api/modelo/info")
async def modelo_info():
    """Información sobre el modelo ML utilizado."""
    info = {
        "set_predictor": None,
        "point_model": None,
        "player_stats": None,
    }

    if set_predictor:
        info["set_predictor"] = {
            "modelo": set_predictor.best_model_name,
            "features": set_predictor.feature_names,
            "resultados_validacion": {
                name: {
                    "accuracy": round(r["accuracy"], 4),
                    "auc_roc": round(r["auc_roc"], 4),
                    "brier_score": round(r["brier_score"], 4),
                }
                for name, r in set_predictor.results.items()
            },
        }

    if player_gen:
        info["player_stats"] = {
            "equipos": len(player_gen.team_profiles),
            "jugadores_total": sum(len(r) for r in player_gen.team_rosters.values()),
        }

    if match_predictor and match_predictor._test_metrics:
        info["match_predictor"] = {
            "modelo": match_predictor.best_model_name,
            "num_features": len(match_predictor.feature_names) if match_predictor.feature_names else 0,
            "test_auc": round(match_predictor._test_metrics.get("auc_roc", 0), 4),
            "test_accuracy": round(match_predictor._test_metrics.get("accuracy", 0), 4),
            "test_brier": round(match_predictor._test_metrics.get("brier_score", 0), 4),
        }

    return info


# ─── Servir frontend estático (si existe) ───
FRONTEND_DIR = BASE_DIR / "src" / "web" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
