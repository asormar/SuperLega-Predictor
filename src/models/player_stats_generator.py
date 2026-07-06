"""
player_stats_generator.py — Generador de estadisticas de jugadores.

Ajusta distribuciones estadisticas a los datos historicos de jugadores
y genera stats realistas para partidos simulados.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"


class PlayerStatsGenerator:
    """
    Genera estadisticas realistas de jugadores para partidos simulados.

    Para cada equipo/temporada, ajusta distribuciones a las stats per-set
    de cada jugador. En simulacion, samplea de estas distribuciones y
    normaliza para que los totales sean consistentes con el marcador.
    """

    # Stats que generamos por jugador por set
    STAT_KEYS = ["puntos", "aces", "ataques_ganados", "bloqueos",
                 "recepciones_exc", "errores_saque"]

    def __init__(self):
        self.team_profiles = {}  # {equipo: {jugador: {stat: {mean, std}}}}
        self.team_rosters = {}   # {equipo: [lista de jugadores]}

    def fit(self, player_stats: pd.DataFrame, team_stats: pd.DataFrame):
        """
        Ajusta las distribuciones a partir de datos historicos.

        Usa la temporada mas reciente disponible para cada equipo.
        """
        # Usar la temporada mas reciente para cada equipo
        if "temporada" in player_stats.columns:
            # Ordenar por temporada descendente
            player_stats = player_stats.sort_values("temporada", ascending=False)

        # Agrupar por equipo (usar equipo_id o inferir del archivo)
        # Primero, necesitamos mapear equipo_id a nombre canonico
        teams_processed = set()

        for equipo_id in player_stats["equipo_id"].unique():
            df_equipo = player_stats[player_stats["equipo_id"] == equipo_id]

            # Tomar la temporada mas reciente
            if "temporada" in df_equipo.columns:
                latest_season = df_equipo["temporada"].iloc[0]
                df_equipo = df_equipo[df_equipo["temporada"] == latest_season]

            # Filtrar jugadores con suficientes partidos (minimo 5 sets)
            df_equipo = df_equipo[
                df_equipo["sets"].fillna(0) >= 5
            ].copy()

            if len(df_equipo) == 0:
                continue

            # Extraer nombre canonico del equipo desde equipo_id
            equipo_name = self._extract_team_name(equipo_id)

            # Calcular stats per set para cada jugador
            profile = {}
            roster = []

            for _, row in df_equipo.iterrows():
                jugador = row["jugador"]
                sets_played = max(row["sets"], 1)

                stats = {}
                for stat_key in self.STAT_KEYS:
                    if stat_key in row and pd.notna(row[stat_key]):
                        per_set = row[stat_key] / sets_played
                        # Usar distribucion Poisson-like (media y std)
                        stats[stat_key] = {
                            "mean": float(per_set),
                            "std": float(max(per_set * 0.4, 0.1)),
                            "total": float(row[stat_key]),
                            "sets": int(sets_played),
                        }
                    else:
                        stats[stat_key] = {
                            "mean": 0.0, "std": 0.1,
                            "total": 0.0, "sets": int(sets_played),
                        }

                # Participacion (que porcentaje de sets juega)
                stats["participation"] = float(sets_played) / float(
                    df_equipo["sets"].max() or 1
                )

                profile[jugador] = stats
                roster.append(jugador)

            self.team_profiles[equipo_name] = profile
            self.team_rosters[equipo_name] = roster
            teams_processed.add(equipo_name)

        print(f"  [PlayerStats] {len(teams_processed)} equipos procesados, "
              f"{sum(len(r) for r in self.team_rosters.values())} jugadores")

    def _extract_team_name(self, equipo_id: str) -> str:
        """Extrae el nombre canonico del equipo desde el ID del CSV.

        Los IDs reales en los CSVs de stats_por_equipo_completo son:
        APG, BASTIA, CIS-VOLLEY, LT, MC, MI-POWER, MIVER, MO, PC,
        PD, PIACENZAYOU, TN-ITAS, VRI, TA, VV, etc.
        """
        from src.data.team_mapper import normalize_team_name

        # Mapa basado en los IDs REALES del CSV
        id_map = {
            # Equipos temporada 2024/2025
            "TN-ITAS": "Trento",
            "APG": "Perugia",
            "MC": "Lube",
            "MI-POWER": "Milano",
            "VRI": "Verona",
            "MIVER": "Monza",
            "MO": "Modena",
            "PIACENZAYOU": "Piacenza",
            "CIS-VOLLEY": "Cisterna",
            "PD": "Padova",
            "TA": "Taranto",
            "BASTIA": "Grottazzolina",
            # Equipos historicos
            "LT": "Cisterna",
            "PC": "Piacenza",
            "RAV-ROB": "Ravenna",
            "VV": "Vibo Valentia",
            "FR-SORA": "Sora",
            "SIENA-EMMAS": "Siena",
            "BAM": "Cuneo",
            "CUNEOSPORT": "Cuneo",
            "CAST-MATER": "Castellana Grotte",
            "ACICASTELLO": "Acicastello",
        }

        eid = str(equipo_id).strip()

        # Busqueda exacta primero
        if eid in id_map:
            return id_map[eid]

        # Busqueda por substring (para variantes con prefijos/sufijos)
        for key, name in id_map.items():
            if key in eid or eid in key:
                return name

        # Fallback: intentar normalizar
        return normalize_team_name(eid)

    def generate_set_stats(
        self,
        team_name: str,
        team_score: int,
        opponent_score: int,
    ) -> list[dict]:
        """
        Genera estadisticas de jugadores para un set simulado.

        Args:
            team_name: nombre canonico del equipo
            team_score: puntos del equipo en este set
            opponent_score: puntos del rival

        Returns:
            Lista de dicts con stats por jugador
        """
        if team_name not in self.team_profiles:
            return []

        profile = self.team_profiles[team_name]
        roster = self.team_rosters[team_name]

        # Generar stats base para cada jugador
        player_stats = []
        for jugador in roster:
            if jugador not in profile:
                continue

            p = profile[jugador]

            # Decidir si el jugador participa (basado en participation rate)
            part = p.get("participacion", 0.8)
            if isinstance(part, dict):
                part_rate = part.get("mean", 0.8)
            else:
                part_rate = part

            if np.random.random() > part_rate:
                continue

            stats = {"jugador": jugador}
            for stat_key in self.STAT_KEYS:
                if stat_key in p and isinstance(p[stat_key], dict):
                    mean = p[stat_key]["mean"]
                    std = p[stat_key]["std"]
                    # Samplear de una distribucion normal truncada en 0
                    value = max(0, int(round(np.random.normal(mean, std))))
                    stats[stat_key] = value
                else:
                    stats[stat_key] = 0

            # Formula estandar de voleibol: PTS = ataques_ganados + aces + bloqueos
            # Los errores del rival no se atribuyen a ningun jugador.
            stats["puntos"] = (
                stats.get("ataques_ganados", 0)
                + stats.get("aces", 0)
                + stats.get("bloqueos", 0)
            )

            player_stats.append(stats)

        return player_stats

    def save(self, path: Optional[Path] = None):
        """Guarda los perfiles de jugadores."""
        if path is None:
            path = MODELS_DIR / "player_stats_params.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "team_profiles": self.team_profiles,
            "team_rosters": self.team_rosters,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"  Perfiles guardados en {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PlayerStatsGenerator":
        """Carga perfiles previamente guardados."""
        if path is None:
            path = MODELS_DIR / "player_stats_params.json"

        with open(path, "r", encoding="utf-8") as f:
            save_data = json.load(f)

        gen = cls()
        gen.team_profiles = save_data["team_profiles"]
        gen.team_rosters = save_data["team_rosters"]
        return gen
