"""
backtest_simulator.py — Backtest end-to-end del SIMULADOR contra una temporada real.

(B1 del PLAN_MEJORAS_CONSOLIDADO)

La "precision del proyecto" que ve el usuario final es la del SIMULADOR completo
(Markov + Monte Carlo), no la del clasificador Elo aislado. Este script mide esa
precision recorriendo una temporada real partido a partido, en orden cronologico,
y actualizando el estado del RuntimeFeatureBuilder con los resultados REALES
anteriores (igual que haria en produccion jornada a jornada).

Para cada partido registra:
  - p_sim: P(gana el local) estimada por monte_carlo_simulate (n>=500).
  - p_elo: P(gana el local) de la senal de Elo con margen (referencia limpia).
  - la distribucion simulada de marcadores (3-0/3-1/3-2).

Y al final compara contra el resultado real: Brier y log-loss del SIMULADOR vs
los del Elo solo, curva de fiabilidad + ECE, y distancia L1 entre la
distribucion simulada de margenes {3-0,3-1,3-2} y la real de la temporada.

────────────────────────────────────────────────────────────────────────────
NOTA CRITICA sobre los datos (hallazgo del 2026-07-15):
`rolling_features._aggregate_matches` agrupa por `partido_id`, pero en
`DB/sets_partidos.csv` el partido_id COLISIONA la ida y la vuelta de cada
enfrentamiento (mismo id para "A vs B" y "B vs A") porque abrevia los nombres
a 5 caracteres y omite la columna `fase` (1st/2nd half). El 82% de los 725
partido_ids funden DOS partidos en uno, sumando sets de ambos y corrompiendo el
target `gana_local`. Por eso este script NO usa `_aggregate_matches`: reconstruye
los partidos reales agrupando por `(partido_id, equipo_local)` (-> 1322 partidos
validos) y siembra el Elo con esa reconstruccion correcta. La consecuencia es
que el backtest mide el simulador con GROUND TRUTH e inputs Elo correctos; el
pipeline de produccion (`get_historical_team_elo`, `build_rolling_match_features`,
las fuerzas del API) sigue consumiendo la agregacion rota hasta que se arregle en
la fuente. Ver el informe del hallazgo para el plan de correccion global.
────────────────────────────────────────────────────────────────────────────

Uso:
    python -m src.models.backtest_simulator                 # season 2024, n=500
    python -m src.models.backtest_simulator --season 2024 --n-sims 500
    python -m src.models.backtest_simulator --use-set-calibration   # clamp ON (lento)
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.metrics import log_loss, brier_score_loss

from src.data.team_mapper import normalize_team_name
from src.data.rolling_features import (
    _elo_expected, _jornada_num,
    ELO_BASE, ELO_K, ELO_HOME_ADV, ELO_SEASON_REGRESS,
)
from src.simulation.simulator import MatchSimulator
from src.simulation.season_simulator import SeasonSimulator
from src.simulation.feature_builder import RuntimeFeatureBuilder
from src.simulation.constants import (
    HOME_ADVANTAGE_STRENGTH_BONUS,
    MATCH_PREDICTOR_DAMPING,
)

MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = MODELS_DIR / "plots"

# Las 6 features que consume el PointProbabilityModel (mismo set que usa el
# SeasonSimulator en produccion, ver season_simulator.py ~:414-422).
POINT_FEATURES = [
    "elo_diff", "diff_win_rate_global", "diff_set_win_rate",
    "diff_dominancia", "diff_set_ratio", "diff_forma_efectiva",
]

# Orden cronologico de las dos vueltas (para separar ida/vuelta colisionadas).
FASE_ORDER = {"1st half": 0, "2nd half": 1}

# Referencia del Elo con margen en el test held-out 2025 (measure_precision).
ELO_REF_2025 = {"brier": 0.200, "logloss": 0.585, "auc": 0.750}

# Presupuesto de tiempo: si la proyeccion supera esto, abortar (salvo --force).
DEFAULT_MAX_SECONDS = 1200.0


# ─────────────────────────────────────────────────────────────
# Carga de partidos REALES (reconstruccion correcta, sin colision)
# ─────────────────────────────────────────────────────────────

def load_real_matches(sp: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye los partidos reales: 1 fila por partido efectivamente jugado.

    Agrupa por `(partido_id, equipo_local)` en lugar de solo `partido_id`, lo
    que separa la ida de la vuelta (que comparten partido_id por el bug de
    abreviacion). Devuelve columnas: temporada_inicio, fnum, jornada_num,
    local, visitante, sets_h, sets_a, pts_h, pts_a, gana_local. Ordenado
    cronologicamente por (temporada_inicio, fnum, jornada_num).
    """
    df = sp.copy()
    df["local"] = df["equipo_local"].apply(normalize_team_name)
    df["visitante"] = df["equipo_visitante"].apply(normalize_team_name)
    df["t"] = df["temporada"].str.split("/").str[0].astype(int)
    df["fnum"] = df["fase"].map(lambda x: FASE_ORDER.get(x, 0))
    df["jnum"] = df["jornada"].apply(_jornada_num)

    rows = []
    for (_pid, _loc), g in df.groupby(["partido_id", "equipo_local"]):
        sh = int((g["ganador_set_local"] == 1).sum())
        sa = int((g["ganador_set_local"] == 0).sum())
        rows.append({
            "temporada_inicio": int(g["t"].iloc[0]),
            "fnum": int(g["fnum"].iloc[0]),
            "jornada_num": int(g["jnum"].iloc[0]),
            "local": g["local"].iloc[0],
            "visitante": g["visitante"].iloc[0],
            "sets_h": sh,
            "sets_a": sa,
            "pts_h": int(g["puntos_local"].sum()),
            "pts_a": int(g["puntos_visitante"].sum()),
            "gana_local": 1 if sh > sa else 0,
        })
    m = pd.DataFrame(rows).sort_values(
        ["temporada_inicio", "fnum", "jornada_num"]).reset_index(drop=True)
    return m


def _replay_elo(matches: pd.DataFrame) -> dict:
    """Elo con margen replayado sobre partidos CORRECTOS, en orden cronologico.

    Misma dinamica que rolling_features (K=28, home_adv=60, regresion 0.25 entre
    temporadas, margin_mult = 1 + 0.15*(|diff_sets|-1)), pero sobre la
    reconstruccion sin colision.
    """
    elo = defaultdict(lambda: ELO_BASE)
    last_season = {}
    for _, r in matches.iterrows():
        h, a, s = r["local"], r["visitante"], r["temporada_inicio"]
        for tm in (h, a):
            if last_season.get(tm) != s and tm in last_season:
                elo[tm] = (1 - ELO_SEASON_REGRESS) * elo[tm] + ELO_SEASON_REGRESS * ELO_BASE
            last_season[tm] = s
        eh, ea = elo[h], elo[a]
        exp = _elo_expected(eh + ELO_HOME_ADV, ea)
        mov = abs(int(r["sets_h"]) - int(r["sets_a"]))
        mm = 1.0 + 0.15 * (mov - 1)
        d = ELO_K * mm * ((1.0 if r["gana_local"] == 1 else 0.0) - exp)
        elo[h] = eh + d
        elo[a] = ea - d
    return dict(elo)


def _seed_state(matches: pd.DataFrame, season: int):
    """Siembra Elo + fuerzas SOLO con historia estrictamente anterior a `season`.

    Devuelve (initial_elo, strengths). Fuerza = logistica del Elo final centrada
    en ELO_BASE (misma formula que get_historical_team_strengths).
    """
    hist = matches[matches["temporada_inicio"] < season]
    if len(hist) == 0:
        raise ValueError(f"No hay historia anterior a la temporada {season}.")
    elo = _replay_elo(hist)
    strengths = {t: 1.0 / (1.0 + 10 ** (-(e - ELO_BASE) / 400.0)) for t, e in elo.items()}
    return elo, strengths


# ─────────────────────────────────────────────────────────────
# Metricas
# ─────────────────────────────────────────────────────────────

def _prob_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Brier, log-loss y accuracy de una probabilidad p contra el resultado y."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "acc": float(np.mean((p >= 0.5).astype(int) == y)),
        "n": int(len(y)),
    }


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 8) -> float:
    """Expected Calibration Error con n_bins equiespaciados."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        ece += (n_bin / total) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def _margin_key(sets_h: int, sets_a: int) -> str:
    """Marca del margen: '3-0' | '3-1' | '3-2' (independiente del ganador)."""
    return f"3-{min(int(sets_h), int(sets_a))}"


def _sim_margin_probs(score_dist: dict) -> dict:
    """Colapsa la distribucion MC de marcadores a margenes {3-0,3-1,3-2}."""
    return {
        "3-0": score_dist.get("3-0", 0.0) + score_dist.get("0-3", 0.0),
        "3-1": score_dist.get("3-1", 0.0) + score_dist.get("1-3", 0.0),
        "3-2": score_dist.get("3-2", 0.0) + score_dist.get("2-3", 0.0),
    }


# ─────────────────────────────────────────────────────────────
# Carga de modelos de produccion
# ─────────────────────────────────────────────────────────────

def _load_point_model():
    """Carga el PointProbabilityModel de produccion (None si no existe)."""
    try:
        from src.models.point_probability import PointProbabilityModel
        return PointProbabilityModel.load(MODELS_DIR / "point_probability.joblib")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] point_probability no disponible ({e}); se usa el fallback interno.")
        return None


def _load_set_predictor():
    """Carga el SetPredictor v2 (None si no existe)."""
    try:
        from src.models.set_predictor_v2 import LogRegSetPredictor
        pred, source = LogRegSetPredictor.try_load_v2(
            MODELS_DIR / "set_predictor_v2.joblib",
            MODELS_DIR / "set_predictor.joblib",
        )
        print(f"  [INFO] SetPredictor para el clamp: {source}")
        return pred
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] set_predictor no disponible ({e}); clamp por defecto.")
        return None


# ─────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────

def run_backtest(
    season: int = 2024,
    n_sims: int = 500,
    use_set_calibration: bool = False,
    damping: float = MATCH_PREDICTOR_DAMPING,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    force: bool = False,
    make_plot: bool = True,
) -> dict:
    """Recorre la temporada `season` real y mide la precision del simulador.

    Returns:
        dict con las metricas agregadas (tambien se guarda en disco).
    """
    print("=" * 70)
    print(f"  BACKTEST DEL SIMULADOR — temporada {season}/{season + 1}")
    print(f"  n_sims={n_sims}  set_calibration={'ON' if use_set_calibration else 'OFF'}"
          f"  damping={damping}")
    print("=" * 70)

    sp = pd.read_csv(BASE_DIR / "DB" / "sets_partidos.csv", encoding="utf-8")

    # Partidos reales reconstruidos SIN colision (ver nota del encabezado).
    matches = load_real_matches(sp)

    # Estado sembrado SOLO con historia < season (anti-leakage).
    initial_elo, strengths = _seed_state(matches, season)
    print(f"  Estado sembrado con {len(initial_elo)} equipos (historia < {season}).")

    m_season = matches[matches["temporada_inicio"] == season].reset_index(drop=True)
    if len(m_season) == 0:
        raise ValueError(f"No hay partidos para la temporada {season}.")
    print(f"  Partidos reales en {season}: {len(m_season)}")

    # Modelos de produccion.
    point_model = _load_point_model()
    set_predictor = _load_set_predictor() if use_set_calibration else None

    # RuntimeFeatureBuilder sembrado (igual que el API en main.py).
    fb = RuntimeFeatureBuilder(initial_elo=initial_elo)
    simulator = MatchSimulator(point_model=point_model, player_stats_gen=None)

    # Acumuladores.
    y = np.zeros(len(m_season), dtype=int)
    p_sim = np.zeros(len(m_season))
    p_elo = np.zeros(len(m_season))
    sim_margin_acc = {"3-0": 0.0, "3-1": 0.0, "3-2": 0.0}
    real_margin_cnt = {"3-0": 0, "3-1": 0, "3-2": 0}
    n_margin = 0

    t_start = time.perf_counter()
    for i, r in m_season.iterrows():
        home, away = r["local"], r["visitante"]
        jornada = int(r["jornada_num"]) if r["jornada_num"] else (i + 1)

        # 1) Features pre-partido con el estado ACUMULADO hasta ahora.
        df = fb.build_features(home, away, jornada)
        row = df.iloc[0]
        p_e = float(row.get("elo_win_prob_h", 0.5))

        # 2) Calibrar fuerzas exactamente como en produccion (season_simulator).
        h_adj = min(strengths.get(home, 0.5) + HOME_ADVANTAGE_STRENGTH_BONUS, 1.0)
        a_str = strengths.get(away, 0.5)
        h_adj, a_str = SeasonSimulator._calibrate_strengths(h_adj, a_str, p_e, damping=damping)

        # 3) Features de punto (las 6) desde el DataFrame de features.
        point_mf = {f: (float(row[f]) if f in df.columns else 0.0) for f in POINT_FEATURES}

        # Contexto de set (solo si el clamp esta activo).
        team_feats = None
        if use_set_calibration and set_predictor is not None:
            team_feats = SeasonSimulator._extract_set_team_features(df)

        # 4) Monte Carlo del partido (semilla por partido -> reproducible).
        mc = simulator.monte_carlo_simulate(
            home_team=home,
            away_team=away,
            home_strength=h_adj,
            away_strength=a_str,
            match_features=point_mf,
            n_simulations=n_sims,
            seed=1000 + i,
            set_predictor=set_predictor if use_set_calibration else None,
            team_features=team_feats,
        )

        # 5) Registrar prediccion y actualizar con el resultado REAL.
        y[i] = int(r["gana_local"])
        p_sim[i] = float(mc["home_win_prob"])
        p_elo[i] = p_e

        sm = _sim_margin_probs(mc["score_distribution"])
        # Solo contamos margenes en partidos con final estandar (ganador a 3 sets).
        if max(int(r["sets_h"]), int(r["sets_a"])) == 3:
            for k in sim_margin_acc:
                sim_margin_acc[k] += sm[k]
            real_margin_cnt[_margin_key(r["sets_h"], r["sets_a"])] += 1
            n_margin += 1

        winner = "home" if r["gana_local"] == 1 else "away"
        fb.update(home, away, int(r["sets_h"]), int(r["sets_a"]), winner,
                  points_local=int(r["pts_h"]), points_visitante=int(r["pts_a"]))

        # Time-box: proyectar tras el primer partido y abortar si se dispara.
        if i == 0:
            dt = time.perf_counter() - t_start
            projected = dt * len(m_season)
            print(f"  [time-box] 1er partido: {dt:.2f}s -> proyeccion total "
                  f"~{projected:.0f}s ({projected / 60:.1f} min)")
            if projected > max_seconds and not force:
                raise SystemExit(
                    f"  ABORTADO: proyeccion {projected:.0f}s > presupuesto {max_seconds:.0f}s. "
                    f"Baja --n-sims, usa clamp OFF, o pasa --force.")

    elapsed = time.perf_counter() - t_start
    print(f"  Simulacion completa en {elapsed:.0f}s ({elapsed / len(m_season):.2f}s/partido).")

    # ── Metricas agregadas ──
    denom = max(n_margin, 1)
    sim_margin = {k: v / denom for k, v in sim_margin_acc.items()}
    real_margin = {k: real_margin_cnt[k] / denom for k in real_margin_cnt}
    l1_margin = float(sum(abs(sim_margin[k] - real_margin[k]) for k in sim_margin))

    results = {
        "season": season,
        "n_matches": int(len(m_season)),
        "n_sims": n_sims,
        "use_set_calibration": use_set_calibration,
        "damping": damping,
        "seconds": round(elapsed, 1),
        "simulator": {**_prob_metrics(y, p_sim), "ece": _ece(y, p_sim)},
        "elo_reference": {**_prob_metrics(y, p_elo), "ece": _ece(y, p_elo)},
        "elo_ref_2025_holdout": ELO_REF_2025,
        "score_margin_distribution": {
            "n_standard_finals": n_margin,
            "simulated": {k: round(v, 4) for k, v in sim_margin.items()},
            "real": {k: round(v, 4) for k, v in real_margin.items()},
            "l1_distance": round(l1_margin, 4),
        },
        "reliability": _reliability_bins(y, p_sim, n_bins=8),
    }

    _print_summary(results)

    out = MODELS_DIR / f"backtest_simulator_{season}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Resultados guardados en {out}")

    if make_plot:
        _plot_reliability(y, p_sim, p_elo, season, use_set_calibration)

    return results


def _reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 8) -> dict:
    """Bins de la curva de fiabilidad (prob predicha media vs frecuencia real)."""
    from sklearn.calibration import calibration_curve
    try:
        prob_true, prob_pred = calibration_curve(y, np.clip(p, 1e-6, 1 - 1e-6),
                                                 n_bins=n_bins, strategy="uniform")
        return {"prob_pred": [round(float(x), 4) for x in prob_pred],
                "prob_true": [round(float(x), 4) for x in prob_true]}
    except Exception:  # noqa: BLE001
        return {"prob_pred": [], "prob_true": []}


def _print_summary(res: dict):
    s, e = res["simulator"], res["elo_reference"]
    print("\n  " + "-" * 60)
    print(f"  {'Metrica':<14}{'SIMULADOR':>14}{'ELO (ref)':>14}{'Delta':>12}")
    print("  " + "-" * 60)
    for k, label in [("brier", "Brier"), ("logloss", "LogLoss"), ("acc", "Accuracy"), ("ece", "ECE")]:
        d = s[k] - e[k]
        print(f"  {label:<14}{s[k]:>14.4f}{e[k]:>14.4f}{d:>+12.4f}")
    print("  " + "-" * 60)
    md = res["score_margin_distribution"]
    print(f"  Distribucion de margenes (sim vs real, n={md['n_standard_finals']}):")
    for k in ["3-0", "3-1", "3-2"]:
        print(f"    {k}:  sim {md['simulated'][k]:.3f}   real {md['real'][k]:.3f}")
    print(f"  L1(margenes) = {md['l1_distance']:.4f}")
    delta_brier = s["brier"] - e["brier"]
    if delta_brier <= 0.005:
        print(f"\n  Lectura: el simulador NO degrada la calidad de probabilidad "
              f"(Brier delta {delta_brier:+.4f}); es fiel al Elo y anade el detalle de marcador.")
    else:
        print(f"\n  Lectura: el pipeline Markov degrada el Brier en {delta_brier:+.4f} "
              f"respecto al Elo; candidato a ajuste (grupo A / B2 / B3).")


def _plot_reliability(y, p_sim, p_elo, season: int, set_cal: bool):
    """Guarda la curva de fiabilidad del simulador y del Elo en models/plots/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for p, label, color in [(p_sim, "Simulador", "#1565C0"), (p_elo, "Elo (ref)", "#E53935")]:
            pt, pp = calibration_curve(y, np.clip(p, 1e-6, 1 - 1e-6), n_bins=8, strategy="uniform")
            ax1.plot(pp, pt, "o-", label=label, color=color)
        ax1.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Perfecta")
        ax1.set_xlabel("Probabilidad predicha")
        ax1.set_ylabel("Frecuencia real de victoria local")
        ax1.set_title(f"Curva de fiabilidad — temporada {season}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.hist(p_sim, bins=20, alpha=0.6, label="Simulador", color="#1565C0")
        ax2.hist(p_elo, bins=20, alpha=0.6, label="Elo", color="#E53935")
        ax2.set_xlabel("P(gana local)")
        ax2.set_ylabel("Nro de partidos")
        ax2.set_title("Distribucion de probabilidades")
        ax2.legend()
        ax2.grid(alpha=0.3)

        suffix = "_clampON" if set_cal else ""
        out = PLOTS_DIR / f"backtest_simulator_{season}{suffix}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  Grafico guardado en {out}")
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] no se pudo generar el grafico ({e}); el JSON tiene los bins.")


def main():
    ap = argparse.ArgumentParser(description="Backtest end-to-end del simulador (B1).")
    ap.add_argument("--season", type=int, default=2024,
                    help="Temporada de inicio a backtestear (default 2024; reservar 2025).")
    ap.add_argument("--n-sims", type=int, default=500,
                    help="Simulaciones Monte Carlo por partido (default 500).")
    ap.add_argument("--use-set-calibration", action="store_true",
                    help="Activar el clamp adaptativo del SetPredictor (lento).")
    ap.add_argument("--damping", type=float, default=MATCH_PREDICTOR_DAMPING,
                    help=f"Damping de _calibrate_strengths (default {MATCH_PREDICTOR_DAMPING}).")
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                    help="Presupuesto de tiempo; aborta si la proyeccion lo supera.")
    ap.add_argument("--force", action="store_true", help="Ignorar el presupuesto de tiempo.")
    ap.add_argument("--no-plot", action="store_true", help="No generar el PNG.")
    args = ap.parse_args()

    run_backtest(
        season=args.season,
        n_sims=args.n_sims,
        use_set_calibration=args.use_set_calibration,
        damping=args.damping,
        max_seconds=args.max_seconds,
        force=args.force,
        make_plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
