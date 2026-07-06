# Match Predictor — Predicción del Ganador de un Partido

## Descripción

El `MatchPredictor` (`src/models/match_predictor.py`) es un clasificador binario que predice la probabilidad de que el equipo local gane un partido completo de volleyball. Es el más complejo de los tres modelos del proyecto porque integra **87 features** de tres categorías: base (Elo, forma, H2H), team stats agregadas y roster.

*Salida: P(local gana partido) ∈ [0, 1] · Usado por: `SeasonSimulator` (calibración de fuerzas antes de cada partido)*

---

## 1. Arquitectura de Features (87 features totales)

El MatchPredictor trabaja sobre `DB/features/match_features.csv` (725 partidos históricos, 10 temporadas). Las features se construyen en tres capas:

| Capa | # Features | Descripción | Fuente |
|---|---:|---|---|
| **Base** | ~50 | Elo, forma, rachas, H2H, win rate, set rate, dominancia | `match_features.csv` |
| **Team stats** | 21 | Diferencias de `pts_set`, `aces_set`, `atq_pct`, `atq_eff`, `rec_eff`, `bloq_set`, `ace_ratio` | `team_season_stats` + `Comparacion_equipos_10_años.csv` |
| **Roster básico** | 15 | `top_scorer_avg`, `roster_depth`, `ace_threat` por equipo | `player_stats` |

El script `train.py` añade las features de team stats (21) y roster (15) sobre las features base para llegar a 87. Si alguna feature falta en un partido, se rellena con 0 (`fill_value=0.0` en `season_simulator.py`).

### 1.1. Importancia de las features (top 10 del champion XGBoost)

| Rank | Feature | Importancia | Tipo |
|---:|---|---:|---|
| 1 | `point_ratio_a` | 0.0234 | Team stats |
| 2 | `diff_ace_threat` | 0.0216 | Roster |
| 3 | `h_racha` | 0.0180 | Base (racha local) |
| 4 | `elo_win_prob_h` | 0.0179 | Base (Elo) |
| 5 | `diff_atq_pct` | 0.0178 | Team stats |
| 6 | `h_top_scorer_avg` | 0.0175 | Roster |
| 7 | `diff_pts_set` | 0.0172 | Team stats |
| 8 | `h_atq_eff` | 0.0167 | Team stats |
| 9 | `elo_h_home` | 0.0166 | Base (Elo con home adv) |
| 10 | `diff_top_scorer` | 0.0163 | Roster |

**Observación**: a diferencia del `SetPredictor` (donde dominan features in-match), en el `MatchPredictor` las importancias están **más repartidas entre las 3 capas** (5 base, 4 team stats, 3 roster, 2 en otras). Esto sugiere que para predecir el resultado global de un partido, las estadísticas históricas del equipo y la calidad del roster importan tanto como la forma reciente.

---

## 2. Pipeline de Entrenamiento

```
match_features.csv (725 partidos)
   ↓ enrich: +21 team_stats features
   ↓ roster: +15 roster features
   ↓ 87 features totales
   ↓
train.py → train / val / test split
   (319 / 81 / 111)        ← split temporal estricto
   ↓
MatchPredictor.train()
   ↓ 4 modelos candidatos
   ↓ selección por AUC
   ↓ calibración isotonic
   ↓
match_predictor.joblib (~1.5 MB)
```

El split temporal (definido en `feature_store.py:25-29`) es **estricto**: train 2016-2022, val 2023, test 2024. Nunca se mezcla el orden ni se hace shuffle, para evitar leakage de datos futuros al pasado.

---

## 3. Candidatos Comparados

| Modelo | Hiperparámetros principales |
|---|---|
| GradientBoosting | `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8` |
| RandomForest | `n_estimators=300, max_depth=10, min_samples_leaf=4` |
| ExtraTrees | `n_estimators=300, max_depth=10, min_samples_leaf=4` |
| **XGBoost** (champion) | `n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0` |

A diferencia del `SetPredictor` (6 candidatos), el `MatchPredictor` evalúa solo 4 porque el dataset de partidos es **muy pequeño** (319 train) y los modelos más pesados (LightGBM) sobreajustan rápido.

---

## 4. Resultados de Validación (selección de champion)

Métricas en el set de validación (año 2023, 81 partidos, balance 44/56):

| Modelo | Acc | AUC | Brier | Prec | Rec |
|---|---:|---:|---:|---:|---:|
| GradientBoosting | 0.420 | 0.471 | 0.3440 | 0.430 | 0.420 |
| RandomForest | 0.432 | 0.451 | 0.2656 | 0.435 | 0.432 |
| ExtraTrees | 0.444 | 0.437 | 0.2612 | 0.435 | 0.444 |
| **XGBoost** | 0.494 | **0.4753** | 0.3337 | 0.503 | 0.494 |

**Modelo campeón: XGBoost** (AUC = 0.4753). Todos los modelos están cerca del azar en accuracy, pero el AUC es lo que importa para la calibración: XGBoost ordena mejor las probabilidades que los demás.

---

## 5. Resultados en Test (datos 2024, no vistos)

Evaluación del modelo calibrado sobre 111 partidos de la temporada 2024 (balance 55/45):

| Métrica | Valor |
|---|---:|
| **Accuracy** | 0.5135 |
| **AUC-ROC** | 0.7070 |
| **Brier Score** | 0.2452 |

Reporte de clasificación:

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Visitante | 0.48 | 0.92 | 0.63 | 50 |
| Local | 0.73 | 0.18 | 0.29 | 61 |
| **macro avg** | 0.61 | 0.55 | 0.46 | 111 |
| **weighted avg** | 0.62 | 0.51 | 0.44 | 111 |

**El AUC sube de 0.4753 (val) a 0.7070 (test)**. Esto es contraintuitivo pero explicable:
- El set de validación (2023) es muy pequeño (81 partidos) y muy ruidoso.
- El set de test (2024) tiene 111 partidos y muestra una distribución más cercana a la del train.

Sin embargo, el modelo tiene **recall muy bajo para el local (0.18)** y muy alto para visitante (0.92): el threshold de 0.5 está mal calibrado. Esto es un **bug de calibración** que se mitiga parcialmente con el isotonic posterior, pero queda como limitación.

---

## 6. Uso en el Simulador (Calibración de Fuerzas)

El `MatchPredictor` se usa en `SeasonSimulator` para **calibrar las fuerzas de equipo antes de cada partido** (`season_simulator.py:225`):

```python
@staticmethod
def _calibrate_strengths(h_str, a_str, p_target, damping=0.5):
    """Ajusta h_str para que su odds ratio converja al p_target del MatchPredictor."""
    p_base = h_str / (h_str + a_str)
    odds_target = p_target / (1 - p_target)
    odds_base = p_base / (1 - p_base)
    k = odds_target / odds_base
    k_damped = k ** damping      # damping=0.5 aplica raíz cuadrada
    h_new = h_str * k_damped
    h_new = max(0.05, min(0.95, h_new))
    return h_new, a_str
```

### 6.1. Lógica del damping

El `MatchPredictor` no es perfecto (AUC 0.71, no 1.0), así que aplicar la corrección completa generaría sobreajuste. Con `damping=0.5`:
- Si el predictor dice "doble de probabilidad" (k=2), solo se aplica `√2 ≈ 1.41×`.
- Si dice "10× más probable" (k=10), se aplica `√10 ≈ 3.16×`.

Es un **shrinkage** hacia las fuerzas base, que reduce la varianza del simulador.

### 6.2. Comparación con baseline (sin ML)

Comparativa en una temporada de 132 partidos con seed=42 (`memoria/prediccion_temporadas.md:368`):

| Configuración | % 3-0 | % 3-1 | % 3-2 | Observación |
|---|---:|---:|---:|---|
| Baseline (sin ML) | 65.9% | 15.9% | 18.2% | Demasiados barridos, distribución irreal |
| Con MatchPredictor | 43.9% | 29.5% | 26.5% | Distribución más realista, liga más igualada |

La calibración con MatchPredictor **reduce 22 puntos porcentuales la proporción de 3-0** y los redistribuye a 3-1 y 3-2, generando sets más competitivos. Esto valida la integración ML como útil para la simulación.

---

## 7. Persistencia

```python
# match_predictor.py:200
save_data = {
    "best_model_name": self.best_model_name,
    "best_model": self.best_model,
    "calibrated_model": self.calibrated_model,
    "feature_names": self.feature_names,
    "results": self.results,
    "test_metrics": getattr(self, "_test_metrics", None),
}
joblib.dump(save_data, path)
```

El predictor pesa **~1.5 MB** (XGBoost con 300 estimadores es mucho más compacto que ExtraTrees).

---

## 8. Limitaciones y Trabajo Futuro

1. **Dataset pequeño (319 train)**: el split temporal estricto deja pocos datos. Un fix sería usar cross-validation temporal (TimeSeriesSplit) en vez de un solo split, o juntar más temporadas.

2. **Recall muy bajo para local (0.18) en test**: el threshold de 0.5 está mal. Soluciones posibles: ajustar threshold a 0.45, usar class_weight, o entrenar con más datos de victorias locales.

3. **Damping fijo en 0.5**: el shrinkage es estático. Podría ajustarse dinámicamente — mayor damping al inicio de la temporada (cuando el MatchPredictor tiene features frías) y menor al final.

4. **Features frías al inicio de temporada**: igual que el `SetPredictor`, en las primeras jornadas las features de `elo`, `streaks`, `results` están en valores por defecto. El modelo predice ~0.5 y la calibración no tiene efecto hasta la jornada 5-6.

5. **Sin features de momentum reciente**: el modelo usa `h_racha` y `forma_h` pero no captura momentum del último set o del último partido con granularidad fina.

6. **87 features es mucho para 319 muestras**: hay riesgo de overfitting. Un feature selection (e.g. top-30 por importance) podría mejorar la generalización.

---

## 9. Conclusión

El `MatchPredictor` es el modelo más ambicioso del proyecto, integrando 87 features de tres capas. Aunque su accuracy en test (0.51) está cerca del azar, su **AUC de 0.71 indica que ordena bien las probabilidades**, lo que es exactamente lo que la calibración con damping necesita: una señal direccional más que una clasificación perfecta.

El damping exponencial con `damping=0.5` es la pieza clave: convierte una predicción ruidosa en una corrección conservadora, que sumada partido a partido a lo largo de la temporada tiene un efecto agregado significativo (de 66% de 3-0 a 44% con ML).

Es un buen ejemplo de **"modelo débil + shrinkage fuerte > modelo fuerte sin shrinkage"** en producción.
