# Point Probability Model — Probabilidad de Ganar un Punto

## Descripción

El `PointProbabilityModel` (`src/models/point_probability.py`) es el modelo más simple del proyecto: convierte las fuerzas relativas de dos equipos en **4 probabilidades de ganar un punto**, según quién esté sacando. Es el bloque fundamental que alimenta el `MatchSimulator` (Markov chain punto a punto).

*Salida: dict con `p_home_serving`, `p_home_receiving`, `p_away_serving`, `p_away_receiving` · Usado por: `MatchSimulator` en cada rally*

---

## 1. Las 4 Probabilidades

En volleyball, la probabilidad de ganar un punto depende de **quién saca**:

| Probabilidad | Significado | Típica |
|---|---|---:|
| `p_home_serving` | P(local gana el punto \| local saca) | ~0.45 |
| `p_home_receiving` | P(local gana el punto \| visitante saca) | ~0.65 |
| `p_away_serving` | P(visitante gana el punto \| visitante saca) | ~0.35 |
| `p_away_receiving` | P(visitante gana el punto \| local saca) | ~0.55 |

**Observación clave**: `p_home_receiving` > `p_home_serving` porque cuando el visitante saca, el local está en **recepción** y en volleyball profesional el equipo que recibe gana ~62% de los rallies (puede organizar un ataque combinado). La diferencia `p_home_receiving - p_home_serving` ≈ 2 × `sideout_rate - 0.5` ≈ 0.24, que es lo que se observa.

---

## 2. Fórmula de Cálculo

```python
# point_probability.py:90
def get_point_probabilities(
    self,
    match_features: Optional[dict] = None,
    home_strength: float = 0.5,
    away_strength: float = 0.5,
) -> dict:
    # 1. Probabilidad base de punto del local
    if match_features and self.is_fitted:
        X = pd.DataFrame([match_features])[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        p_home_dominant = self.model.predict_proba(X_scaled)[0, 1]
        p_home_point = 0.45 + 0.10 * p_home_dominant  # Range: [0.45, 0.55]
    else:
        # Fallback: usar strength directamente
        total = home_strength + away_strength
        p_home_point = home_strength / total if total > 0 else 0.5

    p_away_point = 1.0 - p_home_point

    # 2. Ajuste por sideout rate (0.62)
    sideout = self.DEFAULT_SIDEOUT_RATE  # 0.62

    # 3. P(local gana | local saca) = p_home × (1-sideout) / [p_home × (1-sideout) + p_away × sideout]
    p_home_serving = p_home_point * (1 - sideout) / (
        p_home_point * (1 - sideout) + p_away_point * sideout
    )
    p_home_receiving = p_home_point * sideout / (
        p_home_point * sideout + p_away_point * (1 - sideout)
    )

    # 4. Clamp final
    p_home_serving = np.clip(p_home_serving, 0.25, 0.75)
    p_home_receiving = np.clip(p_home_receiving, 0.25, 0.75)

    return {
        "p_home_serving": p_home_serving,
        "p_home_receiving": p_home_receiving,
        "p_away_serving": 1.0 - p_home_receiving,
        "p_away_receiving": 1.0 - p_home_serving,
    }
```

### 2.1. Desglose de la fórmula

**Paso 1**: `p_home_point` se calcula de dos formas:
- **Con modelo entrenado** (`is_fitted=True`): una `LogisticRegression` predice la probabilidad de que el local sea "dominante en puntos" (P(point_ratio_h > 0.5)) en función de 6 features. El output se mapea a `[0.45, 0.55]` para mantener la predicción conservadora.
- **Sin modelo** (fallback): `p_home_point = home_strength / (home + away)`. Es la probabilidad "naive" basada solo en win rates.

**Paso 2**: `sideout = 0.62`. Es la probabilidad de que el equipo que recibe el saque gane el rally. Valor hardcodeado como `DEFAULT_SIDEOUT_RATE` (point_probability.py:38), estimado de datos históricos de la SuperLega.

**Paso 3**: Se separan las 4 probabilidades aplicando el modelo de sideout:
- Si el local saca: su probabilidad se reduce por `(1-sideout) = 0.38`.
- Si el visitante saca: la probabilidad del local se multiplica por `sideout = 0.62`.
- La normalización por la suma `p_home × adj + p_away × adj_opuesto` garantiza que ambas probabilidades sumen 1.

**Paso 4**: Clamp final a `[0.25, 0.75]` para evitar eventos deterministas en el simulador (un equipo nunca tiene <25% o >75% de probabilidad de ganar un punto individual, sin importar las strengths).

---

## 3. Modelo Subyacente (LogisticRegression)

El modelo entrenado es una `LogisticRegression(max_iter=1000, random_state=42)`. Sus features son:

| Feature | Fuente |
|---|---|
| `elo_diff` | `match_features.csv` |
| `diff_win_rate_global` | `match_features.csv` |
| `diff_set_win_rate` | `match_features.csv` |
| `diff_dominancia` | `match_features.csv` |
| `diff_set_ratio` | `match_features.csv` |
| `diff_forma_efectiva` | `match_features.csv` |

**Target**: binarización de `point_ratio_h` (ratio de puntos del local). `y = (point_ratio_h > 0.5).astype(int)`.

### 3.1. Métricas del último re-entrenamiento

```
[PointProbability] Base home point prob: 0.5298
[PointProbability] Base away point prob: 0.5221
[PointProbability] Default sideout rate: 0.62
```

Es decir, el modelo base estima que el local gana ~53% de los puntos en promedio y el visitante ~52% (no suman 1 porque son medias independientes sobre partidos diferentes, no probabilidades condicionales).

### 3.2. Tamaño del artefacto

`models/point_probability.joblib` pesa solo **~2 KB** porque una `LogisticRegression` con 6 features es extremadamente compacta. Esto es normal y NO indica bug — el archivo contiene:
- `model`: `LogisticRegression(max_iter=1000, random_state=42)` con 6 coeficientes + intercept
- `scaler`: `StandardScaler` con media y std de 6 features
- `feature_cols`: lista de 6 strings
- `base_home_point_prob`, `base_away_point_prob`, `is_fitted`: floats/booleans

El formato de persistencia es un `dict` con esas keys, restaurado en `PointProbabilityModel.load()`.

---

## 4. Uso en el Simulador

El `MatchSimulator` recibe el `PointProbabilityModel` en su constructor (`src/api/main.py:93`) y lo usa en cada rally:

```python
# src/simulation/simulator.py:381 (aprox, _default_point_probs fallback)
# Cuando el point_model está disponible:
probs = self.point_model.get_point_probabilities(
    home_strength=home_strength,
    away_strength=away_strength,
)
p_home_serving = probs["p_home_serving"]
p_home_receiving = probs["p_home_receiving"]

# En el loop punto a punto:
if server == "home":
    p_home_wins = p_home_serving
else:
    p_home_wins = p_home_receiving
```

El modelo aporta el **componente "fuerza"** del Markov chain. La dinámica intra-rally (momentum, sideout) se suma por separado en `MatchSimulator`.

### 4.1. Fallback si el modelo no carga

Si `point_probability.joblib` falta o está corrupto, la API pone `point_model = None` y el `MatchSimulator` usa `_default_point_probs(home_strength, away_strength)` (`simulator.py:381`), que aplica la misma fórmula con `sideout=0.62` hardcodeado pero sin pasar por el LogisticRegression. Es decir, **el simulador sigue funcionando pero sin ajuste por features de partido**.

---

## 5. Limitaciones

1. **Sideout rate constante (0.62)**: el modelo asume que TODOS los equipos tienen el mismo sideout rate. En realidad, los equipos con mejor recepción tienen sideout más alto (Perugia, Trento ~65%) y los débiles más bajo (~58%). Una mejora sería hacer el sideout rate por equipo.

2. **Probabilidad de punto fija en el set**: el modelo devuelve un set de 4 probabilidades al inicio del partido y no las actualiza durante el set. No modela fatiga, cambios tácticos, ni el efecto del marcador parcial (estar 24-20 vs. 0-0 se trata igual).

3. **Mapping conservador [0.45, 0.55]**: la LogisticRegression devuelve P(point_ratio_h > 0.5), pero se mapea a un rango muy estrecho. Esto es intencional para evitar overconfidence, pero pierde capacidad discriminativa entre equipos.

4. **Features limitadas (6)**: el modelo solo usa diferencias de stats agregadas, no features in-match (momentum, score parcial, cansancio). Para el partido suelto, esto es suficiente. Para la temporada, podría complementarse con features de `RuntimeFeatureBuilder`.

5. **Sin reentrenamiento periódico**: el modelo se entrena una vez con todos los datos históricos. Si la liga cambia (nuevo equipo, regla nueva), el modelo se desactualiza.

---

## 6. Conclusión

El `PointProbabilityModel` es el componente más "artesanal" del proyecto: una fórmula con 3 pasos (probabilidad base → ajuste por sideout → clamp) y un `LogisticRegression` simple como pieza opcional. Su valor está en **convertir dos scalars (home_strength, away_strength) en 4 probabilidades físicamente plausibles** que el Markov chain puede consumir.

Es el modelo con menos AUC (no se reporta explícitamente porque no es un clasificador puro, es un regresor) pero **el más estable**: sus predicciones son siempre sensatas gracias al clamp y al sideout hardcodeado, y la `LogisticRegression` aporta una corrección pequeña pero calibrada.

Para el TFG, este modelo es interesante porque muestra que **no siempre se necesita un modelo complejo**: a veces un modelo paramétrico simple con buenos priors (sideout=0.62) supera a un modelo "data-driven" sin esas restricciones.
