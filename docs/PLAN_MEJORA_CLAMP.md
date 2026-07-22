# Plan de Mejora del Clamp Adaptativo del SetPredictor

> **STATUS: HISTÓRICO (2026-07-22)** — Todos los items T1–T6 ejecutados como A1–A6 dentro del [Plan Consolidado](./PLAN_MEJORAS_CONSOLIDADO.md) §GRUPO A (cerrado 2026-07-21). Se conserva este documento como registro del diagnóstico y de las decisiones. Desenlace: `SET_BLEND_WEIGHT_ELO = 1.0` (SetPredictor cableado pero inactivo en runtime). Detalle del desenlace en `memoria/set_predictor.md` §10.5 y `memoria/simulator.md` §4.3.

Investigación del ruido que el clamp del SetPredictor introduce en la
clasificación de temporada, y plan para arreglarlo. Continuación natural de
`PLAN_MEJORA_PRECISION.md` (§7.1 de `memoria/mejora_precision_2026-07.md`
dejó esta pregunta abierta).

## 1. Cómo funciona el mecanismo hoy

Al inicio de cada set (`simulator.py::_simulate_set`):

1. `_eval_set_predictor` construye un contexto con features de equipo
   (`_extract_set_team_features`) + estado del set (marcador 0-0) y pide al
   SetPredictor de producción (ExtraTrees calibrado) `p_set_home`.
2. El clamp de probabilidad de punto se centra ahí:
   `[p_set − 0.20, p_set + 0.20]` recortado a `[0.10, 0.90]`.
3. Cada punto del set se recorta a ese rango
   (`p_home_wins = clip(base_p + momentum, clamp_low, clamp_high)`).

Sin SetPredictor, el clamp es el fijo `DEFAULT_CLAMP_RANGE = (0.20, 0.80)`.

## 2. Diagnóstico: cuatro defectos encontrados

### 2.1 BUG previo (ya arreglado): el sembrado de Elo nunca llegó a producción

Investigando el clamp se descubrió que `rolling_features.py` usaba `Optional`
sin importarlo. El `try/except` del API tragaba el `NameError` y caía
**silenciosamente** al fallback win-rate + Elo plano: ni las fuerzas
margin-Elo ni el Elo sembrado estuvieron nunca activos. Señal delatora:
`elo_win_prob_h` constante 0.592 para los 132 pares de equipos. Arreglado en
el commit `fix(data): importar Optional...`; ahora Perugia vs Grottazzolina
da p_elo=0.949 y las fuerzas son las del margin-Elo real.

> Lección de proceso: verificar integraciones **asertando comportamiento**
> (que `p_elo` varíe por par), no por ausencia de warnings.

### 2.2 Train/serve skew masivo en las features del SetPredictor

Distribución de entrenamiento (`DB/features/set_features.csv`) vs lo que el
runtime le pasa (`_eval_set_predictor` + `_extract_set_team_features`):

| Feature | Entrenamiento | Runtime | Problema |
|---|---|---|---|
| `pts_fav_h/a` | media 3.6, rango [1, 5.1] | marcador del set: **0** al inicio, hasta 25+ | fuera de distribución SIEMPRE (el clamp se evalúa a 0-0) |
| `momentum_h` | [0, 1], media 0.54 | `(sh−sa)/total` ∈ [−1, 1], 0.0 al inicio | escala distinta |
| `h2h_diff` | [−3, 3] (diff de sets) | `(wr−0.5)×2` ∈ [−1, 1] | escala distinta |
| `set_wr_h/a` | rango [0.41, 0.58] | rolling [0, 1] | mucho más ancho |
| `strength_h/a` | media 0.53, std 0.09 | `elo/3000` ≈ [0.42, 0.57] | comprimido |
| `elo_diff` | std 67, max 428 | con Elo sembrado hasta ~440 | borde de distribución |

### 2.3 p_set es casi constante (el modelo no discrimina en runtime)

Evaluado en los 132 pares ordenados de los 12 equipos (contexto 0-0):
`p_set ∈ [0.537, 0.553]`, std 0.007. **El SetPredictor devuelve ~0.54 para
cualquier par de equipos** — con inputs fuera de distribución, el ExtraTrees
regurgita la base rate. El clamp resultante es ~`[0.34, 0.74]` para todos los
sets de todos los partidos.

Consecuencia mecánica: comparado con el clamp por defecto `(0.20, 0.80)`, el
clamp "adaptativo" real **sube el suelo de 0.20 a ~0.34 y baja el techo de
0.80 a ~0.74 por igual para todos** → comprime las diferencias: los débiles
ganan más puntos de los que deberían y los favoritos menos → más upsets →
ruido en la clasificación (Taranto 4º con fuerza 0.35 en el MC de 20
temporadas).

### 2.4 Desajuste conceptual de escala: probabilidad de SET usada como centro de clamp de PUNTO

Aunque el modelo discriminara bien, centrar el clamp de probabilidad de
*punto* en la probabilidad de ganar el *set* es un error de escala: un
favorito con P(set)=0.75 solo necesita P(punto)≈0.55. La relación
set↔punto es una amplificación logística (~25 puntos por set); usar p_set
como si fuera p_punto infla sistemáticamente el rango permitido para el
favorito y lo desinfla para el underdog en el orden de +0.15/−0.15.

### 2.5 Coste: 260 ms por evaluación

`_eval_set_predictor` (ExtraTrees + isotonic sobre DataFrame de 1 fila)
cuesta ~260ms. Una temporada simulada con clamp ON gasta ~137s solo en
predicts (132 partidos × ~4 sets). Hace inviable el Monte Carlo de temporada
con clamp activado (la corrida de 20 temporadas tardó >7 min por esto).

### 2.6 Cuantificación del ruido en la clasificación (ON vs OFF)

Con el sembrado ya arreglado (§2.1), dos mediciones:

**a) La señal del clamp no contiene información.** Sobre los 132 pares
ordenados de 12 equipos (Elo sembrado, contexto 0-0):

| Señal | std | rango | correlación con p_elo |
|---|---:|---|---:|
| `p_elo` (margin-Elo) | 0.220 | [0.10, 0.95] | — |
| `p_set` (SetPredictor) | **0.012** | [0.50, 0.57] | **ρ = −0.006** |

Correlación de Spearman ≈ **cero**: el centro del clamp no sabe qué equipo es
mejor. Todo lo que el clamp hace, lo hace sin información.

**b) Efecto en la clasificación** (6 temporadas de ida simple, mismas seeds):

| Métrica | CLAMP OFF | CLAMP ON | Δ |
|---|---:|---:|---|
| Spearman fuerza→posición media | 0.867 | 0.886 | ≈ empate (ruido de 6 seeds) |
| std de posición entre seeds | **1.75** | **2.13** | **+22% de varianza** |
| pos. media top-3 por fuerza | 3.72 | 4.00 | favoritos comprimidos hacia el centro |
| pos. media bottom-3 | 9.78 | 9.67 | débiles ligeramente inflados |
| coste por temporada | **1 s** | **64 s** | **×60** |

Lectura: con las fuerzas margin-Elo correctas dominando la señal, el clamp ya
no invierte el orden medio (el desastre "Taranto 4º" del MC original se debía
sobre todo al sembrado roto), pero **añade +22% de varianza de posición,
comprime a los favoritos ~0.3 posiciones hacia el centro y multiplica por 60
el coste — a cambio de cero información** (ρ≈0). Es puro ruido caro.

### 2.7 Mitigación inmediata disponible

Mientras no se ejecute el plan, `use_set_calibration=False` en
`simulate_season`/`simulate_jornada` elimina el ruido y el sobrecoste sin
tocar código. No se cambia el default en este commit para no alterar el
comportamiento de la API sin backtest (T5), pero es la palanca si se quiere
la clasificación más fiel hoy.

## 3. Plan de mejora (para ejecutar)

Ordenado por impacto/esfuerzo. Cada tarea es un commit.

### T1 — Reemplazar el centro del clamp: de p_set a p_punto implícito (core fix)

En lugar de centrar el clamp en `p_set`, convertirlo a la probabilidad de
punto equivalente invirtiendo la relación analítica set↔punto:

- Función `p_point_from_p_set(p_set, target=25)`: invertir numéricamente
  P(ganar set a `target` | p punto iid) — la CDF es calculable en cerrado con
  la binomial negativa + deuce geométrico, o por bisección sobre la función
  directa (ya la usa el simulador conceptualmente). Cachear con `lru_cache`
  redondeando p_set a 3 decimales.
- Clamp nuevo: `[p_punto − margen, p_punto + margen]` con margen a re-tunear
  (arrancar con 0.10, ya que la escala de punto es mucho más estrecha).
- Mantener `POINT_PROB_CLIP_ADAPTIVE_HARD` como límite duro.

### T2 — Arreglar el train/serve skew de features (o dejar de alimentar basura)

Dos opciones, elegir por resultado del backtest (T5):

- **Opción A (parche mínimo):** alinear las features de runtime con la
  distribución de entrenamiento: `pts_fav_h/a` = puntos *por rotación*
  esperados (escala 3-5, no marcador); `momentum_h` en [0,1];
  `h2h_diff` en escala [−3,3]; `set_wr` estrechado. Es frágil: cualquier
  cambio futuro vuelve a desalinear.
- **Opción B (recomendada):** sustituir el SetPredictor de producción por el
  **v2 LogReg con recencia** (`set_predictor_v2.joblib`, AUC 0.71 vs 0.65) y
  definir un **contrato de features runtime** único (mismo builder para
  entrenar y servir, como se hizo con el Elo). LogReg además elimina el
  problema de coste: ~0.1ms vs 260ms por predict (T4 gratis).

### T3 — Suavizar el mecanismo: blend en vez de clip duro

El clip actual es un override: si la señal del SetPredictor discrepa de la
del Elo, gana el SetPredictor sin matización. Cambiar a mezcla en espacio de
punto: `p_final = w·p_elo_punto + (1−w)·p_set_punto`, con `w` tuneable
(arrancar 0.7 dado que el Elo demostró logloss 0.585 vs el SetPredictor
legacy). El clamp duro queda solo como salvavidas `[0.10, 0.90]`.

### T4 — Rendimiento: evaluar el set-predictor una vez por partido, no por set

Si tras T2-B el coste ya es trivial, omitir. Si se mantiene ExtraTrees:
evaluar `p_set` una sola vez por partido (contexto 0-0) y ajustar solo el
delta por marcador de sets con la fórmula analítica, en lugar de 4-5
predicts de 260ms.

### T5 — Criterio de aceptación: backtest reproducible

Script `src/models/backtest_clamp.py` que fije el veredicto con números:

1. **Nivel partido:** para los 132 pares, |P_MC(win) − p_elo| con N=200 sims,
   config vieja vs nueva. La nueva no debe degradar la fidelidad al Elo
   (referencia: logloss 0.585 del margin-Elo en test 2025/26).
2. **Nivel temporada:** ≥10 temporadas ida simple ON vs OFF vs NUEVO;
   Spearman(fuerza, posición media) y std de posición. Objetivo: NUEVO ≥ OFF
   en Spearman (el clamp debe dejar de restar) y aportar la señal in-match
   que motivó el mecanismo.
3. Time-boxed: medir coste por llamada antes de dimensionar (lección
   aprendida: 260ms/predict hace explotar los MC ingenuos).

### T6 — Tests y documentación

- Actualizar los tests que fijan el mecanismo (`test_simulator.py` pinea
  ambos clamp ranges y `CLAMP_MARGIN`) — conscientemente, no de pasada.
- Documentar en `memoria/simulator.md` (§ clamp adaptativo) y añadir el
  resultado del backtest a `memoria/mejora_precision_2026-07.md`.
- Corregir §7.1: el MC de 20 temporadas documentado corrió con el sembrado
  roto (bug Optional); re-ejecutar la tabla con la integración real y
  reemplazarla, marcando la anterior como inválida.

## 4. Resultado esperado

- El clamp deja de comprimir artificialmente las diferencias entre equipos
  (fin del "Taranto 4º"), manteniendo la idea original (modular el rango de
  punto según el estado del partido) pero con señal real y en la escala
  correcta.
- Simulaciones de temporada con clamp activado ~1000× más rápidas si se
  adopta T2-B (LogReg v2).
- Un backtest reproducible que impide que el mecanismo vuelva a degradarse
  en silencio.
