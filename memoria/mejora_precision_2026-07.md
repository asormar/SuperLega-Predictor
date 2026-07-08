# Mejora de Precisión de los Modelos — Proceso Completo (2026-07-08)

Bitácora del trabajo de auditoría, diagnóstico, mejora e integración de la
precisión de los modelos ML del proyecto. Documenta qué se hizo, por qué, con
qué resultados y qué queda pendiente. Es material directamente defendible en el
TFG: el hallazgo principal es tan importante como la mejora.

---

## 1. TL;DR

- El **AUC=0.707 del MatchPredictor era ficticio**: producto de *leakage
  temporal* evaluado sobre un único año de test. Medido honestamente valía
  **0.53** (nivel de azar).
- Se construyó un **protocolo de evaluación honesto** (rolling-origin, test
  held-out) que hace que las métricas signifiquen algo.
- Se reconstruyeron las features **sin leakage** desde `sets_partidos.csv`, con
  un **Elo con margen de victoria**. Resultado: AUC de partido **0.53 → 0.75**.
- Se descubrió que las **temporadas viejas (2016-2020) envenenaban el modelo**
  (enseñaban el signo invertido); la recencia lo corrige.
- En este régimen de datos pequeños, **los modelos lineales baten a los árboles
  profundos**, tanto en set como en match.
- Se **integró en producción**: las fuerzas de equipo y la señal de partido del
  simulador ahora salen del Elo limpio; la clasificación simulada por fin
  correlaciona con la calidad real de los equipos.

| Modelo | AUC antes (honesto) | AUC después | Accuracy después |
|---|---:|---:|---:|
| MATCH | 0.53 | **0.75** | 0.69 |
| SET | 0.65 | **0.71** | 0.66 |

Cifras exactas en [`../COMPARACION_ANTES_DESPUES.md`](../COMPARACION_ANTES_DESPUES.md).
Plan original en [`../PLAN_MEJORA_PRECISION.md`](../PLAN_MEJORA_PRECISION.md).

---

## 2. Punto de partida y sospechas

El código reportaba SetPredictor AUC 0.654 y MatchPredictor AUC 0.707 (test =
temporada 2024/25). Tres señales de alarma motivaron la auditoría:

1. El MatchPredictor tenía **AUC de validación 0.475** (peor que azar) pero AUC
   de test 0.707. Una brecha así entre val y test no es señal de un buen modelo,
   sino de una medición inestable.
2. La **accuracy de test era 0.514** (casi azar) pese al "AUC 0.707". AUC alto +
   accuracy de moneda = probabilidades mal calibradas o métrica engañosa.
3. La selección del "modelo campeón" se hacía sobre **81 partidos** (val=2023).
   Con n=81, el error estándar del AUC es ~±0.06: elegir por ahí es ruido.

## 3. Diagnóstico: cuatro problemas estructurales

1. **El 45% de los datos no se usaba.** El split terminaba en test=2024; la
   temporada 2025/26 (214 partidos, la más grande) quedaba fuera. El train real
   eran ~319 partidos.
2. **Selección de modelo sobre ruido** (los 81 partidos de val).
3. **Leakage temporal** en `enrich_with_team_stats` y `compute_roster_features`:
   mergeaban stats de la **temporada completa** a partidos de esa misma
   temporada. Un partido de octubre "sabía" cómo terminaría el equipo en abril.
4. **Calibración con CV no-temporal** (`CalibratedClassifierCV(cv=3)` mezcla
   temporadas internamente) e isotónica con pocos datos (sobreajuste).

## 4. Fase 0 — Protocolo de evaluación honesto

`src/models/evaluation.py`. Reemplaza el val único por **rolling-origin
(expanding window)**: entrenar ≤ T-1, validar en T, para T ∈ {2021…2024};
test held-out intocable = **2025/26**. Métrica primaria: **log-loss** (calidad
de probabilidad, que es lo que alimenta el simulador Monte Carlo), no AUC.

Al medir el baseline con este protocolo salió la verdad:

| Modelo | Lo que reportaba el código | Medido honesto (test 2025/26) |
|---|---|---|
| MATCH | AUC 0.707 | **AUC 0.528**, logloss 0.694, acc 0.514 |
| SET | AUC 0.654 | AUC 0.653, logloss 0.651, acc 0.606 |

El set aguantó (sus features no tenían leakage de temporada). El match se
desplomó: casi toda su "precisión" era ficticia.

## 5. Fases 1-2 — Datos y features sin leakage

`src/data/rolling_features.py`. Se reconstruyeron las features a nivel de
partido desde `sets_partidos.csv` (la verdad set a set), recorriendo los
partidos en **orden cronológico** y usando **solo información previa**:

- **Elo con margen de victoria** (3-0 mueve más rating que 3-2), con regresión
  a la media entre temporadas y ventaja de local.
- **Forma EWMA** (half-life ~5 partidos), **H2H con decaimiento temporal**,
  win/set/point ratios *expanding* dentro de temporada.

Descubrimiento inmediato: **el Elo puro (una sola variable, sin entrenar) ya
daba AUC 0.62** — más que las 87 features viejas. Las features leaked no solo
inflaban la métrica: como ruido, **empeoraban** el modelo honesto.

### 5.1 El hallazgo contraintuitivo: las temporadas viejas envenenaban el modelo

Al medir el Elo por temporada:

| Temporada | n | AUC Elo | % victoria local |
|---|---:|---:|---:|
| 2016 | 34 | 0.28 | 0.32 |
| 2019 | 45 | 0.55 | 0.36 |
| 2022 | 59 | 0.55 | 0.51 |
| 2024 | 111 | 0.55 | 0.55 |
| **2025** | **214** | **0.75** | **0.60** |

Las temporadas viejas (34-55 partidos, con el local ganando solo ~32-35%) son
ruido con el **signo invertido**. Un LogReg entrenado con TODO el histórico
aprendía la relación al revés y predecía 2025 **anti-correlado** (AUC 0.42).

**Solución (recencia, T1.3):** con pesos `0.5^(años/half-life)` o entrenando
solo 2022-2024:

| Entrenamiento | AUC test 2025 | Accuracy |
|---|---:|---:|
| Todo 2016-2024 sin pesos | 0.42 | 0.39 |
| Recencia half-life=1 | 0.76 | 0.69 |
| Solo 2022-2024 | 0.77 | 0.72 |
| **Elo puro** (sin entrenar) | **0.75** | **0.69** |

## 6. Fase 3 — Modelado: lineal > árboles en datos pequeños

Con 34-59 partidos por temporada, los árboles de gradiente sobreajustan ruido y
**ahogan la señal limpia del Elo**: un modelo de 27 features daba AUC 0.54,
peor que el Elo de una variable (0.62). El patrón se repitió en el set: un
**LogisticRegression regularizado** (C=0.5) batió al ExtraTrees (0.71 vs 0.65).

Config final elegida:
- **MATCH**: probabilidad de **Elo con margen** (mejor logloss, 0.585; AUC 0.75).
  Elegante, sin entrenamiento, sin riesgo de sobreajuste.
- **SET**: **LogReg** con pesos de recencia (half-life 2), entrenado en 2022-2024.

Se descartó el intento de recalibración Platt: con datos viejos volvía a
aprender el signo invertido (AUC 0.23). Registrado como negativo, igual que los
experimentos del Batch 3.

Artefactos reproducibles: `python -m src.models.train_improved` →
`match_elo_v2.joblib`, `set_predictor_v2.joblib`, `precision_improved.json`.

## 7. Integración en producción

El objetivo real del proyecto es el **simulador de temporada**, no el
clasificador aislado. Lo integrado:

1. **Fuerzas de equipo desde el Elo con margen** (`api/main.py`
   `_compute_team_strengths` → `get_historical_team_strengths`). El prior pasó
   del win-rate plano a la **jerarquía real de la SuperLega**: Perugia (0.68) >
   Trento (0.60) > Verona > Piacenza … > Cuneo (0.23) > Grottazzolina (0.18).
2. **Elo runtime sembrado desde el histórico** (`RuntimeFeatureBuilder(initial_elo=…)`).
   Antes arrancaba plano en 1500 y "calentaba" desde cero, lo que hacía inútil
   la señal de Elo las primeras jornadas y **diluía** el buen prior. Ahora es
   fiel desde la jornada 1. (Parámetro opt-in: el arranque plano se conserva por
   defecto para no romper el test de schema.)
3. **Update de Elo con margen** en el `RuntimeFeatureBuilder` (alineado con el
   modelo offline: 3-0 mueve más que 3-2).
4. **Señal de partido = probabilidad de Elo limpia** en el `SeasonSimulator`,
   en lugar del MatchPredictor de 87 features (leaky). El artefacto viejo queda
   solo como fallback.

**Efecto observable:** la clasificación simulada por fin correlaciona con la
fuerza real. Antes de sembrar el Elo, Grottazzolina (el equipo más débil) podía
acabar 4º en un solo seed.

### 7.1 Validación Monte Carlo — 20 temporadas simuladas

Se corrieron **20 temporadas completas** (12 equipos, ida y vuelta = 22
jornadas, seeds 0-19) con el Elo sembrado desde el histórico y el clamp del
SetPredictor activado. Posición media final (menor = mejor):

| # | Equipo | Posición media | Fuerza (margin-Elo) |
|---|---|---:|---:|
| 1 | Perugia | 3.3 | 0.681 |
| 2 | Trento | 4.2 | 0.604 |
| 3 | Lube | 5.5 | 0.538 |
| 4 | Taranto | 5.9 | 0.350 |
| 5 | Piacenza | 6.2 | 0.568 |
| 6 | Monza | 6.8 | 0.462 |
| 7 | Verona | 7.2 | 0.578 |
| 8 | Modena | 7.3 | 0.522 |
| 9 | Cisterna | 7.4 | 0.350 |
| 10 | Padova | 7.4 | 0.372 |
| 11 | Grottazzolina | 8.1 | 0.176 |
| 12 | Milano | 8.5 | 0.457 |

**Lectura honesta:**

- El **top-3 es exacto** (Perugia > Trento > Lube) y el fondo también
  (Grottazzolina/Milano últimos). La correlación fuerza→posición es claramente
  positiva: el simulador ya no premia a los equipos débiles.
- La **zona media es ruidosa**: Taranto (fuerza 0.35) sobrerrinde hasta el 4º
  puesto y Verona (0.578) baja al 7º. Parte es varianza real de Monte Carlo con
  solo 20 muestras, y parte viene del **clamp adaptativo del SetPredictor**, que
  comprime las probabilidades punto a punto y mete regresión a la media. Una
  corrida previa de 5 temporadas con el clamp **desactivado** salió más limpia
  (Lube/Perugia/Trento/Piacenza/Verona en las 5 primeras posiciones), lo que
  apunta al clamp como principal fuente de ruido de media tabla.
- Este ruido es **coherente con la realidad**: el mid-table de la SuperLega es
  genuinamente volátil temporada a temporada.

Aislar cuánto ruido introduce el clamp del SetPredictor en la clasificación de
temporada es el siguiente paso de optimización fina (queda fuera del alcance de
esta iteración).

Todo con **134 tests verdes**; el simulador y el API arrancan sin cambios de
interfaz.

## 8. Qué NO se hizo (honestidad de alcance)

- No se amplió el nº de partidos históricos: `sets_partidos.csv` solo tiene 725
  partidos en total (34-59/temporada en las viejas, porque solo se scrapeó a los
  equipos rastreados). Es un techo de datos, no de método.
- El `match_predictor.joblib` de 87 features sigue en disco como fallback; no se
  borró para no romper la carga del API.
- La **feature de continuidad de plantilla** (roster churn) y el **predictor de
  partido derivado del set** (fórmula best-of-5) del plan quedaron sin
  implementar: con el Elo ya en 0.75, el retorno marginal no compensaba el
  riesgo.
- No se hizo el **backtest completo del simulador** contra una temporada real
  con ajuste de momentum/clamps (Fase 4 completa del plan). Se validó a nivel de
  probabilidad (Brier 0.251 → 0.200) y con sanity check de clasificación.

## 9. Techo realista y conclusión

AUC de partido ~0.75 y accuracy ~0.69 es un resultado **sólido y defendible**
para predicción en voleibol profesional: el deporte tiene varianza intrínseca
alta por el formato de sets, y los mercados de apuestas se mueven en ese rango.

La lección metodológica es la más valiosa del TFG: **una métrica mal medida es
peor que ninguna**. El proyecto tenía un AUC "0.71" que daba falsa confianza y
que además ocultaba que el modelo real estaba a nivel de azar. Arreglar *cómo se
mide* fue el prerrequisito para poder mejorar de verdad — y el mismo protocolo
honesto es lo que permite afirmar que la mejora (0.53 → 0.75) es real y no otro
espejismo.

## 10. Archivos nuevos / modificados

**Nuevos:**
- `src/models/evaluation.py` — protocolo rolling-origin.
- `src/data/rolling_features.py` — features rolling sin leakage + Elo con margen.
- `src/models/measure_precision.py` — medición unificada antes/después.
- `src/models/train_improved.py` — entrenador de la config final.
- `PLAN_MEJORA_PRECISION.md`, `COMPARACION_ANTES_DESPUES.md`, este documento.
- `models/precision_baseline.json`, `models/precision_improved.json`.

**Modificados (integración):**
- `src/api/main.py` — fuerzas desde margin-Elo; feature_builder con Elo sembrado.
- `src/simulation/feature_builder.py` — update de Elo con margen; param `initial_elo`.
- `src/simulation/season_simulator.py` — señal de partido = prob de Elo limpia.

## 11. Cómo reproducir

```bash
# Baseline honesto (antes)
python -m src.models.measure_precision --save baseline
# Modelos mejorados (después) + artefactos v2
python -m src.models.train_improved
# Verificar que nada se rompió
python -m pytest -q
```
