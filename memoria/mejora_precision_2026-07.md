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
- La sospecha de envenenamiento de las temporadas viejas quedó **invalidada por
  B0** (era artefacto de la colisión `partido_id`); la recencia operativa
  2022-2024 se mantiene por ciclo de plantillas (half-life 2 temporadas),
  no por sign-flip.
- En este régimen de datos pequeños, **los modelos lineales baten a los árboles
  profundos**, tanto en set como en match.
- Se **integró en producción**: las fuerzas de equipo y la señal de partido del
  simulador ahora salen del Elo limpio; la clasificación simulada por fin
  correlaciona con la calidad real de los equipos.

| Modelo | AUC antes (honesto) | AUC después | Accuracy después |
|---|---:|---:|---:|
| MATCH | 0.53 | **0.75** | 0.69 |
| SET | 0.65 | **0.697** | 0.65 |

El SET v2 tras corrección B0b (2026-07-15): test 2025 AUC **0.697** (n=1193),
CV rolling-origin 2-fold **0.679 ± 0.017**. Detalle en §7.2.

Cifras exactas en [`../docs/COMPARACION_ANTES_DESPUES.md`](../docs/COMPARACION_ANTES_DESPUES.md).
Plan original en [`../docs/PLAN_MEJORA_PRECISION.md`](../docs/PLAN_MEJORA_PRECISION.md).

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

> **⚠️ CORRECCIÓN (2026-07-15) — esta sección era un ARTEFACTO de un bug de datos.**
> Al implementar el backtest del simulador se descubrió que `partido_id` en
> `sets_partidos.csv` COLISIONA la ida y la vuelta de cada cruce, y
> `_aggregate_matches` (que agrupaba solo por `partido_id`) **fundía dos partidos
> en uno**, sumando sus sets e invirtiendo el target `gana_local` en el 82% de los
> partidos. Los "% victoria local ~0.32-0.35" y el "AUC 0.28-0.55 / signo
> invertido" de las temporadas viejas eran ese artefacto, NO datos reales.
> Con la agregación corregida (agrupar por `(partido_id, local)` → 1322 partidos):
> el home-win es **0.48-0.61 en TODAS las temporadas** y el Elo AUC es **~0.76 en
> 2024 y 2025** (2024 pasó de 0.545 a 0.756). **Ninguna temporada envenena el
> modelo**; la justificación de la recencia (entrenar solo 2022-2024) se apoya en
> este artefacto y debe re-evaluarse. La tabla y el texto de abajo se conservan
> como registro histórico. Detalle y plan en `docs/PLAN_MEJORAS_CONSOLIDADO.md` §B0.

Tras la corrección B0 (2026-07-15), la lectura correcta de las temporadas
viejas es:

- **Home-win rate consistente 0.48-0.61 en todas las temporadas**, sin
  envenenamiento sistémico. El "0.32-0.35" era artefacto de la colisión
  `partido_id`.
- **AUC Elo limpio**: 2024 = 0.756, 2025 = 0.762, sin caída en las temporadas
  viejas.
- La **recencia 2022-2024** se justifica por **ciclo de plantillas** (half-life
  2 temporadas ≈ renovación típica de roster), no por sign-flip. Con datos
  limpios, entrenar con todo el histórico ya NO produce AUC 0.42; el problema
  que reveló este análisis era un artefacto del bug.

| Temporada | n_partidos (corregido) | Home-win rate | AUC Elo |
|---|---:|---:|---:|
| 2016-2025 (agregado) | ~1322 | 0.48-0.61 | — |
| 2024 | ~111 | ~0.55 | 0.756 |
| 2025 | ~214 | ~0.60 | 0.762 |

→ **Tabla invalidada y narrativa previa (home-win ~0.32, AUC 0.28-0.55, signo
invertido, AUC 0.42 con todo el histórico):** ver
[`memoria/registro_historico_b0.md`](registro_historico_b0.md) §A.1-A.2.

## 6. Fase 3 — Modelado: lineal > árboles en datos pequeños

Con 34-59 partidos por temporada, los árboles de gradiente sobreajustan ruido y
**ahogan la señal limpia del Elo**: un modelo de 27 features daba AUC 0.54,
peor que el Elo de una variable (0.62). El patrón se repitió en el set: un
**LogisticRegression regularizado** (C=0.5) batió al ExtraTrees en el test 2025
(0.71* vs 0.65). **\* El 0.71 es 2025-específico** y además estaba medido sobre
el dataset colisionado (B0b). Con datos limpios (2026-07-15): test 2025 AUC
**0.697** (n=1193), CV 2-fold **0.679 ± 0.017**. El CV rolling-origin honesto
multi-temporada (0.68) es la magnitud defendible, no el test. Detalle en §7.2.

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
3. **Update de Elo con margen** en el `RuntimeFeatureBuilder` importando
   las constantes canónicas desde `src.data.rolling_features`
   (K=28, HOME_ADV=60, ELO_BASE=1500, ELO_SEASON_REGRESS=0.25). Antes
   de este fix las constantes estaban duplicadas con valores distintos
   (runtime usaba K=32, HOME_ADV=65) — train/serve skew clásico.
   Margen mult `1 + 0.15·(|diff_sets|-1)` (3-0 → 1.30, 3-1 → 1.15,
   3-2 → 1.00) aplicado en ambos lados.
4. **Señal de partido = probabilidad de Elo limpia** en el `SeasonSimulator`,
   en lugar del MatchPredictor de 87 features (leaky). El artefacto viejo queda
   solo como fallback.

**Efecto observable:** la clasificación simulada por fin correlaciona con la
fuerza real. Antes de sembrar el Elo, Grottazzolina (el equipo más débil) podía
acabar 4º en un solo seed.

### 7.1 Validación Monte Carlo — 20 temporadas simuladas

**Corrida vigente (2026-07-21, tras A2/A4).** 20 temporadas completas
(12 equipos, ida y vuelta, seeds 0-19), Elo sembrado desde el histórico real,
clamp en la configuración final del Grupo A (`SET_BLEND_WEIGHT_ELO = 1.0`,
`CLAMP_MARGIN_POINT = 0.10`). Estado dinámico reiniciado en cada temporada.
Reproducible con `python -m src.models.mc_season_validation --n-seeds 20`;
cifras en `models/mc_season_validation.json`.

| # | Equipo | Posición media | Pos. std | Fuerza (margin-Elo) |
|---|---|---:|---:|---:|
| 1 | Perugia | 1.1 | 0.44 | 0.833 |
| 2 | Trento | 2.0 | 0.00 | 0.734 |
| 3 | Verona | 3.4 | 0.86 | 0.672 |
| 4 | Modena | 4.3 | 0.95 | 0.639 |
| 5 | Lube | 4.6 | 0.97 | 0.616 |
| 6 | Piacenza | 5.7 | 0.73 | 0.580 |
| 7 | Taranto | 7.0 | 0.38 | 0.526 |
| 8 | Milano | 7.9 | 0.30 | 0.474 |
| 9 | Monza | 9.2 | 0.43 | 0.374 |
| 10 | Padova | 9.8 | 0.43 | 0.331 |
| 11 | Cisterna | 11.0 | 0.00 | 0.275 |
| 12 | Grottazzolina | 12.0 | 0.00 | 0.183 |

**Spearman fuerza→posición = −1.0000** (p < 1e-16); std media de posición 0.457.

**Lectura honesta:**

- El orden simulado reproduce **exactamente** el orden de fuerza margin-Elo.
  Las dos anomalías de la corrida vieja desaparecen: Taranto ya no sobrerrinde
  al 4º puesto (ahora 7º, coherente con su fuerza 0.526 post-B0) y Verona sube
  al 3º (fuerza 0.672). Ambas eran consecuencia del sembrado roto de Elo y de
  las fuerzas colisionadas de pre-B0, no del simulador.
- **Pero un Spearman de −1.0 es demasiado limpio para ser realista.** Con
  22 jornadas, una liga real reordena el mid-table; aquí cuatro equipos tienen
  std de posición 0.00. El simulador está **sub-disperso**: produce ligas casi
  deterministas. Es la misma patología que cuantifica el backtest B1 (ECE 0.242,
  53% de 3-0 simulados frente a 39% reales) y **no la corrige el Grupo A**,
  porque su origen está en el modelo de punto, no en el clamp. Queda como
  trabajo pendiente (B3: `PointProbabilityModel` con regresión continua).
- Es decir: esta tabla valida que la **señal de fuerza llega íntegra** al
  simulador, no que la *incertidumbre* esté bien calibrada.

<details>
<summary>Corrida histórica invalidada (2026-07-08) — se conserva como registro del proceso</summary>

> **⚠️ TABLA INVALIDADA (2026-07-08, misma tarde).** Al investigar el ruido
> del clamp se descubrió que esta corrida se ejecutó con el sembrado de Elo
> **roto** (bug `Optional` en `rolling_features.py`: el API caía en silencio
> al fallback win-rate + Elo plano). La tabla se conserva como registro del
> proceso, pero NO refleja la integración margin-Elo. El diagnóstico completo,
> la cuantificación ON/OFF con el fix aplicado y el plan de corrección están
> en [`../docs/PLAN_MEJORA_CLAMP.md`](../docs/PLAN_MEJORA_CLAMP.md). Con el fix, 6
> temporadas ida simple dan Spearman fuerza→posición 0.87-0.89, y el clamp
> del SetPredictor demostró aportar cero señal (ρ≈0 con p_elo) y +22% de
> varianza de posición.

Se corrieron **20 temporadas completas** (12 equipos, ida y vuelta = 22
jornadas, seeds 0-19) con el Elo *supuestamente* sembrado desde el histórico
(en realidad plano, ver aviso) y el clamp del SetPredictor activado. Posición
media final (menor = mejor):

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
  puesto y Verona (0.578) baja al 7º. *(Diagnóstico posterior: la causa
  principal era el sembrado roto — ver aviso arriba — más el clamp comprimiendo
  diferencias sin señal.)*
- Este ruido es **coherente con la realidad**: el mid-table de la SuperLega es
  genuinamente volátil temporada a temporada.

El ruido del clamp quedó cuantificado y con plan de corrección en
[`../docs/PLAN_MEJORA_CLAMP.md`](../docs/PLAN_MEJORA_CLAMP.md).

Todo con **142 tests verdes** (134 + 8 nuevos para el adapter v2); el simulador
y el API arrancan sin cambios de interfaz.

</details>

#### Cierre del Grupo A (2026-07-21)

El clamp adaptativo se cierra con un **resultado negativo documentado**: tras
corregir su error de escala (A2: centrar en el p_punto implícito vía
`src/simulation/set_math.py`) y convertirlo en mezcla en vez de override (A4),
el barrido del peso `w ∈ {0.5, 0.7, 0.9, 1.0}` elige **w = 1.0**, es decir,
ignorar por completo al SetPredictor. `w = 0.9` y `w = 1.0` dan métricas
idénticas y coincidentes con la configuración OFF.

Backtest A5 final (`models/backtest_clamp_results.json`):

| Config | \|P_MC − p_elo\| | Spearman | Std pos | Std pts | T(s) |
|---|---:|---:|---:|---:|---:|
| OFF | 0.22470 | −0.9720 | 0.1667 | 0.4940 | 7.5 |
| **NEW (A2+A4)** | **0.22470** | **−0.9720** | **0.0667** | 0.4006 | 9.3 |

Los tres criterios de aceptación del grupo se cumplen por primera vez
(antes de A2 fallaban dos de tres) y el coste cae **14×** respecto al camino ON
anterior, porque con `w = 1.0` la llamada al SetPredictor se cortocircuita.

Lo que sobrevive de valor no es el SetPredictor sino el **reescalado**: centrar
el clamp en la señal Elo viva con ±0.10 en espacio de punto, en lugar del rango
fijo [0.20, 0.80], mejora la estabilidad entre seeds sin coste en fidelidad de
ranking. Detalle del mecanismo en [`simulator.md`](simulator.md) §4.3.

**Corrección metodológica encontrada al hacer A2:** el backtest A5 compartía un
único `RuntimeFeatureBuilder` entre las configs OFF/ON/NEW. Como el builder
acumula estado Elo en cada temporada simulada, las temporadas de una config
contaminaban a la siguiente. El síntoma que lo delató: el nivel-par usa seeds
fijas y debería ser determinista, pero cambiaba al variar `--n-seeds`. Todas las
cifras de esta sección son posteriores al arreglo (commit `fc8aa6b`).

### 7.2 Validación per-year del set v2 (post-integración) — el "0.71" es 2025-específico

> **⚠️ ACTUALIZACIÓN (2026-07-15) — cifras recalculadas sobre datos limpios.**
> El `set_features.csv` sobre el que se midió esta sección estaba colisionado
> (ida+vuelta fundidas; `sets_h_antes` llegaba a 5). Regenerado sin colisión
> (`src/data/set_features_builder.py`) y reentrenado el SetPredictor v2, el CV
> rolling-origin de 2 folds pasa de **0.631 ± 0.078 a 0.679 ± 0.017** (más alto
> y mucho más estable) y el test 2025 de 0.709 a 0.697 (n 853→1193). La lectura
> de fondo NO cambia (el "0.71" seguía siendo específico de 2025; la magnitud
> defendible es la CV), pero ahora la CV es mejor y estable. Números viejos
> abajo como registro. Ver `docs/PLAN_MEJORAS_CONSOLIDADO.md` §B0b.

> **Pregunta que motivó el análisis:** ¿el AUC 0.71 del set v2 es robusto o
> depende de haber tenido "suerte" con la temporada de test? Se hizo un análisis
> per-year deslizando la ventana de validación: para cada temporada T, entrenar
> en los años previos con la misma config (LogReg C=0.5, recencia half-life=2,
> 21 features de `SET_FEATURE_COLS`) y medir en T.

**Resultado per-year (train = años previos, val = T):**

| Temporada | n_sets | AUC | LogLoss | Accuracy |
|---|---:|---:|---:|---:|
| 2018 | 154 | 0.6003 | 0.6675 | 0.6169 |
| 2019 | 186 | 0.6082 | 0.6760 | 0.5968 |
| 2020 | 229 | **0.6407** | 0.6719 | 0.5939 |
| 2021 | 244 | 0.5857 | 0.6726 | 0.5779 |
| 2022 | 252 | 0.5796 | 0.6878 | 0.5754 |
| 2023 | 352 | 0.5743 | 0.6960 | 0.5653 |
| 2024 | 482 | 0.5828 | 0.6810 | 0.5913 |
| **2025** | **853** | **0.7047** | **0.6329** | **0.6600** |

*Tabla calculada sobre el dataset pre-B0b (853 sets en 2025, n total
~2300). Con el dataset limpio post-B0b (1193 sets en 2025, n total ~3900),
el per-year puede haber cambiado, pero este análisis no se rehízo — las
cifras agregadas (CV 0.679±0.017, test 0.697) reemplazan a las de esta tabla
como métricas vigentes.*

**Spearman(val_year, AUC) = −0.17, p=0.69.** La correlación monotónica con el
año es NO significativa y ligeramente negativa: la teoría de "mejora
monotónica con el tiempo" está refutada. **2018-2024 se mueven en una banda
estrecha** (0.57-0.64, con 2020 como el mejor año viejo en 0.64), y el salto a
0.70 aparece **solo en 2025**.

**El salto 2024 → 2025 de +0.122 AUC es real y no es ruido:**
gap demasiado grande para explicarse por `n=482` vs `n=853` (la diferencia de
estabilidad no mueve 0.12 de AUC). El modelo le va notablemente mejor a 2025
que a cualquier temporada anterior.

**Razones más probables del salto 2025:**

1. **2025 es la temporada más grande del dataset** (853 sets, +77% vs 2024).
   La métrica es mucho más estable, pero eso solo no explica +0.12.
2. **La recencia (half-life=2) hace que 2024 sea el training más pesado para
   predecir 2025.** El modelo "memoriza 2024 y predice 2025 ≈ 2024++", y
   funciona porque la dinámica de la SuperLega es estable entre temporadas
   consecutivas.
3. **Para val=2024, el training más pesado son 2022 y 2023** (temporadas
   chicas, 252 y 352 sets), y el modelo tiene poco material reciente para
   aprender. La predicción de 2024 es la que más sufre por este motivo, no la
   de 2025.

**Nota sobre la tabla per-year:** los valores individuales de 2018-2025 en la
tabla de arriba se calcularon sobre el `set_features.csv` original (colisionado
por B0b). Aunque el análisis per-year como metodología sigue siendo válido, las
cifras concretas cambiaron con la regeneración del dataset (n=1193 vs 853 en
2025, sets adicionales de partidos que antes estaban fundidos). La tabla
pre-B0b se conserva en
[`memoria/registro_historico_b0.md`](registro_historico_b0.md) §B.1.

**Lo que esto significa para la narrativa del modelo (con datos limpios):**

- El CV honesto de 2 folds (`train=[2022,2023]→val 2024` + `train=[2023,2024]
  →val 2025`) da **AUC 0.679 ± 0.017** (frente a 0.631 ± 0.078 pre-B0b: más
  alto y mucho más estable). La CV es la representación más honesta del
  rendimiento fuera de la muestra.
- El test 2025 pasa de AUC **0.709 a 0.697** (n 853→1193). La lectura de fondo
  no cambia (el "0.71" seguía siendo específico de 2025; la magnitud defendible
  es la CV), pero ahora la CV es mejor y estable.
- El legacy ExtraTrees (champion anterior) tenía CV AUC 0.62 ± 0.03 sobre 4
  folds (2018-2024). La diferencia **CV v2 (0.68) vs CV legacy (0.62) es
  +0.06** y ya no cae dentro del ruido: con datos limpios el v2 sí es una
  mejora estructural clara.

**Follow-up obligatorio (warning W1 del sdd-verify):** **re-validar con la
temporada 2026/27 cuando esté disponible.** Si el AUC se mantiene por encima
de ~0.65 en una temporada no vista en training, la mejora es estructural (el
modelo aprendió algo real). Si vuelve a ~0.60, el 0.71 de 2025 fue una
coincidencia favorable entre sample-size y ventana de recencia.

Para esta re-validación basta con:

```bash
# Tras descargar la temporada 2026/27 en DB/sets_partidos.csv
python -m src.models.train_improved   # re-genera set_predictor_v2.joblib
                                     # con train incluyendo 2025
python -c "from src.models.measure_precision import measure; print(measure())"
```

Y comparar el AUC en 2026 contra los números de 2018-2025 de la tabla de
arriba. **No tocar el protocolo, no cambiar el modelo: solo agregar datos y
medir igual.**

## 8. Qué NO se hizo (honestidad de alcance)

- No se amplió el nº de partidos históricos: `sets_partidos.csv` tiene ~1322
  partidos reales tras la corrección B0 (frente a los 724 pre-B0, que eran
  artefacto de la colisión `partido_id`). Las temporadas viejas tienen
  34-59/temporada porque solo se scrapeó a los equipos rastreados. Es un techo
  de datos, no de método.
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
