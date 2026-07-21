# Registro histórico del proceso — B0, B0b, B1 (2026-07-15)

## Propósito

Este apéndice preserva la narrativa y los datos previos al fix B0 (colisión
`partido_id`) y B0b (regeneración de `set_features.csv`), retirados del cuerpo
principal de la documentación porque eran artefacto del bug. Se conserva como
trazabilidad del proceso de mejora del TFG para que un evaluador pueda ver el
proceso completo de investigación, incluyendo resultados negativos y
correcciones.

**Nota editorial del autor**: el usuario aprobó mantener este registro en lugar
de borrar la narrativa vieja, para que un evaluador pueda ver el proceso
completo de investigación. Las secciones siguientes contienen datos que fueron
invalidados por los fixes B0/B0b; no deben citarse como vigentes sin verificar
contra `models/precision_improved.json` o `docs/PLAN_MEJORAS_CONSOLIDADO.md`.

---

## A. Datos invalidados por B0 (colisión `partido_id`)

> **⚠️ DATOS INVALIDADOS por B0 (2026-07-15).** La tabla y narrativa siguientes
> eran artefacto de la colisión de `partido_id` en `rolling_features._aggregate_matches`,
> que sumaba los sets de ida y vuelta de un mismo cruce en un solo grupo,
> invirtiendo el target `gana_local` en el 82% de los partidos. Con datos
> limpios (agrupar por `(partido_id, local)` → 1322 partidos reales), el home-win
> es **0.48-0.61 en todas las temporadas** y la narrativa de "temporadas viejas
> envenenadas" se disuelve.

### A.1. Tabla Elo por temporada (invalidada)

| Temporada | n | AUC Elo | % victoria local |
|---|---|---|---:|
| 2016 | 34 | 0.28 | 0.32 |
| 2019 | 45 | 0.55 | 0.36 |
| 2022 | 59 | 0.55 | 0.51 |
| 2024 | 111 | 0.55 | 0.55 |
| **2025** | **214** | **0.75** | **0.60** |

Lectura original (invalidada): las temporadas viejas (34-55 partidos, con el
local ganando solo ~32-35%) eran ruido con el **signo invertido**. Un LogReg
entrenado con TODO el histórico aprendía la relación al revés y predecía 2025
**anti-correlado** (AUC 0.42).

**Lectura vigente (post-B0)**: home-win 0.48-0.61 en todas las temporadas. No
hay envenenamiento sistémico. La recencia 2022-2024 se justifica por ciclo de
plantillas (half-life 2 temporadas), no por sign-flip.

### A.2. Tabla de recencia (invalidada)

| Entrenamiento | AUC test 2025 | Accuracy |
|---|---|---:|---:|
| Todo 2016-2024 sin pesos | 0.42 | 0.39 |
| Recencia half-life=1 | 0.76 | 0.69 |
| Solo 2022-2024 | 0.77 | 0.72 |
| **Elo puro** (sin entrenar) | **0.75** | **0.69** |

Estos datos mostraban que entrenar con todo el histórico empeoraba el modelo
(AUC 0.42). Con datos limpios post-B0, el AUC de "todo el histórico" sube a
un rango coherente. La justificación de la recencia se apoyaba en parte en este
artefacto y hoy se sostiene por el ciclo de plantillas.

### A.3. Sección "Recencia / descarte de datos ruidosos" (de COMPARACION_ANTES_DESPUES.md §3)

Texto original:

> **Recencia / descarte de datos ruidosos**. Las temporadas 2016-2020
> (34-55 partidos, con el local ganando solo ~35% de las veces) enseñaban
> la relación con el **signo invertido**: un modelo entrenado con todo el
> histórico predecía 2025 al revés (AUC 0.42). Entrenando solo con 2022-2024
> (o con pesos de recencia, half-life ≈ 1.5), el AUC salta a 0.75.

Esta lectura quedó invalidada por B0. El texto vigente está en
`docs/COMPARACION_ANTES_DESPUES.md` §3 (actualizado 2026-07-15).

---

## B. Datos invalidados por B0b (regeneración de `set_features.csv`)

> **⚠️ DATOS INVALIDADOS por B0b (2026-07-15).** El `set_features.csv` estaba
> colisionado (ida+vuelta fundidas, `sets_h_antes` llegaba a 5). La sección
> §7.2 de `mejora_precision_2026-07.md` se midió sobre ese CSV. Tras regenerar
> con `src/data/set_features_builder.py`, las cifras cambiaron.

### B.1. Tabla per-year del set v2 (pre-B0b)

| Temporada | n_sets | AUC | LogLoss | Accuracy |
|---|---|---|---:|---:|---:|
| 2018 | 154 | 0.6003 | 0.6675 | 0.6169 |
| 2019 | 186 | 0.6082 | 0.6760 | 0.5968 |
| 2020 | 229 | **0.6407** | 0.6719 | 0.5939 |
| 2021 | 244 | 0.5857 | 0.6726 | 0.5779 |
| 2022 | 252 | 0.5796 | 0.6878 | 0.5754 |
| 2023 | 352 | 0.5743 | 0.6960 | 0.5653 |
| 2024 | 482 | 0.5828 | 0.6810 | 0.5913 |
| **2025** | **853** | **0.7047** | **0.6329** | **0.6600** |

Cifras vigentes (post-B0b): test 2025 AUC 0.697, n=1193; CV 2-fold 0.679±0.017
(ver `models/precision_improved.json`).

### B.2. Métricas de set_predictor.md §10.4 (pre-B0b)

| Métrica | Valor (pre-B0b) |
|---|---:|
| AUC test 2025 (853 sets) | 0.709 |
| CV rolling-origin 2 folds | 0.631 ± 0.078 |
| Accuracy test 2025 | 0.658 |
| Brier Score test 2025 | 0.218 |

Cifras vigentes: AUC 0.697, CV 0.679±0.017, n=1193.

### B.3. Tabla de prediccion_temporadas.md §7 (pre-B0b)

| Métrica | Valor (pre-B0b) |
|---|---:|
| AUC test 2025 (853 sets) | 0.709 |
| CV rolling-origin 2 folds | 0.631 ± 0.078 |
| Accuracy test 2025 | 0.658 |
| Brier Score test 2025 | 0.218 |

---

## C. Datos invalidados por el bug Optional (sembrado Elo roto)

### C.1. Monte Carlo de 20 temporadas con sembrado roto

Se corrieron **20 temporadas completas** (12 equipos, ida y vuelta = 22
jornadas, seeds 0-19) con el Elo *supuestamente* sembrado desde el histórico
(en realidad plano, por bug `Optional` en `rolling_features.py`: el API caía
en silencio al fallback win-rate + Elo plano) y el clamp del SetPredictor
activado. Posición media final (menor = mejor):

| # | Equipo | Posición media | Fuerza (margin-Elo) |
|---|---|---|---:|---:|
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

Con el fix aplicado, 6 temporadas ida simple dan Spearman fuerza→posición
0.87-0.89 (detalle en `docs/PLAN_MEJORAS_CONSOLIDADO.md` §A6).

---

## D. Referencias cruzadas

| Documento | Sección invalidada | Estado actual |
|---|---|---|
| `memoria/mejora_precision_2026-07.md` | §5.1 (tabla Elo por temporada + recencia) | Reescrita con datos limpios. Banner de corrección añadido. |
| `memoria/mejora_precision_2026-07.md` | §7.1 (MC 20 temporadas) | Banner de invalidación. Tabla conservada aquí. |
| `memoria/set_predictor.md` | §10.4 (métricas pre-B0b) | Actualizadas a cifras vigentes. |
| `memoria/prediccion_temporadas.md` | §6-7 (métricas pre-B0b) | Actualizadas a cifras vigentes. |
| `docs/COMPARACION_ANTES_DESPUES.md` | §3 (recencia) | Reescribir con narrativa dual. |
| `lat​ex/front_matter.tex` | Resúmenes (AUC 0.71) | Actualizado a 0.697. |

---

*Fin del registro histórico. Los datos aquí contenidos no deben usarse como
cifras vigentes del proyecto sin verificar contra `models/precision_improved.json`
o la documentación actualizada.*
