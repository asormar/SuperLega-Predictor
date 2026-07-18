# Exploración SDD: Estructura UPV para la memoria del TFG PREDICTOR(2)

**Fecha**: 2026-07-14
**Agente**: sdd-explore
**Próximo step recomendado**: `sdd-propose` → `sdd-design` (esqueleto LaTeX concreto)

---

## 1. Contexto de la exploración

El usuario me pidió analizar la estructura canónica de un TFG de la UPV (Universitat Politècnica de València) a partir de:

- La **plantilla oficial** `latex/TFG-M-Plantilla_sinPortada-3.pdf` (ETSI Telecomunicación).
- **4 PDFs de ejemplo** en `latex/ejemplos/` (3 son TFGs reales de la UPV; 1, `TFG_Alvaro.pdf`, es el propio TFG anterior del usuario sobre sarcoma de Ewing).
- El **esqueleto actual** `latex/main.tex` (504 líneas, 10 capítulos vacíos con TODOs).
- El **mapa de subsistemas** en `memoria/INDICE.md` (vincula 11 archivos `.md` a los subsistemas del proyecto).
- Las **guías de la skill** `latex-document-skill` (clon local) — sección "Workflow de migración desde .md".

El objetivo es destilar la estructura UPV y proponer un plan de migración para los 11 `.md` de `memoria/` al `main.tex` actual, que ahora mismo no compila contenido (todo son `% TODO:`).

## 2. Estructura UPV destilada

A partir de la plantilla oficial y los 4 ejemplos (con foco en TFG_Alvaro por ser del mismo autor), la estructura canónica es:

### 2.1 Front matter (numeración romana, sin numerar portada/resúmenes)

| Sección | Notas |
|---|---|
| **Portada** | `\titlepage` con universidad, escuela, grado, título, autor, director, fecha |
| **Resumen (ES)** | 150-200 palabras + keywords |
| **Resum (CA)** | Mismo contenido, en valenciano |
| **Abstract (EN)** | Mismo, en inglés |
| **Agradecimientos** | 1 página, formal |
| **Resumen ejecutivo (ABET)** | **REQUISITO UPV** para acreditar competencias. Tabla IDENTIFY/FORMULATE/SOLVE con columna "¿Cumple? S/N" y "¿Dónde? (páginas)". Plantilla oficial lo exige para ETSI Telecomunicación. |
| **Lista de acrónimos** | Tabla de siglas |
| **Índice general** (TOC) | |
| **Índice de figuras** | `\listoffigures` |
| **Índice de tablas** | `\listoftables` |

> **NOTA IMPORTANTE**: TFG_Alvaro y Baggetto (UPV inglés) numeran el front matter en **romano** (i, ii, iii…) y arrancan arábigo desde el cap 1. El main.tex actual tiene TODO en arábigo desde la portada — esto es **incorrecto** según la plantilla y debe corregirse.

### 2.2 Cuerpo (numeración arábiga, reinicia en 1)

**6 capítulos** (patrón canónico):

| Cap | Nombre típico | Qué incluye |
|---|---|---|
| 1 | **Introducción** | Motivación, objetivos, estructura del documento |
| 2 | **Marco teórico / Contexto** | Conceptos previos, estado del arte |
| 3 | **Materiales** (o "Materiales y métodos") | Datos, herramientas, entorno |
| 4 | **Métodos** (o integrado en cap 3) | Algoritmos, modelos, arquitectura |
| 5 | **Resultados y discusión** | Experimentación, métricas, comparación |
| 6 | **Conclusiones y trabajo futuro** (a veces "Conclusiones" + "Líneas futuras" como caps separados) | Resumen, contribuciones, limitaciones, futuro |

> **Variantes observadas**:
> - TFG_Alvaro (mismo usuario): **6 caps** (1.Introducción, 2.Contexto, 3.Materiales, 4.Métodos, 5.Resultados y discusión, 6.Conclusiones).
> - Baggetto (UPV inglés): **5 caps** (Materiales+Métodos van juntos en cap 3).
> - Prunonosa: **5 caps** (mismo patrón que Baggetto).
> - Soriano: **7 caps** (rompe la "discusión y resultados", pone resultados+conclusiones juntos y separa líneas futuras).
>
> **Recomendación**: **6 caps** según TFG_Alvaro — es el patrón del propio usuario, mantiene coherencia con su TFG previo.

### 2.3 Back matter

- **Bibliografía** (siempre con `biblatex` + estilo `ieee` según TFG_Alvaro)
- **Apéndices** (típicamente Apéndice A sobre relación con ODS — REQUISITO UPV)

## 3. Comparación con el esqueleto actual (10 cap)

`main.tex` actual:

| # | Cap actual | Destino propuesto |
|---|---|---|
| 1 | Introducción | **= Cap 1** (se mantiene, contenido válido) |
| 2 | Marco teórico | **= Cap 2** (se mantiene) |
| 3 | Arquitectura general | → Absorber en **Cap 3 (Materiales)** como subsección |
| 4 | Predicción de partidos individuales | → Absorber en **Cap 4 (Métodos)** |
| 5 | Predicción de temporadas | → Absorber en **Cap 4 (Métodos)** |
| 6 | Modelos de machine learning | → Absorber en **Cap 4 (Métodos)** |
| 7 | Capa de datos | → **= Cap 3 (Materiales)** |
| 8 | Motor de simulación | → Absorber en **Cap 4 (Métodos)** |
| 9 | Experimentación y resultados | **= Cap 5** (renombrar) |
| 10 | Conclusiones y trabajo futuro | **= Cap 6** (se mantiene) |

**Razón del refactor**: los caps 4, 5, 6 y 8 son todos "métodos" del proyecto (los modelos, las simulaciones, las predicciones). Tener 4 caps separados fragmenta artificialmente el mismo dominio técnico. UPV recomienda caps más sustanciosos y bien delimitados.

## 4. Recomendación: refactorizar el esqueleto

**Recomendación fuerte**: refactorizar de **10 → 6 capítulos** siguiendo el patrón UPV canónico. Razones:

1. **Consistencia con TFG_Alvaro** (el propio TFG previo del usuario usa este patrón).
2. **Coherencia con la plantilla oficial** y con los 4 PDFs revisados.
3. **Mejor narrativa**: caps más sustanciosos, mejor delimitados, menos fragmentados.
4. **Reducción de overhead**: menos `\chapter{}` con contenido ralo, más `\section{}` bien anidadas.

**Trade-off**: hay que reescribir el esqueleto del `main.tex` (504 líneas) y reordenar la migración de los 11 `.md`. Pero es una **inversión de una sola vez** que paga el resto del proyecto.

**Riesgo si NO se refactoriza**: el TFG se defendería contra una estructura que rompe las recomendaciones UPV. El tribunal lo notaría.

## 5. Mapeo de los 11 `.md` a la nueva estructura

| Archivo `.md` | Subsistema | Cap destino | Subsección propuesta |
|---|---|---|---|
| `INDICE.md` | Resumen ejecutivo | Front matter | Resumen + Agradecimientos (síntesis) |
| `prediccion_partidos.md` | Match simulator | **Cap 4 (Métodos)** | 4.2 Simulación de partido individual |
| `prediccion_temporadas.md` | Season simulator | **Cap 4 (Métodos)** | 4.3 Simulación de temporada |
| `simulator.md` | MatchSimulator Markov | **Cap 4 (Métodos)** | 4.1 Motor de Markov (Markov chain + momentum) |
| `point_probability.md` | PointProbabilityModel | **Cap 4 (Métodos)** | 4.4 Modelo de probabilidad de punto |
| `set_predictor.md` | SetPredictor v2 | **Cap 4 (Métodos)** | 4.5 Predictor de set (LogReg + recencia) |
| `match_predictor.md` | MatchPredictor / Elo | **Cap 4 (Métodos)** | 4.6 Predictor de partido (Elo con margen) |
| `data_layer.md` | Capa de datos | **Cap 3 (Materiales)** | 3.x Pipeline de datos + feature store + split temporal |
| `player_stats_generator.md` | Stats sintéticas | **Cap 3 (Materiales)** | 3.x Generación de estadísticas sintéticas |
| `mejora_precision_2026-07.md` | Auditoría 2026-07 | **Cap 5 (Resultados)** | 5.1 Auditoría de precisión (Elo vs MatchPredictor) |
| `benchmark.md` | Comparativa de modelos | **Cap 5 (Resultados)** | 5.2 Benchmark de modelos |
| (no tiene `.md` directo) | Marco teórico | **Cap 2 (Marco teórico)** | 2.1 Cadenas de Markov, 2.2 ELO, 2.3 ML supervisado, 2.4 Validación temporal sin leakage |

## 6. Plan de migración por fases (orden sugerido)

| Fase | Acción | Razonamiento |
|---|---|---|
| **0** | Refactorizar `main.tex`: nuevo preámbulo (sin tocar contenido), 6 capítulos vacíos, 3 apéndices. Compilar iterativamente cada cambio. | Sentar el esqueleto antes de poblarlo |
| **1** | Front matter: portada, resúmenes (ES/CA/EN), agradecimientos, ejecutivo ABET, acrónimos, índices. | Lo más rápido de compilar; valida el preámbulo |
| **2** | **Cap 1 — Introducción**. Mapear motivación (INDICE), objetivos, estructura. | Bajo riesgo, ya está bien esbozado en main.tex |
| **3** | **Cap 2 — Marco teórico**. NO tiene `.md` directo → escribir desde cero usando `set_predictor.md`, `match_predictor.md`, `mejora_precision_2026-07.md` y `AGENTS.md` como fuentes. | Cap autocontenido, baja dependencia de otros |
| **4** | **Cap 3 — Materiales**. Migrar `data_layer.md` + `player_stats_generator.md`. | Materiales va antes que métodos (orden lógico) |
| **5** | **Cap 4 — Métodos y motor** (el más largo). Migrar `simulator.md`, `prediccion_partidos.md`, `prediccion_temporadas.md`, `point_probability.md`, `set_predictor.md`, `match_predictor.md`. | Sección más larga, requiere más iteraciones |
| **6** | **Cap 5 — Resultados y discusión**. Migrar `mejora_precision_2026-07.md` + `benchmark.md`. | Discutir resultados contra los métodos del cap 4 |
| **7** | **Cap 6 — Conclusiones y trabajo futuro**. Síntesis. | Bajo riesgo, formato libre |
| **8** | Apéndices: hiperparámetros, métricas detalladas, API. | Material complementario |
| **9** | Bibliografía. Unificar `references.bib` (ya existe con entries de muestra). | Las refs vienen de los `.md` que ya citan |
| **10** | Compilación final: revisar figuras, tablas, citas, cross-refs, formato de código. | QA antes de imprimir PDF |

**Dependencias críticas entre fases**:
- Fase 5 (Métodos) referencia a Fase 4 (Materiales): las features se construyen en cap 3, los modelos las usan en cap 4.
- Fase 6 (Resultados) referencia a Fase 5 (Métodos): las métricas se interpretan a la luz de los modelos definidos antes.
- Fase 9 (Bibliografía) depende de TODAS las anteriores.

## 7. Convenciones UPV observadas (síntesis)

| Convención | Cómo se aplica en LaTeX | Estado actual en main.tex |
|---|---|---|
| **Tipo de letra** | Times New Roman o similar, ≥11pt cuerpo, ≥9pt pies | `newpxtext` (Palatino, aceptable) ✓ |
| **Márgenes** | A4, ~1 inch | ✓ (`geometry`) |
| **Párrafos justificados** | `\justifying` por defecto | ✓ |
| **Separación entre párrafos** | ≥6pt | **Falta** — añadir `\setlength{\parskip}{6pt}` |
| **Numeración front matter** | Romana (i, ii, iii) | **Falta** — debe ir antes de cap 1 |
| **Numeración cuerpo** | Arábiga, reinicia en 1 | ✓ (debería ser automático con `book` class) |
| **Citas bibliográficas** | Numéricas [1] estilo IEEE | ✓ (`biblatex` + `style=ieee`) |
| **Figuras** | Caption al pie, numeradas independientemente, citadas en orden | ✓ |
| **Tablas** | Caption + booktabs | ✓ |
| **Ecuaciones** | Numeradas en línea independiente, `\eqref{}` | ✓ |
| **Listings de código** | Numerados, con caption | ✓ (estilo `pythonstyle`) |
| **Multi-archivo** | `\input{}` o `\include{}` por cap (recomendado) | **No usado** — todo en un solo .tex de 504 líneas |
| **Front matter estándar book** | `\frontmatter`, `\mainmatter`, `\backmatter` | `\frontmatter` y `\backmatter` presentes, bien |
| **Bibliografía** | `\printbibliography` al final | ✓ |
| **Apéndices** | `\appendix` + `\chapter{}` | ✓ |
| **Resumen ejecutivo ABET** | Tabla con competencias UPV/ETSI | **Falta** en main.tex |
| **Lista de acrónimos** | Tabla propia o paquete `glossaries` | **Falta** en main.tex |
| **3 abstracts obligatorios** | ES + CA + EN | ✓ (definidos, vacíos) |

## 8. Riesgos identificados

| # | Riesgo | Severidad | Mitigación |
|---|---|---|---|
| R1 | El `main.tex` tiene 504 líneas con TODO monolítico — migración capítulo a capítulo significa reescribirlo a multi-archivo (`cap1.tex`, `cap2.tex`, etc.) | Media | Refactorizar a multi-archivo con `\input{}` en Fase 0; main.tex queda como orquestador |
| R2 | Numeración romana en front matter NO está en main.tex actual | Baja | Añadir `\pagenumbering{roman}` antes de Resumen y `\pagenumbering{arabic}` después |
| R3 | `\addcontentsline{toc}{chapter}{...}` para el Resumen puede duplicar la entrada en el TOC | Baja | Verificar después de compilar Fase 1 |
| R4 | `mejora_precision_2026-07.md` documenta una auditoría del 2026-07 — métricas pueden quedar desfasadas si se reentrena | Baja | Documentar versión del modelo + fecha en cada referencia numérica |
| R5 | Hay contradicción potencial entre `INDICE.md` ("AUC 0.71 test 2025" para set predictor) y los valores viejos que pudiera tener `set_predictor.md`. | Media | Auditar métricas al inicio de Fase 5; congelar valores en `mejora_precision_2026-07.md` |
| R6 | El `main.tex` actual define `\usepackage[autostyle=true]{csquotes}` pero los idiomas se setean en babel — hay que verificar que las comillas tipográficas funcionen en ES/CA/EN | Baja | Probar con `\enquote{}` en cada idioma |
| R7 | El `references.bib` actual tiene 6 entries de muestra (Vaswani, Goodfellow, etc.) que NO son del proyecto. Hay que reemplazarlo con las refs reales de los `.md` | Baja | Fase 9: recolectar refs de los `.md` y regenerar el `.bib` |
| R8 | Conflictos potenciales entre el preámbulo actual (biblatex con backend biber) y el estilo de citas [1] numérico que pide UPV | Baja | Verificar que `style=ieee` produce el formato correcto en Fase 9 |
| R9 | El `main.tex` actual tiene 3 apéndices planeados (hiperparámetros, métricas, API) pero no se ha confirmado que haya suficiente material para 3 — quizás se fusionen en 1 o 2 | Baja | Decidir en Fase 8 según contenido real |

## 10. Recomendación de next-step

→ **`sdd-propose`**: con este análisis como input, redactar la propuesta de cambio concreta para el `main.tex` (refactor de 10 → 6 caps + multi-archivo). La propuesta debe ser específica sobre qué archivos `.md` van a qué cap/subsección y en qué orden.

→ Luego, **`sdd-design`**: diseñar la arquitectura LaTeX concreta (multi-archivo, preámbulo actualizado, comandos auxiliares) y los `tex/cap{N}_*.tex` con las migraciones planeadas.

→ **`sdd-apply`** se encargará de la migración efectiva cap por cap.
