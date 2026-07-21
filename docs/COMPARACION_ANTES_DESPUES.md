# Comparación Antes / Después — Ejecución del Plan de Mejora

Todas las cifras se miden con el **mismo protocolo honesto**: validación
rolling-origin y test held-out sobre la temporada **2025/26 (n=214 partidos,
853 sets)**, que nunca se usó para entrenar ni seleccionar.

## Resultado a nivel de partido (MATCH)

| Métrica | Legacy reportado* | Antes (honesto) | Después | Δ real |
|---|---|---|---|---|
| AUC | 0.707 | **0.528** | **0.750** | **+0.222** |
| LogLoss | — | 0.694 | **0.585** | −0.109 |
| Brier | 0.245 | 0.251 | **0.200** | −0.051 |
| Accuracy | 0.514 | 0.514 | **0.692** | **+0.178** |

\* El AUC=0.707 que reportaba el código era **falso**: producto de leakage
temporal (features de temporada completa mezcladas en partidos de esa misma
temporada) evaluado sobre un único año de test afortunado. Medido bien, el
modelo real tenía AUC 0.53 — apenas por encima del azar.

## Resultado a nivel de set (SET)

| Métrica | Antes (honesto) | Después | Δ |
|---|---|---|---|
| AUC | 0.653 | **0.709** | +0.056 |
| LogLoss | 0.651 | **0.625** | −0.026 |
| Brier | 0.230 | **0.218** | −0.012 |
| Accuracy | 0.606 | **0.658** | +0.052 |

## Qué cambió (y por qué funcionó)

1. **Protocolo de evaluación honesto** (`src/models/evaluation.py`). La
   selección de modelo se hacía sobre 81 partidos (error estándar ±0.06 en
   AUC): ruido puro. Se reemplazó por rolling-origin multi-temporada con
   log-loss como métrica primaria. No mejora los números, pero hace que
   signifiquen algo — y destapó que el 0.707 era ficticio.

2. **Features rolling sin leakage** (`src/data/rolling_features.py`). Se
   reconstruyeron desde `sets_partidos.csv` recorriendo los partidos en orden
   cronológico y usando solo información previa. Incluye **Elo con margen de
   victoria**, forma EWMA y H2H con decaimiento. Un Elo limpio (una sola
   variable, sin entrenar) ya daba AUC 0.62 — más que las 87 features viejas.

3. **Recencia operativa (train 2022-2024)**. La ventana de entrenamiento se
   limita a las últimas 3 temporadas.
   - **Justificación vigente**: ciclo de plantillas de la SuperLega (half-life
     2 temporadas ≈ renovación típica de un roster).
   - **Justificación histórica (invalidada por B0, 2026-07-15)**: se creía que
     las temporadas 2016-2020 estaban "envenenadas" y enseñaban el signo
     invertido (home-win ~0.32, AUC 0.42 con todo el histórico). Con el fix B0
     (colisión `partido_id` corregida), el home-win es **0.48-0.61 en todas las
     temporadas** y esa narrativa pierde sustento. La recencia se mantiene por
     la primera razón, no por la segunda.
   - Texto original invalidado: ver
     [`memoria/registro_historico_b0.md`](../memoria/registro_historico_b0.md) §A.3.

4. **Modelos lineales regularizados > árboles profundos** en este régimen
   de datos pequeños. Con 34-59 partidos por temporada, los árboles
   sobreajustan ruido y ahogan la señal. LogReg regularizado (o el Elo
   directo) gana consistentemente en match y en set.

## Integración en producción

Las mejoras se integraron en el simulador (no se quedaron como experimento):

- **Fuerzas de equipo** desde el Elo con margen (`api/main.py`), con la
  jerarquía real de la SuperLega como prior.
- **Elo runtime sembrado** desde el histórico y **update con margen** en el
  `RuntimeFeatureBuilder`.
- **Señal de partido = probabilidad de Elo limpia** en el `SeasonSimulator`;
  el `match_predictor.joblib` de 87 features queda solo como fallback.

### Validación end-to-end: Monte Carlo de 20 temporadas

> **⚠️ Tabla invalidada:** esta corrida se ejecutó con el sembrado de Elo roto
> (bug `Optional`, arreglado después). Diagnóstico y cifras corregidas en
> [`PLAN_MEJORA_CLAMP.md`](PLAN_MEJORA_CLAMP.md) — con el fix: Spearman
> fuerza→posición 0.87-0.89 (6 temporadas), y el clamp del SetPredictor
> demostró aportar cero señal y +22% de varianza.

20 temporadas simuladas (12 equipos, ida y vuelta, seeds 0-19). Posición media
final vs fuerza del equipo:

| # | Equipo | Pos. media | Fuerza | | # | Equipo | Pos. media | Fuerza |
|---|---|---:|---:|---|---|---|---:|---:|
| 1 | Perugia | 3.3 | 0.681 | | 7 | Verona | 7.2 | 0.578 |
| 2 | Trento | 4.2 | 0.604 | | 8 | Modena | 7.3 | 0.522 |
| 3 | Lube | 5.5 | 0.538 | | 9 | Cisterna | 7.4 | 0.350 |
| 4 | Taranto | 5.9 | 0.350 | | 10 | Padova | 7.4 | 0.372 |
| 5 | Piacenza | 6.2 | 0.568 | | 11 | Grottazzolina | 8.1 | 0.176 |
| 6 | Monza | 6.8 | 0.462 | | 12 | Milano | 8.5 | 0.457 |

Top-3 y fondo exactos; correlación fuerza→posición claramente positiva. La zona
media es ruidosa (Taranto sobrerrinde, Verona baja) por varianza de Monte Carlo
y por el clamp adaptativo del SetPredictor. Detalle en
[`../memoria/mejora_precision_2026-07.md`](../memoria/mejora_precision_2026-07.md) §7.1.

## Qué NO se tocó (honestidad de alcance)

- El `match_predictor.joblib` de 87 features sigue en disco como fallback; no se
  borró para no romper la carga del API.
- La feature de continuidad de plantilla (T2.4) y el predictor de partido
  derivado del set (T3.3) quedaron sin implementar: con el Elo ya en 0.75,
  el retorno marginal era bajo frente al riesgo.
- No se hizo el backtest completo del simulador con ajuste de momentum/clamps
  (Fase 4 completa); se validó a nivel de probabilidad (Brier 0.251 → 0.200) y
  con el Monte Carlo de clasificación de arriba.

## Cómo reproducir

```bash
python -m src.models.measure_precision --save baseline   # antes (honesto)
python -m src.models.train_improved                      # después + artefactos v2
python -m pytest -q                                      # 142 tests verdes
```
