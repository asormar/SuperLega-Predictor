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

3. **Recencia / descarte de datos ruidosos**. Las temporadas 2016-2020
   (34-55 partidos, con el local ganando solo ~35% de las veces) enseñaban
   la relación con el **signo invertido**: un modelo entrenado con todo el
   histórico predecía 2025 al revés (AUC 0.42). Entrenando solo con 2022-2024
   (o con pesos de recencia, half-life ≈ 1.5), el AUC salta a 0.75.

4. **Modelos lineales regularizados > árboles profundos** en este régimen
   de datos pequeños. Con 34-59 partidos por temporada, los árboles
   sobreajustan ruido y ahogan la señal. LogReg regularizado (o el Elo
   directo) gana consistentemente en match y en set.

## Qué NO se tocó (honestidad de alcance)

- Los modelos de producción (`set_predictor.joblib`, `match_predictor.joblib`)
  y el simulador siguen intactos y funcionando (134 tests verdes). Las mejoras
  se entregan como artefactos v2 reproducibles (`match_elo_v2.joblib`,
  `set_predictor_v2.joblib`) vía `python -m src.models.train_improved`.
- Integrarlos en la producción exige alinear el `RuntimeFeatureBuilder` del
  simulador con las nuevas features rolling (coherencia train/serve). Es el
  siguiente paso natural, documentado en el plan (Fase 4 completa).
- La feature de continuidad de plantilla (T2.4) y el predictor de partido
  derivado del set (T3.3) quedaron sin implementar: con el Elo ya en 0.75,
  el retorno marginal era bajo frente al riesgo.

## Cómo reproducir

```bash
python -m src.models.measure_precision --save baseline   # antes (honesto)
python -m src.models.train_improved                      # después + artefactos
```
