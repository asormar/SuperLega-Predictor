# Memoria del TFG — Predictor(2)

> **Documento unificado de la memoria.** Este `.md` aúna todo el contenido
> que alimentará la versión final en LaTeX (que gestionas tú manualmente).
> Estructura: front matter + 6 capítulos + 4 apéndices, siguiendo el patrón
> UPV recomendado en `latex/exploration_upv_structure.md` (6 caps, 4
> apéndices).
>
> **Capítulos con contenido propio** (Cap 1, Cap 2): redactados íntegros
> aquí, listos para transcribir a LaTeX.
>
> **Capítulos que viven en otros `.md`** (Cap 3, 4, 5, 6): cada sección
> apunta al archivo de `memoria/` que ya tiene el contenido completo y
> maduro. Cuando los redactes a LaTeX, el contenido está en esos archivos.
>
> El orden de redacción sugerido es el del índice: front matter → Cap 1 →
> Cap 2 → Cap 3 → Cap 4 → Cap 5 → Cap 6 → apéndices → bibliografía.

---

## Front matter (portada, resúmenes, acrónimos)

### Portada institucional

> **Universitat Politècnica de València**
> ETSIT — Escola Tècnica Superior d'Enginyeria de Telecomunicació
> GTDM — Grau en Tecnologia Digital i Multimèdia
>
> **Desarrollo de un modelo predictivo basado en Machine Learning para la
> estimación de resultados y análisis de rendimiento en el voleibol
> profesional**
>
> Trabajo de Fin de Grado presentado por **Alejandro Sorolla Martínez**
> Director: \_\_\_\_ (TODO)
>
> Valencia, 2026
> Curso 2025–2026

### Declaración de autoría y originalidad

Yo, Alejandro Sorolla Martínez, declaro que el Trabajo de Fin de Grado
aquí presentado es de mi autoría original y que no ha sido presentado
previamente para obtener ningún título o calificación. Todo el material
tomado de otras fuentes — bibliografía, referencias o comentarios —
está debidamente reconocido.

### Resumen (ES)

**Desarrollo de un modelo predictivo basado en Machine Learning para la
estimación de resultados y análisis de rendimiento en el voleibol
profesional.**

Este Trabajo de Fin de Grado presenta **Predictor(2)**, un sistema
completo para simular partidos y temporadas de la SuperLega italiana de
voleibol. La arquitectura combina un motor de *cadenas de Markov*
punto a punto con tres modelos de *ML* (regresión logística, gradient
boosting y ensembles) entrenados sobre datos públicos de los últimos
diez años. El sistema se complementa con una interfaz web en React y
una API REST en FastAPI. La evaluación con protocolo *rolling-origin*
honesto (sin leakage temporal) arroja un AUC de 0.762 para la
predicción de partido y 0.697 para la predicción de set (test 2025;
media CV 2-fold 0.679), con tiempos de simulación de temporada
completa del orden de 70 s.

**Palabras clave**: Machine Learning, Analítica deportiva, Voleibol,
Modelos predictivos, Análisis de datos, Rendimiento deportivo.

### Resum (CA)

**Desenvolupament d'un model predictiu basat en Machine Learning per a
l'estimació de resultats i anàlisi de rendiment en el voleibol
professional.**

Aquest Treball de Fi de Grau presenta **Predictor(2)**, un sistema
complet per a simular partits i temporades de la SuperLega italiana de
voleibol. L'arquitectura combina un motor de *cadenes de Markov* punt
a punt amb tres models de *aprenentatge automàtic* (regressió
logística, gradient boosting i ensembles) entrenats sobre dades
públiques dels últims deu anys. El sistema es complementa amb una
interfície web en React i una API REST en FastAPI. L'avaluació amb
protocol *rolling-origin* honest (sense leakage temporal) obté un AUC
de 0.762 per a la predicció de partit i 0.697 per a la predicció de
set (test 2025; mitjana CV 2-fold 0.679), amb temps de simulació de
temporada completa de l'ordre de 70 s.

**Paraules clau**: Aprenentatge automàtic, Analítica esportiva,
Voleibol, Models predictius, Anàlisi de dades, Rendiment esportiu.

### Abstract (EN)

**Development of a Machine Learning-based predictive model for outcome
estimation and performance analysis in professional volleyball.**

This Bachelor's Thesis presents **Predictor(2)**, a complete system
for simulating matches and seasons of the Italian SuperLega volleyball
league. The architecture combines a point-by-point *Markov chain*
engine with three *machine learning* models (logistic regression,
gradient boosting, and ensembles) trained on publicly available data
from the last ten years. The system is complemented with a React web
interface and a FastAPI REST API. Evaluation with an honest
*rolling-origin* protocol (no temporal leakage) yields an AUC of
0.762 for match prediction and 0.697 for set prediction (test 2025;
mean CV 2-fold 0.679), with full-season simulation times in the
order of 70 s.

**Keywords**: Machine Learning, Sports analytics, Volleyball,
Predictive models, Data analysis, Sports performance.

### Resumen ejecutivo ABET (obligatorio ETSIT)

| Competencia | Descripción | Cumple | Dónde |
|---|---|---|---|
| IDENTIFY | Identificar, formular y resolver problemas de ingeniería | TODO | TODO |
| FORMULATE | Formular problemas complejos mediante principios de ingeniería, ciencia y matemáticas | TODO | TODO |
| DESIGN | Diseñar soluciones que satisfagan necesidades específicas considerando salud, seguridad y factores culturales | TODO | TODO |
| IMPLEMENT | Desarrollar e implementar sistemas y procesos basados en análisis de datos | TODO | TODO |
| TEST | Diseñar y ejecutar experimentos, analizar e interpretar datos para mejorar conclusiones | TODO | TODO |

### Acrónimos

- **API** — Application Programming Interface
- **AUC** — Area Under the Curve
- **CSV** — Comma-Separated Values
- **ELO** — Sistema de rating Elo
- **ETSIT** — Escola Tècnica Superior d'Enginyeria de Telecomunicació
- **GTDM** — Grau en Tecnologia Digital i Multimèdia
- **H2H** — Head-to-Head (enfrentamiento directo)
- **JSON** — JavaScript Object Notation
- **MC** — Monte Carlo
- **ML** — Machine Learning
- **ROC** — Receiver Operating Characteristic
- **TFG** — Trabajo de Fin de Grado
- **UPV** — Universitat Politècnica de València

### Agradecimientos

(TODO: completar con personas, director, familia, etc.)

---

## Capítulo 1 — Introducción

### 1.1. Motivación y contexto

El voleibol profesional es un deporte con alto contenido estocástico: un
rally individual tiene una probabilidad de punto dominada por el saque y
la recepción, pero la estructura de marcadores (sets a 25 con
diferencia de 2, mejor de 5) hace que el resultado de un partido sea
difícil de anticipar incluso para expertos. La SuperLega italiana,
máxima competición de clubes del país, dispone de datos públicos
detallados partido a partido desde 2014, lo que la convierte en un
campo natural para experimentar con modelos predictivos.

A pesar de la disponibilidad de datos, son escasos los predictores
calibrados —probabilidad no solo de quién gana, sino de la
distribución de marcadores (3-0, 3-1, 3-2)—. La mayoría de modelos
publicados se quedan en el ranking o la accuracy, sin evaluar la
calidad de probabilidad. Este Trabajo de Fin de Grado ataca esa
brecha combinando dos técnicas que rara vez se ven juntas en el
dominio:

- Un **motor de cadenas de Markov** que simula cada partido punto a
  punto, capturando dinámicas intra-rally (momentum, sideout, rachas).
- Tres **modelos de aprendizaje automático** (regresión logística
  con recencia, ensembles de árboles, gradient boosting) que
  calibran las probabilidades agregadas del simulador.

### 1.2. Pregunta de investigación

> *¿Es posible predecir el resultado de una temporada de la SuperLega
> italiana con buena calibración usando (a) cadenas de Markov para el
> detalle punto a punto, (b) modelos de aprendizaje automático de
> scikit-learn/XGBoost para la calibración, y (c) datos públicos de
> los últimos diez años?*

**Respuesta corta**: sí, con limitaciones. El sistema Predictor(2)
alcanza un AUC de **0.762** en predicción de partido y **0.697** en
predicción de set, con una simulación de temporada completa en 70 s.
La distribución de marcadores generada por el simulador tras la
calibración ML es muy cercana a la real (distancia L1 0.03), frente
a 0.29 sin calibrar.

### 1.3. Objetivos

Los objetivos del trabajo, ordenados de mayor a menor concreción:

1. **O1.** Construir un simulador de partidos y temporadas de la
   SuperLega italiana, basado en cadenas de Markov punto a punto, que
   genere distribuciones realistas de marcadores.
2. **O2.** Entrenar y validar tres modelos de ML (uno por nivel de
   granularidad: partido, set, punto) sobre los datos públicos de
   2014–2024, usando un protocolo de evaluación honesto sin leakage
   temporal.
3. **O3.** Integrar los modelos en el simulador como calibradores de
   la fuerza de los equipos (partido) y del centro del clamp de
   probabilidad de punto (punto), y medir el impacto de esa
   calibración sobre la fidelidad de la temporada simulada.
4. **O4.** Exponer todo el sistema detrás de una API REST en FastAPI
   y una interfaz web en React, para que la simulación sea utilizable
   sin conocimientos de programación.
5. **O5.** Documentar de forma honesta los resultados, incluyendo
   resultados negativos (el SetPredictor no aporta al clamp con los
   datos actuales; el MatchPredictor de 87 features tenía leakage) y
   las limitaciones residuales.

### 1.4. Estructura del documento

El resto de la memoria se organiza en seis capítulos:

- **Cap 2 — Marco teórico.** Conceptos matemáticos y de ML que
  sustentan el trabajo: cadenas de Markov, sistema Elo, regresión
  logística y ensembles, validación temporal sin leakage, métricas de
  evaluación (Brier, log-loss, ECE, AUC), calibración de
  probabilidades y simulación Monte Carlo. Redactado a partir del
  documento unificado `memoria/marco_teorico.md`.

- **Cap 3 — Materiales.** Origen y características de los datos de la
  SuperLega, el pipeline de carga y limpieza, la división temporal en
  train/val/test y el conjunto de features calculadas pre-partido
  (rolling). **Contenido completo en `memoria/data_layer.md` y
  `memoria/player_stats_generator.md`.**

- **Cap 4 — Métodos.** El motor de Markov (MatchSimulator) con sus
  dinámicas de momentum y sideout, el SeasonSimulator con el sistema
  de puntuación italiano, y los tres modelos ML: SetPredictor v2
  (LogReg+recencia), MatchPredictor sustituido en producción por
  **Elo con margen**, y PointProbabilityModel con regresión continua.
  **Contenido completo distribuido en `memoria/simulator.md`,
  `memoria/prediccion_partidos.md`, `memoria/prediccion_temporadas.md`,
  `memoria/point_probability.md`, `memoria/set_predictor.md` y
  `memoria/match_predictor.md`.**

- **Cap 5 — Resultados y discusión.** Métricas de los modelos con
  protocolo rolling-origin honesto, comparativa con la señal Elo
  pura, distribución de marcadores simulada vs. real, y resultados de
  la integración ML en el simulador. **Contenido completo en
  `memoria/mejora_precision_2026-07.md` (proceso) y
  `memoria/simulator.md` §10 (cifras).**

- **Cap 6 — Conclusiones y trabajo futuro.** Contribuciones,
  limitaciones residuales (sub-dispersión, features en frío, dataset
  limitado) y líneas de continuación del trabajo (Grupo B y C del
  plan consolidado). **Síntesis a partir de `memoria/INDICE.md` §
  Limitaciones + `docs/PLAN_MEJORAS_CONSOLIDADO.md` GRUPO B/C.**

- **Apéndices.** A: Relación con ODS de la UPV. B: Hiperparámetros
  del simulador y los modelos. C: Glosario detallado de métricas. D:
  Documentación de la API REST. El contenido de B vive en
  `src/simulation/constants.py`; el de D, en `src/api/main.py` (los
  7 endpoints están documentados en `AGENTS.md`).

### 1.5. Entorno tecnológico

- **Backend**: Python 3.12, FastAPI, Pydantic, scikit-learn 1.x,
  XGBoost, LightGBM, pandas, numpy, scipy, joblib.
- **Frontend**: Vite + React 18, lucide-react para iconos,
  react-router-dom v7 para enrutado, todo con UI en español.
- **Datos**: 22 CSVs públicos de la SuperLega italiana cubriendo 10
  temporadas (2014–2024), almacenados en `DB/`.
- **Persistencia**: modelos serializados con joblib en `models/`
  (gitignored, regenerables con `python -m src.models.train_improved`).

Todo el código y la documentación viven en el repositorio público
`github.com/asormar/SuperLega-Predictor`, con control de versiones
mediante Git, PRs por item de mejora y un protocolo de commits
conventional.

---

## Capítulo 2 — Marco teórico

Este capítulo presenta los fundamentos matemáticos y de aprendizaje
automático sobre los que se construye **Predictor(2)**. Está
redactado a partir del documento unificado
`memoria/marco_teorico.md`, que actúa como fuente única de teoría
para evitar definiciones dispersas en otros `.md` del proyecto. Las
siete secciones que siguen cubren: (1) cadenas de Markov, (2)
sistemas de rating Elo, (3) aprendizaje supervisado, (4) validación
temporal sin leakage, (5) métricas de evaluación, (6) calibración
de probabilidades y (7) simulación Monte Carlo.

### 2.1. Cadenas de Markov y modelado de eventos discretos

#### 2.1.1. Definición formal

Una **cadena de Markov** de tiempo discreto es un proceso estocástico
$\{X_t\}_{t \in \mathbb{N}}$ con espacio de estados finito
$\mathcal{S}$ que satisface la **propiedad de Markov** (memoria de
orden 1):

$$P(X_{t+1} = j \mid X_t = i, X_{t-1} = i_{t-1}, \ldots, X_0 = i_0)
\;=\; P(X_{t+1} = j \mid X_t = i)
\;=:\; p_{ij}.$$

La dinámica queda completamente definida por la **matriz de
transición** $P = (p_{ij})_{i,j \in \mathcal{S}}$, con $p_{ij} \geq 0$ y
$\sum_{j} p_{ij} = 1$. Una trayectoria de longitud $n$ tiene
probabilidad $\prod_{t=0}^{n-1} p_{X_t, X_{t+1}}$.

Para cadenas **irreducibles** y **aperiódicas**, la distribución
converge a una *distribución estacionaria* $\pi$ que satisface
$\pi P = \pi$, independientemente de la distribución inicial.

#### 2.1.2. Aplicación a la simulación deportiva

En Predictor(2) cada *rally* (punto) se modela como un ensayo de
Bernoulli cuyo éxito depende del estado relevante: quién saca, racha
de puntos, y marcador parcial. La propiedad de Markov se asume
*condicional al estado*:

$$P(\text{punto local} \mid \text{marcador}, \text{saca}, \text{rachas})
\;=\; p_{\text{home}} (\text{estado}),$$

es decir, no se depende de la historia más allá de los tres factores
mencionados. Esta asunción es estándar en simulación deportiva de
tenis, voleibol y bádminton, y es razonable en el régimen de rallies
cortos e i.i.d. (con cierto ruido residual que el modelo no captura).

**Limitación reconocida**: se ignora fatiga acumulada, cambios
tácticos mid-set, momentum emocional y efecto del marcador parcial
sobre el riesgo percibido. La propia sección de conclusiones del TFG
enumera esto como limitación atacable.

#### 2.1.3. Cálculo cerrado de probabilidad de set

Dada $p = P(\text{local gana un punto})$, la probabilidad de ganar un
*set* a 25 puntos (con diferencia de 2) admite una forma cerrada como
convolución de binomiales con la regla de勝. En
`src/simulation/set_math.py` la función
`p_set_from_p_point(p, target=25)` la implementa; para el tie-break
se usa `target=15`. La inversa `p_point_from_p_set` se obtiene por
biseción con caché `lru_cache`.

Esta forma cerrada es el *gold standard* que valida la cadena de
Markov del simulador: el test `TestMarkovChainSanity` en
`tests/test_simulator.py` fija $p = 0{,}52$, simula $n = 2000$ veces
y compara con la forma cerrada ($\Delta \approx 0{,}012$ sobre
$0{,}6967$).

### 2.2. Sistemas de rating deportivo

#### 2.2.1. Sistema Elo clásico

El **sistema Elo** asigna a cada jugador o equipo un rating
$R \in \mathbb{R}$ que se actualiza tras cada partido según el
resultado observado $s \in \{0, 1\}$ y la probabilidad esperada de
victoria $E[s]$:

$$R' \;=\; R + K \cdot \bigl( s - E[s] \bigr),$$

donde $K > 0$ es el factor de actualización (típicamente 20 a 32 en
ligas profesionales) y $E[s]$ sigue la función logística:

$$E[s] \;=\; \frac{1}{1 + 10^{(R_{\text{rival}} - R_{\text{local}})/400}}.$$

Una diferencia de 200 puntos equivale a $E[s] \approx 0{,}76$: el
favorito gana aproximadamente 3 de cada 4 partidos en el largo plazo.

#### 2.2.2. Elo con margen de victoria

El Elo clásico ignora *cómo* se ganó. La variante propuesta por
Silver (2015) para deportes de puntuación alta incorpora el margen
de victoria $M = \text{puntos}_{\text{ganador}} - \text{puntos}_{\text{perdedor}}$
mediante un multiplicador logarítmico:

$$R' \;=\; R + K \cdot \ln(1 + M) \cdot \bigl( s - E[s] \bigr).$$

El factor $\ln(1 + M)$ crece con el margen pero con rendimientos
decrecientes: una victoria por 20 puntos actualiza el rating sólo
$\approx 1{,}3$× más que una victoria por 4. Esto refleja que en
voleibol una victoria amplia es más informativa, pero no 5× más.

En Predictor(2) la implementación en
`src/data/rolling_features.py` usa los valores canónicos:

- $K = 28$ (constante del proyecto, pineada en `AGENTS.md`).
- `HOME_ADV = 60` puntos Elo sumados al local antes de calcular
  $E[s]$.
- Margen $\text{margin} = |\text{pts}_h - \text{pts}_a|$ actualizado tras
  cada partido real en el backtest, o simulado en el
  SeasonSimulator.

Evaluado con protocolo rolling-origin sobre 2025/26:
$\text{AUC} = 0{,}762$ (n=314 partidos), Brier $= 0{,}193$,
log-loss $= 0{,}568$.

#### 2.2.3. Probabilidad calibrada por construcción

La $E[s]$ del Elo es una *probabilidad calibrada* en sentido
frecuentista: para todos los partidos con $E[s] \approx 0{,}7$, el
local gana el 70% de las veces a largo plazo. Esto la diferencia de
las salidas brutas de los modelos ML (LogReg, XGBoost) que, sin
calibración post-hoc, dan probabilidades sesgadas o mal calibradas
(ver §2.6). En el backtest del simulador el Elo puro actúa como
*referencia de calibración*: el simulador calibrado debe acercarse a
él, no superarlo por puro *luck*.

### 2.3. Aprendizaje supervisado para predicción deportiva

#### 2.3.1. Regresión logística con regularización L2

La **regresión logística** modela
$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$
con $\sigma(z) = (1 + e^{-z})^{-1}$. La regularización L2 (ridge)
penaliza $\|\mathbf{w}\|_2^2$, dando el problema de optimización:

$$\min_{\mathbf{w}, b} \;
-\sum_{i} \log \sigma\bigl(y_i (\mathbf{w}^\top \mathbf{x}_i + b)\bigr)
\;+\; \frac{\lambda}{2}\|\mathbf{w}\|_2^2.$$

**Por qué se eligió para el SetPredictor v2**: en el régimen de
datos pequeños ($\sim\!1200$ sets tras B0b), los modelos lineales
regularizados ganan a los ensembles de árboles (ver
`mejora_precision_2026-07.md` §6: log-loss CV
$0{,}654 \to 0{,}634$ al pasar de ExtraTrees a LogReg+recencia). El
SetPredictor v2 en producción es una
`LogisticRegression(C = 0{,}5)` (inversa de $\lambda$) con
**pesos de recencia** half-life $= 2$ temporadas.

#### 2.3.2. Árboles de decisión y ensembles

Un *árbol de decisión* particiona recursivamente el espacio de
features por ejes paralelos, eligiendo en cada nodo la división que
maximiza la reducción de impureza (índice de Gini o entropía). Es
interpretable pero muy inestable: pequeños cambios en los datos
producen árboles muy diferentes.

Los *ensembles* mitigan esto promediando muchos árboles:

- **RandomForest** (Breiman, 2001): bagging de árboles con
  subconjuntos aleatorios de features en cada split.
- **ExtraTrees** (Geurts et al., 2006): igual pero con umbrales
  aleatorios, no óptimos. Más rápido, a veces más robusto en
  datasets pequeños.

#### 2.3.3. Gradient boosting

El *gradient boosting* (Friedman, 2001) construye el ensemble
*secuencialmente*: cada árbol nuevo ajusta los residuos del ensemble
actual según el gradiente de la loss. Implementaciones eficientes:

- **XGBoost** (Chen y Guestrin, 2016): regularización L1+L2,
  shrinkage, column subsampling.
- **LightGBM** (Ke et al., 2017): histograma + leaf-wise growth,
  mucho más rápido en datasets grandes.

Predictor(2) los usa como candidatos en el benchmark y en la
calibración de MatchPredictor v1. En el SetPredictor v2 en
producción *no se usan* porque el régimen de datos no los favorece.

#### 2.3.4. Pesos de recencia temporal

En series temporales deportivas los partidos de la temporada actual
son más informativos que los de hace cinco años. Una **función de
peso exponencial** con half-life $= 2$ temporadas asigna a la
temporada $t$ el peso:

$$w(t_{\text{current}} - t) \;=\; 2^{-(t_{\text{current}} - t)/2},$$

de modo que la temporada 2023 pesa $0{,}5$ cuando estamos en 2025, y
la 2018 pesa $\approx 0{,}03$. El SetPredictor v2 entrena con sample
weights así sobre `train 2022–2024` y obtiene un CV AUC
$0{,}679 \pm 0{,}017$ — más alto y estable que el de 4-fold sin
pesos ($0{,}631 \pm 0{,}078$).

### 2.4. Validación temporal y prevención de leakage

#### 2.4.1. El problema del leakage temporal

El **leakage temporal** ocurre cuando el modelo ve, durante el
entrenamiento, información que en producción no estaría disponible.
Ejemplos típicos:

- Usar el *ranking final de la temporada* como feature para predecir
  partidos de esa temporada.
- Calcular `win_rate_last5` sobre los últimos 5 partidos *de la
  temporada completa*, incluyendo partidos futuros desde el punto de
  vista del partido a predecir.
- Entrenar con datos hasta 2025 y validar con datos hasta 2024 (el
  modelo "vió el futuro").

En este TFG se documenta un caso real: el `MatchPredictor` v1
reportaba $\text{AUC} = 0{,}707$ sobre test 2024, pero el valor
honesto con rolling-origin era $\text{AUC} = 0{,}53$. La diferencia
provenía de features que incluían estadísticas de temporada
completa. Reconstruir las features sin leakage (Elo rolling, EWMA
pre-partido, H2H histórico) *bajó* el AUC reportado y *subió* el
AUC honesto a $0{,}762$: el modelo real era mucho mejor de lo que
la métrica inflada sugería. La lección — *una métrica mal medida es
peor que una métrica baja pero honesta* — es la contribución
metodológica central del trabajo.

#### 2.4.2. Protocolo rolling-origin

El **protocolo rolling-origin** es el estándar para evaluar
*forecasts* en series temporales:

1. Fijar un horizonte de test (en Predictor(2), la temporada
   2024/25).
2. Para cada fold $k$:
   - Entrenar con todas las temporadas $\leq t_k$.
   - Predecir la temporada $t_k + 1$ completa.
3. Reportar la métrica agregada sobre todos los folds.

En la práctica del TFG se usa un *rolling-origin de 2 folds* sobre
sets:

- Fold 1: `train 2016–2022 → val 2023`.
- Fold 2: `train 2016–2023 → val 2024`.

La temporada 2025 actúa como *test held-out*: **no se itera** sobre
ella, se evalúa una sola vez al final.

#### 2.4.3. Features rolling pre-partido

Para evitar el leakage, todas las features del pipeline de
producción son *rolling pre-partido*: para un partido de la jornada
$j$ de la temporada $t$, se usan solo partidos de la temporada $t$
con jornada $< j$ (o de temporadas $< t$ si $j$ es muy temprana).
El módulo `src/data/rolling_features.py` implementa:

- `elo_diff`: rating Elo recalculado partido a partido desde el
  inicio del histórico.
- `diff_win_rate`: win rate acumulado hasta la jornada $j - 1$.
- `diff_set_ratio`: ídem para sets.
- `diff_form_ewma`: forma con *exponential weighted moving average*
  con $\alpha$ controlado.

Esto es lo que distingue las features de producción de las del
MatchPredictor v1 (con leakage): las primeras se computan como si
el partido aún no existiera.

### 2.5. Métricas de evaluación

#### 2.5.1. Brier score

El **Brier score** mide el error cuadrático medio entre la
probabilidad predicha y el resultado binario observado:

$$\text{Brier} \;=\; \frac{1}{n} \sum_{i=1}^{n} (p_i - s_i)^2,$$

donde $p_i \in [0, 1]$ es la probabilidad predicha de "local gana" y
$s_i \in \{0, 1\}$ es el resultado. Rango $[0, 1]$, menor es mejor.
Un clasificador perfecto tiene Brier 0; uno constante en $0{,}5$
tiene Brier $0{,}25$.

El Brier es *estrictamente proper*: se minimiza prediciendo la
verdadera probabilidad. Es la métrica preferida del TFG para
evaluar la calidad de probabilidad del simulador, por encima de
accuracy (que es invariante a la calibración).

#### 2.5.2. Log-loss (cross-entropy binaria)

El **log-loss** mide la sorpresa logarítmica media:

$$\text{LogLoss} \;=\; -\frac{1}{n} \sum_{i=1}^{n}
\bigl[ s_i \log p_i + (1 - s_i) \log (1 - p_i) \bigr].$$

Rango $[0, +\infty)$, menor es mejor. Más sensible que el Brier a
predicciones muy seguras pero incorrectas: un clasificador que da
$p = 0{,}99$ y falla paga $-\log(0{,}01) \approx 4{,}6$, mientras
que con Brier paga $0{,}98^2 = 0{,}96$.

#### 2.5.3. Expected Calibration Error (ECE)

El **ECE** mide la calibración con independencia de la
discriminación. Se discretiza el rango $[0, 1]$ en $K$ bins, se
calcula la confianza media $\bar{p}_k$ y la frecuencia observada
$\bar{s}_k$ en cada bin:

$$\text{ECE} \;=\; \sum_{k=1}^{K} \frac{|B_k|}{n}
\bigl| \bar{p}_k - \bar{s}_k \bigr|.$$

Rango $[0, 1]$, menor es mejor. Un modelo perfectamente calibrado
tiene ECE 0. En el TFG, el simulador pre-B3 tenía ECE $0{,}242$
(sobreconfiado) y post-B3 bajó a $0{,}057$, muy cerca del Elo puro
($0{,}045$).

#### 2.5.4. AUC-ROC

El **AUC** es el área bajo la curva ROC (Receiver Operating
Characteristic), que grafica TPR frente a FPR al variar el umbral de
decisión. Equivale a la *probabilidad de que el modelo asigne mayor
score a un positivo aleatorio que a un negativo aleatorio*. Rango
$[0{,}5, 1]$, mayor es mejor, con $0{,}5 =$ azar y $1{,}0 =$
perfecto.

AUC es *invariante a la calibración* (solo mide ranking) y *robusto
a clases desbalanceadas*. En el TFG:
$\text{MATCH AUC} = 0{,}762$, $\text{SET AUC} = 0{,}697$, ambos
con test held-out 2025/26.

#### 2.5.5. Accuracy

La **accuracy** es la fracción de predicciones correctas con un
umbral (típicamente $0{,}5$):

$$\text{Acc} \;=\; \frac{1}{n} \sum_{i} \mathbf{1}\!\left[
s_i = \mathbb{1}[p_i \geq 0{,}5] \right].$$

Rango $[0, 1]$, mayor es mejor. Fácil de interpretar pero
*engañosa en desbalance de clases* y *sensible al umbral*. En el
TFG se reporta siempre junto con Brier y ECE, no sola.

#### 2.5.6. Distancia L1 entre distribuciones de marcador

Para comparar la distribución de marcadores
(3-0 / 3-1 / 3-2) entre lo simulado y lo real, la métrica natural
es la *distancia L1* (variación total):

$$L_1 \;=\; \frac{1}{2} \sum_{m \in \mathcal{M}}
\bigl| P_{\text{sim}}(m) - P_{\text{real}}(m) \bigr|.$$

El factor $1/2$ garantiza $L_1 \in [0, 1]$. En el TFG, pre-B3
$L_1 = 0{,}286$ ($53\%/30\%/17\%$ simulado vs. $39\%/35\%/26\%$
real) y post-B3 $L_1 = 0{,}031$ ($37{,}6\%/34{,}7\%/27{,}7\%$ vs.
$38{,}7\%/35{,}1\%/26{,}1\%$): el simulador pasa de *degradar* la
señal Elo a *superarla*.

### 2.6. Calibración de probabilidades

#### 2.6.1. Platt scaling

El **Platt scaling** aprende una regresión logística de un parámetro
sobre los scores del modelo:

$$P(y = 1 \mid s) \;=\; \sigma(a \cdot s + b),$$

donde $s$ es el score bruto (puede ser un margen o un logit) y
$(a, b)$ se ajustan sobre un conjunto de validación. Es barato,
estable, y suele mejorar la calibración sin perder discriminación.
Asume que la distorsión es *sigmoide*.

#### 2.6.2. Isotonic regression

La **isotonic regression** aprende una función monótona no
paramétrica sobre los scores. Es más flexible que Platt (no asume
forma) pero necesita más datos ($\geq 1000$ muestras) y puede
sobreajustar con datasets pequeños.

En Predictor(2) el SetPredictor legacy ExtraTrees aplicaba
`CalibratedClassifierCV(cv=3, method="isotonic")` tras seleccionar el
campeón por AUC. En el v2 (LogReg) *no se aplica* porque la
regresión logística ya produce probabilidades calibradas por
construcción (ver §2.3).

#### 2.6.3. Por qué LogReg no necesita calibración post-hoc

La salida de LogReg es $\sigma(\mathbf{w}^\top \mathbf{x} + b)$, que
*minimiza la log-loss* sobre los datos de entrenamiento. Si el modelo
está bien especificado (forma funcional correcta), esto converge a
la verdadera $P(y = 1 \mid \mathbf{x})$. En la práctica, las
predicciones LogReg suelen estar bien calibradas *out of the box*,
lo cual es una ventaja sobre los ensembles de árboles (cuyas
salidas son votos o promedios y necesitan calibración post-hoc).

En el TFG esto se manifiesta en que el SetPredictor v2
LogReg+recencia se entrena sin `CalibratedClassifierCV` y obtiene
ECE $0{,}057$ en el backtest — comparable al del Elo puro
($0{,}045$), que es la referencia de calibración.

### 2.7. Simulación Monte Carlo

#### 2.7.1. Estimación de incertidumbre por repetición

La **simulación Monte Carlo** estima expectativas de funcionales de
procesos estocásticos repitiendo el experimento $N$ veces con
semillas distintas y promediando:

$$\hat{\mu} \;=\; \frac{1}{N} \sum_{i=1}^{N} f(\omega_i),
\qquad \omega_i \sim P.$$

El error estándar del estimador es $\sigma / \sqrt{N}$, donde
$\sigma$ es la desviación estándar de $f(\omega)$ entre corridas.

En Predictor(2) el `MatchSimulator.monte_carlo_simulate()` corre $N$
simulaciones independientes de un partido y devuelve la *distribución
empírica* de marcadores y la probabilidad de cada ganador. Para un
partido suelto típico, $N = 2000$ basta para tener errores estándar
$< 1\,\text{pp}$ en las probabilidades de marcador.

#### 2.7.2. Distribución de marcadores frente a resultado puntual

Una sola simulación da un *marcador puntual* (p. ej. 3-1). $N$
simulaciones dan una *distribución* (p. ej. 3-0 en 38%, 3-1 en 35%,
3-2 en 27%). La distribución es la información *útil* para
predicción (probabilidades de cada resultado) y para toma de
decisiones (qué apuesta ofrece valor).

En la UI, el endpoint `POST /api/simular/partido` con
`n_simulaciones_mc > 0` devuelve la distribución agregada; con
`n_simulaciones_mc = 0` devuelve un solo partido. La elección de $N$
es un trade-off *tiempo/precisión* que el propio endpoint expone al
usuario.

### 2.8. Síntesis y referencias cruzadas al proyecto

| Concepto | Aplicación | Fuente original |
|---|---|---|
| Markov (§2.1) | `MatchSimulator._simulate_set` en `simulator.py` | `memoria/simulator.md` §4 |
| Forma cerrada de set (§2.1) | `p_set_from_p_point` en `set_math.py` (A2) | `mejora_precision_2026-07.md` §7 |
| Elo con margen (§2.2) | `get_historical_team_elo`, `_compute_margin_elo` en `rolling_features.py` | `prediccion_temporadas.md` §6 |
| Regresión logística + recencia (§2.3) | `SetPredictor` v2 en `set_predictor_v2.py` | `set_predictor.md` |
| Gradient boosting (§2.3) | `MatchPredictor` v1 legacy (fallback) en `match_predictor.py` | `match_predictor.md` |
| Rolling-origin (§2.4) | `evaluate_model_rolling` en `src/models/evaluation.py` | `mejora_precision_2026-07.md` §1, §5 |
| Brier, log-loss, ECE (§2.5) | `measure_precision.py`, `backtest_simulator.py` | `INDICE.md`, sección "Métricas clave" |
| Isotonic regression (§2.6) | `SetPredictor` legacy con `CalibratedClassifierCV`; v2 *sin* calibración | `set_predictor.md` §3 |
| Monte Carlo (§2.7) | `MatchSimulator.monte_carlo_simulate` | `simulator.md` §6 |

---

## Capítulo 3 — Materiales

**Fuente del contenido completo**:

- `memoria/data_layer.md` — pipeline de datos, feature store, split
  temporal, `RuntimeFeatureBuilder`, `team_mapper`, normalización
  de nombres. Cubre: origen de los 22 CSVs de la SuperLega
  (2014–2024), el script `data_pipeline.py` de carga y limpieza,
  el `feature_store.py` con el split estricto train/val/test
  (2016–2022 / 2023 / 2024), la detección y corrección de la
  colisión `partido_id` que duplicaba 596/725 partidos (B0), y el
  pipeline de features rolling sin leakage en
  `src/data/rolling_features.py`.

- `memoria/player_stats_generator.md` — generación de estadísticas
  sintéticas por jugador. Cubre: el muestreo de distribuciones
  históricas normalizadas al marcador del set, las 8 estadísticas
  que se acumulan por jugador (`puntos`, `aces`, `ataques_ganados`,
  `bloqueos`, `recepciones_exc`, `errores_saque`, etc.) y sus
  limitaciones como generador *post-hoc* (no simulación
  play-by-play).

**Datos clave para la narrativa del capítulo**:

- 22 CSVs en `DB/` con datos de la SuperLega 2014–2024.
- Tras la corrección de la colisión `partido_id` (B0, 2026-07-15):
  **1322 partidos válidos** (frente a 725 "venenados" pre-B0).
- 3 archivos principales que alimentan el pipeline:
  - `sets_partidos.csv` — marcadores punto a punto de cada set.
  - `stats_por_equipo_completo/` — stats históricas por equipo.
  - `Comparacion_equipos_10_años.csv` — comparativa multi-temporada.
- Stack Python: pandas, numpy, scikit-learn, joblib, FastAPI,
  Pydantic.

---

## Capítulo 4 — Métodos

**Fuente del contenido completo** (organizada por subsección del
capítulo):

### 4.1. Motor de Markov del partido individual

→ `memoria/simulator.md` (íntegra el motor)
→ `memoria/prediccion_partidos.md` (uso en el endpoint)

Cubre: el `MatchSimulator` con su estado
`(marcador, saca, rachas, momentum)`, el clamp adaptativo del
Markov con `CLAMP_MARGIN_POINT = 0.10` y `SET_BLEND_WEIGHT_ELO = 1.0`
(SetPredictor cableado pero inactivo en runtime — resultado
negativo de A4), el modo Monte Carlo con `n_simulaciones_mc`, y la
generación de stats por jugador.

### 4.2. Simulación de temporada

→ `memoria/prediccion_temporadas.md`

Cubre: el `SeasonSimulator` con el sistema de puntuación SuperLega
(3-0/3-1 → 3 puntos; 3-2 → 2+1 puntos), el generador de
calendario round-robin (círculo de N equipos), la calibración de
fuerzas con margin-Elo (K=28, HOME_ADV=60), el damping
`MATCH_PREDICTOR_DAMPING = 0.5`, el flujo jornada-a-jornada con
los endpoints `/api/simular/temporada/iniciar` y `/jornada`, y el
método de doble vuelta con `half='first'`/`half='second'`.

### 4.3. Modelo de probabilidad de punto

→ `memoria/point_probability.md`

Cubre: el `PointProbabilityModel` con `Ridge(alpha=1.0)` sobre
target continuo `point_ratio_h` (post-B3, 2026-07-22), el clip
`POINT_RATIO_CLIP = (0.40, 0.60)` solo de salvavidas, el sideout
per-team desde `src/data/team_sideout.py`, y el contrato de
features rolling pre-partido (sin leakage). Detalle del fix B3 que
sustituyó la binarización + mapping `0.45 + 0.10·p_dominante`
(que sesgaba hacia el local y producía 53% de 3-0).

### 4.4. Predictor de set (LogReg + recencia)

→ `memoria/set_predictor.md`

Cubre: el `SetPredictor` v2 con `LogisticRegression(C=0.5)` y
pesos de recencia half-life=2 temporadas, las 21 features
(`SET_FEATURE_COLS`) que incluyen fuerza de equipo, diferencia
Elo, win rate en sets, forma reciente, H2H, y estado in-match
(`set_num_norm`, `sets_h_antes`, `momentum_h`, `es_desempate`).
Métricas: AUC test 2025 = 0.697, CV rolling-origin 2-fold =
0.679 ± 0.017 (n=1193). El adaptador `LogRegSetPredictor` en
`src/models/set_predictor_v2.py` con `try_load_v2` (cascada v2 →
legacy). Discusión del desenlace A4: con `w=1.0` el SetPredictor
no se consulta en el clamp.

### 4.5. Predictor de partido (Elo con margen)

→ `memoria/match_predictor.md`

Cubre: el **margin-Elo** en `src/data/rolling_features.py` con
K=28, HOME_ADV=60 y factor $\ln(1 + M)$, evaluado con protocolo
rolling-origin: AUC 0.762, Brier 0.193, log-loss 0.568 sobre
2025/26 (n=314). El legacy `MatchPredictor` de 87 features
(XGBoost+isotónico) se documenta como antecedente con AUC 0.707
ficticio por leakage; su valor real honesto era ~0.53. Detalle
del proceso de detección y corrección del leakage en
`mejora_precision_2026-07.md` §1.

### 4.6. Generación de estadísticas de jugadores

→ `memoria/player_stats_generator.md`

Cubre: el `PlayerStatsGenerator` con muestreo de distribuciones
históricas normalizadas al marcador del set. Post-hoc, no
play-by-play.

---

## Capítulo 5 — Resultados y discusión

**Fuente del contenido completo**:

- `memoria/mejora_precision_2026-07.md` — proceso completo de
  mejora de precisión (Fases 0-4, B0–B0b, B1, B3, A2–A6). Cubre:
  auditoría inicial del AUC 0.707 ficticio del MatchPredictor,
  reconstrucción de features sin leakage (Elo con margen
  rolling), protocolo rolling-origin honesto, integración de
  Elo en producción, y los resultados del backtest B1 (Brier
  0.273 → 0.1815, ECE 0.242 → 0.057, accuracy 0.649 → 0.721,
  L1 márgenes 0.286 → 0.031). Discusión detallada por item del
  plan consolidado.

- `memoria/simulator.md` §10 — backtest end-to-end (B1) del
  simulador contra la temporada real 2024 (222 partidos, n=500).
  Tablas con cifras vigentes y cifra histórica pre-B3. Lectura
  post-B3: el simulador **no degrada** la calidad de probabilidad,
  la mejora; supera a la señal Elo pura en Brier, logloss y
  accuracy; la distribución de márgenes queda a menos de 2 pp del
  real en los tres marcadores.

- `memoria/benchmark.md` — comparativa de los 9 modelos candidatos
  (LR, RF, ET, GB, XGB, LGBM) sobre features de set y match, con
  ablation de +roster y de equipos 12 vs 16.

- `memoria/INDICE.md` §"Métricas clave" — tabla resumen con las
  cifras vigentes del proyecto:

  | Modelo | AUC | Accuracy | Notas |
  |---|---:|---:|---|
  | MATCH (Elo con margen) | **0.762** | 0.70 | producción; n=314, test 2025/26 |
  | SET (LogReg+recencia v2) | **0.697** | 0.65 | n=1193, CV 0.679 ± 0.017 |

  Y la inversión post-B3 del simulador (cap 5) frente al Elo puro:
  Brier 0.182 vs 0.194; logloss 0.537 vs 0.569; accuracy 0.721 vs
  0.689; ECE 0.057 vs 0.045; L1 márgenes 0.031 vs —.

**Resultados negativos a documentar** (parte de la honestidad del
capítulo):

- **SetPredictor no aporta al clamp** (A4, 2026-07-21): con
  `SET_BLEND_WEIGHT_ELO = 1.0` el cortocircuito evita la
  evaluación. El modelo se mantiene cableado por compatibilidad y
  para futura re-evaluación. Detalle en `mejora_precision_2026-07.md`
  §7.
- **MatchPredictor de 87 features con leakage** (B0): su AUC
  reportado 0.707 era ficticio (valor real 0.53). Reemplazado
  por margin-Elo en producción.
- **Hyperparameter tuning ×2** (Batch 3): ambos intentos
  resultaron negativos con el protocolo honesto. Documentado
  como cierre de línea de investigación.
- **Damping adaptativo como default** (Batch 3): se implementó
  pero su adopción como default no mejoraba las métricas; queda
  como opt-in.

---

## Capítulo 6 — Conclusiones y trabajo futuro

### 6.1. Contribuciones

1. **Sistema end-to-end funcional**: Predictor(2) integra un motor
   de Markov punto a punto, tres modelos ML, una API REST
   (FastAPI) y una UI web (React), con un pipeline de
   entrenamiento reproducible y métricas honestas.

2. **Protocolo de evaluación riguroso**: la auditoría de precisión
   del 2026-07 (Fases 0-4 del plan) demostró que la métrica
   "AUC 0.707" reportada por el MatchPredictor legacy era
   ficticia por leakage temporal. La reconstrucción sin leakage
   (B0) reveló un Elo con margen AUC 0.762 limpio. Este
   *leakage audit* es la contribución metodológica central.

3. **Modelo de punto calibrado** (B3): la sustitución de la
   binarización + mapping sesgado por regresión continua
   (Ridge sobre `point_ratio_h`) cerró la sub-dispersión del
   simulador. El sistema pasa de *degradar* la señal Elo a
   *superarla* en Brier, logloss y accuracy.

4. **Documentación exhaustiva en español** estructurada en
   `memoria/` (12 archivos `.md`) más los `.tex` ahora consolidados
   en este md unificado, listos para transcribir a LaTeX.

5. **Resultados negativos documentados** con el mismo rigor que
   los positivos (SetPredictor cableado pero inactivo, hyperparameter
   tuning ×2, damping adaptativo). Patrón replicable para futuros
   TFGs en analítica deportiva.

### 6.2. Limitaciones

Tomado de `memoria/INDICE.md` y `docs/PLAN_MEJORAS_CONSOLIDADO.md`:

- **Sub-dispersión residual** (origen en el modelo de punto, no
  en el clamp): B6 (ampliar dataset) y B3 (regresión continua)
  redujeron el problema pero ECE 0.057 sigue por encima del Elo
  puro (0.045).
- **Dataset pequeño a nivel de partido** (~1322 tras B0; 34-59 por
  temporada en las viejas): régimen donde modelos lineales baten
  a árboles profundos, pero limita la potencia de los ensembles.
- **Stats de jugadores sintéticas** (muestreadas, no simuladas).
- **Sin Monte Carlo a nivel temporada por defecto** (incertidumbre
  no cuantificada en un solo seed).
- **Sin lesiones ni mercado de fichajes**.
- **MatchPredictor de 87 features (leaky) sigue en disco** como
  fallback; el camino de producción usa la probabilidad de Elo
  limpia.
- **~40 `print()` con caracteres Unicode** que rompen en consola
  Windows (deuda técnica, pendiente C3 del plan).

### 6.3. Trabajo futuro

Agrupado por item del plan consolidado
(`docs/PLAN_MEJORAS_CONSOLIDADO.md`):

**Grupo B — Precisión end-to-end del simulador**:

- **B2 (próxima)**: ajustar las constantes del simulador contra
  el backtest (MOMENTUM_BONUS, GLOBAL_MOMENTUM_FACTOR, MATCH_PREDICTOR_DAMPING)
  con grid 36 combos sobre 2023-2024, validar una vez en 2025.
  Esfuerzo 2-3 h + CPU. El siguiente paso natural.
- **B4**: predictor de partido derivado del SetPredictor
  (best-of-5). Retorno marginal no compensaba; re-evaluar
  post-B2.
- **B5**: feature de continuidad de plantilla (roster churn).
  Única señal pre-temporada de fichajes.
- **B6**: ampliar el dataset de partidos. La palanca grande, la
  más cara; tras B2.
- **B7**: re-validación con 2026/27. Bloqueado por calendario;
  dejar script listo.

**Grupo C — Infra y calidad**:

- **C1**: Ruff + Black + CI.
- **C3-C6**: logging, hardening API, deploy Docker, tests del
  frontend.

**Grupo D — Backlog**:

- D1: arranque en frío de features dinámicas no-Elo.
- D2: PlayerStatsGenerator más realista.
- D3: saneamiento del data layer.
- D4: sideout por forma reciente.

**Grupo E — Extras**:

- E1: MC de temporada en UI. E2: explicabilidad. E3:
  `precision_report.py` unificado. E4: predicción de jornada
  real (demo). E5: blindaje contra regresión silenciosa a
  modelos leaky.

---

## Apéndices

### Apéndice A — Relación con los ODS de la UPV

(TODO: redactar — cubrir al menos ODS 4 "Educación de calidad",
ODS 9 "Industria, innovación e infraestructura", y ODS 17 "Alianzas
para lograr los objetivos". El proyecto es software abierto, por
lo que la alineación con ODS 9 y 17 es directa.)

### Apéndice B — Hiperparámetros

Fuente: `src/simulation/constants.py` y los docstrings de los
modelos. Valores canónicos pineados por `tests/test_team_mapper.py`
y `tests/test_simulator.py`:

**Simulador (constants.py)**:

- `HOME_ADVANTAGE_STRENGTH_BONUS = 0.03`
- `POINT_PROB_CLIP = (0.25, 0.75)`
- `POINT_PROB_CLIP_ADAPTIVE_HARD = (0.10, 0.90)`
- `DEFAULT_CLAMP_RANGE = (0.20, 0.80)`
- `CLAMP_MARGIN = 0.20` (LEGACY)
- `CLAMP_MARGIN_POINT = 0.10` (post-A2)
- `SET_BLEND_WEIGHT_ELO = 1.0` (post-A4)
- `POINT_RATIO_CLIP = (0.40, 0.60)` (post-B3)
- `DEFAULT_SIDEOUT_RATE = 0.62`
- `MATCH_PREDICTOR_DAMPING = 0.5`
- `ADAPTIVE_DAMPING_START = 0.3`
- `ADAPTIVE_DAMPING_END = 0.7`
- `SUPERLEGA_TOTAL_JORNADAS = 26`
- `MOMENTUM_BONUS = 0.015`
- `MOMENTUM_MAX_STREAK = 4`
- `MOMENTUM_DECAY = 0.5`
- `GLOBAL_MOMENTUM_FACTOR = 0.01`
- `STRENGTH_CLAMP_RANGE = (0.05, 0.95)`
- `MAX_MC_ITERATIONS = 5000`

**SetPredictor v2** (de `set_predictor_v2.py` y
`set_features_builder.py`):

- `LogisticRegression(C=0.5, max_iter=2000, random_state=42)`
- 21 features (ver `SET_FEATURE_COLS`)
- Recency: `half-life=2` temporadas
- Train: 2022-2024

**PointProbabilityModel** (post-B3): `Ridge(alpha=1.0, random_state=42)`
con 6 features rolling.

**Margin-Elo** (en `rolling_features.py`): K=28, HOME_ADV=60.

### Apéndice C — Glosario detallado de métricas

(Las definiciones matemáticas completas están en §2.5. Este
apéndice puede ampliar con: cuándo usar cada una, cómo se
relacionan, ejemplos numéricos, y figuras de curvas de
calibración.)

### Apéndice D — API REST

Los 7 endpoints documentados en `AGENTS.md` (sección "API
surface"):

- `GET /api/equipos` — lista de equipos con fuerzas y colores.
- `GET /api/equipos/{nombre}` — roster + promedios por set.
- `POST /api/simular/partido` — simulación de partido suelto.
- `POST /api/simular/temporada` — simulación completa (legacy).
- `POST /api/simular/temporada/iniciar` — inicializa calendario.
- `POST /api/simular/temporada/jornada` — simula una jornada.
- `GET /api/modelo/info` — info del modelo cargado.

Detalle completo de los cuerpos de request/response en
`latex/tex/app_api.tex` (a redactar en LaTeX desde aquí).

---

## Bibliografía

(Las refs se centralizan en `latex/references.bib`. Las que ya
aparecen citadas en el cuerpo del TFG: Elo 1978, Brier 1950,
Platt 1999, Breiman 2001, Friedman 2001, Zadrozny & Elkan 2002,
Hyndman & Athanasopoulos 2018, Fawcett 2006, Tashman 2000,
Naeini et al. 2015, Chen & Guestrin 2016, Ke et al. 2017,
Geurts et al. 2006, Metropolis & Ulam 1949, Silver 2015.)

---

**Cómo transcribir a LaTeX**: copia cada sección a su archivo
correspondiente (`cap1_intro.tex`, `cap2_marco_teorico.tex`, etc.)
y convierte la sintaxis Markdown a LaTeX:

- `$...$` → ya es LaTeX.
- `**bold**` → `\textbf{...}`.
- `*italic*` → `\emph{...}`.
- `` `code` `` → `\texttt{...}`.
- `# Título` → `\chapter{...}` / `\section{...}` / `\subsection{...}`.
- Tablas Markdown → `tabularx` (ver ejemplo en §2.8).
- Ecuaciones de bloque: `$$...$$` → `\[...\]` o `equation`.
