"""
indicators/axes.py

Los 5 ejes del Sistema de Ángulos Cruzados.
Fórmulas exactas: MAESTRO_Patrones_Angulos_v2.docx, Sección 6.

Cada eje toma valores discretos en {-2, -1, 0, +1, +2}:
  +2 = señal alcista fuerte
  +1 = señal alcista moderada
   0 = neutral
  -1 = señal bajista moderada
  -2 = señal bajista fuerte

El score de convergencia = suma de los 5 ejes → rango -10 a +10.

Decisiones de diseño:
- Todas las operaciones son vectorizadas (no hay loops por fila).
- No hay lookahead bias: cada cálculo usa solo datos disponibles en t.
- E5 usa ffill para llevar valores de TFs más lentos al TF de análisis.
  Esto es correcto: al cierre de una vela 15m, la última vela 1h cerrada
  es información ya disponible — no es futuro.
- calcular_ejes() es el punto de entrada principal. Retorna un DataFrame
  con columnas e1-e5 y 'score', listo para el detector de patrones.
"""

import numpy as np
import pandas as pd
from typing import Dict

from indicators.trend import ema, sma
from indicators.momentum import rsi as _rsi


# ── utilidad interna ─────────────────────────────────────────────────────────

def _discretizar(serie: pd.Series, cortes: list[tuple[float, int]]) -> pd.Series:
    """
    Mapea una serie continua a enteros discretos usando np.select.

    cortes: lista de (umbral, valor) ordenada de MAYOR a MENOR umbral.
    La primera condición que se cumple gana — el último valor es el default
    (se aplica cuando ninguna condición anterior fue verdadera).

    Ejemplo para E1:
        cortes = [(3.0, 2), (0.7, 1), (-0.7, 0), (-3.0, -1)]
        → si desviacion > 3.0  → 2
          si desviacion > 0.7  → 1
          si desviacion > -0.7 → 0
          si desviacion > -3.0 → -1
          si ninguna           → -2  (default)
    """
    conditions = [serie > umbral for umbral, _ in cortes]
    choices    = [valor           for _, valor  in cortes]
    return pd.Series(
        np.select(conditions, choices, default=choices[-1] - 1),
        index=serie.index,
        dtype=float,
    )


# ── E1 — Precio vs EMA200 ────────────────────────────────────────────────────

def calcular_e1(df: pd.DataFrame) -> pd.Series:
    """
    E1 mide qué tan lejos está el precio del promedio de largo plazo (EMA200).

    desviacion = (close - EMA200) / EMA200 × 100

    Umbrales:
      +2 : desviación > +3.0%   (precio muy por encima, sobreextendido)
      +1 : +0.7% a +3.0%        (precio moderadamente arriba)
       0 : -0.7% a +0.7%        (precio en zona de la EMA — equilibrio)
      -1 : -3.0% a -0.7%        (precio moderadamente abajo)
      -2 : desviación < -3.0%   (precio muy por debajo, sobreextendido)

    EMA200 en 15m actúa como pivote institucional en BTC — es la referencia
    de largo plazo que los grandes players usan para definir bias de mercado.
    Las primeras 200 velas tendrán NaN (no hay suficiente historia).
    """
    ema200     = ema(df["close"], 200)
    desviacion = (df["close"] - ema200) / ema200 * 100

    return _discretizar(desviacion, [
        ( 3.0,  2),
        ( 0.7,  1),
        (-0.7,  0),
        (-3.0, -1),
    ]).rename("e1")


# ── E2 — Velocidad (ROC5) ────────────────────────────────────────────────────

def calcular_e2(df: pd.DataFrame) -> pd.Series:
    """
    E2 mide la rapidez del movimiento de precio en las últimas 5 velas.

    ROC5 = (close[0] - close[5]) / close[5] × 100

    En 15m: 5 velas = 75 minutos. Un movimiento de ±1.2% en 75min en BTC
    es rápido — de ahí el umbral de ±1.2% para el valor extremo.
    Los mismos umbrales son válidos en 5m y 10m porque ROC es relativo.

    Umbrales:
      +2 : ROC5 > +1.2%    (movimiento rápido y decisivo al alza)
      +1 : +0.3% a +1.2%   (avance moderado con impulso)
       0 : -0.3% a +0.3%   (precio prácticamente quieto)
      -1 : -1.2% a -0.3%   (caída moderada)
      -2 : ROC5 < -1.2%    (caída rápida y decisiva)

    shift(5) usa solo datos pasados → sin lookahead bias.
    Las primeras 5 velas tendrán NaN.
    """
    roc5 = (df["close"] - df["close"].shift(5)) / df["close"].shift(5) * 100

    return _discretizar(roc5, [
        ( 1.2,  2),
        ( 0.3,  1),
        (-0.3,  0),
        (-1.2, -1),
    ]).rename("e2")


# ── E3 — Volumen con dirección ───────────────────────────────────────────────

def calcular_e3(df: pd.DataFrame, serie_e1: pd.Series | None = None) -> pd.Series:
    """
    E3 mide si el volumen valida o contradice el movimiento de precio.

    ratio = volume / SMA(volume, 20)

    El ratio bruto se mapea a un vol_score:
      ratio > 2.5  → +2   (volumen muy por encima del promedio)
      ratio > 1.3  → +1
      ratio ≥ 0.7  →  0   (volumen normal)
      ratio ≥ 0.3  → -1
      ratio < 0.3  → -2   (volumen muy bajo)

    Luego se aplica el signo de la vela:
      Vela alcista (close ≥ open) → E3 = +vol_score
        Ejemplo: ratio=2.0 en vela verde → E3=+1 (compradores activos)
      Vela bajista (close < open)  → E3 = -vol_score
        Ejemplo: ratio=0.4 en vela roja  → E3=+1 (vendedor sin convicción)

    Por qué el signo importa: volumen bajo en una vela bajista es levemente
    alcista porque el vendedor no tiene convicción. Volumen alto en una vela
    bajista es bajista porque el vendedor empuja con fuerza.

    Excepción C3 (Ojo del Huracán — capitulación):
      Si ratio > 3.0 en vela bajista Y E1 ≤ -2 → E3 = +2
      Explicación: volumen extremo en caída extrema = los últimos vendedores
      capitularon, los compradores absorbieron. Es la señal de agotamiento
      bajista. El MAESTRO define esto como señal alcista aunque la vela sea roja.

    serie_e1: pasar el E1 calculado previamente para activar la excepción C3.
    """
    ratio    = df["volume"] / sma(df["volume"], 20)
    es_alcista = (df["close"] >= df["open"])

    # vol_score en bruto (siempre positivo)
    vol_score = _discretizar(ratio, [
        (2.5,  2),
        (1.3,  1),
        (0.7,  0),
        (0.3, -1),
    ])
    # El valor -2 queda como default en _discretizar cuando ratio < 0.3,
    # pero _discretizar retorna choices[-1]-1 = -1-1 = -2. Correcto.

    # Aplicar signo de vela: alcista → positivo, bajista → negativo
    # El +0.0 evita el -0.0 de float cuando vol_score=0 en vela bajista
    e3 = vol_score.where(es_alcista, (-vol_score).replace(-0.0, 0.0))

    # Excepción C3: capitulación
    if serie_e1 is not None:
        capitulacion = (~es_alcista) & (ratio > 3.0) & (serie_e1 <= -2)
        e3 = e3.where(~capitulacion, 2.0)

    return e3.rename("e3")


# ── E4 — Momentum (RSI compuesto) ────────────────────────────────────────────

def calcular_e4(df: pd.DataFrame) -> pd.Series:
    """
    E4 combina el nivel del RSI(14) con su pendiente reciente.

    rsi_score (nivel):
      RSI > 65 → +2    RSI > 55 → +1    RSI 45-55 → 0
      RSI < 45 → -1    RSI < 35 → -2

    slope = (rsi[0] - rsi[3]) / 3   ← cambio promedio por vela en 3 velas
    slope_score (dirección):
      slope > +1.0 → +1     |slope| ≤ 1.0 → 0     slope < -1.0 → -1

    E4 = clamp(rsi_score + slope_score, -2, +2)

    Por qué sumar nivel + pendiente: un RSI en 65 subiendo rápido (E4=+2)
    es más fuerte que un RSI en 65 bajando (E4=+1). La pendiente captura
    si el momentum está acelerando o frenando.

    Ejemplo del MAESTRO:
      RSI=58 subiendo (slope=1.5) → rsi_score=+1, slope_score=+1 → E4=+2
      RSI=65 plano   (slope=0.2) → rsi_score=+2, slope_score=0  → E4=+2
    """
    rsi_vals = _rsi(df["close"], 14)
    slope    = (rsi_vals - rsi_vals.shift(3)) / 3

    rsi_score = _discretizar(rsi_vals, [
        (65,  2),
        (55,  1),
        (45,  0),
        (35, -1),
    ])

    # slope_score: solo 3 valores posibles
    slope_score = pd.Series(
        np.select(
            [slope > 1.0, slope < -1.0],
            [1,           -1],
            default=0,
        ),
        index=df.index,
        dtype=float,
    )

    return (rsi_score + slope_score).clip(-2, 2).rename("e4")


# ── E5 — Temporalidad (alineación multi-TF) ──────────────────────────────────

def _tf_score(df_tf: pd.DataFrame) -> pd.Series:
    """
    Calcula el score de alineación para un timeframe individual.

    TF_bull = EMA20 > EMA50 AND close > EMA50  → +1
    TF_bear = EMA20 < EMA50 AND close < EMA50  → -1
    neutro  = cualquier otro caso              →  0

    Por qué estos dos criterios juntos: el precio por encima del EMA50
    confirma que la vela actúa en el lado alcista del promedio. Solo
    con EMA20 > EMA50 podría ser un cruce reciente sin confirmación.
    """
    ema20 = ema(df_tf["close"], 20)
    ema50 = ema(df_tf["close"], 50)

    tf_bull = (ema20 > ema50) & (df_tf["close"] > ema50)
    tf_bear = (ema20 < ema50) & (df_tf["close"] < ema50)

    return pd.Series(
        np.select([tf_bull, tf_bear], [1, -1], default=0),
        index=df_tf.index,
        dtype=float,
    )


def calcular_e5(
    df_principal: pd.DataFrame,
    todos_tfs: Dict[str, pd.DataFrame],
) -> pd.Series:
    """
    E5 mide la alineación entre los 4 timeframes: 5m, 10m, 15m y 1h.

    Para cada TF calcula si está en modo alcista, bajista o neutro.
    Luego suma los 4 scores (-4 a +4) y los mapea a -2/+2:

      suma ≥ +3  → +2   (todos o casi todos alcistas)
      suma +1/+2 → +1
      suma  = 0  →  0
      suma -1/-2 → -1
      suma ≤ -3  → -2   (todos o casi todos bajistas)

    Manejo del alineamiento temporal (sin lookahead bias):
    Los TFs más lentos (1h) tienen velas que cierran con menos frecuencia.
    Al cierre de una vela 15m, la última vela 1h disponible es la más
    reciente que ya cerró — que es información válida en ese momento.
    Usamos reindex + ffill para llevar el score de cada TF más lento
    al índice del TF principal. Nunca usamos datos futuros.

    todos_tfs: dict con al menos las claves "5m", "10m", "15m", "1h".
    """
    TFS_REQUERIDOS = ["5m", "10m", "15m", "1h"]
    faltantes = [tf for tf in TFS_REQUERIDOS if tf not in todos_tfs]
    if faltantes:
        raise ValueError(f"Faltan timeframes para E5: {faltantes}")

    # Calcular score de alineación para cada TF
    scores_por_tf = {}
    for nombre_tf in TFS_REQUERIDOS:
        score_tf = _tf_score(todos_tfs[nombre_tf])

        # Llevar el score al índice del TF principal
        # reindex con ffill: cada vela del TF principal hereda el score
        # de la última vela cerrada del TF auxiliar
        idx_union  = score_tf.index.union(df_principal.index)
        score_alineado = (
            score_tf
            .reindex(idx_union)
            .ffill()
            .reindex(df_principal.index)
        )
        scores_por_tf[nombre_tf] = score_alineado

    # Sumar los 4 scores → rango -4 a +4
    suma = sum(scores_por_tf.values())

    return _discretizar(suma, [
        ( 2.5,  2),   # suma ≥ 3
        ( 0.5,  1),   # suma ∈ {1, 2}
        (-0.5,  0),   # suma = 0
        (-2.5, -1),   # suma ∈ {-1, -2}
    ]).rename("e5")
    # suma ≤ -3 → default = -2


# ── punto de entrada principal ───────────────────────────────────────────────

def calcular_ejes(
    df_principal: pd.DataFrame,
    todos_tfs: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Calcula los 5 ejes y el score de convergencia para cada vela del
    DataFrame principal.

    Retorna un DataFrame con columnas: e1, e2, e3, e4, e5, score.
    Los NaN al inicio son normales: E1 necesita 200 velas, E2 necesita 5,
    E4 necesita ~17 (14 de RSI + 3 de slope), E5 necesita 50 del TF más lento.

    Uso típico:
        tfs  = cargar_multi_tf("BTC/USDT")
        ejes = calcular_ejes(tfs["15m"], tfs)
        print(ejes.tail())

    df_principal: el TF sobre el que se analizan las señales (ej: 15m).
    todos_tfs:    dict con los 4 TFs — debe incluir el TF principal.
    """
    e1 = calcular_e1(df_principal)
    e2 = calcular_e2(df_principal)
    e3 = calcular_e3(df_principal, serie_e1=e1)
    e4 = calcular_e4(df_principal)
    e5 = calcular_e5(df_principal, todos_tfs)

    df_ejes = pd.DataFrame({
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "e5": e5,
    })

    df_ejes["score"] = df_ejes[["e1", "e2", "e3", "e4", "e5"]].sum(axis=1)

    return df_ejes
