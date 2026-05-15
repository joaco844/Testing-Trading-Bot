"""
indicators/trend.py

Indicadores de tendencia: EMA, SMA, MACD.

Decisión de diseño: cada función recibe un DataFrame y retorna
una Serie de pandas. No modificamos el DataFrame original —
el caller decide si quiere asignarlo como columna o no.
Esto evita efectos secundarios inesperados.

IMPORTANTE — Lookahead bias:
Estos indicadores usan solo datos pasados (rolling hacia atrás),
lo cual es correcto. El error de lookahead bias ocurriría si
usáramos shift(-1) o datos futuros. Acá no lo hacemos.
"""

import pandas as pd
import numpy as np


def ema(serie: pd.Series, periodo: int) -> pd.Series:
    """
    Exponential Moving Average (Media Móvil Exponencial).

    A diferencia de la SMA, la EMA da más peso a los precios recientes.
    El peso decae exponencialmente: el precio más reciente pesa más
    que el de hace 2 períodos, que a su vez pesa más que el de hace 3, etc.

    adjust=False usa la fórmula recursiva estándar:
        EMA_t = precio_t * alpha + EMA_(t-1) * (1 - alpha)
        donde alpha = 2 / (periodo + 1)

    Las primeras `periodo` filas tendrán NaN porque no hay suficiente
    historia para calcular la EMA — esto es correcto y esperado.
    """
    return serie.ewm(span=periodo, adjust=False).mean()


def sma(serie: pd.Series, periodo: int) -> pd.Series:
    """
    Simple Moving Average (Media Móvil Simple).

    Promedio aritmético de los últimos `periodo` valores.
    Más lenta para reaccionar a cambios que la EMA.
    Útil como referencia o para detectar tendencias de largo plazo.

    Las primeras `periodo - 1` filas tendrán NaN.
    """
    return serie.rolling(window=periodo).mean()


def macd(
    serie: pd.Series,
    rapida: int = 12,
    lenta: int = 26,
    señal: int = 9,
) -> pd.DataFrame:
    """
    MACD — Moving Average Convergence Divergence.

    Uno de los indicadores de momentum más usados. Mide la diferencia
    entre dos EMAs para detectar cambios en la fuerza y dirección de tendencia.

    Componentes:
      - macd_line  : EMA rápida - EMA lenta (ej: EMA12 - EMA26)
      - signal_line: EMA de la macd_line (ej: EMA9 de macd_line)
      - histogram  : macd_line - signal_line

    Cómo interpretarlo:
      - Cuando macd_line cruza signal_line hacia arriba → señal alcista
      - Cuando macd_line cruza signal_line hacia abajo → señal bajista
      - El histograma muestra la fuerza de la divergencia

    Retorna un DataFrame con tres columnas para que el caller
    pueda usar las tres o solo las que necesite.
    """
    ema_rapida = ema(serie, rapida)
    ema_lenta = ema(serie, lenta)

    macd_line = ema_rapida - ema_lenta
    signal_line = ema(macd_line, señal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=serie.index)


def adx(df: pd.DataFrame, periodo: int = 14) -> pd.DataFrame:
    """
    ADX — Average Directional Index (Índice Direccional Promedio).

    Mide la FUERZA de la tendencia, no su dirección.
    Rango: 0 a 100.
      - ADX > 25 → tendencia presente (estándar)
      - ADX > 30 → tendencia fuerte (umbral que usa este sistema por la
                   volatilidad estructural de BTC, que genera ADX > 25
                   incluso en fases laterales)
      - ADX < 20 → mercado sin dirección, lateral

    Componentes retornados:
      - adx   : fuerza de tendencia (sin dirección)
      - di_pos: +DI, presión compradora
      - di_neg: -DI, presión vendedora
      - Cuando +DI > -DI → tendencia alcista. Cuando -DI > +DI → bajista.

    Cálculo (método de Wilder):
      1. True Range (igual que ATR)
      2. +DM = max(high - prev_high, 0) si > |low - prev_low|, sino 0
         -DM = max(prev_low - low, 0) si > |high - prev_high|, sino 0
      3. Suavizado Wilder (EWM alpha=1/periodo) sobre TR, +DM, -DM
      4. +DI = 100 * (+DM_smooth / TR_smooth)
         -DI = 100 * (-DM_smooth / TR_smooth)
      5. DX  = 100 * |+DI - -DI| / (+DI + -DI)
      6. ADX = suavizado Wilder de DX
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high - prev_high
    down_move = prev_low - low

    dm_pos = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_neg = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    dm_pos = pd.Series(dm_pos, index=df.index)
    dm_neg = pd.Series(dm_neg, index=df.index)

    alpha = 1 / periodo
    tr_smooth  = tr.ewm(alpha=alpha, adjust=False).mean()
    dmp_smooth = dm_pos.ewm(alpha=alpha, adjust=False).mean()
    dmn_smooth = dm_neg.ewm(alpha=alpha, adjust=False).mean()

    di_pos = 100 * dmp_smooth / tr_smooth
    di_neg = 100 * dmn_smooth / tr_smooth

    dx_denom = (di_pos + di_neg).replace(0, np.nan)
    dx  = 100 * (di_pos - di_neg).abs() / dx_denom
    adx_values = dx.ewm(alpha=alpha, adjust=False).mean()

    return pd.DataFrame({
        "adx":    adx_values,
        "di_pos": di_pos,
        "di_neg": di_neg,
    }, index=df.index)
