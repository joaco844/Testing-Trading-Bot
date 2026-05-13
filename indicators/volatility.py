"""
indicators/volatility.py

Indicadores de volatilidad: Bollinger Bands, ATR.

La volatilidad mide cuánto se mueve el precio. Es clave para:
  - Dimensionar posiciones (cuánto arriesgar en cada trade)
  - Detectar rangos de precio esperados
  - Identificar períodos de calma antes de grandes movimientos
"""

import pandas as pd
import numpy as np


def bollinger_bands(
    serie: pd.Series,
    periodo: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands (Bandas de Bollinger).

    Componentes:
      - media  : SMA de `periodo` períodos (banda central)
      - banda_superior : media + num_std * desvío estándar
      - banda_inferior : media - num_std * desvío estándar
      - ancho  : (banda_superior - banda_inferior) / media (volatilidad relativa)
      - %b     : posición del precio dentro de las bandas (0=inferior, 1=superior)

    Cómo interpretarlas:
      - El precio tiende a rebotar entre las bandas (reversión a la media)
      - Bandas muy juntas (squeeze) → baja volatilidad → posible explosión inminente
      - Precio tocando banda superior → sobrecomprado en el contexto de las bandas
      - Precio tocando banda inferior → sobrevendido en el contexto de las bandas

    num_std=2.0 es el estándar: cubre ~95% de los precios en distribución normal.
    """
    media = serie.rolling(window=periodo).mean()
    std = serie.rolling(window=periodo).std()

    banda_superior = media + (num_std * std)
    banda_inferior = media - (num_std * std)
    ancho = (banda_superior - banda_inferior) / media
    pct_b = (serie - banda_inferior) / (banda_superior - banda_inferior)

    return pd.DataFrame({
        "bb_media": media,
        "bb_superior": banda_superior,
        "bb_inferior": banda_inferior,
        "bb_ancho": ancho,
        "bb_pct_b": pct_b,
    }, index=serie.index)


def atr(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    """
    ATR — Average True Range (Rango Verdadero Promedio).

    Mide la volatilidad del mercado en términos absolutos de precio.
    No indica dirección, solo magnitud del movimiento.

    El True Range de cada vela es el mayor de:
      1. high - low (rango de la vela actual)
      2. |high - close_anterior| (gap alcista entre velas)
      3. |low - close_anterior|  (gap bajista entre velas)

    Se usa para:
      - Calcular stop-loss dinámicos (ej: stop = precio - 2 * ATR)
      - Dimensionar posiciones: arriesgar X% del capital por cada ATR de movimiento
      - Comparar volatilidad entre distintos activos o períodos

    Retorna el ATR suavizado con EWM (método de Wilder, alpha=1/periodo).
    """
    high = df["high"]
    low = df["low"]
    close_anterior = df["close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - close_anterior).abs(),
        (low - close_anterior).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / periodo, adjust=False).mean()
