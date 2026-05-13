"""
indicators/momentum.py

Indicadores de momentum: RSI.

El momentum mide la velocidad y fuerza de los movimientos de precio.
Los indicadores de momentum ayudan a detectar si un activo está
sobrecomprado (posible reversión bajista) o sobrevendido (posible rebote).
"""

import pandas as pd
import numpy as np


def rsi(serie: pd.Series, periodo: int = 14) -> pd.Series:
    """
    RSI — Relative Strength Index (Índice de Fuerza Relativa).

    Rango: 0 a 100.
      - RSI > 70 → sobrecomprado (el precio subió demasiado rápido)
      - RSI < 30 → sobrevendido (el precio bajó demasiado rápido)
      - RSI = 50 → zona neutral

    Cómo se calcula:
      1. Calcular el cambio de precio en cada vela (delta)
      2. Separar ganancias (subidas) de pérdidas (bajadas)
      3. Calcular la media exponencial de ganancias y pérdidas
      4. RS = media_ganancias / media_perdidas
      5. RSI = 100 - (100 / (1 + RS))

    Por qué ewm en lugar de rolling:
      Wilder (creador del RSI) usó una media exponencial específica
      con alpha = 1/periodo, que en pandas se logra con
      ewm(alpha=1/periodo, adjust=False).
      Usar rolling().mean() daría valores ligeramente distintos
      a la fórmula original de Wilder.

    Las primeras `periodo` filas tendrán NaN.
    """
    delta = serie.diff()  # cambio precio a precio

    ganancias = delta.clip(lower=0)   # solo subidas, bajadas = 0
    perdidas = (-delta).clip(lower=0) # solo bajadas (positivas), subidas = 0

    media_ganancias = ganancias.ewm(alpha=1 / periodo, adjust=False).mean()
    media_perdidas = perdidas.ewm(alpha=1 / periodo, adjust=False).mean()

    # Evitar división por cero: si no hubo pérdidas, RS → infinito → RSI = 100
    rs = media_ganancias / media_perdidas.replace(0, np.nan)

    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values
