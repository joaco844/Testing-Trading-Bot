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
