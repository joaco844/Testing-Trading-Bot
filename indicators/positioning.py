"""
indicators/positioning.py

Indicadores de posicionamiento de mercado — datos independientes del precio.

E6 — Funding Rate:
    En futuros perpetuos, cada 8h se produce un pago entre longs y shorts
    para anclar el precio perpetuo al spot. Si hay más longs que shorts,
    los longs pagan a los shorts (funding positivo). Si hay más shorts,
    los shorts pagan (funding negativo).

    Por qué es útil:
      Cuando el funding está muy positivo, el mercado está congestionado
      de longs. Cualquier bajada produce liquidaciones en cascada. Entrar
      long en ese contexto es comprar en la cima del entusiasmo.
      Cuando el funding es negativo, los shorts están pagando — el mercado
      espera una baja que no llega, lo que puede generar un short squeeze.

    Valores discretos:
      +2 : funding muy negativo → shorts pagando fuerte, favorece longs
      +1 : funding levemente negativo → leve edge para longs
       0 : funding neutral (±0.02%)
      -1 : funding elevado → demasiados longs, mercado en riesgo
      -2 : funding extremo → NO entrar long, riesgo de squeeze bajista

    Umbrales (por evento de 8h en BTC perpetuos):
      Normal bull market   : +0.005% a +0.015% → neutro
      Mercado overheated   : > +0.05%          → -1
      Euforia extrema      : > +0.10%          → -2
      Shorts pagando       : < -0.01%          → +1
      Short squeeze activo : < -0.03%          → +2
"""

import numpy as np
import pandas as pd


def calcular_e6(
    df_principal: pd.DataFrame,
    df_funding: pd.DataFrame,
) -> pd.Series:
    """
    Calcula E6 (posicionamiento via Funding Rate) alineado al TF principal.

    El funding ocurre cada 8h → se propaga con ffill al TF principal.
    Esta propagación es correcta: al cierre de una vela de 1h, el último
    funding rate publicado es información ya disponible (sin lookahead bias).

    df_principal : DataFrame del TF de análisis (ej: 1h)
    df_funding   : DataFrame con columna 'funding_rate' (índice = timestamp)
    """
    # Alinear funding rate al índice del TF principal
    idx_union = df_funding.index.union(df_principal.index)
    funding = (
        df_funding["funding_rate"]
        .reindex(idx_union)
        .ffill()
        .reindex(df_principal.index)
    )

    # Discretizar en 5 niveles — thresholds de mayor a menor
    # np.select evalúa condiciones en orden, primera que se cumple gana
    conditions = [
        funding > 0.00100,   # > +0.10% → euforia extrema
        funding > 0.00050,   # > +0.05% → mercado sobrecargado de longs
        funding > -0.00010,  # > -0.01% → zona neutral
        funding > -0.00030,  # > -0.03% → shorts pagando levemente
    ]
    choices = [-2, -1, 0, +1]

    return pd.Series(
        np.select(conditions, choices, default=2),  # ≤ -0.03% → +2
        index=df_principal.index,
        dtype=float,
    ).rename("e6")
