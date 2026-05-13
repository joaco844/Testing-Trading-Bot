"""
strategies/rsi_mean_reversion.py

Estrategia: RSI Mean Reversion con filtro de tendencia opcional.

Lógica base:
  - Compramos cuando RSI sale de sobrevendido (< umbral_bajo)
  - Vendemos cuando RSI sale de sobrecomprado (> umbral_alto)

Filtro de tendencia (EMA de largo plazo):
  El problema sin filtro: en tendencia bajista el RSI marca
  sobrevendido constantemente y compramos en caída libre.

  Con filtro: solo compramos si precio > EMA_largo.
  Esto asegura que operamos a favor de la tendencia principal.
  Si el precio está debajo de la EMA200, el mercado está en
  tendencia bajista y no queremos comprar dips — son trampas.

  Por qué EMA200:
    La EMA200 es el indicador de tendencia de largo plazo más
    usado en todos los mercados. En cripto, separa bien los
    períodos bull (precio encima) de los bear (precio debajo).
    No es un número mágico — es una convención ampliamente
    adoptada que tiene sentido económico.
"""

import pandas as pd
from strategies.base import EstrategiaBase
from indicators.momentum import rsi
from indicators.volatility import atr
from indicators.trend import ema


class RSIMeanReversion(EstrategiaBase):

    def __init__(
        self,
        periodo: int = 14,
        umbral_bajo: float = 30.0,
        umbral_alto: float = 70.0,
        ema_tendencia: int = None,   # None = sin filtro | 200 = solo comprar sobre EMA200
    ):
        nombre = f"RSI_{periodo}_{umbral_bajo}_{umbral_alto}"
        if ema_tendencia:
            nombre += f"_EMA{ema_tendencia}filter"
        super().__init__(nombre)

        self.periodo       = periodo
        self.umbral_bajo   = umbral_bajo
        self.umbral_alto   = umbral_alto
        self.ema_tendencia = ema_tendencia

    def calcular_señales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales con filtro de tendencia opcional.

        Sin filtro:   señal=1 cada vez que RSI sale de sobrevendido
        Con filtro:   señal=1 solo si además precio > EMA_tendencia

        Anti-lookahead bias:
          La EMA se calcula sobre datos hasta el momento t.
          El filtro precio > EMA usa los valores del momento actual,
          no del futuro. No hay bias.
        """
        df = df.copy()

        df["rsi"] = rsi(df["close"], self.periodo)
        df["atr"] = atr(df, periodo=14)

        # Filtro de tendencia — calculado siempre, aplicado solo si se configuró
        if self.ema_tendencia:
            df["ema_tendencia"] = ema(df["close"], self.ema_tendencia)
            tendencia_alcista = df["close"] > df["ema_tendencia"]
        else:
            tendencia_alcista = pd.Series(True, index=df.index)  # sin filtro = siempre True

        rsi_bajo_hoy  = df["rsi"] < self.umbral_bajo
        rsi_bajo_ayer = rsi_bajo_hoy.shift(1).fillna(False).astype(bool)

        rsi_alto_hoy  = df["rsi"] > self.umbral_alto
        rsi_alto_ayer = rsi_alto_hoy.shift(1).fillna(False).astype(bool)

        cruce_salida_sobrevendido  = ~rsi_bajo_hoy & rsi_bajo_ayer
        cruce_salida_sobrecomprado = ~rsi_alto_hoy & rsi_alto_ayer

        df["señal"] = 0

        # Compra: RSI sale de sobrevendido Y estamos en tendencia alcista
        df.loc[cruce_salida_sobrevendido & tendencia_alcista, "señal"] = 1

        # Venta: RSI sale de sobrecomprado (sin filtro de tendencia —
        # siempre cerramos si el RSI indica sobrecomprado)
        df.loc[cruce_salida_sobrecomprado, "señal"] = -1

        return df
