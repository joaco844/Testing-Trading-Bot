"""
strategies/ema_crossover.py

Estrategia: EMA Crossover (Cruce de Medias Móviles Exponenciales)

Lógica:
  - Compramos cuando la EMA rápida cruza POR ARRIBA a la EMA lenta
    → el precio de corto plazo supera al de largo plazo: tendencia alcista
  - Vendemos cuando la EMA rápida cruza POR ABAJO a la EMA lenta
    → el precio de corto plazo cae por debajo del de largo plazo: tendencia bajista

Es una estrategia de seguimiento de tendencia (trend following).
Funciona bien en mercados con tendencia clara, mal en mercados laterales
(genera muchas señales falsas = "whipsaws").

Parámetros por defecto: EMA9 y EMA21 (populares en cripto por sesgo de 24h).
"""

import pandas as pd
from strategies.base import EstrategiaBase
from indicators.trend import ema
from indicators.volatility import atr


class EMACrossover(EstrategiaBase):

    def __init__(self, periodo_rapida: int = 9, periodo_lenta: int = 21):
        super().__init__(f"EMA_Crossover_{periodo_rapida}_{periodo_lenta}")
        self.periodo_rapida = periodo_rapida
        self.periodo_lenta = periodo_lenta

    def calcular_señales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genera señales de compra/venta por cruce de EMAs.

        Paso a paso:
        1. Calculamos EMA rápida y EMA lenta
        2. Detectamos cuándo una supera a la otra (cruce)
        3. Generamos señal=1 en el cruce alcista, señal=-1 en el bajista

        Anti-lookahead bias:
        Usamos .shift(1) para comparar la relación de AYER con la de HOY.
        Así la señal se genera cuando el cruce ya ocurrió, no antes.
        """
        df = df.copy()

        df["ema_rapida"] = ema(df["close"], self.periodo_rapida)
        df["ema_lenta"]  = ema(df["close"], self.periodo_lenta)
        df["atr"]        = atr(df, periodo=14)

        # ¿Está la EMA rápida por encima de la lenta en esta vela?
        rapida_arriba_hoy  = df["ema_rapida"] > df["ema_lenta"]

        # ¿Estaba por encima en la vela ANTERIOR? (shift(1) = un paso atrás)
        # fillna(False): la primera fila queda NaN tras el shift — la tratamos
        # como "no estaba arriba" para que no genere señal falsa.
        rapida_arriba_ayer = rapida_arriba_hoy.shift(1).fillna(False).astype(bool)

        # Cruce alcista: ayer estaba abajo, hoy está arriba
        cruce_alcista = rapida_arriba_hoy & ~rapida_arriba_ayer

        # Cruce bajista: ayer estaba arriba, hoy está abajo
        cruce_bajista = ~rapida_arriba_hoy & rapida_arriba_ayer

        df["señal"] = 0
        df.loc[cruce_alcista, "señal"] = 1
        df.loc[cruce_bajista, "señal"] = -1

        return df
