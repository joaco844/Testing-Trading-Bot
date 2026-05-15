"""
strategies/tendencia_pullback.py

Sistema Tendencia + Pullback — BTC/USDT 1h

Arquitectura de dos capas que ataca el "Consenso Tardío" del sistema anterior:

  Capa 1 — Bias (4h):
    Solo operar long cuando el mercado grande es alcista.
    Condición: precio en 1h > EMA50 en 4h, Y EMA50-4h tiene pendiente positiva.
    La pendiente evita entrar en mercados donde el precio está sobre EMA50
    pero la EMA50 ya está girando a la baja.

  Capa 2 — Timing (1h):
    Dentro de una tendencia, los retrocesos a la EMA20 son oportunidades.
    Entramos cuando la vela toca EMA20 (wick baja hasta ahí) pero CIERRA
    por encima — eso es el nivel aguantando como soporte.
    El RSI < 55 asegura que entramos en la recuperación temprana,
    no cuando el rebote ya corrió.

Por qué es anticipatorio:
  No esperamos que los indicadores "confirmen" el movimiento (tarde).
  Esperamos a que el precio LLEGUE a una zona específica (EMA20) y dé
  una señal de rechazo. El movimiento grande todavía no empezó.

IMPORTANTE — Lookahead bias en multi-TF:
  La barra 4h "00:00" contiene las 1h bars 00:00, 01:00, 02:00, 03:00.
  Si mapeamos directamente, la barra 1h 01:00 "vería" datos de 02:00 y 03:00.
  Solución: shift(1) en la serie 4h antes del reindex. El costo es un lag
  de hasta 4 horas en el filtro de tendencia — irrelevante para un bias.
"""

import pandas as pd
from indicators.trend import ema as calc_ema
from indicators.volatility import atr as calc_atr
from indicators.momentum import rsi as calc_rsi


def _resamplear_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el DataFrame de 4h a partir del 1h.

    closed='left', label='left': la barra etiquetada "08:00" agrupa
    las 1h candles 08:00, 09:00, 10:00, 11:00 (cierra al abrirse 12:00).
    """
    return df_1h.resample("4h", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()


def calcular_señales(df: pd.DataFrame, atr_min_pct: float = 0.70) -> pd.DataFrame:
    """
    Recibe el DataFrame de 1h limpio (OHLCV con DatetimeIndex).
    Retorna el mismo DataFrame con columnas adicionales:
      - señal    : 1 (entrar long), 0 (sin acción). -1 no se usa; salidas via SL/TP.
      - ema20    : EMA20 en 1h (zona de valor para el pullback)
      - rsi14    : RSI14 en 1h
      - atr      : ATR14 en 1h (requerido por engine para SL/TP dinámicos)
      - atr_pct  : ATR como % del precio (filtro de régimen)
      - ema50_4h : EMA50 en 4h mapeado a 1h (para análisis y debugging)

    atr_min_pct:
        Umbral mínimo de ATR expresado como % del precio de cierre.
        Solo se generan señales cuando el mercado tiene suficiente movimiento.
        Por debajo de este umbral el mercado está en rango y los pullbacks
        a EMA20 son ruido, no señales de continuación.
        Fijado en 0.70% tras análisis en development set (fijado — no ajustar
        hasta correr holdout).
    """
    df = df.copy()

    # --- Capa 1: bias en 4h ---
    df_4h = _resamplear_4h(df)
    df_4h["ema50"] = calc_ema(df_4h["close"], 50)

    # Pendiente: EMA50 actual vs 3 barras de 4h atrás (= 12 horas).
    # True si la EMA50 está subiendo. Filtra mercados laterales donde
    # el precio está sobre EMA50 pero la tendencia ya perdió fuerza.
    df_4h["slope_pos"] = df_4h["ema50"] > df_4h["ema50"].shift(3)

    # shift(1) en la serie 4h antes de reindexar a 1h: evita lookahead.
    # En la barra 1h 04:00 usamos el EMA de la barra 4h "00:00" (ya cerrada).
    # En las barras 1h 01:00, 02:00, 03:00 usamos la barra 4h anterior a "00:00".
    ema50_en_1h = df_4h["ema50"].shift(1).reindex(df.index, method="ffill")
    slope_en_1h = df_4h["slope_pos"].shift(1).reindex(df.index, method="ffill")

    # --- Capa 2: indicadores en 1h ---
    df["ema20"]    = calc_ema(df["close"], 20)
    df["rsi14"]    = calc_rsi(df["close"], 14)
    df["atr"]      = calc_atr(df, 14)
    df["atr_pct"]  = df["atr"] / df["close"] * 100
    df["ema50_4h"] = ema50_en_1h

    # --- Condiciones de entrada (todas deben ser True) ---

    # 1. Bias alcista: precio 1h sobre EMA50-4h y la EMA50-4h está subiendo
    bias_alcista = (df["close"] > ema50_en_1h) & slope_en_1h.fillna(False)

    # 2. Pullback: el wick de la vela bajó hasta EMA20 o más abajo
    toco_ema20 = df["low"] <= df["ema20"]

    # 3. Rebote: la vela CERRÓ sobre EMA20 — el nivel aguantó como soporte
    cerro_sobre_ema20 = df["close"] > df["ema20"]

    # 4. No sobrecomprado: entramos en la recuperación temprana del pullback
    no_sobrecomprado = df["rsi14"] < 55

    # 5. Régimen: solo operar cuando el mercado tiene suficiente movimiento.
    # ATR bajo = mercado en rango = pullbacks a EMA20 son ruido, no señales.
    mercado_activo = df["atr_pct"] >= atr_min_pct

    señal_entrada = (bias_alcista & toco_ema20 & cerro_sobre_ema20
                     & no_sobrecomprado & mercado_activo)

    df["señal"] = 0
    df.loc[señal_entrada, "señal"] = 1

    return df


class TendenciaPullback:
    """Wrapper de clase para compatibilidad con el framework."""
    nombre = "Tendencia + Pullback (EMA50-4h / EMA20-1h)"

    def __init__(self, atr_min_pct: float = 0.70):
        self.atr_min_pct = atr_min_pct

    def calcular_señales(self, df: pd.DataFrame) -> pd.DataFrame:
        return calcular_señales(df, atr_min_pct=self.atr_min_pct)
