"""
strategies/trend_following_diario.py

Sistema Trend Following Diario — BTC/USDT

Filosofía: entrar con criterio sólido y permanecer en la tendencia el mayor
tiempo posible dentro de la sesión. No busca timing perfecto — busca participar
de las tendencias diarias de BTC con gestión activa de parciales.

Arquitectura de tres capas:

  Capa 1 — Confirmación de tendencia (2 de 3 requeridas):
    - Estructura semanal: máximos y mínimos semanales crecientes
    - EMA200 diaria: precio por encima para largos
    - ADX(14) diario ≥ 30: tendencia genuina, no lateral

  Capa 2 — Calidad de la vela del día anterior (ambas requeridas):
    - Cuerpo de la vela ≥ 0.5 × ATR(14) diario: recorrido suficiente
    - Volumen del día anterior > media de 20 días: participación real

  Capa 3 — Timing (intradía):
    - Entrada al 50% del cuerpo de la vela anterior (retroceso al nivel de valor)
    - SL justo debajo del piso del cuerpo (open de la vela anterior en tendencia alcista)
    - Parciales en prev_day_high y swing highs recientes por encima

Por qué el umbral ADX es 30 y no 25:
  BTC tiene volatilidad estructural alta. El ADX supera 25 incluso en fases
  laterales volátiles. 30 asegura que existe tendencia genuina, no solo ruido.
"""

import pandas as pd
import numpy as np
from indicators.trend import ema as calc_ema, adx as calc_adx
from indicators.volatility import atr as calc_atr


def _swing_highs(serie_high: pd.Series, ventana: int = 3) -> pd.Series:
    """
    Detecta máximos locales (swing highs) en una serie de highs.

    Un bar es swing high si su high es el mayor en una ventana de
    ±ventana barras. Retorna una Serie con los valores de los swing highs
    y NaN en el resto.

    ventana=3 significa que el high debe ser el mayor en las 3 barras
    anteriores y las 3 siguientes. Para datos diarios, esto detecta
    máximos locales de ~1 semana de alcance.
    """
    rolling_max = serie_high.rolling(window=2 * ventana + 1, center=True).max()
    es_swing    = serie_high == rolling_max
    return serie_high.where(es_swing)


def preparar_datos_diarios(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el DataFrame diario a partir de los datos 1h.

    Retorna un DataFrame con índice de fecha (UTC) y columnas:
      - open, high, low, close, volume (OHLCV diario)
      - ema200      : EMA200 sobre el close diario
      - adx, di_pos, di_neg : ADX(14) sobre datos diarios
      - atr14       : ATR(14) sobre datos diarios
      - vol_ma20    : media de 20 días del volumen
      - swing_high  : swing high local (NaN si no es swing high)

    La lógica de resampling para el daily:
      closed='left', label='left' significa que la barra diaria "2024-01-01"
      agrupa las 1h candles desde 00:00 hasta 23:00 de ese día (24 candles).
    """
    df_d = df_1h.resample("1D", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    df_d["ema200"]   = calc_ema(df_d["close"], 200)
    df_d["atr14"]    = calc_atr(df_d, 14)
    df_d["vol_ma20"] = df_d["volume"].rolling(20).mean()

    adx_df          = calc_adx(df_d, 14)
    df_d["adx"]     = adx_df["adx"]
    df_d["di_pos"]  = adx_df["di_pos"]
    df_d["di_neg"]  = adx_df["di_neg"]

    df_d["swing_high"] = _swing_highs(df_d["high"], ventana=3)

    return df_d


def _estructura_semanal_alcista(df_diario: pd.DataFrame, fecha: pd.Timestamp) -> bool:
    """
    Verifica si la estructura semanal es alcista en la fecha dada.

    Resamplea las últimas semanas disponibles hasta `fecha` y compara:
    - High de la semana más reciente completa > high de 3 semanas antes
    - Low de la semana más reciente completa > low de 3 semanas antes

    Si ambas condiciones se cumplen → estructura semanal alcista.
    Necesitamos al menos 4 semanas de historia para esta comparación.

    Por qué "2 de 3" en lugar de exigir todos los highs/lows:
      Los mercados no forman tendencias perfectas. Exigir que cada semana
      sea estrictamente mayor a la anterior filtraría demasiado — incluso
      las tendencias alcistas más fuertes tienen semanas de corrección.
      Comparar la semana reciente con la de hace 3 semanas capta la
      dirección general sin sensibilidad excesiva al ruido semanal.
    """
    datos_previos = df_diario.loc[:fecha]
    if len(datos_previos) < 28:   # necesitamos al menos 4 semanas
        return False

    semanal = datos_previos.resample("1W").agg({"high": "max", "low": "min"}).dropna()
    # Excluir la semana actual (puede estar incompleta) → usar solo semanas completas
    semanal_completas = semanal.iloc[:-1]

    if len(semanal_completas) < 4:
        return False

    semana_reciente = semanal_completas.iloc[-1]
    semana_hace_3   = semanal_completas.iloc[-4]

    high_creciente = semana_reciente["high"] > semana_hace_3["high"]
    low_creciente  = semana_reciente["low"]  > semana_hace_3["low"]

    return high_creciente and low_creciente


def calcular_señales_diarias(df_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Evalúa las condiciones de entrada para cada día y genera el DataFrame
    de señales que consume el engine_parciales.

    Columnas generadas:
      - señal       : 1 si se cumplen las condiciones, 0 si no
      - entry_level : precio de entrada (50% del cuerpo de la vela anterior)
      - sl_precio   : precio de stop loss (por debajo del piso del cuerpo)
      - niveles     : lista de precios objetivo para cierres parciales
      - conf_score  : cuántas de las 3 confirmaciones de tendencia se cumplen (0-3)
    """
    df = df_diario.copy()

    df["señal"]       = 0
    df["entry_level"] = np.nan
    df["sl_precio"]   = np.nan
    df["niveles"]     = [[] for _ in range(len(df))]
    df["conf_score"]  = 0

    for i in range(1, len(df)):
        hoy     = df.index[i]
        ayer    = df.index[i - 1]
        fila_ayer = df.iloc[i - 1]
        fila_hoy  = df.iloc[i]

        # --- Calidad de la vela anterior ---
        cuerpo = abs(fila_ayer["close"] - fila_ayer["open"])
        if pd.isna(fila_ayer["atr14"]) or fila_ayer["atr14"] == 0:
            continue
        cuerpo_ok  = cuerpo >= 0.5 * fila_ayer["atr14"]
        volumen_ok = (
            not pd.isna(fila_ayer["vol_ma20"]) and
            fila_ayer["volume"] > fila_ayer["vol_ma20"]
        )
        if not cuerpo_ok or not volumen_ok:
            continue

        # Solo operar en días alcistas (close > open de la vela anterior)
        # Los cortos se pueden agregar después; por ahora long only.
        if fila_ayer["close"] <= fila_ayer["open"]:
            continue

        # --- Confirmaciones de tendencia (necesitamos ≥ 2 de 3) ---
        # IMPORTANTE: usamos fila_ayer para EMA200 y ADX — son los valores
        # conocidos al inicio de la sesión de hoy (00:00 UTC). Usar fila_hoy
        # sería lookahead porque el close de hoy no se conoce hasta las 23:59.
        confirmaciones = 0

        # 1. Close de ayer sobre EMA200 de ayer
        if not pd.isna(fila_ayer["ema200"]) and fila_ayer["close"] > fila_ayer["ema200"]:
            confirmaciones += 1

        # 2. ADX ≥ 30 de ayer — tendencia fuerte confirmada al cierre de ayer
        if not pd.isna(fila_ayer["adx"]) and fila_ayer["adx"] >= 30:
            confirmaciones += 1

        # 3. Estructura semanal alcista — se evalúa hasta ayer (inclusive)
        if _estructura_semanal_alcista(df, ayer):
            confirmaciones += 1

        df.at[hoy, "conf_score"] = confirmaciones
        if confirmaciones < 2:
            continue

        # --- Niveles del trade ---
        # Entrada al 50% del cuerpo de la vela anterior
        cuerpo_piso = fila_ayer["open"]   # open es el piso en vela alcista
        cuerpo_techo = fila_ayer["close"]
        entry_level  = (cuerpo_piso + cuerpo_techo) / 2.0

        # SL justo debajo del piso del cuerpo (0.1% de buffer)
        sl_precio = cuerpo_piso * 0.999

        # Niveles para parciales:
        # 1. High de la vela anterior (primer objetivo natural)
        # 2. Swing highs recientes (últimos 10 días) por encima del prev_high
        niveles = []
        if fila_ayer["high"] > entry_level:
            niveles.append(fila_ayer["high"])

        # Swing highs de los últimos 10 días hábiles por encima de prev_high
        ventana_swings = df.iloc[max(0, i - 10):i]
        swings_relevantes = ventana_swings["swing_high"].dropna()
        swings_sobre_prev = sorted([
            s for s in swings_relevantes
            if s > fila_ayer["high"]
        ])
        niveles.extend(swings_sobre_prev[:2])   # máximo 2 swing highs adicionales

        niveles = sorted(set(niveles))   # ordenados, sin duplicados

        if not niveles:
            continue

        df.at[hoy, "señal"]       = 1
        df.at[hoy, "entry_level"] = entry_level
        df.at[hoy, "sl_precio"]   = sl_precio
        df.at[hoy, "niveles"]     = niveles

    return df


class TrendFollowingDiario:
    """Wrapper de clase para compatibilidad con el framework."""
    nombre = "Trend Following Diario (EMA200 / ADX30 / Estructura Semanal)"

    def preparar_datos(self, df_1h: pd.DataFrame) -> tuple:
        """
        Retorna (df_diario_con_señales, df_1h_limpio).
        El engine_parciales necesita ambos.
        """
        df_diario  = preparar_datos_diarios(df_1h)
        df_señales = calcular_señales_diarias(df_diario)
        return df_señales, df_1h
