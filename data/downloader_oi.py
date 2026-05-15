"""
data/downloader_oi.py

Descarga datos de posicionamiento del mercado desde Binance Futures:

  Open Interest (OI):
    Total de contratos abiertos en futuros perpetuos.
    Un OI creciente junto con precio subiendo = nuevos longs abriendo
    (convicción real). OI cayendo con precio subiendo = short covering
    (movimiento débil, sin nueva convicción).

  Funding Rate:
    En futuros perpetuos, cada 8h se produce un pago entre longs y shorts
    para anclar el precio al spot. Si el funding es muy positivo, los longs
    están pagando a los shorts — significa demasiados longs, mercado
    vulnerable a squeeze bajista. Información de posicionamiento puro.

Ambos endpoints son públicos — no requieren API key.
"""

import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_FUTURES = "https://fapi.binance.com"
CARPETA_DATOS   = Path(__file__).parent


# ── Open Interest ────────────────────────────────────────────────────────────

def descargar_open_interest(
    simbolo: str = "BTCUSDT",
    period:  str = "1h",
    meses:   int = 24,
) -> pd.DataFrame:
    """
    Descarga el Open Interest histórico de Binance Futures.

    Endpoint: GET /futures/data/openInterestHist
    Máximo 500 registros por request — paginamos hacia adelante.

    Retorna un DataFrame con columnas:
      oi       : Open Interest en BTC (número de contratos)
      oi_usd   : Open Interest en USDT (valor nocional)

    period soportados: "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"
    """
    # Calcular cantidad total de registros según el period
    minutos_por_periodo = {
        "5m": 5, "15m": 15, "30m": 30, "1h": 60,
        "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }
    if period not in minutos_por_periodo:
        raise ValueError(f"Period '{period}' no soportado. Opciones: {list(minutos_por_periodo)}")

    mins     = minutos_por_periodo[period]
    # Binance solo tiene OI histórico de los últimos ~30 días.
    # Capamos a 30 días para evitar el error 400.
    meses_cap = min(meses, 1)
    total     = int(meses_cap * 30 * 24 * 60 / mins)
    ahora_ms  = int(time.time() * 1000)
    since_ms  = ahora_ms - total * mins * 60 * 1000

    print(f"[Binance Futures] Descargando OI {period} para {simbolo}...")
    print(f"  Desde: {pd.to_datetime(since_ms, unit='ms', utc=True).strftime('%Y-%m-%d')}")
    print(f"  Total registros estimados: {total:,}")

    todos = []
    start = since_ms

    while True:
        resp = requests.get(
            f"{BINANCE_FUTURES}/futures/data/openInterestHist",
            params={
                "symbol":    simbolo,
                "period":    period,
                "limit":     500,
                "startTime": start,
            },
            timeout=10,
        )
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        todos.extend(batch)
        ultimo_ts = batch[-1]["timestamp"]
        print(f"  Acumulado: {len(todos):,} | hasta: "
              f"{pd.to_datetime(ultimo_ts, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M')}")

        if len(batch) < 500 or ultimo_ts >= ahora_ms:
            break

        start = ultimo_ts + mins * 60 * 1000
        time.sleep(0.3)

    if not todos:
        raise ValueError(f"No se recibieron datos de OI para {simbolo}")

    df = pd.DataFrame(todos)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df.rename(columns={
        "sumOpenInterest":      "oi",
        "sumOpenInterestValue": "oi_usd",
    })[["oi", "oi_usd"]]
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    print(f"OK — {len(df):,} registros de OI descargados.")
    return df


# ── Funding Rate ─────────────────────────────────────────────────────────────

def descargar_funding_rate(
    simbolo: str = "BTCUSDT",
    meses:   int = 24,
) -> pd.DataFrame:
    """
    Descarga el historial de Funding Rate de Binance Futures.

    Endpoint: GET /fapi/v1/fundingRate
    Máximo 1000 registros por request. El funding ocurre cada 8h → 3/día.
    24 meses × 3/día × 30 días = ~2,160 registros (3 requests).

    Retorna un DataFrame con columna:
      funding_rate : valor del funding rate (ej: 0.0001 = 0.01%)

    Interpretación:
      > +0.05%  → demasiados longs, mercado caro para comprar
        +0.01%  → leve bias alcista, normal
        ≈  0    → equilibrio
       -0.01%  → leve bias bajista
      < -0.02%  → demasiados shorts, potencial long squeeze
    """
    ahora_ms = int(time.time() * 1000)
    since_ms = ahora_ms - int(meses * 30 * 24 * 3600 * 1000)

    print(f"[Binance Futures] Descargando Funding Rate para {simbolo}...")

    todos = []
    start = since_ms

    while True:
        resp = requests.get(
            f"{BINANCE_FUTURES}/fapi/v1/fundingRate",
            params={
                "symbol":    simbolo,
                "startTime": start,
                "limit":     1000,
            },
            timeout=10,
        )
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        todos.extend(batch)
        ultimo_ts = batch[-1]["fundingTime"]
        print(f"  Acumulado: {len(todos):,} | hasta: "
              f"{pd.to_datetime(ultimo_ts, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M')}")

        if len(batch) < 1000 or ultimo_ts >= ahora_ms:
            break

        start = ultimo_ts + 1
        time.sleep(0.3)

    if not todos:
        raise ValueError(f"No se recibieron datos de funding rate para {simbolo}")

    df = pd.DataFrame(todos)
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[["fundingRate"]].rename(columns={"fundingRate": "funding_rate"})
    df = df.astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()

    print(f"OK — {len(df):,} registros de funding rate descargados.")
    return df


# ── Guardar / cargar ─────────────────────────────────────────────────────────

def guardar_oi(df: pd.DataFrame, simbolo: str, period: str) -> Path:
    nombre = f"{simbolo}_OI_{period}.csv"
    ruta   = CARPETA_DATOS / nombre
    df.to_csv(ruta)
    print(f"OI guardado en: {ruta}")
    return ruta


def guardar_funding(df: pd.DataFrame, simbolo: str) -> Path:
    nombre = f"{simbolo}_funding.csv"
    ruta   = CARPETA_DATOS / nombre
    df.to_csv(ruta)
    print(f"Funding rate guardado en: {ruta}")
    return ruta


def cargar_oi(simbolo: str, period: str) -> pd.DataFrame:
    ruta = CARPETA_DATOS / f"{simbolo}_OI_{period}.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Corré primero descargar_open_interest()."
        )
    df = pd.read_csv(ruta, index_col="timestamp", parse_dates=True)
    df.index = pd.to_datetime(df.index, format="ISO8601", utc=True)
    return df


def cargar_funding(simbolo: str) -> pd.DataFrame:
    ruta = CARPETA_DATOS / f"{simbolo}_funding.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Corré primero descargar_funding_rate()."
        )
    df = pd.read_csv(ruta, index_col="timestamp", parse_dates=True)
    df.index = pd.to_datetime(df.index, format="ISO8601", utc=True)
    return df


# ── Descarga completa ────────────────────────────────────────────────────────

def descargar_datos_posicionamiento(
    simbolo: str = "BTCUSDT",
    meses:   int = 24,
) -> None:
    """
    Descarga y guarda datos de posicionamiento para el símbolo indicado.

    Limitaciones de Binance:
      - OI histórico: solo últimos ~30 días. No útil para backtesting largo.
      - Funding Rate: historia completa (24+ meses). ESTE es el dato principal.

    Para backtesting de 24 meses usamos Funding Rate como proxy de
    posicionamiento. OI se descarga solo si querés análisis reciente.
    """
    print("=" * 55)
    print("  DATOS DE POSICIONAMIENTO")
    print("=" * 55)

    # Funding Rate — historia completa, es el dato principal para backtest
    df_funding = descargar_funding_rate(simbolo, meses=meses)
    guardar_funding(df_funding, simbolo)

    print("\nDatos de posicionamiento listos.")
    print(f"  Funding: {len(df_funding):,} registros | "
          f"{df_funding.index[0].strftime('%Y-%m-%d')} → "
          f"{df_funding.index[-1].strftime('%Y-%m-%d')}")
