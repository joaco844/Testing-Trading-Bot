"""
data/downloader.py

Responsabilidad única: descargar datos OHLCV de un exchange via CCXT
y persistirlos en CSV.

Decisiones de diseño:
- get_exchange() está separado para que en el futuro se pueda
  inyectar una instancia ya configurada (con API keys, por ejemplo).
- guardar_csv() y cargar_csv() están separados de la descarga para
  que en el futuro podamos cambiar el storage (SQLite, PostgreSQL)
  sin tocar la lógica de descarga.
- resamplear_ohlcv() genera 10m a partir de 5m porque BingX no tiene
  10m nativo. OHLCV tiene reglas de agregación específicas:
    open=primero, high=máximo, low=mínimo, close=último, volume=suma.
- descargar_multi_tf() orquesta la descarga de los 4 TFs que necesita
  el sistema de Ángulos Cruzados (E5 requiere 5m, 10m, 15m y 1H).
"""

import ccxt
import pandas as pd
from pathlib import Path
import time
from typing import Dict


def get_exchange(exchange_id: str = "bingx") -> ccxt.Exchange:
    """
    Crea y retorna una instancia del exchange solicitado.

    enableRateLimit=True hace que ccxt espere automáticamente
    entre requests para no superar el límite de la API del exchange.
    Sin esto, podríamos recibir un ban temporal.
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        "enableRateLimit": True,
    })
    return exchange


LIMITE_POR_REQUEST = 1000  # máximo por request en la mayoría de exchanges


def descargar_ohlcv(
    simbolo: str = "BTC/USDT",
    timeframe: str = "1h",
    limite: int = 1000,
    exchange_id: str = "bingx",
) -> pd.DataFrame:
    """
    Descarga velas OHLCV con paginación hacia adelante.

    Estrategia correcta de paginación con ccxt:
      - `since` significa "dame velas CON timestamp >= since" (hacia adelante)
      - Para paginar: calculamos el punto de inicio (ahora - N velas atrás)
        y avanzamos batch a batch hasta el presente.

    Flujo:
      1. Calculamos since_inicial = ahora - (limite * duración_vela_en_ms)
      2. Pedimos 1000 velas desde since_inicial
      3. El próximo since = último timestamp del batch + 1 duración de vela
      4. Repetimos hasta tener suficientes velas o llegar al presente

    Args:
        simbolo:     Par de trading, ej: "BTC/USDT"
        timeframe:   Duración de cada vela: "1m", "5m", "1h", "4h", "1d"
        limite:      Cantidad total de velas deseadas
        exchange_id: Nombre del exchange en ccxt
    """
    exchange = get_exchange(exchange_id)

    # Convertir timeframe a milisegundos
    # parse_timeframe retorna segundos → multiplicamos x1000
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    # Punto de inicio: N velas atrás desde ahora
    ahora_ms = exchange.milliseconds()
    since = ahora_ms - (limite * timeframe_ms)

    print(f"[{exchange_id}] Descargando {limite} velas de {simbolo} ({timeframe})...")
    print(f"  Buscando desde: {pd.to_datetime(since, unit='ms', utc=True).strftime('%Y-%m-%d')}")

    todos_los_datos = []

    while len(todos_los_datos) < limite:
        cantidad = min(LIMITE_POR_REQUEST, limite - len(todos_los_datos))

        datos_crudos = exchange.fetch_ohlcv(
            simbolo,
            timeframe=timeframe,
            since=since,
            limit=cantidad,
        )

        if not datos_crudos:
            print("  No se recibieron más datos.")
            break

        todos_los_datos.extend(datos_crudos)

        ultimo_ts = datos_crudos[-1][0]
        since = ultimo_ts + timeframe_ms  # próximo batch empieza después del último

        print(f"  Batch: {len(datos_crudos)} velas | "
              f"acumulado: {len(todos_los_datos)} | "
              f"hasta: {pd.to_datetime(ultimo_ts, unit='ms', utc=True).strftime('%Y-%m-%d %H:%M')}")

        # Si llegamos al presente, paramos
        if ultimo_ts >= ahora_ms:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not todos_los_datos:
        raise ValueError(f"No se recibieron datos para {simbolo} en {exchange_id}")

    df = pd.DataFrame(
        todos_los_datos,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Eliminar duplicados en bordes de batch y ordenar
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    # Forzamos tipos numéricos
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    print(f"OK — {len(df)} velas descargadas.")
    print(f"   Desde: {df.index[0]}")
    print(f"   Hasta: {df.index[-1]}")

    return df


def guardar_csv(df: pd.DataFrame, simbolo: str, timeframe: str) -> Path:
    """
    Guarda el DataFrame como CSV en la carpeta data/.

    Convención de nombre: BTC_USDT_1h.csv
    Usamos '/' -> '_' en el símbolo para que sea válido como nombre de archivo.
    """
    carpeta = Path(__file__).parent  # carpeta data/
    nombre = f"{simbolo.replace('/', '_')}_{timeframe}.csv"
    ruta = carpeta / nombre

    df.to_csv(ruta)
    print(f"Datos guardados en: {ruta}")
    return ruta


def cargar_csv(simbolo: str, timeframe: str) -> pd.DataFrame:
    """
    Carga un CSV previamente guardado y lo retorna como DataFrame.

    parse_dates=True convierte automáticamente el índice a datetime.
    """
    carpeta = Path(__file__).parent
    nombre = f"{simbolo.replace('/', '_')}_{timeframe}.csv"
    ruta = carpeta / nombre

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}\n"
            f"Primero corré descargar_ohlcv() para descargarlo."
        )

    df = pd.read_csv(ruta, index_col="timestamp", parse_dates=True)
    return df


# ---------------------------------------------------------------------------
# Multi-timeframe — soporte para el sistema de Ángulos Cruzados (E5)
# ---------------------------------------------------------------------------

# Meses → cantidad de velas por timeframe
# 24 meses × 30 días × 24h × 60min / N_min_por_vela
_VELAS_POR_MES = {
    "5m":  8_640,   # 12 velas/hora × 24h × 30 días
    "15m": 2_880,   #  4 velas/hora × 24h × 30 días
    "1h":    720,   #  1 vela/hora  × 24h × 30 días
}


def resamplear_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Agrega velas de menor timeframe a uno mayor.

    Reglas de agregación OHLCV (no son arbitrarias — reflejan qué pasó
    en el período):
      open   → primera vela del período (precio al que abrió)
      high   → máximo entre todas las velas (precio más alto tocado)
      low    → mínimo entre todas las velas (precio más bajo tocado)
      close  → última vela del período (precio al que cerró)
      volume → suma (volumen total operado en el período)

    Ejemplo: 2 velas de 5m → 1 vela de 10m
      5m #1: O=100 H=105 L=99  C=103 V=50
      5m #2: O=103 H=107 L=102 C=106 V=80
      10m:   O=100 H=107 L=99  C=106 V=130  ← así queda la vela compuesta

    freq: string de pandas offset, ej: "10min", "15min", "1h"

    label="left" + closed="left": la vela de 10:00 contiene datos de
    10:00 a 10:09, NO de 10:01 a 10:10. Convención estándar en trading.
    """
    return df.resample(freq, label="left", closed="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()


def descargar_multi_tf(
    simbolo: str = "BTC/USDT",
    exchange_id: str = "bingx",
    meses: int = 24,
) -> Dict[str, pd.DataFrame]:
    """
    Descarga y persiste los 4 timeframes que necesita el sistema de
    Ángulos Cruzados para calcular E5 (alineación multi-TF).

    Timeframes resultantes:
      "5m"  → descargado directo de la API
      "10m" → generado resampleando desde 5m (BingX no tiene 10m nativo)
      "15m" → descargado directo de la API
      "1h"  → descargado directo de la API

    Por qué descargamos 5m y 15m por separado en vez de resamplear todo
    desde 5m: el 15m del exchange tiene su historia más larga disponible
    y los datos son exactamente los que usa el exchange para sus propios
    cálculos. Resamplear introduce diferencias mínimas en los bordes de
    vela que pueden acumularse en 24 meses de historia.

    Retorna un dict con los 4 DataFrames listos para usar.
    """
    tfs_a_descargar = ["5m", "15m", "1h"]
    resultado: Dict[str, pd.DataFrame] = {}

    for tf in tfs_a_descargar:
        limite = _VELAS_POR_MES[tf] * meses
        print(f"\n{'='*55}")
        print(f"Descargando {tf} — {limite:,} velas ({meses} meses)...")
        df = descargar_ohlcv(simbolo, tf, limite, exchange_id)
        guardar_csv(df, simbolo, tf)
        resultado[tf] = df

    # Generar 10m resampleando desde 5m
    print(f"\n{'='*55}")
    print("Generando 10m desde 5m (resampleo)...")
    df_10m = resamplear_ohlcv(resultado["5m"], "10min")
    guardar_csv(df_10m, simbolo, "10m")
    resultado["10m"] = df_10m

    print(f"\n{'='*55}")
    print("Multi-TF completo:")
    for tf, df in sorted(resultado.items()):
        print(f"  {tf:4s} → {len(df):>7,} velas | "
              f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    return resultado


def cargar_multi_tf(simbolo: str = "BTC/USDT") -> Dict[str, pd.DataFrame]:
    """
    Carga los 4 timeframes desde los CSVs guardados previamente.
    Si alguno no existe, lanza FileNotFoundError con instrucciones.

    Uso típico:
        tfs = cargar_multi_tf("BTC/USDT")
        df_5m  = tfs["5m"]
        df_10m = tfs["10m"]
        df_15m = tfs["15m"]
        df_1h  = tfs["1h"]
    """
    tfs = ["5m", "10m", "15m", "1h"]
    resultado: Dict[str, pd.DataFrame] = {}

    for tf in tfs:
        try:
            resultado[tf] = cargar_csv(simbolo, tf)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No se encontró el archivo de {tf} para {simbolo}.\n"
                f"Corré primero: descargar_multi_tf('{simbolo}')"
            )

    return resultado
