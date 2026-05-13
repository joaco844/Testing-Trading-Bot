"""
data/inspector.py

Responsabilidad única: inspeccionar y limpiar DataFrames de velas OHLCV.

Regla de oro: nunca confiar en que los datos del exchange están limpios.
Los exchanges pueden tener:
  - Velas duplicadas por reconexiones de API
  - Velas con volumen 0 por falta de actividad o errores
  - Precios inconsistentes (high < low, etc.)
  - Gaps temporales cuando el exchange tuvo downtime
"""

import pandas as pd


def inspeccionar(df: pd.DataFrame) -> None:
    """
    Imprime un resumen completo del DataFrame.
    Siempre llamar esto antes de usar datos nuevos.
    """
    print("=" * 55)
    print("INSPECCIÓN DE DATOS")
    print("=" * 55)
    print(f"  Filas x Columnas : {df.shape}")
    print(f"  Desde            : {df.index[0]}")
    print(f"  Hasta            : {df.index[-1]}")
    print(f"  Intervalo aprox  : {df.index[1] - df.index[0]}")
    print()
    print("-- Primeras 3 filas --")
    print(df.head(3).to_string())
    print()
    print("-- Últimas 3 filas --")
    print(df.tail(3).to_string())
    print()
    print("-- Estadísticas --")
    print(df.describe().to_string())
    print()
    print("-- Valores nulos --")
    nulos = df.isnull().sum()
    print(nulos.to_string())
    if nulos.sum() == 0:
        print("  (ninguno)")
    print()

    # Detectar gaps temporales (velas faltantes)
    diffs = df.index.to_series().diff().dropna()
    intervalo_esperado = diffs.mode()[0]
    gaps = diffs[diffs > intervalo_esperado * 1.5]
    if len(gaps) > 0:
        print(f"-- ADVERTENCIA: {len(gaps)} gap(s) temporal(es) detectado(s) --")
        for ts, gap in gaps.items():
            print(f"  {ts} → gap de {gap}")
    else:
        print("  Sin gaps temporales detectados.")

    print("=" * 55)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza básica de datos OHLCV.
    Retorna un DataFrame nuevo sin modificar el original.

    Pasos:
    1. Eliminar duplicados por timestamp
    2. Ordenar cronológicamente
    3. Eliminar filas con NaN en columnas críticas
    4. Eliminar velas con precios inconsistentes (high < low, etc.)
    5. Eliminar velas con precios cero o negativos
    """
    df = df.copy()
    filas_originales = len(df)

    # 1. Duplicados por índice
    df = df[~df.index.duplicated(keep="first")]

    # 2. Orden cronológico (siempre, aunque parezca innecesario)
    df.sort_index(inplace=True)

    # 3. NaN en columnas de precio/volumen
    df.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)

    # 4. Precios inconsistentes
    #    En una vela válida: low <= open, close <= high
    mask_invalido = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    df = df[~mask_invalido]

    # 5. Precios cero o negativos (error del exchange)
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]

    filas_eliminadas = filas_originales - len(df)
    if filas_eliminadas > 0:
        print(f"Limpieza: {filas_eliminadas} fila(s) problemática(s) eliminada(s).")
    else:
        print("Limpieza: datos OK, no se eliminó ninguna fila.")

    return df
