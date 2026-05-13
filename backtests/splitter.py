"""
backtests/splitter.py

Separación correcta de datos para investigación cuantitativa.

El problema que resuelve:
  Si usás todos los datos para experimentar, cada decisión que tomás
  (qué indicador agregar, qué parámetro cambiar) está influenciada
  por esos datos. El modelo "aprende" el dataset aunque no lo veas
  explícitamente. Cuando lo corrés en datos nuevos, falla.

La solución:
  Separar UN conjunto de datos que NO se toca durante el desarrollo.
  Solo se usa UNA vez, al final, para validar la estrategia final.
  Si el resultado es malo → la estrategia no funciona. Punto.
  No se vuelve a ajustar mirando ese resultado.

Analogía:
  El examen final de la materia. Podés estudiar todo lo que quieras
  con los ejercicios del libro (development set). Pero el examen
  (holdout set) lo rendís una sola vez y el resultado es definitivo.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class DataSplit:
    """Contiene los dos conjuntos de datos bien separados."""
    development: pd.DataFrame   # para experimentar libremente
    holdout: pd.DataFrame       # sagrado — no tocar hasta el final

    def info(self) -> None:
        print("=" * 55)
        print("  SEPARACIÓN DE DATOS")
        print("=" * 55)
        print(f"  Development set:")
        print(f"    Velas : {len(self.development)}")
        print(f"    Desde : {self.development.index[0].strftime('%Y-%m-%d')}")
        print(f"    Hasta : {self.development.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Holdout set (NO TOCAR hasta estrategia final):")
        print(f"    Velas : {len(self.holdout)}")
        print(f"    Desde : {self.holdout.index[0].strftime('%Y-%m-%d')}")
        print(f"    Hasta : {self.holdout.index[-1].strftime('%Y-%m-%d')}")
        print("=" * 55)


def split_datos(
    df: pd.DataFrame,
    holdout_pct: float = 0.20,
) -> DataSplit:
    """
    Divide el DataFrame en development set y holdout set.

    Siempre cortamos por tiempo, no al azar.
    Por qué no al azar: los datos financieros tienen dependencia temporal
    (el mercado de hoy influye en el de mañana). Si mezclamos fechas
    al azar, el modelo puede "ver el futuro" durante el entrenamiento.
    Siempre: datos más viejos para development, más nuevos para holdout.

    Args:
        df:          DataFrame con OHLCV ordenado cronológicamente
        holdout_pct: Fracción de datos a reservar para holdout (default 20%)

    Returns:
        DataSplit con los dos conjuntos
    """
    if not 0 < holdout_pct < 1:
        raise ValueError("holdout_pct debe estar entre 0 y 1")

    n_total      = len(df)
    n_holdout    = int(n_total * holdout_pct)
    n_development = n_total - n_holdout

    development = df.iloc[:n_development].copy()
    holdout     = df.iloc[n_development:].copy()

    return DataSplit(development=development, holdout=holdout)
