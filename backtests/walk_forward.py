"""
backtests/walk_forward.py

Walk-forward testing — la forma correcta de validar una estrategia.

El problema con el backtesting normal:
  Diseñás la estrategia mirando los datos históricos.
  La testeas en esos mismos datos.
  Los resultados parecen buenos.
  La corrés en el mercado real → falla.

  ¿Por qué? Porque sin darte cuenta, tus decisiones de diseño
  (qué indicadores usar, qué parámetros, qué reglas) se ajustaron
  a los patrones de ESE período específico. El modelo "memorizó"
  en lugar de "aprender".

Cómo lo resuelve el walk-forward:
  Dividís el tiempo en ventanas. En cada ventana:
    - Training set: acá observás el mercado, ajustás parámetros
    - Test set (out-of-sample): acá corrés la estrategia con
      lo que aprendiste en el training, pero en datos que NUNCA viste

  El resultado que importa es SOLO el de los test sets.
  Si la estrategia funciona en todos los test sets por separado,
  hay evidencia real de que encontraste algo que generaliza.

Esquema visual con 5 folds:

  |-- Fold 1 --|-- Fold 2 --|-- Fold 3 --|-- Fold 4 --|-- Fold 5 --|
  |  TRAIN     |  TEST      |            |            |            |  ← Iteración 1
  |  TRAIN     |  TRAIN     |  TEST      |            |            |  ← Iteración 2
  |  TRAIN     |  TRAIN     |  TRAIN     |  TEST      |            |  ← Iteración 3
  |  TRAIN     |  TRAIN     |  TRAIN     |  TRAIN     |  TEST      |  ← Iteración 4

  El training set crece (expanding window). El test siempre es un fold nuevo.
  Así, el test set de cada iteración nunca fue visto en la iteración anterior.

En esta implementación NO optimizamos parámetros automáticamente —
usamos los mismos parámetros en todas las iteraciones. Lo que medimos
es si la CONSISTENCIA del resultado se mantiene en distintos períodos.
(La optimización por fold es el siguiente paso, requiere más datos.)
"""

import pandas as pd
import numpy as np
from typing import Callable, List
from dataclasses import dataclass

from backtests.engine import correr_backtest, ResultadoBacktest


@dataclass
class ResultadoFold:
    """Resultado de una iteración del walk-forward."""
    fold: int
    fecha_inicio: pd.Timestamp
    fecha_fin: pd.Timestamp
    num_velas: int
    resultado: ResultadoBacktest

    def __str__(self):
        r = self.resultado
        return (
            f"  Fold {self.fold} | "
            f"{self.fecha_inicio.strftime('%Y-%m-%d')} → {self.fecha_fin.strftime('%Y-%m-%d')} | "
            f"retorno: {r.retorno_total_pct:+.2f}% | "
            f"trades: {r.num_trades} | "
            f"WR: {r.win_rate:.1f}% | "
            f"DD: {r.max_drawdown_pct:.2f}%"
        )


@dataclass
class ResultadoWalkForward:
    """Resultado agregado de todas las iteraciones."""
    nombre_estrategia: str
    folds: List[ResultadoFold]

    @property
    def retornos(self) -> List[float]:
        return [f.resultado.retorno_total_pct for f in self.folds]

    @property
    def retorno_medio(self) -> float:
        return np.mean(self.retornos)

    @property
    def retorno_std(self) -> float:
        return np.std(self.retornos, ddof=1) if len(self.retornos) > 1 else 0.0

    @property
    def folds_positivos(self) -> int:
        return sum(1 for r in self.retornos if r > 0)

    @property
    def consistencia_pct(self) -> float:
        """% de folds con retorno positivo. La métrica más importante del walk-forward."""
        return self.folds_positivos / len(self.folds) * 100

    @property
    def sharpe_entre_folds(self) -> float:
        """
        Sharpe calculado entre los retornos de los folds (no dentro de cada fold).
        Mide si los retornos son estables o muy variables entre períodos.
        Un valor alto acá significa que la estrategia es consistente en el tiempo.
        """
        if len(self.retornos) < 2 or self.retorno_std == 0:
            return 0.0
        return self.retorno_medio / self.retorno_std

    def resumen(self) -> str:
        lineas = [
            f"\n{'='*65}",
            f"  WALK-FORWARD: {self.nombre_estrategia}",
            f"{'='*65}",
            f"  Folds testeados       : {len(self.folds)}",
            f"  Folds positivos       : {self.folds_positivos} / {len(self.folds)}",
            f"  Consistencia          : {self.consistencia_pct:.0f}%",
            f"  Retorno medio/fold    : {self.retorno_medio:+.2f}%",
            f"  Desvío entre folds    : {self.retorno_std:.2f}%",
            f"  Sharpe entre folds    : {self.sharpe_entre_folds:.2f}",
            f"",
            f"  Detalle por fold (OUT-OF-SAMPLE):",
        ]
        for fold in self.folds:
            lineas.append(str(fold))
        lineas.append(f"{'='*65}")
        return "\n".join(lineas)


def walk_forward(
    df: pd.DataFrame,
    calcular_señales: Callable[[pd.DataFrame], pd.DataFrame],
    nombre_estrategia: str,
    n_folds: int = 5,
    capital_inicial: float = 10_000.0,
    fee_pct: float = 0.001,
    stop_loss_pct: float = None,
    take_profit_pct: float = None,
    atr_sl_mult: float = None,
    atr_tp_mult: float = None,
) -> ResultadoWalkForward:
    """
    Ejecuta el walk-forward test sobre un DataFrame.

    Args:
        df:                 DataFrame con OHLCV limpio
        calcular_señales:   Función que recibe un df y retorna df con columna 'señal'
                            (usar estrategia.calcular_señales)
        nombre_estrategia:  Nombre para mostrar en el resumen
        n_folds:            En cuántos segmentos dividir el tiempo.
                            Con 5 folds: 1 fold de training + 4 folds de test.
                            Más folds = más iteraciones pero menos datos por fold.
        capital_inicial:    Capital de inicio de CADA fold (no acumulado).
                            Así comparamos rendimientos porcentuales en igualdad
                            de condiciones, independiente del orden.

    IMPORTANTE — capital por fold vs capital acumulado:
        Usamos capital_inicial fijo por fold para medir si la estrategia
        genera retornos positivos en cada período por separado.
        Si acumuláramos el capital, un fold muy bueno al principio
        distorsionaría los siguientes (el % de ganancia/pérdida sería
        sobre una base diferente en cada fold).
    """
    total_velas = len(df)
    tamaño_fold = total_velas // n_folds

    print(f"\nWalk-forward: {nombre_estrategia}")
    print(f"  Total velas: {total_velas} | Folds: {n_folds} | "
          f"Velas por fold: {tamaño_fold} (~{tamaño_fold}h = "
          f"~{tamaño_fold//24}d)")

    folds_resultado = []

    # Iteración i: train = folds 0..i-1 | test = fold i
    # Empezamos en i=1 para tener al menos 1 fold de contexto previo
    for i in range(1, n_folds):
        # --- Definir ventanas ---
        idx_test_inicio = i * tamaño_fold
        idx_test_fin    = (i + 1) * tamaño_fold if i < n_folds - 1 else total_velas

        # El training incluye TODO lo anterior al test
        # Importante: calculamos señales sobre el training completo + test
        # para que los indicadores al inicio del test tengan suficiente historia.
        # Luego filtramos solo el período de test para el backtest.
        df_completo_hasta_test = df.iloc[:idx_test_fin]
        df_con_señales = calcular_señales(df_completo_hasta_test)

        # Solo el período out-of-sample va al backtest
        df_test = df_con_señales.iloc[idx_test_inicio:idx_test_fin]

        resultado = correr_backtest(
            df_test,
            nombre_estrategia=f"{nombre_estrategia} fold {i}",
            capital_inicial=capital_inicial,
            fee_pct=fee_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
        )

        fold = ResultadoFold(
            fold=i,
            fecha_inicio=df_test.index[0],
            fecha_fin=df_test.index[-1],
            num_velas=len(df_test),
            resultado=resultado,
        )
        folds_resultado.append(fold)

    return ResultadoWalkForward(
        nombre_estrategia=nombre_estrategia,
        folds=folds_resultado,
    )
