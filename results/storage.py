"""
results/storage.py

Capa de persistencia de resultados de backtesting y walk-forward.

Decisión de diseño — por qué separar storage de la lógica:
  El motor de backtesting no sabe ni le importa dónde se guardan
  los resultados. Si mañana cambiamos de CSV a SQLite o PostgreSQL,
  solo cambia este archivo. El resto del sistema no se toca.

Estructura de archivos generados:

  results/
  ├── runs.csv              → registro de cada ejecución (una fila por run)
  ├── trades/
  │   └── {run_id}.csv      → todos los trades de esa ejecución
  └── walk_forward/
      └── {run_id}.csv      → detalle por fold de esa ejecución

Qué es un "run":
  Cada vez que corrés un backtest o walk-forward con una estrategia
  y parámetros específicos, se registra como un run con ID único.
  Así podés comparar "RSI sin SL" vs "RSI con ATR SL" vs
  "RSI con filtro de tendencia" y ver la evolución histórica.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from backtests.engine import ResultadoBacktest
from backtests.walk_forward import ResultadoWalkForward


# Carpeta base de resultados
RESULTS_DIR = Path(__file__).parent


def _asegurar_carpetas():
    (RESULTS_DIR / "trades").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "walk_forward").mkdir(parents=True, exist_ok=True)


def _generar_run_id(nombre_estrategia: str) -> str:
    """
    ID único por ejecución. Formato: YYYYMMDD_HHMMSS_nombre
    Legible por humanos y ordenable cronológicamente.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    nombre_limpio = nombre_estrategia.replace(" ", "_").replace("/", "-")[:30]
    return f"{ts}_{nombre_limpio}"


def guardar_backtest(
    resultado: ResultadoBacktest,
    simbolo: str,
    timeframe: str,
    parametros: Optional[dict] = None,
    notas: str = "",
) -> str:
    """
    Persiste el resultado de un backtest simple.

    Guarda:
      - Una fila en runs.csv con el resumen y los parámetros
      - Un CSV en trades/ con el detalle de cada trade

    Args:
        resultado:   ResultadoBacktest del motor
        simbolo:     Par testeado, ej: "BTC/USDT"
        timeframe:   Timeframe usado, ej: "1h"
        parametros:  Dict con los parámetros de la estrategia (libre formato)
        notas:       Texto libre para recordar por qué se hizo este test

    Returns:
        run_id: identificador único de esta ejecución
    """
    _asegurar_carpetas()
    run_id = _generar_run_id(resultado.nombre_estrategia)

    # 1. Guardar resumen en runs.csv
    fila = {
        "run_id":            run_id,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "tipo":              "backtest",
        "estrategia":        resultado.nombre_estrategia,
        "simbolo":           simbolo,
        "timeframe":         timeframe,
        "parametros":        json.dumps(parametros or {}),
        "capital_inicial":   resultado.capital_inicial,
        "capital_final":     resultado.capital_final,
        "retorno_pct":       round(resultado.retorno_total_pct, 4),
        "num_trades":        resultado.num_trades,
        "win_rate":          round(resultado.win_rate, 2),
        "profit_factor":     round(resultado.profit_factor, 4) if resultado.profit_factor != float("inf") else 999,
        "max_drawdown_pct":  round(resultado.max_drawdown_pct, 4),
        "sharpe_ratio":      round(resultado.sharpe_ratio, 4),
        "duracion_media":    resultado.duracion_media_trades,
        "notas":             notas,
    }

    _append_csv(RESULTS_DIR / "runs.csv", fila)

    # 2. Guardar trades individuales
    if resultado.trades:
        trades_data = []
        for t in resultado.trades:
            trades_data.append({
                "run_id":          run_id,
                "entrada_tiempo":  t.entrada_tiempo,
                "entrada_precio":  round(t.entrada_precio, 4),
                "salida_tiempo":   t.salida_tiempo,
                "salida_precio":   round(t.salida_precio, 4),
                "duracion_horas":  round(t.duracion.total_seconds() / 3600, 2),
                "pnl_neto":        round(t.pnl_neto, 4),
                "retorno_pct":     round(t.retorno_pct, 4),
                "motivo_salida":   t.motivo_salida,
            })
        pd.DataFrame(trades_data).to_csv(
            RESULTS_DIR / "trades" / f"{run_id}.csv", index=False
        )

    print(f"Backtest guardado → run_id: {run_id}")
    return run_id


def guardar_walk_forward(
    resultado: ResultadoWalkForward,
    simbolo: str,
    timeframe: str,
    parametros: Optional[dict] = None,
    notas: str = "",
) -> str:
    """
    Persiste el resultado de un walk-forward test.

    Guarda:
      - Una fila en runs.csv con el resumen agregado
      - Un CSV en walk_forward/ con el detalle por fold
    """
    _asegurar_carpetas()
    run_id = _generar_run_id(resultado.nombre_estrategia)

    # 1. Resumen en runs.csv
    fila = {
        "run_id":            run_id,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "tipo":              "walk_forward",
        "estrategia":        resultado.nombre_estrategia,
        "simbolo":           simbolo,
        "timeframe":         timeframe,
        "parametros":        json.dumps(parametros or {}),
        "capital_inicial":   resultado.folds[0].resultado.capital_inicial if resultado.folds else 0,
        "capital_final":     None,   # no aplica en walk-forward (capital por fold)
        "retorno_pct":       round(resultado.retorno_medio, 4),
        "num_trades":        sum(f.resultado.num_trades for f in resultado.folds),
        "win_rate":          None,
        "profit_factor":     None,
        "max_drawdown_pct":  None,
        "sharpe_ratio":      round(resultado.sharpe_entre_folds, 4),
        "duracion_media":    None,
        "consistencia_pct":  round(resultado.consistencia_pct, 2),
        "retorno_std":       round(resultado.retorno_std, 4),
        "folds_positivos":   resultado.folds_positivos,
        "n_folds":           len(resultado.folds),
        "notas":             notas,
    }

    _append_csv(RESULTS_DIR / "runs.csv", fila)

    # 2. Detalle por fold
    folds_data = []
    for fold in resultado.folds:
        r = fold.resultado
        folds_data.append({
            "run_id":           run_id,
            "fold":             fold.fold,
            "fecha_inicio":     fold.fecha_inicio,
            "fecha_fin":        fold.fecha_fin,
            "num_velas":        fold.num_velas,
            "num_trades":       r.num_trades,
            "retorno_pct":      round(r.retorno_total_pct, 4),
            "win_rate":         round(r.win_rate, 2),
            "max_drawdown_pct": round(r.max_drawdown_pct, 4),
            "sharpe_ratio":     round(r.sharpe_ratio, 4),
            "sl_hits":          sum(1 for t in r.trades if t.motivo_salida == "stop_loss"),
            "tp_hits":          sum(1 for t in r.trades if t.motivo_salida == "take_profit"),
        })
    pd.DataFrame(folds_data).to_csv(
        RESULTS_DIR / "walk_forward" / f"{run_id}.csv", index=False
    )

    print(f"Walk-forward guardado → run_id: {run_id}")
    return run_id


def cargar_historial() -> pd.DataFrame:
    """
    Carga y retorna todos los runs guardados como DataFrame.
    Útil para comparar experimentos históricos.
    """
    ruta = RESULTS_DIR / "runs.csv"
    if not ruta.exists():
        print("No hay historial guardado todavía.")
        return pd.DataFrame()
    return pd.read_csv(ruta, parse_dates=["timestamp"])


def cargar_trades(run_id: str) -> pd.DataFrame:
    """Carga los trades de un run específico."""
    ruta = RESULTS_DIR / "trades" / f"{run_id}.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No hay trades para run_id: {run_id}")
    return pd.read_csv(ruta, parse_dates=["entrada_tiempo", "salida_tiempo"])


def cargar_walk_forward(run_id: str) -> pd.DataFrame:
    """Carga el detalle por fold de un walk-forward específico."""
    ruta = RESULTS_DIR / "walk_forward" / f"{run_id}.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No hay walk-forward para run_id: {run_id}")
    return pd.read_csv(ruta, parse_dates=["fecha_inicio", "fecha_fin"])


def mostrar_historial(top_n: int = 20) -> None:
    """Imprime el historial de runs de forma legible."""
    df = cargar_historial()
    if df.empty:
        return

    cols = ["timestamp", "tipo", "estrategia", "simbolo", "timeframe",
            "retorno_pct", "num_trades", "sharpe_ratio", "notas"]
    cols_existentes = [c for c in cols if c in df.columns]

    print(f"\n{'='*70}")
    print(f"  HISTORIAL DE EXPERIMENTOS ({len(df)} runs totales)")
    print(f"{'='*70}")
    print(df[cols_existentes].tail(top_n).to_string(index=False))
    print(f"{'='*70}")


def _append_csv(ruta: Path, fila: dict) -> None:
    """
    Agrega una fila a un CSV existente o crea el archivo si no existe.
    Maneja automáticamente las columnas — si el CSV ya existe y tiene
    columnas distintas, agrega las que faltan con NaN.
    """
    df_nueva = pd.DataFrame([fila])

    if ruta.exists():
        df_existente = pd.read_csv(ruta)
        # Unir columnas de ambos para compatibilidad hacia adelante
        df_combinado = pd.concat([df_existente, df_nueva], ignore_index=True)
        df_combinado.to_csv(ruta, index=False)
    else:
        df_nueva.to_csv(ruta, index=False)
