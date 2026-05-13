"""
paper_trading/state.py

Manejo del estado persistente del paper trader.

El estado se guarda en un JSON para sobrevivir entre ejecuciones.
Cada vez que corrés el bot, carga el estado anterior, procesa
la vela más reciente, y guarda el estado actualizado.

Estructura del estado:
{
  "capital":         10000.0,    <- capital disponible (sin posición)
  "en_posicion":     false,      <- si hay una posición abierta
  "entrada_precio":  null,       <- precio de entrada de la posición actual
  "entrada_tiempo":  null,       <- timestamp de entrada
  "trades":          [],         <- historial de trades cerrados
  "ultima_vela":     null        <- timestamp de la última vela procesada
}
"""

import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


ESTADO_PATH = Path(__file__).parent / "estado.json"


def cargar_estado(capital_inicial: float = 10_000.0) -> dict:
    """
    Carga el estado desde disco. Si no existe, crea uno nuevo.
    """
    if ESTADO_PATH.exists():
        with open(ESTADO_PATH, "r") as f:
            estado = json.load(f)
        print(f"Estado cargado. Capital: ${estado['capital']:,.2f} | "
              f"En posición: {estado['en_posicion']}")
        return estado

    # Primera vez — estado inicial
    estado = {
        "capital":        capital_inicial,
        "en_posicion":    False,
        "entrada_precio": None,
        "entrada_tiempo": None,
        "trades":         [],
        "ultima_vela":    None,
    }
    print(f"Estado nuevo creado. Capital inicial: ${capital_inicial:,.2f}")
    guardar_estado(estado)
    return estado


def guardar_estado(estado: dict) -> None:
    """Persiste el estado en disco."""
    with open(ESTADO_PATH, "w") as f:
        json.dump(estado, f, indent=2, default=str)


def resetear_estado(capital_inicial: float = 10_000.0) -> dict:
    """
    Borra el estado actual y empieza de cero.
    Usar cuando querés empezar un nuevo paper trading desde cero.
    """
    if ESTADO_PATH.exists():
        ESTADO_PATH.unlink()
        print("Estado anterior borrado.")
    return cargar_estado(capital_inicial)


def registrar_trade(estado: dict, entrada_precio: float, entrada_tiempo: str,
                    salida_precio: float, salida_tiempo: str,
                    capital_usado: float, fee_pct: float, motivo: str) -> float:
    """
    Registra un trade cerrado en el historial y retorna el PnL neto.
    """
    pnl_bruto = (salida_precio - entrada_precio) / entrada_precio * capital_usado
    fee_total = (capital_usado * fee_pct) + ((capital_usado + pnl_bruto) * fee_pct)
    pnl_neto  = pnl_bruto - fee_total
    retorno_pct = pnl_neto / capital_usado * 100

    trade = {
        "entrada_tiempo":  entrada_tiempo,
        "entrada_precio":  round(entrada_precio, 2),
        "salida_tiempo":   salida_tiempo,
        "salida_precio":   round(salida_precio, 2),
        "pnl_neto":        round(pnl_neto, 2),
        "retorno_pct":     round(retorno_pct, 4),
        "motivo":          motivo,
    }
    estado["trades"].append(trade)
    return pnl_neto


def mostrar_estado(estado: dict, precio_actual: Optional[float] = None) -> None:
    """Imprime el estado actual del paper trader."""
    capital = estado["capital"]
    trades  = estado["trades"]

    # Valor de la posición abierta (si hay)
    valor_posicion = 0.0
    pnl_no_realizado = 0.0
    if estado["en_posicion"] and precio_actual and estado["entrada_precio"]:
        pnl_no_realizado = (precio_actual - estado["entrada_precio"]) / \
                           estado["entrada_precio"] * capital
        valor_posicion = capital + pnl_no_realizado

    capital_total = capital if not estado["en_posicion"] else valor_posicion

    print("\n" + "=" * 55)
    print("  PAPER TRADER — ESTADO ACTUAL")
    print("=" * 55)
    print(f"  Capital disponible  : ${capital:,.2f}")
    if estado["en_posicion"]:
        print(f"  POSICIÓN ABIERTA    : entrada ${estado['entrada_precio']:,.2f}")
        print(f"  Precio actual       : ${precio_actual:,.2f}" if precio_actual else "")
        print(f"  PnL no realizado    : ${pnl_no_realizado:+,.2f}  "
              f"({pnl_no_realizado/capital*100:+.2f}%)")
        print(f"  Valor total         : ${capital_total:,.2f}")
    else:
        print(f"  Sin posición abierta")

    if trades:
        pnl_total = sum(t["pnl_neto"] for t in trades)
        ganadores = sum(1 for t in trades if t["pnl_neto"] > 0)
        print(f"\n  Trades cerrados     : {len(trades)}")
        print(f"  Ganadores           : {ganadores} / {len(trades)}")
        print(f"  PnL realizado total : ${pnl_total:+,.2f}")
        print(f"\n  Últimos 3 trades:")
        for t in trades[-3:]:
            print(f"    {t['entrada_tiempo'][:10]} → {t['salida_tiempo'][:10]} | "
                  f"${t['entrada_precio']:,.0f} → ${t['salida_precio']:,.0f} | "
                  f"PnL: {t['retorno_pct']:+.2f}% | {t['motivo']}")
    else:
        print(f"  Sin trades cerrados aún")
    print("=" * 55)
