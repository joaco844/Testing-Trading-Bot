"""
exchange/ordenes.py

Funciones para ejecutar órdenes en BingX demo (perpetuos/swap).

Decisiones de diseño:
- Usamos perpetuos (swap) porque el demo de BingX solo da USDT virtual ahí.
- Leverage fijo en 1x — sin apalancamiento, comportamiento equivalente a spot.
- Tamaño de posición en USDT fijo por trade — simple y controlable.
- SL y TP se mandan junto con la orden de entrada (soportado por BingX).
- Una sola posición abierta a la vez — el bot cierra antes de abrir otra.
"""

import ccxt


SIMBOLO_SWAP = "BTC/USDT:USDT"   # formato CCXT para perpetuo BTC/USDT en BingX
LEVERAGE     = 1                  # sin apalancamiento


def configurar_leverage(exchange: ccxt.bingx, simbolo: str = SIMBOLO_SWAP) -> None:
    """Establece leverage 1x antes de operar."""
    exchange.set_leverage(LEVERAGE, simbolo)


def obtener_precio_actual(exchange: ccxt.bingx, simbolo: str = SIMBOLO_SWAP) -> float:
    """Precio de mercado actual."""
    ticker = exchange.fetch_ticker(simbolo)
    return ticker["last"]


def obtener_posicion(exchange: ccxt.bingx, simbolo: str = SIMBOLO_SWAP) -> dict | None:
    """
    Retorna la posición abierta en el símbolo, o None si no hay.
    BingX devuelve una lista de posiciones — filtramos por símbolo y tamaño > 0.
    """
    posiciones = exchange.fetch_positions([simbolo])
    for pos in posiciones:
        if pos["symbol"] == simbolo and pos["contracts"] and pos["contracts"] > 0:
            return pos
    return None


def abrir_long(
    exchange: ccxt.bingx,
    usdt_a_usar: float,
    sl_precio: float,
    tp_precio: float,
    simbolo: str = SIMBOLO_SWAP,
) -> dict:
    """
    Abre una posición long a mercado con SL y TP.

    usdt_a_usar: capital en USDT a destinar a este trade.
    sl_precio:   precio de stop-loss (calculado como entrada - 1×ATR).
    tp_precio:   precio de take-profit (calculado como entrada + 1.5×ATR).

    El tamaño en BTC se calcula dividiendo usdt_a_usar por el precio actual.
    BingX requiere el tamaño en contratos (BTC), no en USDT.
    """
    configurar_leverage(exchange, simbolo)

    precio = obtener_precio_actual(exchange, simbolo)

    # Tamaño en BTC, redondeado a 4 decimales (mínimo de BingX)
    mercado = exchange.market(simbolo)
    precision = mercado["precision"]["amount"]
    cantidad  = exchange.amount_to_precision(simbolo, usdt_a_usar / precio)

    print(f"  Abriendo long: {cantidad} BTC a ~${precio:,.2f}")
    print(f"  SL: ${sl_precio:,.2f} | TP: ${tp_precio:,.2f}")

    orden = exchange.create_order(
        symbol=simbolo,
        type="market",
        side="buy",
        amount=float(cantidad),
        params={
            "stopLoss":   {"type": "market", "triggerPrice": sl_precio},
            "takeProfit": {"type": "market", "triggerPrice": tp_precio},
            "positionSide": "LONG",
        },
    )

    return orden


def cerrar_long(
    exchange: ccxt.bingx,
    simbolo: str = SIMBOLO_SWAP,
) -> dict | None:
    """
    Cierra la posición long abierta a mercado.
    Si no hay posición abierta, no hace nada.
    """
    posicion = obtener_posicion(exchange, simbolo)
    if posicion is None:
        print("  Sin posición abierta para cerrar.")
        return None

    cantidad = posicion["contracts"]
    print(f"  Cerrando long: {cantidad} BTC a mercado")

    orden = exchange.create_order(
        symbol=simbolo,
        type="market",
        side="sell",
        amount=cantidad,
        params={"positionSide": "LONG", "reduceOnly": True},
    )

    return orden
