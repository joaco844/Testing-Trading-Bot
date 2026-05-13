"""
paper_trading/engine.py

Motor de paper trading — corre la estrategia en datos reales
sin ejecutar órdenes reales.

Diferencia con backtesting:
  - Backtesting: corre sobre datos históricos completos de una vez
  - Paper trading: corre sobre datos actuales, vela a vela,
    en tiempo real (o semi-real si lo corrés manualmente)

Cómo funciona cada ejecución:
  1. Descarga las últimas N velas del exchange
  2. Aplica la estrategia para obtener señales
  3. Mira SOLO la última vela completada (no la vela en curso)
  4. Si hay señal nueva → simula entrada/salida
  5. Guarda el estado actualizado

Por qué la anteúltima vela y no la última:
  La última vela está en curso — no está cerrada todavía.
  Sus valores de close, high, low van a cambiar hasta que cierre.
  Si tomamos señal sobre una vela abierta, estamos usando datos
  incompletos. Siempre operamos sobre la última vela CERRADA.
"""

from datetime import datetime, timezone
from data.downloader import descargar_ohlcv, resamplear_ohlcv
from paper_trading.state import (
    cargar_estado, guardar_estado, registrar_trade, mostrar_estado
)


def correr_paso(
    estrategia,
    simbolo: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_id: str = "bingx",
    capital_inicial: float = 10_000.0,
    fee_pct: float = 0.001,
    velas_contexto: int = 300,    # velas para calentar indicadores
) -> None:
    """
    Ejecuta un paso del paper trader.

    Llamá a esta función cada vez que quieras que el bot evalúe
    el mercado. Puede ser manualmente, o más adelante con un
    scheduler cada 1h (cuando la vela cierra).

    Args:
        estrategia:      Instancia de una estrategia (tiene .calcular_señales)
        simbolo:         Par a operar
        timeframe:       Timeframe de las velas
        exchange_id:     Exchange para datos (binance para histórico)
        capital_inicial: Capital inicial si es la primera vez
        fee_pct:         Comisión por operación
        velas_contexto:  Cuántas velas descargar para dar contexto a indicadores
    """
    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] "
          f"Evaluando {simbolo} ({timeframe})...")

    # 1. Descargar velas recientes
    df = descargar_ohlcv(simbolo, timeframe, velas_contexto, exchange_id)

    # 2. Aplicar estrategia
    df_señales = estrategia.calcular_señales(df)

    # 3. La señal que nos importa es la de la ANTEÚLTIMA vela (última cerrada)
    #    La última puede estar aún en curso
    vela_cerrada = df_señales.iloc[-2]
    señal        = vela_cerrada["señal"]
    precio       = vela_cerrada["close"]
    timestamp    = str(df_señales.index[-2])

    precio_actual = df_señales["close"].iloc[-1]  # precio de la vela en curso

    print(f"  Última vela cerrada : {timestamp}")
    print(f"  Precio cierre       : ${precio:,.2f}")
    print(f"  Señal               : {_describir_señal(señal)}")

    # 4. Cargar estado
    estado = cargar_estado(capital_inicial)

    # 5. Verificar si ya procesamos esta vela (evitar procesar dos veces)
    if estado["ultima_vela"] == timestamp:
        print("  Esta vela ya fue procesada. Nada que hacer.")
        mostrar_estado(estado, precio_actual)
        return

    # 6. Actuar según la señal
    if señal == 1 and not estado["en_posicion"]:
        # COMPRAR
        estado["en_posicion"]    = True
        estado["entrada_precio"] = precio
        estado["entrada_tiempo"] = timestamp
        print(f"  ✓ COMPRA simulada a ${precio:,.2f}")

    elif señal == -1 and estado["en_posicion"]:
        # VENDER
        pnl = registrar_trade(
            estado,
            entrada_precio=estado["entrada_precio"],
            entrada_tiempo=estado["entrada_tiempo"],
            salida_precio=precio,
            salida_tiempo=timestamp,
            capital_usado=estado["capital"],
            fee_pct=fee_pct,
            motivo="señal",
        )
        estado["capital"]        += pnl
        estado["en_posicion"]    = False
        estado["entrada_precio"] = None
        estado["entrada_tiempo"] = None
        print(f"  ✓ VENTA simulada a ${precio:,.2f} | PnL: ${pnl:+,.2f}")

    else:
        print(f"  Sin acción.")

    # 7. Guardar estado actualizado
    estado["ultima_vela"] = timestamp
    guardar_estado(estado)

    # 8. Mostrar resumen
    mostrar_estado(estado, precio_actual)


def correr_paso_angulos(
    simbolo: str = "BTC/USDT",
    exchange_id: str = "binance",
    capital_inicial: float = 10_000.0,
    fee_pct: float = 0.001,
    umbral_entrada: float = 70.0,
    umbral_salida: float = 50.0,
) -> None:
    """
    Paper trading del Sistema de Ángulos Cruzados en 15m.

    Descarga los 4 TFs en vivo, calcula los 5 ejes, detecta patrones
    y evalúa la última vela cerrada. Llamar cada 15 minutos (al cierre
    de cada vela de 15m).

    Velas de contexto descargadas:
      - 5m : 600 velas (~50h) — suficiente para EMA50 del 10m resampleado
      - 15m: 300 velas (~75h) — suficiente para EMA200
      - 1h : 300 velas (300h) — suficiente para EMA50
    """
    from strategies.angulos_cruzados import AngulosCruzados

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}] "
          f"Ángulos Cruzados — {simbolo} 15m...")

    # 1. Descargar los 4 TFs en vivo
    print("  Descargando datos multi-TF...")
    df_5m  = descargar_ohlcv(simbolo, "5m",  600, exchange_id)
    df_15m = descargar_ohlcv(simbolo, "15m", 300, exchange_id)
    df_1h  = descargar_ohlcv(simbolo, "1h",  300, exchange_id)
    df_10m = resamplear_ohlcv(df_5m, "10min")

    todos_tfs = {"5m": df_5m, "10m": df_10m, "15m": df_15m, "1h": df_1h}

    # 2. Calcular señales
    estrategia = AngulosCruzados(todos_tfs, umbral_entrada, umbral_salida)
    df_señales = estrategia.calcular_señales(df_15m)

    # 3. Última vela CERRADA (anteúltima del df — la última está en curso)
    vela       = df_señales.iloc[-2]
    señal      = int(vela["señal"])
    precio     = vela["close"]
    timestamp  = str(df_señales.index[-2])
    patron     = vela["patron"]
    confianza  = vela["confianza"]
    score      = vela["score"]

    precio_actual = df_señales["close"].iloc[-1]

    print(f"  Vela cerrada : {timestamp}")
    print(f"  Precio       : ${precio:,.2f}")
    print(f"  Patrón       : {patron} (confianza {confianza:.0f}% | score {score:+.0f})")
    print(f"  Ejes         : E1={vela['e1']:+.0f} E2={vela['e2']:+.0f} "
          f"E3={vela['e3']:+.0f} E4={vela['e4']:+.0f} E5={vela['e5']:+.0f}")
    print(f"  Señal        : {_describir_señal(señal)}")

    # 4. Cargar estado (archivo JSON con posición abierta y capital)
    estado = cargar_estado(capital_inicial)

    # 5. Evitar procesar la misma vela dos veces
    if estado["ultima_vela"] == timestamp:
        print("  Esta vela ya fue procesada.")
        mostrar_estado(estado, precio_actual)
        return

    # 6. Actuar según la señal
    if señal == 1 and not estado["en_posicion"]:
        estado["en_posicion"]    = True
        estado["entrada_precio"] = precio
        estado["entrada_tiempo"] = timestamp
        print(f"  → COMPRA simulada a ${precio:,.2f}  [{patron}]")

    elif señal == -1 and estado["en_posicion"]:
        pnl = registrar_trade(
            estado,
            entrada_precio=estado["entrada_precio"],
            entrada_tiempo=estado["entrada_tiempo"],
            salida_precio=precio,
            salida_tiempo=timestamp,
            capital_usado=estado["capital"],
            fee_pct=fee_pct,
            motivo="señal",
        )
        estado["capital"]        += pnl
        estado["en_posicion"]    = False
        estado["entrada_precio"] = None
        estado["entrada_tiempo"] = None
        print(f"  → VENTA simulada a ${precio:,.2f} | PnL: ${pnl:+,.2f}")

    else:
        print("  Sin acción.")

    estado["ultima_vela"] = timestamp
    guardar_estado(estado)
    mostrar_estado(estado, precio_actual)


def _describir_señal(señal: int) -> str:
    if señal == 1:
        return "1 → COMPRA"
    elif señal == -1:
        return "-1 → VENTA"
    return "0 → sin acción"
