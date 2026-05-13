"""
main.py — Punto de entrada del proyecto.

Modos de uso:
  python main.py            → corre el paper trader (modo por defecto)
  python main.py backtest   → corre el walk-forward en development set
  python main.py angulos    → backtest del Sistema de Ángulos Cruzados
"""

import sys
from strategies.rsi_mean_reversion import RSIMeanReversion
from paper_trading.engine import correr_paso
from paper_trading.state import resetear_estado


def modo_paper_trading():
    """
    Paper trading: evalúa el mercado actual y simula trades.
    Corré esto cada vez que cierra una vela de 1h.
    """
    estrategia = RSIMeanReversion(
        periodo=14,
        umbral_bajo=30,
        umbral_alto=70,
        # Sin filtro EMA por ahora — lo agregamos cuando tengamos
        # una versión validada de la estrategia
    )

    correr_paso(
        estrategia=estrategia,
        simbolo="BTC/USDT",
        timeframe="1h",
        exchange_id="binance",   # datos de Binance (más historia)
        capital_inicial=10_000.0,
        fee_pct=0.001,
        velas_contexto=300,      # 300 velas = ~12 días de contexto para indicadores
    )


def modo_backtest():
    """Walk-forward en development set — para seguir investigando."""
    from data.downloader import descargar_ohlcv, guardar_csv, cargar_csv
    from data.inspector import limpiar
    from backtests.splitter import split_datos
    from backtests.walk_forward import walk_forward
    from results.storage import guardar_walk_forward, mostrar_historial

    SIMBOLO   = "BTC/USDT"
    TIMEFRAME = "1h"
    CAPITAL   = 10_000.0
    FEE       = 0.001
    N_FOLDS   = 5

    df = descargar_ohlcv(SIMBOLO, TIMEFRAME, 5000, "binance")
    guardar_csv(df, SIMBOLO, TIMEFRAME)
    df = cargar_csv(SIMBOLO, TIMEFRAME)
    df = limpiar(df)

    split  = split_datos(df, holdout_pct=0.20)
    split.info()
    df_dev = split.development

    strat  = RSIMeanReversion(periodo=14, umbral_bajo=30, umbral_alto=70)
    result = walk_forward(df_dev, strat.calcular_señales, strat.nombre,
                          N_FOLDS, CAPITAL, FEE)
    print(result.resumen())
    guardar_walk_forward(result, SIMBOLO, TIMEFRAME,
                         {"periodo": 14, "umbral_bajo": 30, "umbral_alto": 70},
                         "backtest desde main")
    mostrar_historial()


def modo_angulos():
    """
    Backtest completo del Sistema de Ángulos Cruzados sobre BTC/USDT 15m.

    Requiere que los 4 CSVs de datos existan en data/.
    Si no existen, corré primero:
        from data.downloader import descargar_multi_tf
        descargar_multi_tf("BTC/USDT", exchange_id="binance", meses=24)
    """
    from data.downloader import cargar_multi_tf
    from data.inspector import limpiar
    from strategies.angulos_cruzados import AngulosCruzados, CATALOGO
    from backtests.engine import correr_backtest
    from backtests.splitter import split_datos
    from indicators.volatility import atr

    SIMBOLO        = "BTC/USDT"
    CAPITAL        = 10_000.0
    FEE            = 0.001      # 0.1% Binance maker/taker
    ATR_SL         = 1.0        # SL = 1× ATR  (definido en MAESTRO sección 7.2)
    ATR_TP         = 2.0        # TP = 2× ATR → R:R 2:1 (break-even baja a 33.3%)
    COOLDOWN_SL    = 4          # velas bloqueadas tras SL (frenar churn)
    UMBRAL_ENTRADA = 70.0
    UMBRAL_SALIDA  = 50.0

    print("=" * 60)
    print("  SISTEMA DE ÁNGULOS CRUZADOS — BTC/USDT 15m")
    print("=" * 60)
    print(f"  Patrones cargados : {len(CATALOGO)}")
    print(f"  Capital inicial   : ${CAPITAL:,.0f}")
    print(f"  Fee por trade     : {FEE*100:.1f}%")
    print(f"  SL / TP           : {ATR_SL}× ATR / {ATR_TP}× ATR  (R:R {ATR_TP/ATR_SL:.1f}:1)")
    print(f"  Cooldown post-SL  : {COOLDOWN_SL} velas")
    print(f"  Umbral entrada    : {UMBRAL_ENTRADA:.0f}% confianza")
    print()

    # 1. Cargar datos
    print("Cargando datos multi-TF...")
    tfs    = cargar_multi_tf(SIMBOLO)
    df_15m = limpiar(tfs["15m"])
    df_15m["atr"] = atr(df_15m, periodo=14)
    print(f"  15m: {len(df_15m):,} velas | "
          f"{df_15m.index[0].strftime('%Y-%m-%d')} → "
          f"{df_15m.index[-1].strftime('%Y-%m-%d')}")
    print()

    # 2. Calcular señales
    print("Calculando ejes y detectando patrones...")
    estrategia = AngulosCruzados(tfs, UMBRAL_ENTRADA, UMBRAL_SALIDA)
    df_señales = estrategia.calcular_señales(df_15m)

    n_entradas = (df_señales["señal"] == 1).sum()
    print(f"  Señales de entrada: {n_entradas:,}")
    print()

    # 3. Split train/test — solo usamos development set
    split  = split_datos(df_señales, holdout_pct=0.20)
    df_dev = split.development
    print(f"Development set (80%): {len(df_dev):,} velas | "
          f"{df_dev.index[0].strftime('%Y-%m-%d')} → "
          f"{df_dev.index[-1].strftime('%Y-%m-%d')}")
    print()

    # 4. Backtest
    print("Corriendo backtest...")
    resultado = correr_backtest(
        df_señales=df_dev,
        nombre_estrategia="Ángulos Cruzados 15m",
        capital_inicial=CAPITAL,
        fee_pct=FEE,
        atr_sl_mult=ATR_SL,
        atr_tp_mult=ATR_TP,
        cooldown_sl_velas=COOLDOWN_SL,
    )
    print(resultado.resumen())

    # 5. Win rate por patrón
    # Para cada trade, buscamos qué patrón estaba activo al momento de entrada.
    # El df_señales tiene la columna 'patron' por timestamp — hacemos lookup directo.
    import pandas as pd

    stats: dict = {}
    for trade in resultado.trades:
        ts = trade.entrada_tiempo
        # Buscar el patrón activo en el timestamp de entrada (o el más cercano anterior)
        if ts in df_dev.index:
            patron = df_dev.loc[ts, "patron"]
        else:
            idx_pos = df_dev.index.searchsorted(ts)
            if idx_pos == 0:
                patron = "—"
            else:
                patron = df_dev.iloc[idx_pos - 1]["patron"]

        if patron not in stats:
            stats[patron] = {"trades": 0, "ganadores": 0, "pnl": 0.0,
                             "stop_loss": 0, "take_profit": 0, "señal": 0, "fin_datos": 0}
        stats[patron]["trades"]    += 1
        stats[patron]["pnl"]       += trade.pnl_neto
        stats[patron]["ganadores"] += 1 if trade.pnl_neto > 0 else 0
        stats[patron][trade.motivo_salida] += 1

    print("\n" + "=" * 70)
    print("  WIN RATE POR PATRÓN")
    print("=" * 70)
    print(f"  {'Patrón':<6} {'Trades':>7} {'Win%':>6} {'PnL neto':>11} {'SL':>5} {'TP':>5}")
    print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*11} {'-'*5} {'-'*5}")

    # Ordenar por PnL neto descendente
    for patron, s in sorted(stats.items(), key=lambda x: x[1]["pnl"], reverse=True):
        win_pct = s["ganadores"] / s["trades"] * 100 if s["trades"] > 0 else 0
        sl_n    = s.get("stop_loss", 0)
        tp_n    = s.get("take_profit", 0)
        print(f"  {patron:<6} {s['trades']:>7,} {win_pct:>5.1f}% "
              f"  ${s['pnl']:>+9.2f} {sl_n:>5} {tp_n:>5}")

    print("=" * 70)


def modo_paper_angulos():
    """
    Paper trading local del Sistema de Ángulos Cruzados.
    Simula órdenes internamente sin tocar ningún exchange.
    """
    from paper_trading.engine import correr_paso_angulos
    correr_paso_angulos(
        simbolo="BTC/USDT",
        exchange_id="binance",
        capital_inicial=10_000.0,
        fee_pct=0.001,
        umbral_entrada=70.0,
        umbral_salida=50.0,
    )


def modo_demo_bingx():
    """
    Paper trading REAL contra BingX demo.
    Manda órdenes reales a la cuenta demo con USDT virtual.
    Requiere BINGX_DEMO_API_KEY y BINGX_DEMO_SECRET en .env
    Correr manualmente cada 15 minutos al cierre de cada vela.
    """
    from data.downloader import descargar_ohlcv, resamplear_ohlcv
    from strategies.angulos_cruzados import AngulosCruzados
    from indicators.volatility import atr
    from exchange.conexion import get_exchange_demo
    from exchange.ordenes import (
        obtener_posicion, abrir_long, cerrar_long,
        obtener_precio_actual, SIMBOLO_SWAP
    )
    from datetime import datetime, timezone

    USDT_POR_TRADE = 1_000.0   # capital por operación en USDT virtual
    UMBRAL_ENTRADA = 70.0
    UMBRAL_SALIDA  = 50.0

    print(f"\n[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}]")
    print("  BingX Demo — Ángulos Cruzados BTC/USDT 15m")
    print(f"  Capital por trade: ${USDT_POR_TRADE:,.0f} USDT virtual")

    # 1. Conectar al exchange demo
    exchange = get_exchange_demo()
    precio_actual = obtener_precio_actual(exchange)
    print(f"  Precio BTC actual: ${precio_actual:,.2f}")

    # 2. Descargar datos en vivo desde Binance (datos públicos, no requiere key)
    print("  Descargando datos...")
    df_5m  = descargar_ohlcv("BTC/USDT", "5m",  600, "binance")
    df_15m = descargar_ohlcv("BTC/USDT", "15m", 300, "binance")
    df_1h  = descargar_ohlcv("BTC/USDT", "1h",  300, "binance")
    df_10m = resamplear_ohlcv(df_5m, "10min")
    df_15m["atr"] = atr(df_15m, periodo=14)

    todos_tfs = {"5m": df_5m, "10m": df_10m, "15m": df_15m, "1h": df_1h}

    # 3. Calcular señales
    estrategia = AngulosCruzados(todos_tfs, UMBRAL_ENTRADA, UMBRAL_SALIDA)
    df_señales = estrategia.calcular_señales(df_15m)

    # Última vela cerrada (anteúltima — la última está en curso)
    vela      = df_señales.iloc[-2]
    señal     = int(vela["señal"])
    precio_sl_raw = vela["close"] - vela["atr"]   # 1× ATR abajo
    precio_tp_raw = vela["close"] + vela["atr"] * 1.5  # 1.5× ATR arriba

    print(f"  Vela cerrada : {df_señales.index[-2]}")
    print(f"  Patrón       : {vela['patron']} (confianza {vela['confianza']:.0f}% | score {vela['score']:+.0f})")
    print(f"  Señal        : {señal:+d}")

    # 4. Ver posición actual en el exchange
    posicion = obtener_posicion(exchange)
    en_posicion = posicion is not None
    print(f"  Posición abierta: {'SÍ' if en_posicion else 'NO'}")

    # 5. Actuar
    if señal == 1 and not en_posicion:
        print("  → ABRIENDO LONG en BingX demo...")
        orden = abrir_long(
            exchange,
            usdt_a_usar=USDT_POR_TRADE,
            sl_precio=round(precio_sl_raw, 1),
            tp_precio=round(precio_tp_raw, 1),
        )
        print(f"  Orden ID: {orden.get('id', '—')}")
        print(f"  Estado  : {orden.get('status', '—')}")

    elif señal == -1 and en_posicion:
        print("  → CERRANDO LONG en BingX demo...")
        orden = cerrar_long(exchange)
        if orden:
            print(f"  Orden ID: {orden.get('id', '—')}")

    else:
        print("  Sin acción.")

    # 6. Balance final
    # BingX demo usa "VST" (Virtual Standard Token) como moneda de margen
    balance = exchange.fetch_balance()
    usdt    = balance["total"].get("VST", balance["total"].get("USDT", 0))
    print(f"\n  Balance demo: ${usdt:,.2f} VST (USDT virtual)")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "paper"

    if modo == "backtest":
        modo_backtest()
    elif modo == "angulos":
        modo_angulos()
    elif modo == "paper_angulos":
        modo_paper_angulos()
    elif modo == "demo":
        modo_demo_bingx()
    elif modo == "reset":
        resetear_estado(10_000.0)
        print("Paper trader reseteado.")
    else:
        modo_paper_trading()
