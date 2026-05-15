"""
backtests/engine_parciales.py

Motor de backtesting para estrategias con cierres parciales progresivos.

Diseñado para el Sistema Trend Following Diario donde:
  - La posición se divide en N partes iguales al entrar
  - Cada parte se cierra en un nivel técnico distinto (o al fin de sesión)
  - Todo cierra dentro de la sesión del día (UTC 00:00-23:59)
  - Si el precio baja hasta el SL antes de llegar a los niveles, se cierra todo

Diferencias clave respecto a engine.py:
  - Un trade puede tener múltiples cierres (uno por nivel + cierre de sesión)
  - El PnL es la suma de todos los cierres parciales
  - La posición remanente se actualiza con cada cierre parcial
  - El capital se actualiza solo al cerrar el trade completo (última fracción)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CierreParcial:
    """Un cierre parcial de la posición."""
    tiempo:   pd.Timestamp
    precio:   float
    fraccion: float    # fracción del capital original cerrada (ej: 0.333)
    motivo:   str      # "nivel", "stop_loss", "fin_sesion"


@dataclass
class TradeParcial:
    """
    Un trade completo con uno o más cierres parciales.

    La posición se divide en partes_totales partes iguales al entrar.
    Cada cierre reduce la fracción abierta en 1/partes_totales.
    """
    entrada_tiempo:  pd.Timestamp
    entrada_precio:  float
    sl_precio:       float
    niveles:         List[float]   # niveles de toma de ganancias, en orden ascendente
    partes_totales:  int           # N partes en que se divide la posición
    capital_entrada: float         # capital total comprometido en la entrada
    fee_pct:         float
    cierres:         List[CierreParcial] = field(default_factory=list)

    @property
    def fraccion_cerrada(self) -> float:
        return sum(c.fraccion for c in self.cierres)

    @property
    def fraccion_abierta(self) -> float:
        return max(0.0, 1.0 - self.fraccion_cerrada)

    @property
    def esta_cerrado(self) -> bool:
        return self.fraccion_abierta < 1e-9

    @property
    def pnl_neto(self) -> float:
        total = 0.0
        for c in self.cierres:
            capital_parcial = self.capital_entrada * c.fraccion
            pnl_bruto = (c.precio - self.entrada_precio) / self.entrada_precio * capital_parcial
            fee = capital_parcial * self.fee_pct + (capital_parcial + pnl_bruto) * self.fee_pct
            total += pnl_bruto - fee
        return total

    @property
    def retorno_pct(self) -> float:
        return self.pnl_neto / self.capital_entrada * 100

    @property
    def duracion(self) -> pd.Timedelta:
        if not self.cierres:
            return pd.Timedelta(0)
        return self.cierres[-1].tiempo - self.entrada_tiempo

    def motivo_resumen(self) -> str:
        """Describe cómo terminó el trade (para el resumen)."""
        motivos = [c.motivo for c in self.cierres]
        if "stop_loss" in motivos:
            return "stop_loss"
        if all(m == "fin_sesion" for m in motivos):
            return "fin_sesion"
        return "niveles"


@dataclass
class ResultadoParciales:
    """Resultado completo del backtest con parciales."""
    nombre_estrategia: str
    capital_inicial:   float
    capital_final:     float
    trades:            List[TradeParcial] = field(default_factory=list)
    equity_curve:      pd.Series = field(default_factory=pd.Series)

    @property
    def retorno_total_pct(self) -> float:
        return (self.capital_final - self.capital_inicial) / self.capital_inicial * 100

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def trades_ganadores(self) -> List[TradeParcial]:
        return [t for t in self.trades if t.pnl_neto > 0]

    @property
    def trades_perdedores(self) -> List[TradeParcial]:
        return [t for t in self.trades if t.pnl_neto <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.trades_ganadores) / self.num_trades * 100

    @property
    def profit_factor(self) -> float:
        ganancias = sum(t.pnl_neto for t in self.trades_ganadores)
        perdidas  = abs(sum(t.pnl_neto for t in self.trades_perdedores))
        if perdidas == 0:
            return float("inf")
        return ganancias / perdidas

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        pico      = self.equity_curve.cummax()
        drawdown  = (self.equity_curve - pico) / pico * 100
        return drawdown.min()

    @property
    def sharpe_ratio(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        retornos = [t.retorno_pct for t in self.trades]
        media = np.mean(retornos)
        std   = np.std(retornos, ddof=1)
        if std == 0:
            return 0.0
        return media / std

    @property
    def duracion_media(self) -> str:
        if not self.trades:
            return "N/A"
        media = pd.Series([t.duracion for t in self.trades]).mean()
        horas = int(media.total_seconds() // 3600)
        return f"{horas}h"

    def resumen(self) -> str:
        por_motivo = {}
        for t in self.trades:
            m = t.motivo_resumen()
            por_motivo[m] = por_motivo.get(m, 0) + 1

        lineas = [
            f"\n{'='*55}",
            f"  BACKTEST: {self.nombre_estrategia}",
            f"{'='*55}",
            f"  Capital inicial   : ${self.capital_inicial:,.2f}",
            f"  Capital final     : ${self.capital_final:,.2f}",
            f"  Retorno total     : {self.retorno_total_pct:+.2f}%",
            f"  Trades totales    : {self.num_trades}",
            f"  Win rate          : {self.win_rate:.1f}%",
            f"  Profit factor     : {self.profit_factor:.2f}",
            f"  Max drawdown      : {self.max_drawdown_pct:.2f}%",
            f"  Sharpe ratio      : {self.sharpe_ratio:.2f}",
            f"  Duración media    : {self.duracion_media}",
            f"  Cerrados por SL       : {por_motivo.get('stop_loss', 0)}",
            f"  Cerrados en niveles   : {por_motivo.get('niveles', 0)}",
            f"  Cerrados fin sesión   : {por_motivo.get('fin_sesion', 0)}",
            f"{'='*55}",
        ]
        return "\n".join(lineas)


def correr_backtest_parciales(
    df_1h:        pd.DataFrame,
    df_diario:    pd.DataFrame,
    nombre:       str,
    capital_inicial: float = 10_000.0,
    fee_pct:      float = 0.001,
    riesgo_pct:   float = 0.01,
) -> ResultadoParciales:
    """
    Simula el sistema Trend Following Diario usando datos 1h para timing intradía.

    df_1h     : OHLCV en 1h con DatetimeIndex UTC (para simular la sesión intradía)
    df_diario : DataFrame con las señales diarias precalculadas por la estrategia.
                Columnas requeridas:
                  - señal          : 1 (día con condiciones alcistas), 0 (sin señal)
                  - entry_level    : precio de entrada (50% del cuerpo de la vela anterior)
                  - sl_precio      : precio de stop loss
                  - niveles        : lista de precios objetivo para parciales (ordenados asc)
    riesgo_pct: fracción del capital a arriesgar por trade (ej: 0.01 = 1%).
                El capital usado en el trade = riesgo_pct / distancia_sl_pct.
                Si la distancia SL es 1% del precio de entrada, el capital usado
                = riesgo_pct / 0.01 = 100% del capital disponible.
                Para evitar over-leverage, se clampa a máx 50% del capital.

    Lógica de simulación:
      Para cada día con señal=1:
        1. Durante las velas 1h de ese día (00:00-22:00 UTC):
           a. Si no hay trade abierto: buscar que low <= entry_level (entrada al nivel)
           b. Si hay trade abierto:
              - Si low <= sl_precio → cerrar todo con SL
              - Si high >= nivel_siguiente → cerrar 1/N partes al nivel
        2. A las 23:00 UTC: cerrar todo lo que quede abierto (fin de sesión)
    """
    capital = capital_inicial
    trades: List[TradeParcial] = []
    equity: List[tuple] = []   # (timestamp, capital)

    # Iterar por cada día de señal
    for fecha_dia, fila_dia in df_diario.iterrows():
        if fila_dia["señal"] != 1:
            continue

        entry_level = fila_dia["entry_level"]
        sl_precio   = fila_dia["sl_precio"]
        niveles     = fila_dia["niveles"]   # lista de floats, puede estar vacía

        if not niveles or entry_level <= sl_precio:
            continue

        # Filtrar velas 1h de este día (UTC 00:00 a 23:59)
        fecha_str = str(fecha_dia.date())
        velas_dia = df_1h.loc[fecha_str]
        if velas_dia.empty:
            continue

        # Calcular tamaño de posición basado en riesgo fijo
        distancia_sl_pct = (entry_level - sl_precio) / entry_level
        if distancia_sl_pct <= 0:
            continue
        capital_trade = min(capital * riesgo_pct / distancia_sl_pct, capital * 0.50)

        partes_totales = len(niveles) + 1   # N niveles + 1 cierre de sesión
        fraccion_por_parte = 1.0 / partes_totales

        trade: Optional[TradeParcial] = None
        nivel_idx = 0  # índice del próximo nivel a alcanzar
        entrada_intentada = False  # solo una entrada por sesión

        # El precio debe ABRIR por encima del entry_level para que haya
        # un retroceso real durante la sesión. Si ya abre debajo, es un
        # gap down — no hay pullback, hay caída directa. Saltamos el día.
        primera_vela = velas_dia.iloc[0]
        if primera_vela["open"] <= entry_level:
            continue

        for ts, vela in velas_dia.iterrows():
            hora_utc = ts.hour

            if trade is None and not entrada_intentada:
                # Buscar entrada: la vela debe tocar el nivel (low <= entry_level)
                # Solo intentamos entrar antes de las 20:00 para dejar margen de gestión
                if hora_utc < 20 and vela["low"] <= entry_level:
                    trade = TradeParcial(
                        entrada_tiempo  = ts,
                        entrada_precio  = entry_level,
                        sl_precio       = sl_precio,
                        niveles         = niveles,
                        partes_totales  = partes_totales,
                        capital_entrada = capital_trade,
                        fee_pct         = fee_pct,
                    )
                    nivel_idx = 0
                    entrada_intentada = True  # no re-entrar en esta sesión

                    # Si la misma vela de entrada también toca el SL,
                    # cerramos inmediatamente: el precio bajó al entry
                    # y siguió hacia abajo sin rebotar.
                    if vela["low"] <= sl_precio:
                        trade.cierres.append(CierreParcial(
                            tiempo   = ts,
                            precio   = sl_precio,
                            fraccion = 1.0,
                            motivo   = "stop_loss",
                        ))
                        capital += trade.pnl_neto
                        trades.append(trade)
                        equity.append((ts, capital))
                        trade     = None
                        nivel_idx = 0
            else:
                # Gestión del trade abierto
                # 1. Stop loss — tiene prioridad
                if vela["low"] <= sl_precio:
                    fraccion_restante = trade.fraccion_abierta
                    trade.cierres.append(CierreParcial(
                        tiempo   = ts,
                        precio   = sl_precio,
                        fraccion = fraccion_restante,
                        motivo   = "stop_loss",
                    ))

                # 2. Niveles de ganancia — en orden ascendente
                elif nivel_idx < len(niveles) and vela["high"] >= niveles[nivel_idx]:
                    trade.cierres.append(CierreParcial(
                        tiempo   = ts,
                        precio   = niveles[nivel_idx],
                        fraccion = fraccion_por_parte,
                        motivo   = "nivel",
                    ))
                    nivel_idx += 1

                # 3. Fin de sesión — cierre forzado en la última vela del día
                if hora_utc >= 23 and not trade.esta_cerrado:
                    trade.cierres.append(CierreParcial(
                        tiempo   = ts,
                        precio   = vela["close"],
                        fraccion = trade.fraccion_abierta,
                        motivo   = "fin_sesion",
                    ))

                if trade.esta_cerrado:
                    capital += trade.pnl_neto
                    trades.append(trade)
                    equity.append((ts, capital))
                    trade = None
                    nivel_idx = 0

        # Si la sesión terminó con trade aún abierto (ej: no hubo vela 23:00)
        if trade is not None and not trade.esta_cerrado:
            ultima_vela = velas_dia.iloc[-1]
            trade.cierres.append(CierreParcial(
                tiempo   = velas_dia.index[-1],
                precio   = ultima_vela["close"],
                fraccion = trade.fraccion_abierta,
                motivo   = "fin_sesion",
            ))
            capital += trade.pnl_neto
            trades.append(trade)
            equity.append((velas_dia.index[-1], capital))

    if equity:
        eq_index  = [e[0] for e in equity]
        eq_values = [e[1] for e in equity]
        equity_curve = pd.Series(eq_values, index=eq_index)
    else:
        equity_curve = pd.Series(dtype=float)

    return ResultadoParciales(
        nombre_estrategia = nombre,
        capital_inicial   = capital_inicial,
        capital_final     = capital,
        trades            = trades,
        equity_curve      = equity_curve,
    )
