"""
backtests/engine.py

Motor de backtesting propio — simple, transparente y sin magia.

Supuestos del motor:
  - Operamos con todo el capital disponible en cada trade (1 posición a la vez)
  - Ejecutamos al precio de CIERRE de la vela de la señal
  - Fee aplicado en entrada y en salida
  - Stop-loss y take-profit chequeados usando high/low de cada vela
  - Sin apalancamiento (spot puro)
  - Solo posiciones largas (LONG) por ahora

Por qué usar high/low para chequear SL/TP y no el close:
  En una vela de 1 hora pueden pasar muchas cosas. Si el precio
  tocó nuestro SL en algún momento de esa hora, el close puede
  estar más arriba y nunca nos enteraríamos. Usar low para SL
  y high para TP es más realista. Es una aproximación — con datos
  de tick sería exacto, pero para backtesting con OHLCV es el
  estándar aceptado.

  Caso donde ambos se tocan en la misma vela (high >= TP y low <= SL):
  Asumimos que primero se tocó el SL (caso conservador/pesimista).
  En la realidad podría haber sido cualquiera. Esta suposición nos
  protege de ser demasiado optimistas.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Trade:
    """Representa un trade completo (entrada + salida)."""
    entrada_tiempo: pd.Timestamp
    entrada_precio: float
    salida_tiempo: pd.Timestamp
    salida_precio: float
    capital_usado: float
    fee_pct: float
    motivo_salida: str = "señal"   # "señal", "stop_loss", "take_profit", "fin_datos"

    @property
    def pnl_bruto(self) -> float:
        """Ganancia/pérdida sin fees."""
        return (self.salida_precio - self.entrada_precio) / self.entrada_precio * self.capital_usado

    @property
    def fee_total(self) -> float:
        """Fee de entrada + fee de salida."""
        return (self.capital_usado * self.fee_pct) + \
               ((self.capital_usado + self.pnl_bruto) * self.fee_pct)

    @property
    def pnl_neto(self) -> float:
        """Ganancia/pérdida real después de fees."""
        return self.pnl_bruto - self.fee_total

    @property
    def retorno_pct(self) -> float:
        """Retorno porcentual del trade."""
        return self.pnl_neto / self.capital_usado * 100

    @property
    def duracion(self) -> pd.Timedelta:
        return self.salida_tiempo - self.entrada_tiempo


@dataclass
class ResultadoBacktest:
    """Resultado completo de un backtest."""
    nombre_estrategia: str
    capital_inicial: float
    capital_final: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    @property
    def retorno_total_pct(self) -> float:
        return (self.capital_final - self.capital_inicial) / self.capital_inicial * 100

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def trades_ganadores(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl_neto > 0]

    @property
    def trades_perdedores(self) -> List[Trade]:
        return [t for t in self.trades if t.pnl_neto <= 0]

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return len(self.trades_ganadores) / self.num_trades * 100

    @property
    def profit_factor(self) -> float:
        """
        Ganancias brutas / Pérdidas brutas.
        > 1 = rentable. Un PF de 1.5 significa que por cada $1 perdido ganás $1.50.
        """
        ganancias = sum(t.pnl_neto for t in self.trades_ganadores)
        perdidas  = abs(sum(t.pnl_neto for t in self.trades_perdedores))
        if perdidas == 0:
            return float("inf")
        return ganancias / perdidas

    @property
    def max_drawdown_pct(self) -> float:
        """
        Máxima caída desde un pico hasta el valle siguiente.
        Métrica de riesgo más importante del backtest.
        """
        if self.equity_curve.empty:
            return 0.0
        pico_acumulado = self.equity_curve.cummax()
        drawdown = (self.equity_curve - pico_acumulado) / pico_acumulado * 100
        return drawdown.min()

    @property
    def sharpe_ratio(self) -> float:
        """Retorno ajustado por riesgo. >1 bueno, >2 muy bueno, <0 malo."""
        if len(self.trades) < 2:
            return 0.0
        retornos = [t.retorno_pct for t in self.trades]
        media = np.mean(retornos)
        std   = np.std(retornos, ddof=1)
        if std == 0:
            return 0.0
        return media / std

    @property
    def duracion_media_trades(self) -> str:
        if not self.trades:
            return "N/A"
        media = pd.Series([t.duracion for t in self.trades]).mean()
        horas = int(media.total_seconds() // 3600)
        return f"{horas}h"

    def resumen(self) -> str:
        # Contar salidas por motivo
        motivos = {}
        for t in self.trades:
            motivos[t.motivo_salida] = motivos.get(t.motivo_salida, 0) + 1

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
            f"  Duración media    : {self.duracion_media_trades}",
            f"  Salidas por señal : {motivos.get('señal', 0)}",
            f"  Salidas por SL    : {motivos.get('stop_loss', 0)}",
            f"  Salidas por TP    : {motivos.get('take_profit', 0)}",
            f"  Salidas expiradas : {motivos.get('expiracion', 0)}",
            f"{'='*55}",
        ]
        return "\n".join(lineas)


def correr_backtest(
    df_señales: pd.DataFrame,
    nombre_estrategia: str,
    capital_inicial: float = 10_000.0,
    fee_pct: float = 0.001,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    atr_sl_mult: Optional[float] = None,
    atr_tp_mult: Optional[float] = None,
    cooldown_sl_velas: int = 0,
    expiracion_velas: int = 0,
) -> ResultadoBacktest:
    """
    Ejecuta el backtest con gestión de riesgo (SL/TP opcionales).

    cooldown_sl_velas:
        Cuántas velas esperar antes de permitir una nueva entrada
        después de un stop-loss. 0 = sin cooldown (comportamiento original).
        Evita re-entradas inmediatas tras SL, que generan churn.

    expiracion_velas:
        Si el trade no tocó SL ni TP en N velas, se cierra a mercado.
        0 = sin expiración. Según MAESTRO sección 7.3: 20 velas.
        Evita trades "zombie" que permanecen abiertos indefinidamente.

    Dos modos de SL/TP (mutuamente excluyentes, ATR tiene prioridad):

    Modo 1 — Porcentaje fijo:
        stop_loss_pct=0.02  → SL a 2% debajo de la entrada
        take_profit_pct=0.04 → TP a 4% arriba de la entrada

    Modo 2 — Basado en ATR (requiere columna 'atr' en el DataFrame):
        atr_sl_mult=1.5 → SL a 1.5 × ATR debajo de la entrada
        atr_tp_mult=3.0 → TP a 3.0 × ATR arriba de la entrada

        Ventaja: se adapta a la volatilidad actual. En momentos de alta
        volatilidad el SL se aleja para no ser tocado por ruido normal.
        En momentos de calma se acerca para proteger mejor el capital.

    Orden de chequeo en cada vela con posición abierta:
      1. SL (low de la vela) — caso conservador primero
      2. TP (high de la vela)
      3. Señal de salida de la estrategia
    Si SL y TP se tocan en la misma vela → SL gana.
    """
    capital        = capital_inicial
    en_posicion    = False
    entrada_tiempo = None
    entrada_precio = None
    precio_sl      = None
    precio_tp      = None
    velas_cooldown  = 0   # velas restantes de bloqueo post-SL
    velas_en_trade  = 0   # velas transcurridas desde la entrada actual
    trades: List[Trade] = []
    equity: List[float] = []

    usar_atr = atr_sl_mult is not None or atr_tp_mult is not None
    if usar_atr and "atr" not in df_señales.columns:
        raise ValueError(
            "Para usar ATR-based SL/TP el DataFrame debe tener columna 'atr'. "
            "Asegurate de que la estrategia la calcule."
        )

    for timestamp, fila in df_señales.iterrows():
        precio = fila["close"]
        high   = fila["high"]
        low    = fila["low"]
        señal  = fila["señal"]

        if velas_cooldown > 0:
            velas_cooldown -= 1

        if en_posicion:
            salida_precio = None
            motivo_salida = None
            velas_en_trade += 1

            # 1. Stop-loss — usando el LOW de la vela
            if precio_sl is not None and low <= precio_sl:
                salida_precio = precio_sl
                motivo_salida = "stop_loss"

            # 2. Take-profit — usando el HIGH de la vela
            elif precio_tp is not None and high >= precio_tp:
                salida_precio = precio_tp
                motivo_salida = "take_profit"

            # 3. Expiración — MAESTRO sección 7.3: cerrar a mercado a las N velas
            elif expiracion_velas > 0 and velas_en_trade >= expiracion_velas:
                salida_precio = precio
                motivo_salida = "expiracion"

            # 4. Señal de salida de la estrategia
            elif señal == -1:
                salida_precio = precio
                motivo_salida = "señal"

            if salida_precio is not None:
                trade = Trade(
                    entrada_tiempo=entrada_tiempo,
                    entrada_precio=entrada_precio,
                    salida_tiempo=timestamp,
                    salida_precio=salida_precio,
                    capital_usado=capital,
                    fee_pct=fee_pct,
                    motivo_salida=motivo_salida,
                )
                capital += trade.pnl_neto
                trades.append(trade)
                en_posicion    = False
                precio_sl      = None
                precio_tp      = None
                velas_en_trade = 0
                if motivo_salida == "stop_loss":
                    velas_cooldown = cooldown_sl_velas

        # Abrir posición — respetar cooldown post-SL
        if señal == 1 and not en_posicion and velas_cooldown == 0:
            entrada_tiempo = timestamp
            entrada_precio = precio
            en_posicion    = True

            if usar_atr:
                # SL/TP dinámicos basados en la volatilidad actual (ATR al momento de entrada)
                atr_actual = fila["atr"]
                precio_sl  = (entrada_precio - atr_actual * atr_sl_mult) if atr_sl_mult else None
                precio_tp  = (entrada_precio + atr_actual * atr_tp_mult) if atr_tp_mult else None
            else:
                # SL/TP fijos en porcentaje
                precio_sl  = entrada_precio * (1 - stop_loss_pct)   if stop_loss_pct   else None
                precio_tp  = entrada_precio * (1 + take_profit_pct) if take_profit_pct else None

        equity.append(capital)

    # Posición abierta al final → cerrar al último precio
    if en_posicion:
        ultimo_precio    = df_señales["close"].iloc[-1]
        ultimo_timestamp = df_señales.index[-1]
        trade = Trade(
            entrada_tiempo=entrada_tiempo,
            entrada_precio=entrada_precio,
            salida_tiempo=ultimo_timestamp,
            salida_precio=ultimo_precio,
            capital_usado=capital,
            fee_pct=fee_pct,
            motivo_salida="fin_datos",
        )
        capital += trade.pnl_neto
        trades.append(trade)

    equity_curve = pd.Series(equity, index=df_señales.index)

    return ResultadoBacktest(
        nombre_estrategia=nombre_estrategia,
        capital_inicial=capital_inicial,
        capital_final=capital,
        trades=trades,
        equity_curve=equity_curve,
    )
