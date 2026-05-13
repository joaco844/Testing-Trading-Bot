"""
strategies/angulos_cruzados.py

Detector de los 21 patrones del Sistema de Ángulos Cruzados.
Spec completa: MAESTRO_Patrones_Angulos_v2.docx

Flujo por vela:
  1. Calcular E1-E5 y score (indicators/axes.py)
  2. Para cada patrón, calcular confianza:
       - Si el eje mandatorio no se cumple → confianza = 0%
       - Confianza = (ejes que coinciden con firma ±1) / 5 × 100
  3. Seleccionar el patrón de mayor grupo (A > B > C > D > E > F)
     con confianza ≥ umbral_entrada (default 70%)
  4. Un patrón se activa solo si supera el umbral en 2 velas consecutivas
     (persistencia — evita señales de una sola vela)
  5. Un patrón se cierra cuando la confianza cae < umbral_salida (50%)
     en 2 velas consecutivas

Señales generadas (por ahora solo LONG — engine actual es solo spot):
  +1 → entrar LONG (patrón A/B/C de dirección long activo)
  -1 → salir (patrón long ya no activo)
   0 → esperar

Patrones SHORT detectados pero no operados todavía (requiere margen).
Patrones D/E/F: informativos, nunca generan señal de entrada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Dict

from strategies.base import EstrategiaBase
from indicators.axes import calcular_ejes


# ── Definición de un patrón ──────────────────────────────────────────────────

@dataclass
class Patron:
    """
    Representa un patrón del catálogo.

    firma:      valores esperados de los 5 ejes (para calcular confianza).
    mandatorio: función vectorizada que recibe el df de ejes y retorna
                una Serie booleana. Si es False → confianza = 0.
    prioridad:  A=1 (más alta) … F=6 (más baja). Determina cuál patrón
                se reporta cuando varios superan el umbral.
    """
    codigo:    str
    nombre:    str
    grupo:     str        # A, B, C, D, E, F
    prioridad: int        # 1-6
    direccion: str        # "long", "short", "none"
    firma:     Dict[str, int]
    mandatorio: Callable[[pd.DataFrame], pd.Series]
    score_min: int
    score_max: int


# ── Helpers para condiciones mandatorias vectorizadas ────────────────────────

def _eq(eje: str, val: int):
    """Eje == val exactamente."""
    return lambda ejes: ejes[eje] == val

def _gte(eje: str, val: int):
    """Eje >= val."""
    return lambda ejes: ejes[eje] >= val

def _lte(eje: str, val: int):
    """Eje <= val."""
    return lambda ejes: ejes[eje] <= val

def _and(*fns):
    """Todas las condiciones deben cumplirse."""
    return lambda ejes: np.logical_and.reduce([fn(ejes) for fn in fns])


# ── Condiciones especiales ────────────────────────────────────────────────────

def _mandatorio_c1(ejes: pd.DataFrame) -> pd.Series:
    """
    C1 Giro: E3=+2 Y el score subió ≥+5 en las últimas 5 velas.
    Captura la velocidad del cambio de dirección — lo que diferencia
    un giro real de un rebote lento.
    """
    velocidad = ejes["score"] - ejes["score"].shift(5)
    return (ejes["e3"] == 2) & (velocidad >= 5)

def _mandatorio_c2(ejes: pd.DataFrame) -> pd.Series:
    """C2 Vuelta: E3=-2 Y el score bajó ≤-5 en las últimas 5 velas."""
    velocidad = ejes["score"] - ejes["score"].shift(5)
    return (ejes["e3"] == -2) & (velocidad <= -5)

def _mandatorio_e2_tension(ejes: pd.DataFrame) -> pd.Series:
    """
    E2 Tensión: E1=+2 sostenido por ≥8 velas consecutivas.
    rolling(8).min() == 2 significa que las últimas 8 velas todas
    tuvieron E1=+2 — es decir, el precio lleva mucho tiempo
    sobreextendido respecto a la EMA200.
    """
    e1_sostenido = ejes["e1"].rolling(8).min() >= 2
    return e1_sostenido & (ejes["e1"] == 2)

def _mandatorio_e3_embudo(ejes: pd.DataFrame) -> pd.Series:
    """
    E3 Embudo: máximos decrecientes Y mínimos crecientes (triángulo).
    Simplificado: E3 cayendo por ≥4 velas (volumen decreciente
    mientras el rango se comprime) y velocidad de precio baja.
    """
    vol_decreciente = ejes["e3"].rolling(4).max() <= 0
    precio_quieto   = ejes["e2"].abs() <= 1
    return vol_decreciente & precio_quieto

def _mandatorio_c5_grieta(ejes: pd.DataFrame) -> pd.Series:
    """
    C5 Grieta: E4≤-1 Y E1≥+1 sostenidos por ≥5 velas.
    La grieta no es un evento de una vela — necesita confirmación.
    """
    divergencia  = (ejes["e4"] <= -1) & (ejes["e1"] >= 1)
    sostenido    = divergencia.rolling(5).min().astype(bool)
    return sostenido

def _mandatorio_d3(ejes: pd.DataFrame) -> pd.Series:
    """D3 Eco Rojo: E3≤-1 Y score≥+3 (rally sin volumen)."""
    return (ejes["e3"] <= -1) & (ejes["score"] >= 3)


# ── Catálogo de los 21 patrones ───────────────────────────────────────────────
# Orden A→F, dentro de cada grupo por número.
# La prioridad determina cuál se reporta cuando varios superen el umbral.

CATALOGO: list[Patron] = [

    # ── Grupo A — Señales direccionales fuertes ──────────────────────────────

    Patron(
        codigo="A1", nombre="Tormenta", grupo="A", prioridad=1, direccion="long",
        firma={"e1": 2, "e2": 2, "e3": 2, "e4": 2, "e5": 1},
        mandatorio=_gte("e3", 1),
        score_min=7, score_max=10,
    ),
    Patron(
        codigo="A2", nombre="Tormenta Roja", grupo="A", prioridad=1, direccion="short",
        firma={"e1": -2, "e2": -2, "e3": -2, "e4": -2, "e5": -1},
        mandatorio=_lte("e3", -1),
        score_min=-10, score_max=-7,
    ),
    Patron(
        codigo="A3", nombre="Corriente", grupo="A", prioridad=1, direccion="long",
        firma={"e1": 1, "e2": 1, "e3": 1, "e4": 1, "e5": 2},
        mandatorio=_eq("e5", 2),
        score_min=4, score_max=7,
    ),
    Patron(
        codigo="A4", nombre="Marea", grupo="A", prioridad=1, direccion="short",
        firma={"e1": -1, "e2": -1, "e3": -1, "e4": -1, "e5": -2},
        mandatorio=_eq("e5", -2),
        score_min=-7, score_max=-4,
    ),

    # ── Grupo B — Patrones trampa ────────────────────────────────────────────

    Patron(
        codigo="B1", nombre="Espejismo", grupo="B", prioridad=2, direccion="short",
        firma={"e1": 1, "e2": 2, "e3": -1, "e4": 1, "e5": -1},
        mandatorio=_and(_eq("e2", 2), _lte("e3", -1)),
        score_min=1, score_max=3,
    ),
    Patron(
        codigo="B2", nombre="Trampa Alta", grupo="B", prioridad=2, direccion="short",
        firma={"e1": 1, "e2": 2, "e3": -1, "e4": 1, "e5": -1},
        mandatorio=_and(_lte("e5", -1), _lte("e3", 0)),
        score_min=1, score_max=3,
    ),
    Patron(
        codigo="B3", nombre="Trampa Baja", grupo="B", prioridad=2, direccion="long",
        firma={"e1": -1, "e2": -2, "e3": -1, "e4": -1, "e5": 1},
        mandatorio=_and(_gte("e5", 1), _eq("e2", -2)),
        score_min=-5, score_max=-3,
    ),
    Patron(
        # B4 simplificado: el componente de reversión en ≤3 velas no puede
        # detectarse solo con ejes — requeriría price action adicional.
        # Aquí usamos la firma de ejes pura con el mandatorio de E3=+2.
        codigo="B4", nombre="Caza", grupo="B", prioridad=2, direccion="long",
        firma={"e1": 0, "e2": 2, "e3": 2, "e4": 0, "e5": -1},
        mandatorio=_eq("e3", 2),
        score_min=-2, score_max=4,
    ),

    # ── Grupo C — Transiciones de alto valor ─────────────────────────────────

    Patron(
        codigo="C1", nombre="Giro", grupo="C", prioridad=3, direccion="long",
        firma={"e1": -1, "e2": 1, "e3": 2, "e4": 1, "e5": 0},
        mandatorio=_mandatorio_c1,
        score_min=-2, score_max=4,
    ),
    Patron(
        codigo="C2", nombre="Vuelta", grupo="C", prioridad=3, direccion="short",
        firma={"e1": 1, "e2": -1, "e3": -2, "e4": -1, "e5": 0},
        mandatorio=_mandatorio_c2,
        score_min=-4, score_max=2,
    ),
    Patron(
        # C3: informativo — precede recuperación pero no es señal de compra inmediata
        codigo="C3", nombre="Ojo del Huracán", grupo="C", prioridad=3, direccion="none",
        firma={"e1": -2, "e2": -2, "e3": 2, "e4": -2, "e5": -2},
        mandatorio=_and(_eq("e3", 2), _lte("e1", -2), _lte("e4", -2)),
        score_min=-8, score_max=-6,
    ),
    Patron(
        codigo="C4", nombre="Resurgimiento", grupo="C", prioridad=3, direccion="long",
        firma={"e1": -1, "e2": 2, "e3": 2, "e4": 2, "e5": 0},
        mandatorio=_and(_eq("e2", 2), _eq("e3", 2)),
        score_min=3, score_max=6,
    ),
    Patron(
        codigo="C5", nombre="Grieta", grupo="C", prioridad=3, direccion="short",
        firma={"e1": 1, "e2": 0, "e3": -1, "e4": -1, "e5": -1},
        mandatorio=_mandatorio_c5_grieta,
        score_min=-3, score_max=-1,
    ),

    # ── Grupo D — Divergencias ───────────────────────────────────────────────

    Patron(
        codigo="D1", nombre="Fatiga", grupo="D", prioridad=4, direccion="none",
        firma={"e1": 1, "e2": 1, "e3": 1, "e4": -1, "e5": 1},
        mandatorio=_and(_lte("e4", -1), _gte("e1", 1)),
        score_min=2, score_max=4,
    ),
    Patron(
        codigo="D2", nombre="Eco", grupo="D", prioridad=4, direccion="none",
        firma={"e1": -1, "e2": -1, "e3": 1, "e4": 1, "e5": -1},
        mandatorio=_and(_gte("e4", 1), _lte("e1", -1)),
        score_min=-2, score_max=-1,
    ),
    Patron(
        codigo="D3", nombre="Eco Rojo", grupo="D", prioridad=4, direccion="none",
        firma={"e1": 1, "e2": 1, "e3": -1, "e4": 1, "e5": 1},
        mandatorio=_mandatorio_d3,
        score_min=2, score_max=4,
    ),

    # ── Grupo E — Estados especiales ─────────────────────────────────────────

    Patron(
        codigo="E1", nombre="El Ojo", grupo="E", prioridad=5, direccion="none",
        firma={"e1": 0, "e2": -1, "e3": -2, "e4": 0, "e5": 0},
        mandatorio=_eq("e3", -2),
        score_min=-4, score_max=-2,
    ),
    Patron(
        codigo="E2", nombre="Tensión", grupo="E", prioridad=5, direccion="none",
        firma={"e1": 2, "e2": 1, "e3": 0, "e4": 0, "e5": 1},
        mandatorio=_mandatorio_e2_tension,
        score_min=3, score_max=5,
    ),
    Patron(
        codigo="E3", nombre="Embudo", grupo="E", prioridad=5, direccion="none",
        firma={"e1": 0, "e2": -1, "e3": -1, "e4": 0, "e5": 0},
        mandatorio=_mandatorio_e3_embudo,
        score_min=-3, score_max=-2,
    ),

    # ── Grupo F — Señales secundarias ────────────────────────────────────────

    Patron(
        codigo="F1", nombre="Arena Viva", grupo="F", prioridad=6, direccion="none",
        firma={"e1": 0, "e2": -1, "e3": 1, "e4": 0, "e5": 0},
        mandatorio=_and(_eq("e3", 1), _lte("e2", -1)),
        score_min=-1, score_max=1,
    ),
    Patron(
        # F2 no se opera — rebote dentro de tendencia bajista,
        # solo válido para scalp con objetivo ajustado
        codigo="F2", nombre="Ola Corta", grupo="F", prioridad=6, direccion="none",
        firma={"e1": -1, "e2": 1, "e3": 0, "e4": 1, "e5": -2},
        mandatorio=_eq("e5", -2),
        score_min=-1, score_max=0,
    ),
    Patron(
        codigo="F3", nombre="Pausa", grupo="F", prioridad=6, direccion="none",
        firma={"e1": 1, "e2": -1, "e3": 0, "e4": -1, "e5": 2},
        mandatorio=_eq("e5", 2),
        score_min=0, score_max=1,
    ),
]


# ── Motor de detección ────────────────────────────────────────────────────────

def _confianza_patron(ejes: pd.DataFrame, patron: Patron) -> pd.Series:
    """
    Calcula la confianza (0-100%) de un patrón para cada fila del DataFrame.

    Un eje "coincide" si su valor está dentro de ±1 del valor en la firma.
    El mandatorio no tiene tolerancia — si no se cumple, confianza = 0.

    Todas las operaciones son vectorizadas sobre el DataFrame completo.
    """
    # Chequeo del eje mandatorio
    cumple_mandatorio = patron.mandatorio(ejes)

    # Contar coincidencias (tolerancia ±1) para los 5 ejes
    ejes_nombres = ["e1", "e2", "e3", "e4", "e5"]
    coincidencias = sum(
        (ejes[eje] - patron.firma[eje]).abs() <= 1
        for eje in ejes_nombres
    )

    confianza = (coincidencias / 5 * 100).astype(float)

    # Anular la confianza si el mandatorio no se cumple
    return confianza.where(cumple_mandatorio, 0.0)


def detectar_patrones(ejes: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la confianza de los 21 patrones para cada vela.

    Retorna un DataFrame con una columna por patrón (código como nombre),
    más columnas 'patron_activo' y 'direccion_activa' con el patrón de
    mayor prioridad que supera el 60% de confianza.

    Útil para análisis exploratorio — ver cuándo aparece cada patrón.
    """
    resultado = {}
    for patron in CATALOGO:
        resultado[patron.codigo] = _confianza_patron(ejes, patron)

    df_conf = pd.DataFrame(resultado, index=ejes.index)

    # Patrón de mayor prioridad con confianza ≥ 60%
    patron_activo  = pd.Series("—", index=ejes.index)
    direccion_activa = pd.Series("none", index=ejes.index)

    # Evaluar de mayor a menor prioridad (F→A), sobreescribiendo hacia A
    for patron in reversed(CATALOGO):
        supera = df_conf[patron.codigo] >= 60
        patron_activo   = patron_activo.where(~supera,   patron.codigo)
        direccion_activa = direccion_activa.where(~supera, patron.direccion)

    df_conf["patron_activo"]   = patron_activo
    df_conf["direccion_activa"] = direccion_activa

    return df_conf


# ── Estrategia ────────────────────────────────────────────────────────────────

class AngulosCruzados(EstrategiaBase):
    """
    Estrategia basada en el Sistema de Ángulos Cruzados.

    Genera señales LONG cuando un patrón de dirección 'long' (grupos A,
    B o C) tiene confianza ≥ umbral_entrada en 2 velas consecutivas.

    Cierra la posición cuando la confianza del patrón activo cae
    por debajo de umbral_salida en 2 velas consecutivas.

    Parámetros:
        todos_tfs:      dict con los 4 DataFrames (5m, 10m, 15m, 1h).
                        Necesarios para calcular E5.
        umbral_entrada: confianza mínima para abrir posición (default 70%).
        umbral_salida:  confianza mínima para mantener posición (default 50%).

    Uso:
        tfs      = cargar_multi_tf("BTC/USDT")
        estrategia = AngulosCruzados(tfs)
        df_con_señales = estrategia.calcular_señales(tfs["15m"])
    """

    def __init__(
        self,
        todos_tfs: Dict[str, pd.DataFrame],
        umbral_entrada: float = 70.0,
        umbral_salida:  float = 50.0,
    ):
        super().__init__("Ángulos Cruzados")
        self.todos_tfs      = todos_tfs
        self.umbral_entrada = umbral_entrada
        self.umbral_salida  = umbral_salida

    def calcular_señales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe el DataFrame del TF principal (mismo que todos_tfs["15m"]).
        Retorna el DataFrame con columnas adicionales:
          - e1..e5, score     → los 5 ejes y su suma
          - patron            → código del patrón activo (ej: "A1")
          - confianza         → porcentaje de confianza del patrón
          - señal             → +1 (entrar), -1 (salir), 0 (esperar)

        Solo genera señal=+1 para patrones LONG.
        Los patrones SHORT y los informativos (D/E/F) generan señal=0.
        """
        # ── 1. Calcular los 5 ejes ───────────────────────────────────────────
        ejes = calcular_ejes(df, self.todos_tfs)

        # ── 2. Confianza de cada patrón ──────────────────────────────────────
        confianzas: Dict[str, pd.Series] = {}
        for patron in CATALOGO:
            confianzas[patron.codigo] = _confianza_patron(ejes, patron)

        # ── 3. Patrón de mayor prioridad con confianza ≥ umbral_entrada ──────
        # "Mayor prioridad" = grupo A primero (prioridad 1), F último (6).
        # Iteramos de menor a mayor prioridad para que A sobreescriba a F.
        mejor_codigo    = pd.Series("—",    index=df.index)
        mejor_confianza = pd.Series(0.0,    index=df.index)
        mejor_direccion = pd.Series("none", index=df.index)

        for patron in reversed(CATALOGO):
            conf = confianzas[patron.codigo]
            supera = conf >= self.umbral_entrada
            mejor_codigo    = mejor_codigo.where(~supera,    patron.codigo)
            mejor_confianza = mejor_confianza.where(~supera, conf)
            mejor_direccion = mejor_direccion.where(~supera, patron.direccion)

        # ── 4. Persistencia de 2 velas ────────────────────────────────────────
        # El patrón solo se activa si supera el umbral en 2 velas consecutivas.
        # rolling(2).min() sobre un bool: True solo si las 2 últimas son True.
        # Esto evita señales de ruido de una sola vela.
        patron_long  = mejor_direccion == "long"
        patron_short = mejor_direccion == "short"

        # 2 velas consecutivas con patrón long activo → run confirmado
        long_activo = (
            patron_long
            & patron_long.shift(1).fillna(False)
        )

        # Entrada solo en la transición: primera vela del run (edge detection).
        # Evita re-entradas mid-pattern después de un SL/TP.
        long_se_activa = long_activo & ~long_activo.shift(1).fillna(False)

        # Salida: 2 velas consecutivas donde la confianza del mejor patrón
        # cayó por debajo del umbral de salida
        sin_señal = mejor_confianza < self.umbral_salida
        cierre    = (
            sin_señal
            & sin_señal.shift(1).fillna(False)
        )

        # ── 5. Generar señal ─────────────────────────────────────────────────
        # Regla: si en la misma vela hay señal de entrada Y de cierre,
        # la entrada tiene prioridad (caso raro pero posible al inicio).
        señal = pd.Series(0, index=df.index, dtype=int)
        señal[cierre]        = -1
        señal[long_se_activa] = 1

        # ── 6. Ensamblar resultado ────────────────────────────────────────────
        df_out = df.copy()
        df_out["e1"]        = ejes["e1"]
        df_out["e2"]        = ejes["e2"]
        df_out["e3"]        = ejes["e3"]
        df_out["e4"]        = ejes["e4"]
        df_out["e5"]        = ejes["e5"]
        df_out["score"]     = ejes["score"]
        df_out["patron"]    = mejor_codigo
        df_out["confianza"] = mejor_confianza
        df_out["señal"]     = señal

        return df_out
