"""
strategies/base.py

Clase base que deben heredar todas las estrategias.

Por qué una clase base:
  - Obliga a que todas las estrategias tengan la misma interfaz
  - El backtester puede trabajar con cualquier estrategia sin saber
    cuál es: solo llama a .calcular_señales(df)
  - Si mañana agregamos 10 estrategias, todas funcionan igual desde afuera
"""

from abc import ABC, abstractmethod
import pandas as pd


class EstrategiaBase(ABC):
    """
    Todas las estrategias deben heredar de esta clase
    e implementar el método calcular_señales().
    """

    def __init__(self, nombre: str):
        self.nombre = nombre

    @abstractmethod
    def calcular_señales(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recibe un DataFrame con OHLCV y retorna el mismo DataFrame
        con una columna adicional 'señal':
          1  → comprar (LONG)
         -1  → vender / cerrar posición
          0  → no hacer nada

        CRÍTICO — Lookahead bias:
        Las señales deben calcularse usando SOLO datos disponibles
        en el momento t, nunca datos de t+1 en adelante.
        En pandas: nunca usar shift(-n) para generar señales.
        """
        pass

    def __repr__(self):
        return f"<Estrategia: {self.nombre}>"
