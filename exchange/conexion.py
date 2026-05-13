"""
exchange/conexion.py

Conexión autenticada a BingX — modo demo (sandbox) o producción.

Las API keys se leen del archivo .env — nunca hardcodeadas en el código.
El modo demo usa la URL open-api-vst.bingx.com en lugar de la real.
"""

import os
import ccxt
from dotenv import load_dotenv
from pathlib import Path


def get_exchange_demo() -> ccxt.bingx:
    """
    Retorna una instancia autenticada de BingX en modo DEMO.

    Lee las keys de .env:
        BINGX_DEMO_API_KEY=...
        BINGX_DEMO_SECRET=...

    El sandbox de BingX opera contra el mercado real pero con
    fondos virtuales — los precios son reales, el dinero no.
    """
    _cargar_env()

    api_key = os.getenv("BINGX_DEMO_API_KEY")
    secret  = os.getenv("BINGX_DEMO_SECRET")

    if not api_key or not secret:
        raise ValueError(
            "Faltan BINGX_DEMO_API_KEY o BINGX_DEMO_SECRET en el archivo .env\n"
            "Asegurate de que el .env esté en trading_bot/ con el formato:\n"
            "  BINGX_DEMO_API_KEY=tu_key\n"
            "  BINGX_DEMO_SECRET=tu_secret"
        )

    exchange = ccxt.bingx({
        "apiKey":          api_key,
        "secret":          secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # demo de BingX es en futuros perpetuos
        },
    })

    # Activar modo sandbox — cambia la URL base a open-api-vst.bingx.com
    exchange.set_sandbox_mode(True)

    return exchange


def _cargar_env():
    """Carga el .env desde la carpeta del proyecto."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
