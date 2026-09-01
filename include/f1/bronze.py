"""
Orquestador de prefetch a capa Bronze para las 3 fuentes.

Cada función es idempotente: si el archivo ya existe en disco, no se
vuelve a descargar.  Una corrida interrumpida puede continuarse sin
perder lo ya guardado.
"""

import time

from f1.jolpica import fetch_json
from f1.openf1 import prefetch_openf1
from f1.fastf1_source import prefetch_fastf1

# --- Jolpica ---

SEASONS_JOLPICA = list(range(2010, 2025))  # 2010–2024


def calendario(season: int) -> list[dict]:
    """Devuelve la lista de carreras de una temporada (Jolpica)."""
    data = fetch_json(f"{season}.json?limit=100")
    return data["MRData"]["RaceTable"]["Races"]


def prefetch_jolpica(season: int) -> dict:
    """Baja calendario + results + driverStandings de cada ronda.

    Retorna ``{rondas: int, errores: int}``.
    """
    rondas_ok = 0
    errores = 0

    carreras = calendario(season)
    for carrera in carreras:
        round_ = int(carrera["round"])

        # results.json
        try:
            fetch_json(f"{season}/{round_}/results.json?limit=100")
        except RuntimeError as e:
            print(f"  jolpica: error results {season}/{round_}: {e}")
            errores += 1

        # driverStandings de la ronda previa (anti-leakage).
        if round_ > 1:
            try:
                fetch_json(f"{season}/{round_ - 1}/driverStandings.json?limit=100")
            except RuntimeError as e:
                print(f"  jolpica: error standings {season}/{round_ - 1}: {e}")
                errores += 1

        rondas_ok += 1
        time.sleep(0.3)  # no golpear la API de más

    return {"rondas": rondas_ok, "errores": errores}
