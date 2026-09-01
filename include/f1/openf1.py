"""
Cliente de OpenF1 (api.openf1.org) con caché en disco y manejo de 429.

Cobertura: solo desde 2023.
Endpoints: meetings, sessions, weather, pit, stints, position (para
starting_grid y session results que apliquen).
"""

import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.openf1.org/v1"
HEADERS = {"User-Agent": "utn-frm-ciencia-de-datos-proyecto-f1/1.0"}

BRONZE_DIR = Path(
    os.environ.get(
        "F1_BRONZE_DIR",
        Path(__file__).resolve().parents[2] / "include" / "bronze",
    )
)

# ponytail: set mínimo de endpoints por sesión de carrera. Agregar cuando
# se necesiten features concretas (car_data, intervals, etc.).
SESSION_ENDPOINTS = ["weather", "pit", "stints", "position"]

# Solo años con cobertura en OpenF1.
YEARS = list(range(2023, 2025))  # 2023–2024


def _cache_path(endpoint: str, params_key: str) -> Path:
    """Ruta de caché: ``bronze/openf1/{endpoint}/{params_key}.json``."""
    return BRONZE_DIR / "openf1" / endpoint / f"{params_key}.json"


def _params_to_key(params: dict) -> str:
    """Clave determinística a partir de los query params ordenados."""
    return "_".join(f"{k}={v}" for k, v in sorted(params.items()))


def fetch_openf1(endpoint: str, params: dict,
                 retries: int = 5, backoff: float = 2.0) -> list:
    """GET ``BASE_URL/{endpoint}?{params}`` con caché en disco.

    OpenF1 devuelve listas JSON (no objetos envolventes tipo MRData).
    """
    key = _params_to_key(params)
    destino = _cache_path(endpoint, key)

    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))

    url = f"{BASE_URL}/{endpoint}"
    ultimo_error = None

    for intento in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                espera = float(retry_after) if retry_after else backoff ** intento
                time.sleep(espera)
                continue

            resp.raise_for_status()
            data = resp.json()

            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(json.dumps(data), encoding="utf-8")
            return data

        except requests.RequestException as e:
            ultimo_error = e
            time.sleep(backoff ** intento)

    raise RuntimeError(
        f"No se pudo bajar {url} params={params}: {ultimo_error}"
    )


def prefetch_openf1(year: int) -> dict:
    """Baja índice de meetings/sessions y datos por sesión de carrera.

    Retorna un resumen ``{sessions_bajadas: int, errores: int}``.
    """
    meetings = fetch_openf1("meetings", {"year": year})
    sessions = fetch_openf1("sessions", {"year": year})

    # Solo sesiones de carrera (session_type "Race").
    race_sessions = [
        s for s in sessions
        if s.get("session_type") == "Race" and s.get("session_key")
    ]

    bajadas = 0
    errores = 0

    for s in race_sessions:
        session_key = s["session_key"]
        for endpoint in SESSION_ENDPOINTS:
            try:
                fetch_openf1(endpoint, {"session_key": session_key})
                bajadas += 1
            except RuntimeError as e:
                print(f"  openf1: error {endpoint} session_key={session_key}: {e}")
                errores += 1
            time.sleep(0.3)

    return {"sessions_bajadas": bajadas, "errores": errores}


if __name__ == "__main__":
    print("Self-check: bajando meetings 2024 ...")
    data = fetch_openf1("meetings", {"year": 2024})
    archivo = _cache_path("meetings", "year=2024")
    assert archivo.exists(), f"Caché no creada: {archivo}"
    print(f"OK — {len(data)} meetings, caché en {archivo}")
