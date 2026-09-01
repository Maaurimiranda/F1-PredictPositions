"""
Cliente de la API Jolpica-F1 (Ergast) con caché en disco y manejo de 429.

Migrado de proyecto_f1_colab.ipynb (celda 1 — fetch_json).
El único cambio respecto del notebook es la capa de caché y el manejo
explícito de HTTP 429 / Retry-After.
"""

import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
HEADERS = {"User-Agent": "utn-frm-ciencia-de-datos-proyecto-f1/1.0"}

# include/ se monta como volumen en el contenedor de Airflow.
# Override con F1_BRONZE_DIR para desarrollo local fuera del repo.
BRONZE_DIR = Path(
    os.environ.get(
        "F1_BRONZE_DIR",
        Path(__file__).resolve().parents[2] / "include" / "bronze",
    )
)


def _cache_path(path: str) -> Path:
    """Espejo del path de la API bajo bronze/jolpica/, sin query string.

    Seguro porque la única query usada es ``?limit=100`` y es constante.
    """
    # ponytail: se descarta el query para la clave; si algún día varía, hashear.
    limpio = path.lstrip("/").split("?", 1)[0]
    return BRONZE_DIR / "jolpica" / limpio


def fetch_json(path: str, retries: int = 5, backoff: float = 2.0) -> dict:
    """Pide ``BASE_URL/<path>`` y devuelve el JSON parseado.

    1. Construye la URL.
    2. Calcula la clave determinística (espejo del path en disco).
    3. Si el archivo ya existe en caché, lo lee sin llamar a la API.
    4. Si no existe, hace el request.
    5. Si recibe 429, respeta Retry-After (o backoff exponencial).
    6. Al obtener respuesta exitosa, guarda el JSON en disco antes de devolver.
    7. Una ejecución interrumpida puede continuar sin re-descargar lo guardado.
    """
    destino = _cache_path(path)

    # Cache hit → no llamar a la API.
    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))

    url = f"{BASE_URL}/{path.lstrip('/')}"
    ultimo_error = None

    for intento in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                espera = float(retry_after) if retry_after else backoff ** intento
                time.sleep(espera)
                continue

            resp.raise_for_status()
            data = resp.json()

            # Guardar antes de devolver → reanudable.
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(json.dumps(data), encoding="utf-8")
            return data

        except requests.RequestException as e:
            ultimo_error = e
            time.sleep(backoff ** intento)

    raise RuntimeError(f"No se pudo bajar {url}: {ultimo_error}")


# --- Self-check -----------------------------------------------------------

if __name__ == "__main__":
    print("Self-check: bajando 2024.json ...")
    fetch_json("2024.json?limit=100")
    archivo = _cache_path("2024.json?limit=100")
    assert archivo.exists(), f"El archivo de caché no se creó: {archivo}"
    mtime1 = archivo.stat().st_mtime

    print("Self-check: segundo llamado (debe ser cache hit) ...")
    fetch_json("2024.json?limit=100")
    mtime2 = archivo.stat().st_mtime
    assert mtime1 == mtime2, "El archivo se re-descargó (mtime cambió)"

    print(f"OK — caché funciona: {archivo}")
