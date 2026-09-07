"""
Ingesta FastF1 a capa Bronze.

FastF1 es librería Python, no REST. Su caché nativo se apunta a
``bronze/fastf1_cache/``. Además volcamos resultados y laps resumidos a CSV
dentro de ``bronze/fastf1/{year}/{round}/`` para que Silver los lea sin
recargar la librería.

Cobertura de timing completo: 2018+.

IMPORTANTE: este archivo NO se llama ``fastf1.py`` — ese nombre colisiona
con ``import fastf1``.
"""

import datetime
import os
from pathlib import Path

BRONZE_DIR = Path(
    os.environ.get(
        "F1_BRONZE_DIR",
        Path(__file__).resolve().parents[2] / "include" / "output" / "bronze",
    )
)

# Habilitar caché nativo de FastF1 antes de cualquier carga de sesión.
# ponytail: el caché nativo ya es Bronze suficiente; los CSV volcados son
# para que Silver no dependa de FastF1 en runtime.
FASTF1_CACHE_DIR = BRONZE_DIR / "fastf1_cache"

YEARS = list(range(2018, 2025))  # 2018–2024


def _ensure_cache():
    """Habilita el caché nativo de FastF1."""
    import fastf1
    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))


def _dump_session(year: int, round_num: int, session_type: str):
    """Carga una sesión y vuelca results + laps a CSV.

    ``session_type``: ``"Q"`` (qualifying) o ``"R"`` (race).
    """
    import fastf1

    out_dir = BRONZE_DIR / "source=fastf1" / f"year={year}" / f"round={round_num}" / f"session={session_type}"
    results_path = out_dir / "results.csv"
    laps_path = out_dir / "laps.csv"

    # Si ambos CSV ya existen, no recargar (idempotente).
    if results_path.exists() and laps_path.exists():
        return

    _ensure_cache()
    session = fastf1.get_session(year, round_num, session_type)
    session.load()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Results (clasificación/carrera).
    if session.results is not None and not session.results.empty:
        session.results.to_csv(results_path, index=False)

    # Laps resumidos (sin telemetría — ponytail: agregar telemetría solo si
    # se necesita como feature; es pesada y lenta).
    if session.laps is not None and not session.laps.empty:
        cols = [
            c for c in [
                "DriverNumber", "Driver", "LapNumber", "LapTime",
                "Sector1Time", "Sector2Time", "Sector3Time",
                "Compound", "TyreLife", "FreshTyre",
                "IsPersonalBest", "Position",
            ]
            if c in session.laps.columns
        ]
        session.laps[cols].to_csv(laps_path, index=False)


def prefetch_fastf1(year: int) -> dict:
    """Carga Q y R de cada ronda del año y vuelca a CSV.

    Retorna ``{rondas: int, errores: int}``.
    """
    import pandas as pd
    import fastf1

    _ensure_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    today = datetime.date.today()

    rondas = 0
    errores = 0

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        if round_num < 1:
            continue

        for stype in ("Q", "R"):
            try:
                try:
                    sdate = event.get_session_date(stype)
                except Exception:
                    sdate = event.get("EventDate")

                if pd.notna(sdate):
                    if hasattr(sdate, "date"):
                        sdate = sdate.date()
                    if sdate > today:
                        continue

                _dump_session(year, round_num, stype)
                rondas += 1
            except Exception as e:
                print(f"  fastf1: error {year}/{round_num}/{stype}: {e}")
                errores += 1

    return {"rondas": rondas, "errores": errores}


if __name__ == "__main__":
    print("Self-check: cargando 2024 R1 Race ...")
    _ensure_cache()
    _dump_session(2024, 1, "R")
    r = BRONZE_DIR / "source=fastf1" / "year=2024" / "round=1" / "session=R" / "results.csv"
    assert r.exists(), f"No se creó {r}"
    print(f"OK — {r}")
