"""
DAG de ingesta Bronze — baja datos crudos de Jolpica, OpenF1 y FastF1.

Cada tarea es idempotente: si los archivos ya existen en include/bronze/,
no se vuelven a descargar.  Disparar de nuevo = SUCCESS rápido sin
requests nuevos.

schedule=None → se dispara manualmente (para la demo de la entrega,
la corrida ya está hecha previamente).
"""

import sys
from pathlib import Path
from datetime import datetime

from airflow.decorators import dag, task

# PYTHONPATH ya incluye include/ (via .env y Dockerfile).
# Para desarrollo local sin eso, agregar include/ al path.
_include = Path(__file__).resolve().parents[1] / "include"
if str(_include) not in sys.path:
    sys.path.insert(0, str(_include))

# Rangos de cobertura de cada fuente.
SEASONS_JOLPICA = list(range(2010, 2025))  # 2010–2024
YEARS_OPENF1 = list(range(2023, 2025))      # 2023–2024
YEARS_FASTF1 = list(range(2018, 2025))      # 2018–2024


@dag(
    dag_id="f1_ingest",
    description="Ingesta de datos crudos de F1 a capa Bronze (Jolpica, OpenF1, FastF1).",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["f1", "bronze", "ingesta"],
    default_args={"retries": 2},
)
def f1_ingest():
    """Pipeline Bronze: baja los datos crudos de las 3 fuentes y los
    persiste en include/bronze/, particionados por fuente y season/round.

    Para la entrega: disparar UNA VEZ para poblar Bronze. Las re-ejecuciones
    son instantáneas porque todo queda cacheado.
    """

    @task()
    def ingest_jolpica(season: int) -> dict:
        """Baja calendario, results y driverStandings de una temporada Jolpica."""
        from f1.bronze import prefetch_jolpica
        return prefetch_jolpica(season)

    @task()
    def ingest_openf1(year: int) -> dict:
        """Baja meetings, sessions y datos por carrera de OpenF1 (2023+)."""
        from f1.openf1 import prefetch_openf1
        return prefetch_openf1(year)

    @task()
    def ingest_fastf1(year: int) -> dict:
        """Carga sesiones Q y R de FastF1 y vuelca results/laps a CSV (2018+)."""
        from f1.fastf1_source import prefetch_fastf1
        return prefetch_fastf1(year)

    @task()
    def verificar_bronze(jolpica_results, openf1_results, fastf1_results) -> str:
        """Cuenta archivos en include/bronze/ por fuente y loguea la cobertura.

        Downstream de las 3 fuentes: solo corre si todas terminaron.
        Evidencia legible del SUCCESS para la demo.
        """
        from f1.jolpica import BRONZE_DIR
        resumen = []
        for subfolder in ["jolpica", "openf1", "fastf1", "fastf1_cache"]:
            ruta = BRONZE_DIR / subfolder
            if ruta.exists():
                archivos = list(ruta.rglob("*"))
                archivos_reales = [a for a in archivos if a.is_file()]
                resumen.append(f"{subfolder}: {len(archivos_reales)} archivos")
            else:
                resumen.append(f"{subfolder}: no existe")

        msg = "Bronze coverage:\n" + "\n".join(f"  {r}" for r in resumen)
        print(msg)
        return msg

    # Dynamic task mapping (Airflow ≥ 2.3).
    jolpica = ingest_jolpica.expand(season=SEASONS_JOLPICA)
    openf1 = ingest_openf1.expand(year=YEARS_OPENF1)
    fastf1 = ingest_fastf1.expand(year=YEARS_FASTF1)

    verificar_bronze(jolpica, openf1, fastf1)


f1_ingest()
