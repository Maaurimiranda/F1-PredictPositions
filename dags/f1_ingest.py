"""
DAG de ingesta Bronze para F1.

La idea es construir la capa cruda de datos de Fórmula 1 a partir de tres
fuentes: Jolpica, OpenF1 y FastF1. Cada tarea es idempotente: si los archivos
ya existen en include/bronze/, no se re-descargan y la ejecución vuelve a ser
rápida.

Este DAG se dispara manualmente para la demo: no tiene schedule porque la
corrida se controla desde la UI de Airflow o por trigger explícito.
"""

import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task

# El proyecto importa los módulos como `f1.*`; agregamos include/ al path para
# que funcione tanto en Docker como en un entorno local sin instalación editable.
_include = Path(__file__).resolve().parents[1] / "include"
if str(_include) not in sys.path:
    sys.path.insert(0, str(_include))

# Rangos de cobertura por fuente.
SEASONS_JOLPICA = list(range(2010, 2025))  # 2010–2024
YEARS_OPENF1 = list(range(2023, 2025))      # 2023–2024
YEARS_FASTF1 = list(range(2018, 2025))      # 2018–2024


@dag(
    dag_id="f1_ingest",
    description="Ingesta de datos crudos de F1 a la capa Bronze (Jolpica, OpenF1, FastF1).",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["f1", "bronze", "ingesta"],
    default_args={"retries": 2},
)
def f1_ingest():
    """Pipeline Bronze del proyecto F1.

    La idea central es poblar include/bronze/ con archivos particionados por
    fuente y, para cada una de ellas, por temporada o año. Las reejecuciones
    son seguras porque la lógica de descarga evita duplicar trabajo.
    """

    @task(task_id="ingest_jolpica")
    def ingest_jolpica(season: int) -> dict:
        """Baja calendario, resultados y standings de una temporada Jolpica."""
        from f1.bronze import prefetch_jolpica

        return prefetch_jolpica(season)

    @task(task_id="ingest_openf1")
    def ingest_openf1(year: int) -> dict:
        """Descarga meetings, sessions y resultados por carrera para OpenF1."""
        from f1.openf1 import prefetch_openf1

        return prefetch_openf1(year)

    @task(task_id="ingest_fastf1")
    def ingest_fastf1(year: int) -> dict:
        """Carga sesiones Q y R de FastF1 y escribe resultados/laps en disco."""
        from f1.fastf1_source import prefetch_fastf1

        return prefetch_fastf1(year)

    @task(task_id="verificar_bronze")
    def verificar_bronze(jolpica_results, openf1_results, fastf1_results) -> str:
        """Cuenta los archivos por fuente y deja evidencia legible del éxito."""
        from f1.jolpica import BRONZE_DIR

        resumen = []
        for subfolder in ["jolpica", "openf1", "fastf1", "fastf1_cache"]:
            ruta = BRONZE_DIR / subfolder
            if ruta.exists():
                archivos = list(ruta.rglob("*"))
                archivos_reales = [archivo for archivo in archivos if archivo.is_file()]
                resumen.append(f"{subfolder}: {len(archivos_reales)} archivos")
            else:
                resumen.append(f"{subfolder}: no existe")

        msg = "Bronze coverage:\n" + "\n".join(f"  {r}" for r in resumen)
        print(msg)
        return msg

    # Mapeo dinámico: una tarea por temporada/año para cada fuente.
    jolpica = ingest_jolpica.expand(season=SEASONS_JOLPICA)
    openf1 = ingest_openf1.expand(year=YEARS_OPENF1)
    fastf1 = ingest_fastf1.expand(year=YEARS_FASTF1)

    verificar_bronze(jolpica, openf1, fastf1)


f1_ingest()
