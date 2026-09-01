import pendulum
from airflow.sdk import dag, task

SEASONS_JOLPICA = list(range(2022, 2027))
YEARS_OPENF1 = list(range(2022, 2027))
YEARS_FASTF1 = list(range(2022, 2027))


@dag(
    dag_id="f1_ingest",
    description="Ingesta de datos crudos de F1 a capa Bronze (Jolpica, OpenF1, FastF1).",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["f1", "bronze", "ingesta", "ciencia-de-datos"],
    max_active_tasks=3,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=1)},
)
def f1_ingest():

    @task()
    def ingest_jolpica(season: int) -> dict:
        from f1.bronze import prefetch_jolpica
        return prefetch_jolpica(season)

    @task()
    def ingest_openf1(year: int) -> dict:
        from f1.openf1 import prefetch_openf1
        return prefetch_openf1(year)

    @task()
    def ingest_fastf1(year: int) -> dict:
        from f1.fastf1_source import prefetch_fastf1
        return prefetch_fastf1(year)

    @task()
    def verificar_bronze(jolpica_results, openf1_results, fastf1_results) -> str:
        from f1.jolpica import BRONZE_DIR
        resumen = []
        for subfolder in ["jolpica", "openf1", "fastf1", "fastf1_cache"]:
            ruta = BRONZE_DIR / subfolder
            if ruta.exists():
                n = sum(1 for _ in ruta.rglob("*") if _.is_file())
                resumen.append(f"{subfolder}: {n} archivos")
            else:
                resumen.append(f"{subfolder}: no existe")

        msg = "Bronze coverage:\n" + "\n".join(f"  {r}" for r in resumen)
        print(msg)
        return msg

    jolpica = ingest_jolpica.expand(season=SEASONS_JOLPICA)
    openf1 = ingest_openf1.expand(year=YEARS_OPENF1)
    fastf1 = ingest_fastf1.expand(year=YEARS_FASTF1)

    verificar_bronze(jolpica, openf1, fastf1)


f1_ingest()
