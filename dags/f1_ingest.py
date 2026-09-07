"""
### Pipeline de datos F1 — Ciencia de Datos, UTN FRM 2026

Construye el dataset de carreras de Fórmula 1 para predecir el Top 10
antes del inicio de cada carrera, siguiendo el modelo medallón en dos capas.

**Dos capas, dos bloques de tareas en el grafo:**

* **Bronze** (`ingest_jolpica`, `ingest_openf1`, `ingest_fastf1`) — datos
  crudos tal como los devuelven las APIs, guardados en disco y particionados
  por fuente/año/ronda. Es el único bloque que toca la red. No interpreta nada.

* **Silver** (`build_silver_race`, `build_silver_laps`, `build_unified_silver`) —
  tablas tipadas y deduplicadas. No toca la red: lee del bronce y enriquece
  las vueltas con toda la información disponible a nivel de carrera.

La separación no es decorativa. Si aparece un bug en el parseo, se corrige
Silver y se reprocesa el bronce que ya está en disco, sin volver a llamar a
las APIs. Una corrida interrumpida puede continuarse sin re-descargar nada.

**Tres fuentes, cobertura distinta:**

* **Jolpica-F1** (API Ergast): resultados, grid, standings — 1950 a hoy.
* **FastF1**: qualifying detallado, vueltas, tiempos por sector — 2018+.
* **OpenF1**: clima, pits, stints en tiempo real — 2023+.

**Modo `subset`** baja solo la temporada 2024 para validar el pipeline rápido
(~minutos). **Modo `full`** baja todas las temporadas configuradas (~horas).

La capa Oro — features del modelo, predicciones — no se construye acá.
Se arma en las unidades siguientes sobre esta misma plata.
"""
from __future__ import annotations

import logging

import pendulum
from airflow.sdk import Param, dag, task

from f1.openf1 import YEARS as YEARS_OPENF1

log = logging.getLogger(__name__)

SEASONS_JOLPICA = list(range(2022, 2027))
YEARS_FASTF1 = list(range(2022, 2027))

# Temporada de referencia para el modo subset.
_SUBSET_SEASON = 2024


@dag(
    dag_id="f1_ingest",
    description="Pipeline F1 completo: ingesta Bronze (Jolpica, OpenF1, FastF1) + construcción Silver.",
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    tags=["f1", "bronze", "silver", "ingesta", "ciencia-de-datos"],
    max_active_tasks=3,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=1)},
    doc_md=__doc__,
    params={
        "mode": Param(
            "full",
            enum=["subset", "full"],
            title="Modo de corrida",
            description=(
                "subset: solo la temporada 2024 (~minutos, para validar el pipeline). "
                "full: todas las temporadas configuradas (2022-2026, ~horas)."
            ),
        ),
        "force": Param(
            False,
            type="boolean",
            title="Forzar re-descarga",
            description=(
                "Si es True, ignora la caché en disco y vuelve a descargar todo. "
                "Útil si se sospecha que los datos en Bronze están corruptos."
            ),
        ),
    },
)
def f1_ingest():

    @task(map_index_template="Temporada {{ task.op_kwargs['season'] }}")
    def ingest_jolpica(season: int) -> dict:
        from f1.bronze import prefetch_jolpica
        log.info("Iniciando ingesta Jolpica para temporada %s", season)
        result = prefetch_jolpica(season)
        log.info("Jolpica %s: %s rondas, %s errores", season, result.get("rondas"), result.get("errores"))
        return result

    @task(map_index_template="Año {{ task.op_kwargs['year'] }}")
    def ingest_openf1(year: int) -> dict:
        from f1.openf1 import prefetch_openf1
        log.info("Iniciando ingesta OpenF1 para año %s", year)
        result = prefetch_openf1(year)
        if result.get("omitido"):
            log.warning("OpenF1 %s: año fuera de cobertura, omitido", year)
        else:
            log.info("OpenF1 %s: %s sesiones bajadas, %s errores", year, result.get("sessions_bajadas"), result.get("errores"))
        return result

    @task(map_index_template="Año {{ task.op_kwargs['year'] }}")
    def ingest_fastf1(year: int) -> dict:
        from f1.fastf1_source import prefetch_fastf1
        log.info("Iniciando ingesta FastF1 para año %s", year)
        result = prefetch_fastf1(year)
        log.info("FastF1 %s: %s rondas, %s errores", year, result.get("rondas"), result.get("errores"))
        return result

    @task
    def verificar_bronze(jolpica_results, openf1_results, fastf1_results) -> str:
        from f1.jolpica import BRONZE_DIR

        resumen = []
        for subfolder in ["source=jolpica", "source=openf1", "source=fastf1", "fastf1_cache"]:
            ruta = BRONZE_DIR / subfolder
            if ruta.exists():
                n = sum(1 for _ in ruta.rglob("*") if _.is_file())
                resumen.append(f"{subfolder}: {n} archivos")
                log.info("Bronze %s: %s archivos", subfolder, n)
            else:
                resumen.append(f"{subfolder}: no existe todavía")
                log.warning("Bronze %s: carpeta no encontrada", subfolder)

        omitidos = [r for r in openf1_results if r.get("omitido")]
        if omitidos:
            resumen.append(f"openf1: {len(omitidos)} año(s) omitido(s) por falta de cobertura")

        msg = "Bronze coverage:\n" + "\n".join(f"  {r}" for r in resumen)
        log.info(msg)
        return msg

    @task
    def build_silver_race(**context) -> str:
        from f1.silver_builder import build_driver_race_features
        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        ds = momento.date().isoformat()
        log.info("Construyendo Silver race features con sufijo %s", ds)
        ruta = build_driver_race_features(date_suffix=ds)
        log.info("Silver race features escrito en %s", ruta)
        return ruta

    @task
    def build_silver_laps(**context) -> str:
        from f1.silver_builder import build_driver_lap_features
        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        ds = momento.date().isoformat()
        log.info("Construyendo Silver lap features con sufijo %s", ds)
        ruta = build_driver_lap_features(date_suffix=ds)
        log.info("Silver lap features escrito en %s", ruta)
        return ruta

    @task
    def validate_silver_race(ruta: str, **context) -> str:
        import pandas as pd
        from f1.silver_builder import validate_silver_dataset
        mode = context["params"].get("mode", "full")
        min_rows = 400 if mode == "subset" else 1000
        log.info("Validando Silver race features desde %s (modo %s, min_rows=%s)", ruta, mode, min_rows)
        df = pd.read_csv(ruta)
        validate_silver_dataset(
            df, key_cols=["season", "round", "driver_id"], min_rows=min_rows,
            required_non_null=["season", "round", "driver_id", "finish_position"]
        )
        log.info("Validación race features OK: %s filas x %s columnas", len(df), df.shape[1])
        return "Validación Race: OK"

    @task
    def validate_silver_laps(ruta: str, **context) -> str:
        import pandas as pd
        from f1.silver_builder import validate_silver_dataset
        mode = context["params"].get("mode", "full")
        min_rows = 10000 if mode == "subset" else 20000
        log.info("Validando Silver lap features desde %s (modo %s, min_rows=%s)", ruta, mode, min_rows)
        df = pd.read_csv(ruta)
        validate_silver_dataset(
            df, key_cols=["season", "round", "driver_code", "lap_number"],
            min_rows=min_rows,
            required_non_null=["season", "round", "driver_code", "lap_number", "lap_time_s"]
        )
        log.info("Validación lap features OK: %s filas x %s columnas", len(df), df.shape[1])
        return "Validación Laps: OK"

    @task
    def build_unified_silver(race_path: str, laps_path: str, **context) -> str:
        from f1.silver_builder import build_unified_lap_race_features
        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        ds = momento.date().isoformat()
        log.info("Construyendo Silver unificado (laps + race) con sufijo %s", ds)
        ruta = build_unified_lap_race_features(race_path, laps_path, date_suffix=ds)
        log.info("Silver unificado escrito en %s", ruta)
        return ruta

    @task
    def validate_unified_silver(ruta: str, **context) -> str:
        import pandas as pd
        mode = context["params"].get("mode", "full")
        log.info("Validando Silver unificado desde %s (modo %s)", ruta, mode)
        df = pd.read_csv(ruta)

        # 1. Clave sin duplicados (Test operativo de la unidad de análisis)
        key_cols = ["season", "round", "driver_code", "lap_number"]
        faltantes = [c for c in key_cols if c not in df.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas de la clave primaria: {faltantes}")
        
        is_key_unique = df.set_index(key_cols).index.is_unique
        if not is_key_unique:
            dup_count = int(df.duplicated(subset=key_cols).sum())
            raise ValueError(
                f"Clave primaria duplicada sobre {key_cols}: {dup_count} filas repetidas. "
                "La unidad de análisis está comprometida o el pipeline duplica registros."
            )
        log.info("1. Clave sin duplicados OK sobre %s: is_unique=True", key_cols)

        # 2. Volumen suficiente (> 1.000 filas)
        min_rows = 1000
        n_rows = len(df)
        if n_rows <= min_rows:
            raise ValueError(f"Volumen insuficiente: {n_rows} filas (se esperaban más de {min_rows}).")
        log.info("2. Volumen suficiente OK: %s filas (> %s)", n_rows, min_rows)

        # 3. Ancho suficiente (5 o más columnas útiles)
        n_cols = df.shape[1]
        if n_cols < 5:
            raise ValueError(f"Ancho insuficiente: {n_cols} columnas (se esperaban 5 o más).")
        log.info("3. Ancho suficiente OK: shape=%s (%s columnas)", df.shape, n_cols)

        # 4. Mezcla de tipos (numéricas, categóricas y fechas)
        type_counts = df.dtypes.value_counts()
        log.info("4. Mezcla de tipos de datos:\n%s", type_counts.to_string())
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        date_candidates = [c for c in df.columns if "date" in c.lower()]
        if len(numeric_cols) == 0:
            raise ValueError("No se encontraron columnas numéricas en el dataset.")
        if len(categorical_cols) == 0:
            raise ValueError("No se encontraron columnas categóricas en el dataset.")
        if len(date_candidates) == 0:
            log.warning("No se identificaron columnas con nombre de fecha ('date').")

        # 5. Nulos conocidos (saber cuáles y por qué)
        null_series = df.isna().mean().sort_values(ascending=False)
        top_nulls = null_series[null_series > 0]
        log.info(
            "5. Nulos conocidos (porcentaje descendente):\n%s",
            top_nulls.round(4).to_string() if not top_nulls.empty else "Sin valores nulos."
        )

        # 6. Sin columnas vacías (ninguna al 100 % de nulos)
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            raise ValueError(
                f"Columnas 100%% vacías detectadas en Silver unificado: {empty_cols}"
            )
        log.info("6. Sin columnas vacías OK: ninguna columna al 100%% de nulos.")

        log.info("Validación completa Silver unificado EXITOSA: %s filas x %s columnas", n_rows, n_cols)
        return "Validación Silver Unificado: OK"

    @task
    def get_seasons_jolpica(**context) -> list[int]:
        mode = context["params"].get("mode", "full")
        seasons = [_SUBSET_SEASON] if mode == "subset" else SEASONS_JOLPICA
        log.info("Modo %s: temporadas Jolpica a procesar: %s", mode, seasons)
        return seasons

    @task
    def get_years_openf1(**context) -> list[int]:
        mode = context["params"].get("mode", "full")
        years = [_SUBSET_SEASON] if mode == "subset" else YEARS_OPENF1
        log.info("Modo %s: años OpenF1 a procesar: %s", mode, years)
        return years

    @task
    def get_years_fastf1(**context) -> list[int]:
        mode = context["params"].get("mode", "full")
        years = [_SUBSET_SEASON] if mode == "subset" else YEARS_FASTF1
        log.info("Modo %s: años FastF1 a procesar: %s", mode, years)
        return years

    # --- Orquestación ---
    # El modo 'subset' restringe a una sola temporada para correr rápido.
    # El modo 'full' procesa todas las temporadas configuradas.
    seasons_jolpica = get_seasons_jolpica()
    years_openf1 = get_years_openf1()
    years_fastf1 = get_years_fastf1()

    jolpica = ingest_jolpica.expand(season=seasons_jolpica)
    openf1 = ingest_openf1.expand(year=years_openf1)
    fastf1 = ingest_fastf1.expand(year=years_fastf1)

    v_bronze = verificar_bronze(jolpica, openf1, fastf1)

    race_path = build_silver_race()
    laps_path = build_silver_laps()

    v_bronze >> [race_path, laps_path]

    val_race = validate_silver_race(race_path)
    val_laps = validate_silver_laps(laps_path)

    unified_path = build_unified_silver(race_path, laps_path)
    [val_race, val_laps] >> unified_path

    validate_unified_silver(unified_path)


f1_ingest()