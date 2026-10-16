# F1-PredictPositions

Pipeline de datos de Fórmula 1 para predecir la **posición de llegada** de cada piloto en cuatro momentos de la carrera: antes de largar y al 25, 50 y 75 %.

## Pregunta de investigación

> ¿Dónde termina un piloto? Predecir `finish_position` (la posición oficial de llegada) en cuatro momentos de cada carrera —antes de largar y cuando el líder completa el 25, el 50 y el 75 % de las vueltas— usando sólo lo que se sabe hasta ese momento.

**Unidad de análisis:** una fila = un piloto en una carrera en un momento (`season × round × driver_code × lap_cutoff`), en el Silver D (`driver_race_snapshots_YYYY-MM-DD.csv`). Hay un modelo por momento.

## Fuentes de datos

| Fuente | Cobertura | Qué aporta |
|---|---|---|
| [Jolpica-F1](https://api.jolpi.ca/ergast/f1) | 1950–actual | Resultados, grid, standings |
| [FastF1](https://docs.fastf1.dev/) | 2018+ | Qualifying detallado, laps, tiempos por sector |
| [OpenF1](https://openf1.org/) | 2023+ | Clima, pits, stints |

### Cobertura y Limitaciones Conocidas de los Datos

- **OpenF1 (`pit_count` en 2023):**
  - La API de OpenF1 comenzó a registrar el endpoint `/v1/pit` a partir del **Gran Premio de España 2023 (Ronda 7, `session_key=9102`)**.
  - Las **primeras 6 rondas de 2023** (Bahrain, Arabia Saudita, Australia, Azerbaiyán, Miami y Mónaco) devuelven `404 Not Found` en la API pública por falta de telemetría de pits upstream, produciendo valores nulos (`NaN`) en `pit_count` para esas carreras (~27.5% de las vueltas de 2023). A partir de la ronda 7 y en las temporadas 2024+, la cobertura de pits es completa.
- **Tiempos de Vuelta y Sectores (`sector_1_s`, `lap_time_s`):**
  - Presentan valores nulos estructurales típicos del cronometraje de carreras (~1% a 2%):
    - `sector_1_s` es nulo en la **vuelta 1** de cada carrera debido a la largada detenida desde el cajón de salida (no existe registro de sector lanzado).
    - `lap_time_s` es nulo en la vuelta en la cual un piloto entra a boxes y **abandona la carrera** (vuelta incompleta).
- **Temporada 2022:**
  - FastF1 y Jolpica aportan telemetría de vueltas y resultados completos; las columnas dependientes de OpenF1 (`air_temp`, `track_temp`, `pit_count`, etc.) son nulas ya que OpenF1 solo tiene soporte a partir de 2023.

## Arquitectura

```
API (Jolpica / FastF1 / OpenF1)
        │
        ▼
  include/output/bronze/    ← datos crudos (JSON/CSV), particionados (source=X/year=Y/round=Z)
        │
        ▼
  include/output/silver/    ← dataset construido, CSV fechado (ej: driver_race_features_2026-09-03.csv)
```

El DAG de Airflow (`dags/f1_ingest.py`) orquesta el pipeline completo siguiendo el modelo medallón:
- **Bronze**: `ingest_jolpica`, `ingest_openf1`, `ingest_fastf1` → `verificar_bronze`
- **Silver A y B**: `build_silver_race`, `build_silver_laps` → `validate_silver_race`, `validate_silver_laps`
- **Silver C** (vueltas con contexto de carrera): `build_unified_silver` → `validate_unified_silver`
- **Silver D** (snapshots con el target): `build_objective_columns` → `validate_objective_columns`

## Silver D: el estado de cada piloto en cada momento

Cinco filas por piloto y carrera: antes de largar (`lap_cutoff = 0`) y al cerrar la vuelta del 25, 50, 75 y 100 % de la carrera. Los modelos predicen en los cuatro primeros; el 100 % queda como control. Todo lo "al corte" sale sólo de las vueltas `<= lap_cutoff` (`tests/test_snapshots.py` lo verifica).

| Columna en el pipeline | Nombre en el notebook | Qué es |
|---|---|---|
| `current_position` | `posicion_al_corte` | Puesto al corte (1..N en cada carrera). Antes de largar, el orden de largada: los que salen de boxes van al fondo y los empates se resuelven por la qualy. Durante la carrera, primero los que completaron más vueltas y, a igualdad, la posición de timing de FastF1 en su última vuelta |
| `laps_done_at_cutoff` | `ultima_vuelta` | Vueltas completadas al corte |
| `timing_position_at_cutoff` | `posicion_timing` | Posición de timing (`lap_position`) en su última vuelta |
| `position_10_laps_before` | `pos_hace_10` | Posición de timing diez vueltas antes del corte |
| `tyre_life_at_cutoff` | `vueltas_neumatico` | Edad del neumático al corte |
| `tyre_changes_so_far` | `paradas_al_corte` | Cambios de neumático. Sale de las vueltas, así que cubre todas las temporadas (las paradas de OpenF1 empiezan en 2023) |
| `dry_compounds_so_far` | `compuestos_secos` | Compuestos de seco distintos usados (SOFT, MEDIUM, HARD) |
| `used_wet_tyres_so_far` | `uso_lluvia` | Usó intermedios o de lluvia |
| `pace_last_5_laps_s` | `ritmo_5` | Mediana del tiempo de vuelta en las 5 vueltas que terminan en el corte |
| `laps_without_time_so_far` | `vueltas_sin_tiempo` | Vueltas sin tiempo registrado hasta el corte |

`schema.NOMBRES_NOTEBOOK` tiene esta equivalencia: `snapshots.rename(columns=NOMBRES_NOTEBOOK)` deja el snapshot con los nombres que esperan los modelos de la Entrega 3. `schema.MODEL_INPUT_COLUMNS` lista las columnas que leen esos modelos.

Cuidado con dos columnas:
- `cumulative_time_s` sigue en el snapshot, pero **no sirve para ordenar**: las vueltas sin tiempo no suman, y quien tiene alguna aparece adelante.
- `laps_total` es **fuga**: son las vueltas que completó el piloto en toda la carrera, no el largo de la carrera. Está en `schema.LEAKY_COLS`, junto con todas las columnas de carrera completa, que son fuga en cualquier momento.

**Control al 100 %:** al cerrar la última vuelta, `current_position` coincide con `finish_position` en el 85 % de las filas de 2022-2026. Las diferencias son penalizaciones post-carrera y descalificaciones. Con el orden anterior por `cumulative_time_s` coincidía en el 66 %. `validate_objective_columns` avisa si baja del 80 %.

## Cambios para la Entrega 3

- **`current_position` corregida.** Antes se ordenaba por `cumulative_time_s`. Ahora usa las vueltas completadas y la posición de timing (y antes de largar, la grilla). Los empates se resuelven por `driver_code`, explícitamente: el Silver A viene ordenado por `finish_position`, y desempatar por el orden de las filas metería el resultado en la posición.
- **Columnas nuevas de estado al corte** (tabla de arriba): las mismas que calcula el notebook de modelado. Con los CSV del 23/09 dan exactamente lo mismo que el notebook en los cinco momentos.
- **`laps_total` pasa a fuga**, y `schema.LEAKY_COLS` aclara que las columnas de carrera completa son fuga en cualquier momento, no sólo antes de largar.
- **Validación del Silver D** en `silver_builder.validate_snapshot_dataset`, con chequeos nuevos: columnas de los modelos presentes, `current_position` = 1..N en cada carrera y corte, nada posterior al corte y el control al 100 %.
- **Tests sin Airflow** en `tests/test_snapshots.py` (vueltas sin tiempo, orden de largada, empates, nada posterior al corte, neumáticos): `astro dev pytest`, o `pytest tests/test_snapshots.py` con `pandas` y `pytest` instalados.

## Estructura del repositorio

```
├── dags/
│   └── f1_ingest.py              # DAG Airflow — pipeline completo Bronze + Silver
├── include/
│   ├── f1/
│   │   ├── __init__.py
│   │   ├── jolpica.py            # Cliente Jolpica con caché + 429
│   │   ├── openf1.py             # Cliente OpenF1 con caché + 429
│   │   ├── fastf1_source.py      # Ingesta FastF1 (caché nativo)
│   │   ├── bronze.py             # Orquestador de prefetch Bronze
│   │   ├── schema.py             # Columnas de cada tabla Silver, fuga y nombres del notebook
│   │   └── silver_builder.py     # Construcción y validación de la capa Silver
│   └── output/                   # Datos generados (gitignored)
│       ├── bronze/               # Datos crudos particionados por fuente/año/ronda
│       └── silver/               # Datasets CSV fechados
├── include/frozen/               # Respaldo frío del último Silver exitoso
├── tests/
│   ├── dags/test_dag_integrity.py  # El DAG importa sin errores
│   └── test_snapshots.py           # Lógica del Silver D (sin Airflow)
├── requirements.txt
├── .gitignore
└── README.md
```

## Requisitos previos

- **Python 3.10+**
- **Docker Desktop** corriendo
- **Airflow** (con Docker o Astronomer/Astro CLI)
- Git

## Instalación paso a paso

### 1. Clonar el repositorio

```bash
git clone https://github.com/Maaurimiranda/F1-PredictPositions.git
cd F1-PredictPositions
```

### 2. Instalar dependencias Python

Si querés probar los clientes fuera de Airflow (local):

```bash
pip install -r requirements.txt
```

### 3. Verificar que los clientes funcionan

Probá que el cliente Jolpica baja y cachea correctamente:

```bash
python include/f1/jolpica.py
```

Debería imprimir `OK — caché funciona: include/output/bronze/source=jolpica/...`.

### 4. Configurar Airflow

El proyecto necesita que Airflow vea las carpetas `dags/` e `include/`.

**Con Astronomer (Astro CLI):**

```bash
astro dev start
```

**Con docker-compose (Airflow vanilla):**

Asegurate de que los volúmenes monten:

- `./dags` → `/opt/airflow/dags`
- `./include` → `/opt/airflow/include`

Y que `include/` sea un **volumen persistente** (si no, la caché Bronze se pierde al recrear el contenedor).

### 5. Ejecutar el pipeline

Desde la UI de Airflow:

1. Buscar el DAG **`f1_ingest`**.
2. Activarlo (toggle ON).
3. Disparar manualmente (botón "Trigger DAG").
4. Opcional: en la pantalla de trigger, elegir **modo `subset`** para una corrida rápida (solo 2024, ~minutos) o **`full`** para todo el histórico.

O desde la terminal:

```bash
airflow dags trigger f1_ingest
```

### 6. Verificar la ejecución

- En la UI de Airflow, todas las tareas deben terminar en **SUCCESS** (verde).
- La tarea `verificar_bronze` loguea la cobertura de archivos por fuente.
- En disco, `include/output/bronze/` debe tener subcarpetas `source=jolpica/`, `source=openf1/`, `source=fastf1/`.
- En disco, `include/output/silver/` debe tener los CSV fechados: `driver_race_features_YYYY-MM-DD.csv`, `driver_lap_features_...`, `driver_lap_race_features_...` y `driver_race_snapshots_...`.
- El log de `validate_objective_columns` muestra los chequeos del Silver D, incluido el control al 100 %.

### 7. Reejecución

Disparar el DAG de nuevo es seguro: **no vuelve a descargar** lo que ya está en `include/output/bronze/`. Solo baja lo faltante. Una corrida interrumpida puede continuarse sin perder trabajo.

## Variables de entorno (opcionales)

| Variable | Descripción | Default |
|---|---|---|
| `F1_BRONZE_DIR` | Ruta donde se guardan los datos crudos | `include/output/bronze/` (relativo al repo) |
| `F1_SILVER_DIR` | Ruta donde se guardan los datasets Silver | `include/output/silver/` (relativo al repo) |

## Parámetros del DAG (en la UI de Airflow)

| Parámetro | Valores | Descripción |
|---|---|---|
| `mode` | `full` (default) / `subset` | `subset` procesa solo 2024 para validar rápido |
| `force` | `false` (default) / `true` | Fuerza re-descarga ignorando la caché en Bronze |

## Notas para la entrega

- **NO** depender de ejecutar una descarga completa en vivo durante la presentación. La corrida debe estar hecha previamente.
- Cada integrante debe poder explicar qué hace cada tarea del DAG.
- El notebook `proyecto_f1_colab.ipynb` contiene la lógica original probada y el chequeo de calidad (`chequear_calidad`).
