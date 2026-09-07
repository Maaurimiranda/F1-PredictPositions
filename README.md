# F1-PredictPositions

Pipeline de datos de Fórmula 1 para predecir, **antes del inicio de cada carrera**, el ranking del Top 10.

## Pregunta de investigación

> ¿Es posible predecir, antes del inicio de una carrera de F1, el ranking final de los diez primeros pilotos usando exclusivamente información disponible antes de la carrera?

**Unidad de análisis:** una fila = un piloto en una carrera (`season × round × driver_id`).

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
- **Silver**: `build_silver_race`, `build_silver_laps` → `validate_silver_race`, `validate_silver_laps`

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
│   │   └── silver_builder.py     # Construcción y validación de la capa Silver
│   └── output/                   # Datos generados (gitignored)
│       ├── bronze/               # Datos crudos particionados por fuente/año/ronda
│       └── silver/               # Datasets CSV fechados
├── include/frozen/               # Respaldo frío del último Silver exitoso
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
- En disco, `include/output/silver/` debe tener los CSV fechados: `driver_race_features_YYYY-MM-DD.csv`.

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
