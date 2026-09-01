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

## Arquitectura

```
API (Jolpica / FastF1 / OpenF1)
        │
        ▼
   include/bronze/       ← datos crudos (JSON/CSV), particionados por fuente y season/round
        │
        ▼  (próxima iteración)
   include/silver/        ← dataset construido, CSV
```

El DAG de Airflow (`dags/f1_ingest.py`) orquesta la ingesta Bronze.

## Estructura del repositorio

```
├── dags/
│   └── f1_ingest.py              # DAG Airflow — ingesta Bronze
├── include/
│   ├── f1/
│   │   ├── __init__.py
│   │   ├── jolpica.py            # Cliente Jolpica con caché + 429
│   │   ├── openf1.py             # Cliente OpenF1 con caché + 429
│   │   ├── fastf1_source.py      # Ingesta FastF1 (caché nativo)
│   │   └── bronze.py             # Orquestador de prefetch
│   └── bronze/                   # Datos crudos (gitignored)
├── proyecto_f1_colab.ipynb       # Prototipo original en Colab
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

Debería imprimir `OK — caché funciona: include/bronze/jolpica/2024.json`.

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

Instalar las dependencias dentro del contenedor:

```bash
docker exec -it <airflow-worker> pip install -r /opt/airflow/requirements.txt
```

### 5. Ejecutar el pipeline

Desde la UI de Airflow:

1. Buscar el DAG **`f1_ingest`**.
2. Activarlo (toggle ON).
3. Disparar manualmente (botón "Trigger DAG").

O desde la terminal:

```bash
airflow dags trigger f1_ingest
```

### 6. Verificar la ejecución

- En la UI de Airflow, todas las tareas deben terminar en **SUCCESS** (verde).
- La tarea `verificar_bronze` loguea la cobertura de archivos por fuente.
- En disco, `include/bronze/` debe tener subcarpetas `jolpica/`, `openf1/`, `fastf1/`, `fastf1_cache/`.

### 7. Reejecución

Disparar el DAG de nuevo es seguro: **no vuelve a descargar** lo que ya está en `include/bronze/`. Solo baja lo faltante. Una corrida interrumpida puede continuarse sin perder trabajo.

## Variables de entorno (opcionales)

| Variable | Descripción | Default |
|---|---|---|
| `F1_BRONZE_DIR` | Ruta donde se guardan los datos crudos | `include/bronze/` (relativo al repo) |

## Notas para la entrega

- **NO** depender de ejecutar una descarga completa en vivo durante la presentación. La corrida debe estar hecha previamente.
- Cada integrante debe poder explicar qué hace cada tarea del DAG.
- El notebook `proyecto_f1_colab.ipynb` contiene la lógica original probada y el chequeo de calidad (`chequear_calidad`).

## Próximos pasos (Silver)

- Migrar `resultados_a_filas`, `driver_standings_before` y rolling features a `include/f1/transform.py` y `include/f1/features.py`.
- Agregar tarea al DAG que construya el dataset desde Bronze y exporte a `include/silver/dataset.csv`.
- Gate de calidad automatizado en el DAG.
