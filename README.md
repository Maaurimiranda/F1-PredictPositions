# F1-PredictPositions

Pipeline de Airflow para construir la capa Bronze de datos de Fórmula 1 y
servir como base para un modelo predictivo del Top 10 antes de cada carrera.

## Pregunta de investigación

> ¿Es posible predecir, antes del inicio de una carrera de F1, el ranking final
> de los diez primeros pilotos usando únicamente información disponible antes de
> la carrera?

**Unidad de análisis:** una fila representa a un piloto en una carrera
(`season × round × driver_id`).

## Fuentes de datos

| Fuente | Cobertura | Qué aporta |
|---|---|---|
| [Jolpica-F1](https://api.jolpi.ca/ergast/f1) | 1950–actual | Resultados, grid, standings |
| [FastF1](https://docs.fastf1.dev/) | 2018+ | Qualifying, laps y tiempos por sector |
| [OpenF1](https://openf1.org/) | 2023+ | Clima, pits y stints |

## Arquitectura

```text
API (Jolpica / FastF1 / OpenF1)
        │
        ▼
   include/bronze/       ← datos crudos (JSON/CSV), particionados por fuente
        │
        ▼
   include/silver/       ← dataset construido (próxima etapa)
```

El DAG principal es [dags/f1_ingest.py](dags/f1_ingest.py), que orquesta la ingesta
Bronze desde las tres fuentes.

## Estructura del repositorio

```text
├── dags/
│   └── f1_ingest.py              # DAG Airflow — ingesta Bronze
├── include/
│   ├── f1/
│   │   ├── __init__.py
│   │   ├── bronze.py             # Orquestador de prefetch
│   │   ├── fastf1_source.py      # Ingesta FastF1
│   │   ├── jolpica.py            # Cliente Jolpica con caché + retry
│   │   ├── openf1.py             # Cliente OpenF1 con caché + retry
│   │   └── __init__.py
│   └── bronze/                   # Datos crudos (gitignored)
├── proyecto_f1_colab.ipynb       # Prototipo original en Colab
├── docker-compose.yaml           # Stack local de Airflow + Postgres
├── Dockerfile                    # Imagen base para el entorno
├── requirements.txt
├── packages.txt
├── airflow_settings.yaml
├── .env
├── .astro/
├── .gitignore
└── README.md
```

## Requisitos previos

- Python 3.10+
- Docker Desktop levantado
- Git
- Airflow o Astro CLI para ejecución local

## Arranque rápido

### Opción A: Docker Compose

```bash
cd "C:\Proyecto Ciencia de Datos\F1-PredictPositions"
docker compose up -d --build
```

Luego abrir:

```text
http://localhost:8080
```

Credenciales por defecto del stack local:

```text
usuario: admin
password: admin
```

### Opción B: Astro CLI

```bash
astro dev start
```

## Ejecutar el pipeline

Desde la UI de Airflow:

1. Buscar el DAG `f1_ingest`
2. Activarlo (toggle ON)
3. Dispararlo manualmente

O desde terminal:

```bash
airflow dags trigger f1_ingest
```

## Verificación

Tras la corrida, debería quedar en disco una estructura similar a:

```text
include/bronze/
├── jolpica/
├── openf1/
├── fastf1/
├── fastf1_cache/
```

La tarea `verificar_bronze` imprime el resumen de cobertura por fuente.

## Reejecución

La ingesta es idempotente. Si los archivos ya existen, no se vuelven a descargar;
la corrida reintenta sólo lo faltante. Eso hace que el pipeline sea seguro para
repetir ejecuciones en una demo o en una presentación.

## Variables de entorno

| Variable | Descripción | Default |
|---|---|---|
| `F1_BRONZE_DIR` | Ruta donde se guardan los datos crudos | `include/bronze/` |

## Próximos pasos

- Implementar la capa Silver con `transform.py` y `features.py`
- Generar `include/silver/dataset.csv`
- Agregar validaciones automáticas de calidad
- Refinar la predicción del Top 10 con features previas a la carrera

## Notas de entrega

- No depender de ejecutar una descarga completa en vivo durante la presentación.
- Cada integrante debe poder explicar cada tarea del DAG.
- El notebook `proyecto_f1_colab.ipynb` conserva la lógica prototipada y los
  chequeos de calidad iniciales.
