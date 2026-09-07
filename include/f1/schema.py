"""
Schema canónico del dataset de carreras F1.

Define las columnas de cada tabla Silver, cuáles son obligatorias y cuáles
contienen información post-carrera que NO se puede usar como feature predictivo
(ver LEAKY_COLS).

Centralizar el schema acá evita tener estas listas dispersas en silver_builder.py
y en el DAG — si se agrega una columna, el cambio vive en un solo lugar.
"""
from __future__ import annotations

# ──────────────────────────────────────────────
# Silver A: piloto × carrera
# ──────────────────────────────────────────────

# Orden canónico de columnas del dataset de carreras.
RACE_COLUMNS: list[str] = [
    "season", "round", "race_date", "race_name", "circuit_id", "circuit_name",
    "driver_id", "driver_code", "driver_name",
    "constructor_id", "constructor_name", "team_name",
    "grid_position", "grid_position_ff1", "pit_lane_start",
    "quali_position", "q1_s", "q2_s", "q3_s", "quali_best_s",
    "driver_standing_before", "driver_points_before", "driver_wins_before",
    "air_temp", "track_temp", "humidity", "pressure", "wind_speed",
    "rainfall_max",
    "finish_position", "classified", "points", "status",
    "laps_completed", "laps_completed_jolpica",
    "avg_lap_time_s", "median_lap_time_s", "best_lap_time_s", "std_lap_time_s",
    "pit_count", "pit_duration_mean", "pit_duration_min",
    "stint_count", "compounds_used", "tyre_age_start_mean",
    "tyre_change_count",
]

# Clave primaria de la tabla de carreras.
RACE_KEY: list[str] = ["season", "round", "driver_id"]

# Columnas que NO pueden ser nulas en un dataset válido.
RACE_OBLIGATORIAS: list[str] = ["season", "round", "driver_id", "finish_position"]

# Columnas que describen lo ocurrido DURANTE la carrera.
# No usar como features para predecir el resultado de esa misma carrera.
LEAKY_COLS: list[str] = [
    "finish_position", "classified", "points", "status",
    "laps_completed", "laps_completed_jolpica",
    "avg_lap_time_s", "median_lap_time_s", "best_lap_time_s", "std_lap_time_s",
    "pit_count", "pit_duration_mean", "pit_duration_min",
    "stint_count", "compounds_used", "tyre_age_start_mean", "tyre_change_count",
]

# ──────────────────────────────────────────────
# Silver B: piloto × vuelta × carrera
# ──────────────────────────────────────────────

# Orden canónico de columnas del dataset de vueltas.
LAP_COLUMNS: list[str] = [
    "season", "round", "race_name", "driver_id", "driver_code", "driver_name",
    "constructor_name", "lap_number", "lap_time_s",
    "sector_1_s", "sector_2_s", "sector_3_s",
    "compound", "tyre_life", "fresh_tyre", "is_personal_best", "lap_position",
    "grid_position", "finish_position",
    "air_temp", "track_temp", "humidity", "rainfall_max", "pit_count",
]

# Clave primaria de la tabla de vueltas.
LAP_KEY: list[str] = ["season", "round", "driver_code", "lap_number"]

# Columnas que NO pueden ser nulas en un dataset de vueltas válido.
LAP_OBLIGATORIAS: list[str] = ["season", "round", "driver_code", "lap_number", "lap_time_s"]


# ──────────────────────────────────────────────
# Silver C: vueltas enriquecidas con contexto de carrera (1 a N)
# ──────────────────────────────────────────────

# Orden canónico del dataset unificado: columnas de vueltas + columnas de carrera no redundantes
UNIFIED_LAP_RACE_COLUMNS: list[str] = [
    # Identificadores y contexto de carrera
    "season", "round", "race_date", "race_name", "circuit_id", "circuit_name",
    # Piloto y equipo
    "driver_id", "driver_code", "driver_name",
    "constructor_id", "constructor_name", "team_name",
    # Vuelta y tiempos de vuelta
    "lap_number", "lap_time_s", "sector_1_s", "sector_2_s", "sector_3_s",
    "compound", "tyre_life", "fresh_tyre", "is_personal_best", "lap_position",
    # Parrilla y clasificación
    "grid_position", "grid_position_ff1", "pit_lane_start",
    "quali_position", "q1_s", "q2_s", "q3_s", "quali_best_s",
    # Standings previos a la carrera
    "driver_standing_before", "driver_points_before", "driver_wins_before",
    # Clima
    "air_temp", "track_temp", "humidity", "pressure", "wind_speed", "rainfall_max",
    # Resultado de carrera
    "finish_position", "classified", "points", "status",
    # Métricas agregadas de carrera (post-carrera)
    "laps_completed", "laps_completed_jolpica",
    "avg_lap_time_s", "median_lap_time_s", "best_lap_time_s", "std_lap_time_s",
    "pit_count", "pit_duration_mean", "pit_duration_min",
    "stint_count", "compounds_used", "tyre_age_start_mean", "tyre_change_count",
]

UNIFIED_LAP_RACE_KEY: list[str] = ["season", "round", "driver_code", "lap_number"]
UNIFIED_LAP_RACE_OBLIGATORIAS: list[str] = ["season", "round", "driver_code", "lap_number", "lap_time_s"]

