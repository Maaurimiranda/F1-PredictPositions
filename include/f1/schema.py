"""
Schema canónico del dataset de carreras F1.

Define las columnas de cada tabla Silver, cuáles son obligatorias y cuáles
contienen información post-carrera que NO se puede usar como feature predictivo
(ver LEAKY_COLS).

Centralizar el schema acá evita tener estas listas dispersas en silver_builder.py
y en el DAG — si se agrega una columna, el cambio vive en un solo lugar.

Silver D (driver_race_snapshots):
  Una fila por (season, round, driver_code, lap_cutoff). El lap_cutoff define el
  momento de la carrera en que se "congela" la observación: 0 = antes de largar,
  N = al cerrar la vuelta N. Hay cinco momentos: 0, 25, 50, 75 y 100 % de las
  vueltas de la carrera. Los modelos predicen en los cuatro primeros; el 100 %
  queda como control (ahí la posición al corte tiene que coincidir con la de
  llegada). El target es finish_position (posición oficial FIA).
  total_race_time_s se mantiene como columna auxiliar — el EDA lo analiza para
  justificar por qué NO es adecuado como target (Sección 1.2 del notebook).
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

# Columnas que se completan al TERMINAR la carrera (resultado y agregados de la
# carrera entera). Son fuga en CUALQUIER momento, también con lap_cutoff>0: por
# ejemplo, pit_count cuenta paradas que todavía no ocurrieron al 50 %.
# Lo que sí se puede usar durante la carrera son las columnas "al corte" del
# Silver D (current_position, laps_done_at_cutoff, ...), que salen sólo de las
# vueltas <= lap_cutoff. laps_total y total_race_time_s viven en el Silver D.
LEAKY_COLS: list[str] = [
    "finish_position", "classified", "points", "status",
    "laps_completed", "laps_completed_jolpica",
    "avg_lap_time_s", "median_lap_time_s", "best_lap_time_s", "std_lap_time_s",
    "pit_count", "pit_duration_mean", "pit_duration_min",
    "stint_count", "compounds_used", "tyre_age_start_mean", "tyre_change_count",
    "laps_total", "total_race_time_s",
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


# ──────────────────────────────────────────────
# Silver D: piloto × carrera × lap_cutoff (snapshots con target)
# ──────────────────────────────────────────────

# Clave primaria del dataset de snapshots.
SILVER_SNAPSHOT_KEY: list[str] = ["season", "round", "driver_code", "lap_cutoff"]

# Columna objetivo: posición oficial de llegada según la FIA (fuente: Jolpica).
# Entero entre 1 y 22, sin nulos (la FIA asigna posición también a los DNFs).
# Se predice en cuatro momentos: antes de largar y al 25, 50 y 75 % de la carrera
# (pct_race_complete = 0, 0.25, 0.5 y 0.75), con un modelo por momento.
# total_race_time_s se mantiene en el dataset como auxiliar para el análisis de
# la Sección 1.2 del EDA, pero NO es el target del modelo.
SILVER_TARGET: str = "finish_position"

# Orden canónico de columnas del dataset de snapshots.
SNAPSHOT_COLUMNS: list[str] = [
    # Identificadores y contexto de carrera
    "season", "round", "race_date", "race_name", "circuit_id", "circuit_name",
    # Piloto y equipo
    "driver_id", "driver_code", "driver_name",
    "constructor_id", "constructor_name", "team_name",
    # --- Dimensión de snapshot ---
    "lap_cutoff",        # vuelta en que se congela la observación: round(vueltas de la carrera × pct); 0 = antes de largar
    "pct_race_complete", # 0, 0.25, 0.5, 0.75 (momentos en que predice el modelo) y 1.0 (control)
    # --- Features pre-carrera (siempre disponibles) ---
    "grid_position", "grid_position_ff1", "pit_lane_start",
    "quali_position", "q1_s", "q2_s", "q3_s", "quali_best_s",
    "driver_standing_before", "driver_points_before", "driver_wins_before",
    "air_temp", "track_temp", "humidity", "pressure", "wind_speed", "rainfall_max",
    # --- Estado al momento del corte: sale sólo de las vueltas <= lap_cutoff ---
    "current_position",          # puesto al corte (1..N en cada carrera). Antes de largar: orden de largada, con los que
                                 # salen de boxes al fondo. Después: más vueltas completadas primero y, a igualdad, la
                                 # posición de timing de FastF1 en su última vuelta
    "position_gain_loss",        # grid_position - current_position
    "laps_done_at_cutoff",       # vueltas completadas al corte (0 antes de largar)
    "timing_position_at_cutoff", # lap_position de FastF1 en la última vuelta completada (NaN antes de largar)
    "position_10_laps_before",   # lap_position en la última vuelta <= lap_cutoff - 10
    "tyre_life_at_cutoff",       # edad del neumático en la última vuelta completada
    "tyre_changes_so_far",       # cambios de neumático detectados en las vueltas (todas las temporadas, sin OpenF1)
    "dry_compounds_so_far",      # compuestos de seco distintos usados hasta el corte (SOFT, MEDIUM, HARD)
    "used_wet_tyres_so_far",     # True si usó INTERMEDIATE o WET hasta el corte
    "compound_at_cutoff",        # último compuesto registrado al corte
    "pace_last_5_laps_s",        # mediana de lap_time_s en las 5 vueltas que terminan en el corte
    "laps_without_time_so_far",  # vueltas sin lap_time_s (timing perdido) hasta el corte
    "cumulative_time_s",         # Σ lap_time_s hasta el corte. Las vueltas sin tiempo no suman: NO sirve para ordenar
    "avg_lap_time_s_so_far",     # ritmo promedio hasta el corte
    "pits_done_so_far",          # paradas según OpenF1 (desde 2023 ronda 7; ver pit_time_available)
    "pit_time_so_far_s",         # tiempo total en boxes según OpenF1
    # --- Target ---
    "finish_position",        # posición oficial FIA (1-22), sin nulos — TARGET del modelo
    # --- Auxiliares de resultado (fuga: se completan al terminar la carrera; nunca como features) ---
    "laps_total",             # vueltas que completó el piloto en TODA la carrera
    "total_race_time_s",      # Σ lap_time_s clasificados; auxiliar para análisis EDA 1.2
    "pit_time_available",     # True si hay datos de pit de OpenF1 para esta carrera
    "classified", "points", "status",
    "laps_completed", "laps_completed_jolpica",
]

SNAPSHOT_OBLIGATORIAS: list[str] = [
    "season", "round", "driver_code", "lap_cutoff", "pct_race_complete",
    "current_position", "finish_position",
]

# Columnas del snapshot que leen los modelos guardados en la Entrega 3
# (modelos_llegada_por_momento.joblib). "carrera" no está porque el notebook la
# arma como season-round. Si falta alguna, validate_snapshot_dataset falla.
MODEL_INPUT_COLUMNS: list[str] = [
    "current_position", "grid_position", "pit_lane_start", "quali_best_s",
    "driver_points_before", "driver_standing_before", "constructor_name",
    "laps_done_at_cutoff", "tyre_changes_so_far", "tyre_life_at_cutoff",
    "dry_compounds_so_far", "used_wet_tyres_so_far", "position_10_laps_before",
    "pace_last_5_laps_s", "compound_at_cutoff",
]

# Nombres que usa el notebook de modelado (Entrega 3) para las mismas columnas:
#   snapshots.rename(columns=NOMBRES_NOTEBOOK)
# deja el snapshot con los nombres que esperan los modelos guardados.
NOMBRES_NOTEBOOK: dict[str, str] = {
    "current_position": "posicion_al_corte",
    "timing_position_at_cutoff": "posicion_timing",
    "laps_done_at_cutoff": "ultima_vuelta",
    "tyre_life_at_cutoff": "vueltas_neumatico",
    "position_10_laps_before": "pos_hace_10",
    "pace_last_5_laps_s": "ritmo_5",
    "laps_without_time_so_far": "vueltas_sin_tiempo",
    "tyre_changes_so_far": "paradas_al_corte",
    "dry_compounds_so_far": "compuestos_secos",
    "used_wet_tyres_so_far": "uso_lluvia",
}
