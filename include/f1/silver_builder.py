"""
Silver: normaliza Bronze (Jolpica + OpenF1 + FastF1) a datasets de features.

Clave canónica: (season, round, driver_code) con las 3 letras FIA.
  - Jolpica: Driver.code
  - FastF1:  Abbreviation (results) / Driver (laps)
  - OpenF1:  driver_number -> traducido con el mapa que arma FastF1

Estructura Bronze esperada:
  bronze/jolpica/**/results.json, driverStandings.json
  bronze/openf1/{sessions,weather,pit,stints}/*.json
  bronze/fastf1/{year}/{round}/{Q|R}/{results,laps}.csv

OJO CON LA FUGA DE DATOS: varias columnas describen lo que pasó DURANTE la
carrera (ritmo, pits, stints, posición por vuelta). Sirven para análisis
descriptivo, pero NO se pueden usar como features para predecir el
resultado de esa misma carrera. Ver LEAKY_COLS al final del archivo.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from f1 import schema

log = logging.getLogger(__name__)

BRONZE = Path(
    os.environ.get(
        "F1_BRONZE_DIR",
        Path(__file__).resolve().parents[2] / "include" / "output" / "bronze",
    )
)
SILVER = Path(os.environ.get("F1_SILVER_DIR", BRONZE.parent / "silver"))


# =============================
# Helpers genéricos
# =============================
def _safe_json_load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("No se pudo leer %s: %s", path, e)
        return None


def _empty(cols: list[str]) -> pd.DataFrame:
    """DataFrame vacío pero CON columnas, para no romper los merges."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _norm_code(value) -> str | None:
    """Código de piloto a 3 letras mayúsculas."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().upper()
    return s or None


def _to_int(value, default=None):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_seconds(serie: pd.Series) -> pd.Series:
    """'0 days 00:01:32.123000' o '92.123' -> float de segundos."""
    if serie is None or len(serie) == 0:
        return pd.Series(dtype="float64")
    numerico = pd.to_numeric(serie, errors="coerce")
    try:
        td = pd.to_timedelta(serie, errors="coerce")
        return td.dt.total_seconds().fillna(numerico)
    except Exception:
        return numerico


def _pick(df: pd.DataFrame, *nombres: str) -> pd.Series:
    """Primera columna existente entre `nombres`; si ninguna, serie de NA."""
    for n in nombres:
        if n in df.columns:
            return df[n]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def _align_keys(df: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in on:
        if c not in df.columns:
            continue
        if c in ("season", "round", "lap_number", "driver_number"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        else:
            df[c] = df[c].astype("object")
    return df


def _safe_merge(left: pd.DataFrame, right: pd.DataFrame,
                on: list[str], label: str) -> pd.DataFrame:
    """Merge left que garantiza no duplicar filas ni perderlas."""
    if right is None or right.empty:
        log.warning("[%s] vacío — merge omitido", label)
        return left

    faltantes = [c for c in on if c not in right.columns]
    if faltantes:
        log.warning("[%s] sin columnas %s — merge omitido", label, faltantes)
        return left

    left = _align_keys(left, on)
    right = _align_keys(right, on)

    dup = int(right.duplicated(subset=on).sum())
    if dup:
        raise ValueError(
            f"[{label}] tiene {dup} filas duplicadas sobre {on}. "
            "Falta agregar antes de mergear (si no, se multiplica el dataset)."
        )

    n_antes = len(left)
    out = left.merge(right, on=on, how="left")
    if len(out) != n_antes:
        raise ValueError(
            f"[{label}] el merge cambió las filas: {n_antes} -> {len(out)}."
        )

    nuevas = [c for c in right.columns if c not in on]
    if nuevas:
        cob = (1 - out[nuevas].isna().mean().mean()) * 100
        log.info("[%s] %.1f%% de cobertura sobre %s filas", label, cob, len(out))
    return out


# =============================
# Jolpica
# =============================
JOLPICA_COLS = [
    "season", "round", "race_date", "race_name", "circuit_id", "circuit_name",
    "driver_id", "driver_code", "driver_name", "constructor_id",
    "constructor_name", "grid_position", "pit_lane_start", "finish_position",
    "classified", "points", "status", "laps_completed_jolpica",
]


def load_jolpica_results() -> pd.DataFrame:
    rows = []
    base = BRONZE / "source=jolpica"
    if not base.exists():
        log.warning("No existe bronze/source=jolpica")
        return _empty(JOLPICA_COLS)

    for p in sorted(base.rglob("results.json")):
        data = _safe_json_load(p)
        if not data:
            continue
        try:
            races = data["MRData"]["RaceTable"]["Races"]
        except (KeyError, TypeError):
            continue

        for race in races:
            season = _to_int(race.get("season"))
            round_num = _to_int(race.get("round"))
            if season is None or round_num is None:
                continue
            circuito = race.get("Circuit") or {}

            for result in race.get("Results", []):
                driver = result.get("Driver") or {}
                constructor = result.get("Constructor") or {}
                st = result.get("status")
                status_value = (st.get("status", "") if isinstance(st, dict)
                                else str(st or ""))
                # grid 0 = largada desde pit lane; como número es engañoso.
                grid = _to_int(result.get("grid"))
                pos_text = str(result.get("positionText", "")).strip()

                rows.append({
                    "season": season,
                    "round": round_num,
                    "race_date": race.get("date"),
                    "race_name": race.get("raceName"),
                    "circuit_id": circuito.get("circuitId"),
                    "circuit_name": circuito.get("circuitName"),
                    "driver_id": driver.get("driverId"),
                    "driver_code": _norm_code(driver.get("code")),
                    "driver_name": f"{driver.get('givenName', '')} "
                                   f"{driver.get('familyName', '')}".strip(),
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "grid_position": None if grid in (None, 0) else grid,
                    "pit_lane_start": grid == 0,
                    "finish_position": _to_int(result.get("position")),
                    "classified": pos_text.isdigit(),
                    "points": float(result.get("points") or 0),
                    "status": status_value,
                    "laps_completed_jolpica": _to_int(result.get("laps")),
                })

    if not rows:
        return _empty(JOLPICA_COLS)

    df = pd.DataFrame(rows)
    antes = len(df)
    df = df.drop_duplicates(subset=["season", "round", "driver_id"], keep="last")
    if len(df) != antes:
        log.info("jolpica: %s duplicados descartados", antes - len(df))

    sin_code = int(df["driver_code"].isna().sum())
    if sin_code:
        log.warning("%s filas de Jolpica sin driver_code (no van a matchear con FastF1/OpenF1)", sin_code)
    return df


def load_calendario(results: pd.DataFrame) -> pd.DataFrame:
    """(season, round) -> fecha de carrera, para mapear OpenF1."""
    if results.empty:
        return _empty(["season", "round", "race_date"])
    cal = results[["season", "round", "race_date"]].dropna().drop_duplicates()
    cal["race_date"] = pd.to_datetime(cal["race_date"], errors="coerce").dt.date
    return cal.dropna(subset=["race_date"])


def load_jolpica_standings_after() -> pd.DataFrame:
    """Standings TAL COMO QUEDARON después de correrse esa ronda."""
    cols = ["season", "round", "driver_id", "standing_after", "points_after",
            "wins_after"]
    rows = []
    base = BRONZE / "source=jolpica"
    if not base.exists():
        return _empty(cols)

    for p in sorted(base.rglob("driverStandings.json")):
        data = _safe_json_load(p)
        if not data:
            continue
        try:
            listas = data["MRData"]["StandingsTable"]["StandingsLists"]
        except (KeyError, TypeError):
            continue

        for item in listas:
            season = _to_int(item.get("season"))
            round_num = _to_int(item.get("round"))
            if season is None or round_num is None:
                continue
            for entry in item.get("DriverStandings", []):
                driver = entry.get("Driver") or {}
                rows.append({
                    "season": season,
                    "round": round_num,
                    "driver_id": driver.get("driverId"),
                    "standing_after": _to_int(entry.get("position")),
                    "points_after": float(entry.get("points") or 0),
                    "wins_after": _to_int(entry.get("wins"), 0),
                })

    if not rows:
        return _empty(cols)
    return pd.DataFrame(rows).drop_duplicates(
        subset=["season", "round", "driver_id"], keep="last")


def build_standings_before() -> pd.DataFrame:
    """Corre las standings una ronda: las de N-1 son el estado PREVIO a N.

    Sin este shift hay fuga: las standings de la ronda N ya incluyen el
    resultado de la carrera que se quiere predecir.
    """
    cols = ["season", "round", "driver_id", "driver_standing_before",
            "driver_points_before", "driver_wins_before"]
    std = load_jolpica_standings_after()
    if std.empty:
        return _empty(cols)

    prev = std.copy()
    prev["round"] = prev["round"] + 1
    prev = prev.rename(columns={
        "standing_after": "driver_standing_before",
        "points_after": "driver_points_before",
        "wins_after": "driver_wins_before",
    })
    return prev[cols]


# =============================
# FastF1
# =============================
def _parse_fastf1_path(path: Path):
    """.../source=fastf1/year={year}/round={round}/session={session}/{archivo}.csv -> (year, round, ses)."""
    partes = path.parts
    if len(partes) < 4:
        return None
    session_part = partes[-2]
    rnd_part = partes[-3]
    season_part = partes[-4]
    
    if not (session_part.startswith("session=") and rnd_part.startswith("round=") and season_part.startswith("year=")):
        return None
        
    session = session_part.split("=")[1]
    rnd = _to_int(rnd_part.split("=")[1])
    season = _to_int(season_part.split("=")[1])
    if season is None or rnd is None or not (1950 < season < 2100) or rnd < 1:
        return None
    return season, rnd, session


def _leer_fastf1(nombre: str, session: str) -> list[tuple[int, int, pd.DataFrame]]:
    """Lee todos los `nombre` de la sesión `session` ('Q' o 'R')."""
    base = BRONZE / "source=fastf1"
    if not base.exists():
        log.warning("No existe bronze/source=fastf1")
        return []

    salida, malas = [], 0
    for f in sorted(base.rglob(nombre)):
        meta = _parse_fastf1_path(f)
        if meta is None:
            malas += 1
            continue
        season, rnd, ses = meta
        if ses != session:
            continue
        try:
            raw = pd.read_csv(f)
        except Exception as e:
            log.warning("No se pudo leer %s: %s", f, e)
            continue
        if not raw.empty:
            salida.append((season, rnd, raw))

    if malas:
        log.warning("%s archivo(s) %s con ruta no parseable", malas, nombre)
    return salida


def load_fastf1_driver_map() -> pd.DataFrame:
    """(season, round, driver_number) -> driver_code.

    Reemplaza al endpoint `drivers` de OpenF1: FastF1 ya trae el número y la
    abreviatura en el mismo archivo.
    """
    cols = ["season", "round", "driver_number", "driver_code"]
    frames = []
    for session in ("R", "Q"):
        for season, rnd, raw in _leer_fastf1("results.csv", session):
            frames.append(pd.DataFrame({
                "season": season,
                "round": rnd,
                "driver_number": pd.to_numeric(_pick(raw, "DriverNumber"),
                                               errors="coerce"),
                "driver_code": _pick(raw, "Abbreviation", "Driver").apply(_norm_code),
            }))

    if not frames:
        log.warning("Sin mapa de números FastF1 — pit/stints van a quedar vacíos")
        return _empty(cols)

    df = pd.concat(frames, ignore_index=True).dropna(
        subset=["driver_number", "driver_code"])
    df["driver_number"] = df["driver_number"].astype(int)
    return df.drop_duplicates(subset=["season", "round", "driver_number"],
                              keep="first")


def load_fastf1_quali() -> pd.DataFrame:
    """Features de clasificación: pre-carrera, sin fuga."""
    cols = ["season", "round", "driver_code", "quali_position",
            "q1_s", "q2_s", "q3_s", "quali_best_s"]
    frames = []
    for season, rnd, raw in _leer_fastf1("results.csv", "Q"):
        frames.append(pd.DataFrame({
            "season": season,
            "round": rnd,
            "driver_code": _pick(raw, "Abbreviation", "Driver").apply(_norm_code),
            "quali_position": pd.to_numeric(_pick(raw, "Position"),
                                            errors="coerce"),
            "q1_s": _to_seconds(_pick(raw, "Q1")),
            "q2_s": _to_seconds(_pick(raw, "Q2")),
            "q3_s": _to_seconds(_pick(raw, "Q3")),
        }))

    if not frames:
        return _empty(cols)

    df = pd.concat(frames, ignore_index=True).dropna(subset=["driver_code"])
    df["quali_best_s"] = df[["q1_s", "q2_s", "q3_s"]].min(axis=1, skipna=True)
    return df.drop_duplicates(subset=["season", "round", "driver_code"],
                              keep="last")[cols]


def load_fastf1_race_results() -> pd.DataFrame:
    """Resultado de carrera según FastF1 (respaldo de Jolpica)."""
    cols = ["season", "round", "driver_code", "team_name", "grid_position_ff1"]
    frames = []
    for season, rnd, raw in _leer_fastf1("results.csv", "R"):
        frames.append(pd.DataFrame({
            "season": season,
            "round": rnd,
            "driver_code": _pick(raw, "Abbreviation", "Driver").apply(_norm_code),
            "team_name": _pick(raw, "TeamName"),
            "grid_position_ff1": pd.to_numeric(_pick(raw, "GridPosition"),
                                               errors="coerce"),
        }))

    if not frames:
        return _empty(cols)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["driver_code"])
    return df.drop_duplicates(subset=["season", "round", "driver_code"],
                              keep="last")


def load_fastf1_laps(session: str = "R") -> pd.DataFrame:
    """Vueltas vectorizadas (sin iterrows)."""
    cols = ["season", "round", "driver_code", "driver_number", "lap_number",
            "lap_time_s", "sector_1_s", "sector_2_s", "sector_3_s",
            "compound", "tyre_life", "fresh_tyre", "is_personal_best",
            "lap_position"]
    frames = []
    for season, rnd, raw in _leer_fastf1("laps.csv", session):
        frames.append(pd.DataFrame({
            "season": season,
            "round": rnd,
            "driver_code": _pick(raw, "Driver", "Abbreviation").apply(_norm_code),
            "driver_number": pd.to_numeric(_pick(raw, "DriverNumber"),
                                           errors="coerce"),
            "lap_number": pd.to_numeric(_pick(raw, "LapNumber"),
                                        errors="coerce"),
            "lap_time_s": _to_seconds(_pick(raw, "LapTime")),
            "sector_1_s": _to_seconds(_pick(raw, "Sector1Time")),
            "sector_2_s": _to_seconds(_pick(raw, "Sector2Time")),
            "sector_3_s": _to_seconds(_pick(raw, "Sector3Time")),
            "compound": _pick(raw, "Compound"),
            "tyre_life": pd.to_numeric(_pick(raw, "TyreLife"), errors="coerce"),
            "fresh_tyre": _pick(raw, "FreshTyre"),
            "is_personal_best": _pick(raw, "IsPersonalBest"),
            "lap_position": pd.to_numeric(_pick(raw, "Position"),
                                          errors="coerce"),
        }))

    if not frames:
        return _empty(cols)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["driver_code", "lap_number"])
    df["lap_number"] = df["lap_number"].astype(int)
    return df.drop_duplicates(
        subset=["season", "round", "driver_code", "lap_number"], keep="last")


def build_race_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """Ritmo de carrera por piloto. OJO: es información POST-carrera."""
    cols = ["season", "round", "driver_code", "avg_lap_time_s",
            "median_lap_time_s", "best_lap_time_s", "std_lap_time_s",
            "laps_completed"]
    if laps.empty:
        return _empty(cols)

    agg = laps.groupby(["season", "round", "driver_code"]).agg(
        avg_lap_time_s=("lap_time_s", "mean"),
        median_lap_time_s=("lap_time_s", "median"),
        best_lap_time_s=("lap_time_s", "min"),
        std_lap_time_s=("lap_time_s", "std"),
        laps_completed=("lap_number", "max"),
    ).reset_index()
    return agg


# =============================
# OpenF1
# =============================
def _openf1_concat(endpoint: str) -> pd.DataFrame:
    d = BRONZE / "source=openf1" / f"endpoint={endpoint}"
    if not d.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(d.glob("*.json")):
        data = _safe_json_load(p)
        if data:
            frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_openf1_session_index(calendario: pd.DataFrame) -> pd.DataFrame:
    """session_key -> (season, round), matcheando por fecha con Jolpica.

    Se matchea por fecha y no por orden: si a OpenF1 le falta una carrera,
    rankear por fecha correría todos los rounds siguientes.
    """
    cols = ["session_key", "season", "round"]
    df = _openf1_concat("sessions")
    if df.empty or "session_name" not in df.columns:
        log.warning("Sin índice de sesiones de OpenF1")
        return _empty(cols)

    # session_type "Race" también marca los Sprint: filtramos por nombre.
    races = df[df["session_name"] == "Race"].copy()
    if races.empty:
        return _empty(cols)

    races["date_start"] = pd.to_datetime(races["date_start"], errors="coerce",
                                         utc=True)
    races = races.dropna(subset=["date_start", "session_key", "year"])
    races = races.drop_duplicates(subset=["session_key"])
    races["season"] = pd.to_numeric(races["year"], errors="coerce").astype("Int64")
    races["race_date"] = races["date_start"].dt.date

    if calendario.empty:
        log.warning("Sin calendario de Jolpica, se usa orden por fecha para OpenF1")
        races["round"] = (races.groupby("season")["date_start"]
                          .rank(method="first").astype(int))
        return races[cols]

    cal = calendario.copy()
    cal["season"] = pd.to_numeric(cal["season"], errors="coerce").astype("Int64")
    out = races.merge(cal, on=["season", "race_date"], how="left")

    sin_match = out["round"].isna()
    if sin_match.any():
        rank = (out.groupby("season")["date_start"].rank(method="first"))
        out.loc[sin_match, "round"] = rank[sin_match]
        log.warning("openf1: %s sesión(es) sin fecha coincidente en Jolpica, resueltas por orden", int(sin_match.sum()))

    out["round"] = pd.to_numeric(out["round"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["round"]).drop_duplicates(subset=["session_key"])
    log.info("openf1: %s sesiones de carrera mapeadas a (season, round)", len(out))
    return out[cols]


def load_openf1_weather(sessions: pd.DataFrame) -> pd.DataFrame:
    """Clima agregado por sesión (no depende del piloto)."""
    cols = ["season", "round", "air_temp", "track_temp", "humidity",
            "pressure", "wind_speed", "rainfall_max"]
    df = _openf1_concat("weather")
    if df.empty or sessions.empty or "session_key" not in df.columns:
        return _empty(cols)

    medias = {
        "air_temperature": "air_temp",
        "track_temperature": "track_temp",
        "humidity": "humidity",
        "pressure": "pressure",
        "wind_speed": "wind_speed",
    }
    presentes = {k: v for k, v in medias.items() if k in df.columns}
    if not presentes:
        return _empty(cols)

    for c in presentes:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = (df.groupby("session_key")[list(presentes)].mean()
           .rename(columns=presentes).reset_index())

    if "rainfall" in df.columns:
        lluvia = (pd.to_numeric(df["rainfall"], errors="coerce")
                  .groupby(df["session_key"]).max().rename("rainfall_max"))
        agg = agg.merge(lluvia.reset_index(), on="session_key", how="left")

    out = agg.merge(sessions, on="session_key", how="inner")
    return out.drop(columns=["session_key"]).drop_duplicates(
        subset=["season", "round"])


def _a_piloto(agg: pd.DataFrame, sessions: pd.DataFrame,
              dmap: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """(session_key, driver_number) -> (season, round, driver_code)."""
    if agg.empty or sessions.empty or dmap.empty:
        return _empty(cols)

    out = agg.merge(sessions, on="session_key", how="inner")
    out["driver_number"] = pd.to_numeric(out["driver_number"],
                                         errors="coerce").astype("Int64")
    dmap = dmap.copy()
    dmap["driver_number"] = dmap["driver_number"].astype("Int64")
    out = out.merge(dmap, on=["season", "round", "driver_number"], how="inner")
    out = out.drop(columns=["session_key", "driver_number"])
    return out.drop_duplicates(subset=["season", "round", "driver_code"])[cols]


def load_openf1_pits(sessions, dmap) -> pd.DataFrame:
    cols = ["season", "round", "driver_code", "pit_count",
            "pit_duration_mean", "pit_duration_min"]
    df = _openf1_concat("pit")
    if df.empty or "driver_number" not in df.columns:
        return _empty(cols)

    df["pit_duration"] = pd.to_numeric(_pick(df, "pit_duration"),
                                       errors="coerce")
    agg = df.groupby(["session_key", "driver_number"]).agg(
        pit_count=("driver_number", "size"),
        pit_duration_mean=("pit_duration", "mean"),
        pit_duration_min=("pit_duration", "min"),
    ).reset_index()
    return _a_piloto(agg, sessions, dmap, cols)


def load_openf1_stints(sessions, dmap) -> pd.DataFrame:
    cols = ["season", "round", "driver_code", "stint_count",
            "compounds_used", "tyre_age_start_mean"]
    df = _openf1_concat("stints")
    if df.empty or "driver_number" not in df.columns:
        return _empty(cols)

    df["tyre_age_at_start"] = pd.to_numeric(_pick(df, "tyre_age_at_start"),
                                            errors="coerce")
    if "stint_number" not in df.columns:
        df["stint_number"] = 1
    if "compound" not in df.columns:
        df["compound"] = pd.NA

    agg = df.groupby(["session_key", "driver_number"]).agg(
        stint_count=("stint_number", "nunique"),
        compounds_used=("compound", "nunique"),
        tyre_age_start_mean=("tyre_age_at_start", "mean"),
    ).reset_index()
    return _a_piloto(agg, sessions, dmap, cols)


# =============================
# Validación
# =============================
def validate_silver_dataset(df: pd.DataFrame, key_cols: list[str],
                            min_rows: int = 1000,
                            required_non_null: list[str] | None = None):
    """Falla duro en lo estructural; avisa en lo opcional."""
    if df.empty:
        raise ValueError("El dataset está vacío.")

    dup = int(df.duplicated(subset=key_cols).sum()) if key_cols else 0
    if dup:
        raise ValueError(f"Clave duplicada: {dup} filas sobre {key_cols}.")

    if len(df) < min_rows:
        raise ValueError(f"Volumen insuficiente: {len(df)} < {min_rows} filas.")

    for c in (required_non_null or []):
        if c not in df.columns:
            raise ValueError(f"Falta la columna obligatoria '{c}'.")
        if df[c].isna().all():
            raise ValueError(f"La columna obligatoria '{c}' está 100% vacía.")

    cobertura = ((1 - df.isna().mean()) * 100).sort_values()
    log.info("Cobertura por columna (10 peores):\n%s", cobertura.head(10).round(1).to_string())

    vacias = [c for c in df.columns if df[c].isna().all()]
    if vacias:
        log.warning("ATENCIÓN: columnas 100%% vacías -> %s", vacias)
    return df


# =============================
# Silver A: piloto-carrera
# =============================
# Importados desde schema.py — el único lugar donde viven estas listas.
RACE_COL_ORDER = schema.RACE_COLUMNS
LEAKY_COLS = schema.LEAKY_COLS


def build_driver_race_features(date_suffix: str = "") -> str:
    log.info("== Silver A: driver_race_features ==")
    SILVER.mkdir(parents=True, exist_ok=True)

    df = load_jolpica_results()
    if df.empty:
        raise ValueError("No hay resultados de Jolpica en Bronze.")
    log.info("base jolpica: %s filas", len(df))

    calendario = load_calendario(df)
    sessions = load_openf1_session_index(calendario)
    dmap = load_fastf1_driver_map()
    log.info("mapa de números: %s combinaciones (season, round, nro)", len(dmap))

    df = _safe_merge(df, build_standings_before(),
                     ["season", "round", "driver_id"], "standings_before")
    df = _safe_merge(df, load_fastf1_quali(),
                     ["season", "round", "driver_code"], "fastf1_quali")
    df = _safe_merge(df, load_fastf1_race_results(),
                     ["season", "round", "driver_code"], "fastf1_race")
    df = _safe_merge(df, build_race_pace(load_fastf1_laps("R")),
                     ["season", "round", "driver_code"], "fastf1_ritmo")
    df = _safe_merge(df, load_openf1_weather(sessions),
                     ["season", "round"], "openf1_weather")
    df = _safe_merge(df, load_openf1_pits(sessions, dmap),
                     ["season", "round", "driver_code"], "openf1_pits")
    df = _safe_merge(df, load_openf1_stints(sessions, dmap),
                     ["season", "round", "driver_code"], "openf1_stints")

    if "stint_count" in df.columns:
        df["tyre_change_count"] = df["stint_count"] - 1

    orden = [c for c in RACE_COL_ORDER if c in df.columns]
    resto = [c for c in df.columns if c not in orden]
    df = df[orden + resto].sort_values(["season", "round", "finish_position"])

    filename = f"driver_race_features_{date_suffix}.csv" if date_suffix else "driver_race_features.csv"
    destino = SILVER / filename
    df.to_csv(destino, index=False)
    log.info("OK -> %s (%s filas, %s columnas)", destino, len(df), df.shape[1])
    return str(destino)


# =============================
# Silver B: piloto-vuelta-carrera
# =============================
LAP_COL_ORDER = schema.LAP_COLUMNS


def build_driver_lap_features(date_suffix: str = "") -> str:
    log.info("== Silver B: driver_lap_features ==")
    SILVER.mkdir(parents=True, exist_ok=True)

    df = load_fastf1_laps("R")
    if df.empty:
        raise ValueError("No hay vueltas de carrera de FastF1 en Bronze.")
    log.info("base fastf1 laps: %s filas", len(df))
    df = df.drop(columns=["driver_number"], errors="ignore")

    resultados = load_jolpica_results()
    contexto = resultados[[
        "season", "round", "race_name", "driver_id", "driver_code",
        "driver_name", "constructor_name", "grid_position", "finish_position",
    ]].dropna(subset=["driver_code"]).drop_duplicates(
        subset=["season", "round", "driver_code"])

    calendario = load_calendario(resultados)
    sessions = load_openf1_session_index(calendario)
    dmap = load_fastf1_driver_map()

    df = _safe_merge(df, contexto,
                     ["season", "round", "driver_code"], "jolpica_contexto")
    df = _safe_merge(df, load_openf1_weather(sessions),
                     ["season", "round"], "openf1_weather")
    df = _safe_merge(df, load_openf1_pits(sessions, dmap)[
                         ["season", "round", "driver_code", "pit_count"]],
                     ["season", "round", "driver_code"], "openf1_pits")

    orden = [c for c in LAP_COL_ORDER if c in df.columns]
    resto = [c for c in df.columns if c not in orden]
    df = df[orden + resto].sort_values(
        ["season", "round", "driver_code", "lap_number"])

    filename = f"driver_lap_features_{date_suffix}.csv" if date_suffix else "driver_lap_features.csv"
    destino = SILVER / filename
    df.to_csv(destino, index=False)
    log.info("OK -> %s (%s filas, %s columnas)", destino, len(df), df.shape[1])
    return str(destino)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_driver_race_features()
    build_driver_lap_features()
    log.info("RECORDATORIO: columnas post-carrera (no usar como features para predecir): %s", ", ".join(LEAKY_COLS))