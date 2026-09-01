from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BRONZE = Path("include/bronze")
SILVER = Path("include/silver")
SILVER.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers de carga
# -----------------------------
def _safe_json_load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jolpica_results() -> pd.DataFrame:
    rows = []
    for p in (BRONZE / "jolpica").rglob("results.json"):
        data = _safe_json_load(p)
        if not data:
            continue

        try:
            races = data["MRData"]["RaceTable"]["Races"]
        except KeyError:
            continue

        for race in races:
            season = int(race.get("season", 0) or 0)
            round_num = int(race.get("round", 0) or 0)
            race_name = race.get("raceName")
            circuit_name = race.get("Circuit", {}).get("circuitName")
            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                constructor = result.get("Constructor", {})
                status_obj = result.get("status")
                status_value = status_obj.get("status", "") if isinstance(status_obj, dict) else str(status_obj or "")
                driver_id = driver.get("driverId")
                rows.append(
                    {
                        "season": season,
                        "round": round_num,
                        "race_name": race_name,
                        "circuit_name": circuit_name,
                        "driver_id": driver_id,
                        "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                        "constructor_id": constructor.get("constructorId"),
                        "constructor_name": constructor.get("name"),
                        "grid_position": int(result.get("grid", 0) or 0),
                        "finish_position": int(result.get("position", 0) or 0),
                        "points": float(result.get("points", 0) or 0),
                        "status": status_value,
                    }
                )

    return pd.DataFrame(rows)


def load_jolpica_standings_before() -> pd.DataFrame:
    rows = []
    for p in (BRONZE / "jolpica").rglob("driverStandings.json"):
        data = _safe_json_load(p)
        if not data:
            continue

        try:
            standings_lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        except KeyError:
            continue

        for item in standings_lists:
            season = int(item.get("season", 0) or 0)
            round_num = int(item.get("round", 0) or 0)
            for entry in item.get("DriverStandings", []):
                driver = entry.get("Driver", {})
                rows.append(
                    {
                        "season": season,
                        "round": round_num,
                        "driver_id": driver.get("driverId"),
                        "driver_standing_before": int(entry.get("position", 0) or 0),
                        "driver_points_before": float(entry.get("points", 0) or 0),
                    }
                )

    return pd.DataFrame(rows)


def _normalize_driver_id(value):
    if value is None:
        return None
    return str(value).strip().lower()


def validate_silver_dataset(
    df: pd.DataFrame,
    key_cols: list[str],
    min_rows: int = 1000,
    min_columns: int = 5,
    allow_all_null_cols: list[str] | None = None,
):
    if df.empty:
        raise ValueError("El dataset está vacío.")

    duplicated = int(df.duplicated(subset=key_cols).sum()) if key_cols else 0
    if duplicated > 0:
        raise ValueError(f"La clave primaria está duplicada: {duplicated} filas repetidas sobre {key_cols}.")

    if len(df) < min_rows:
        raise ValueError(f"El dataset no alcanza el volumen mínimo esperado: {len(df)} filas < {min_rows}.")

    if df.shape[1] < min_columns:
        raise ValueError(f"El dataset no alcanza el ancho mínimo esperado: {df.shape[1]} columnas < {min_columns}.")

    dtype_counts = df.dtypes.value_counts()
    if len(dtype_counts) < 2:
        raise ValueError("El dataset no tiene mezcla suficiente de tipos de datos (numérico/categórico).")

    null_summary = df.isna().mean().sort_values(ascending=False)
    if not null_summary.empty and (null_summary > 0).any():
        print("Resumen de nulos:")
        print(null_summary.head(10))

    allowed_nulls = set(allow_all_null_cols or [])
    all_null_cols = [c for c in df.columns if c not in allowed_nulls and df[c].isna().all()]
    if all_null_cols:
        raise ValueError(f"Hay columnas completamente vacías y no permitidas: {all_null_cols}")

    return df


def load_fastf1_results() -> pd.DataFrame:
    rows = []
    for file in (BRONZE / "fastf1").rglob("results.csv"):
        try:
            df = pd.read_csv(file)
        except Exception:
            continue

        if df.empty:
            continue

        season = int(file.parts[-4])
        round_num = int(file.parts[-3])
        for _, r in df.iterrows():
            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "driver_id": _normalize_driver_id(r.get("Driver")),
                    "driver_name": r.get("Driver"),
                    "avg_lap_time": r.get("AverageLapTime"),
                    "best_lap_time": r.get("FastestLapTime"),
                }
            )

    return pd.DataFrame(rows)


def load_fastf1_laps() -> pd.DataFrame:
    rows = []
    for file in (BRONZE / "fastf1").rglob("laps.csv"):
        try:
            df = pd.read_csv(file)
        except Exception:
            continue

        if df.empty:
            continue

        season = int(file.parts[-4])
        round_num = int(file.parts[-3])
        for _, r in df.iterrows():
            rows.append(
                {
                    "season": season,
                    "round": round_num,
                    "driver_id": _normalize_driver_id(r.get("Driver")),
                    "driver_name": r.get("Driver"),
                    "lap_number": int(r.get("LapNumber", 0) or 0),
                    "lap_time": r.get("LapTime"),
                    "sector_1": r.get("Sector1Time"),
                    "sector_2": r.get("Sector2Time"),
                    "sector_3": r.get("Sector3Time"),
                    "compound": r.get("Compound"),
                    "tyre_life": r.get("TyreLife"),
                    "fresh_tyre": r.get("FreshTyre"),
                    "position": r.get("Position"),
                }
            )

    return pd.DataFrame(rows)


def load_openf1_weather() -> pd.DataFrame:
    rows = []
    weather_dir = BRONZE / "openf1" / "weather"
    for p in weather_dir.glob("*.json"):
        data = _safe_json_load(p)
        if not data:
            continue

        for item in data:
            rows.append(
                {
                    "season": int(item.get("year", 0) or 0),
                    "round": int(item.get("round", 0) or 0),
                    "driver_id": _normalize_driver_id(item.get("driver")),
                    "weather_temp": item.get("air_temp"),
                    "track_temp": item.get("track_temp"),
                    "humidity": item.get("humidity"),
                }
            )

    return pd.DataFrame(rows)


def load_openf1_pits() -> pd.DataFrame:
    rows = []
    pit_dir = BRONZE / "openf1" / "pit"
    for p in pit_dir.glob("*.json"):
        data = _safe_json_load(p)
        if not data:
            continue

        for item in data:
            rows.append(
                {
                    "season": int(item.get("year", 0) or 0),
                    "round": int(item.get("round", 0) or 0),
                    "driver_id": _normalize_driver_id(item.get("driver")),
                    "pit_count": int(item.get("pit_count", 0) or 0),
                }
            )

    return pd.DataFrame(rows)


def load_openf1_stints() -> pd.DataFrame:
    rows = []
    stints_dir = BRONZE / "openf1" / "stints"
    for p in stints_dir.glob("*.json"):
        data = _safe_json_load(p)
        if not data:
            continue

        for item in data:
            rows.append(
                {
                    "season": int(item.get("year", 0) or 0),
                    "round": int(item.get("round", 0) or 0),
                    "driver_id": _normalize_driver_id(item.get("driver")),
                    "stint_count": int(item.get("stint_count", 0) or 0),
                    "tyre_change_count": int(item.get("tyre_changes", 0) or 0),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------
# Silver A: piloto-carrera
# -----------------------------
def build_driver_race_features() -> pd.DataFrame:
    results = load_jolpica_results()
    standings = load_jolpica_standings_before()
    fastf1 = load_fastf1_results()
    weather = load_openf1_weather()
    pits = load_openf1_pits()
    stints = load_openf1_stints()

    results["driver_id"] = results["driver_id"].map(_normalize_driver_id)
    standings["driver_id"] = standings["driver_id"].map(_normalize_driver_id)
    fastf1["driver_id"] = fastf1["driver_id"].map(_normalize_driver_id)
    weather["driver_id"] = weather["driver_id"].map(_normalize_driver_id)
    pits["driver_id"] = pits["driver_id"].map(_normalize_driver_id)
    stints["driver_id"] = stints["driver_id"].map(_normalize_driver_id)

    df = results.merge(standings, on=["season", "round", "driver_id"], how="left")
    df = df.merge(fastf1, on=["season", "round", "driver_id"], how="left")
    df = df.merge(weather, on=["season", "round", "driver_id"], how="left")
    df = df.merge(pits, on=["season", "round", "driver_id"], how="left")
    df = df.merge(stints, on=["season", "round", "driver_id"], how="left")

    if "driver_name_x" in df.columns and "driver_name_y" in df.columns:
        df["driver_name"] = df["driver_name_x"].fillna(df["driver_name_y"])
        df = df.drop(columns=["driver_name_x", "driver_name_y"])
    elif "driver_name_x" in df.columns:
        df = df.rename(columns={"driver_name_x": "driver_name"})
    elif "driver_name_y" in df.columns:
        df = df.rename(columns={"driver_name_y": "driver_name"})

    if "constructor_name_x" in df.columns and "constructor_name_y" in df.columns:
        df["constructor_name"] = df["constructor_name_x"].fillna(df["constructor_name_y"])
        df = df.drop(columns=["constructor_name_x", "constructor_name_y"])
    elif "constructor_name_x" in df.columns:
        df = df.rename(columns={"constructor_name_x": "constructor_name"})
    elif "constructor_name_y" in df.columns:
        df = df.rename(columns={"constructor_name_y": "constructor_name"})

    cols = [
        "season",
        "round",
        "race_name",
        "circuit_name",
        "driver_id",
        "driver_name",
        "constructor_id",
        "constructor_name",
        "grid_position",
        "finish_position",
        "points",
        "driver_standing_before",
        "driver_points_before",
        "avg_lap_time",
        "best_lap_time",
        "weather_temp",
        "track_temp",
        "humidity",
        "pit_count",
        "stint_count",
        "tyre_change_count",
    ]

    df = df[cols]
    validate_silver_dataset(
        df,
        key_cols=["season", "round", "driver_id"],
        min_rows=1000,
        min_columns=5,
        allow_all_null_cols=[
            "avg_lap_time",
            "best_lap_time",
            "weather_temp",
            "track_temp",
            "humidity",
            "pit_count",
            "stint_count",
            "tyre_change_count",
        ],
    )
    df.to_csv(SILVER / "driver_race_features.csv", index=False)
    print(f"driver_race_features.csv generado: {len(df)} filas")
    return df


# -----------------------------
# Silver B: piloto-vuelta-carrera
# -----------------------------
def build_driver_lap_features() -> pd.DataFrame:
    laps = load_fastf1_laps()
    results = load_jolpica_results()[["season", "round", "driver_id", "grid_position", "finish_position"]]
    weather = load_openf1_weather()
    pits = load_openf1_pits()

    laps["driver_id"] = laps["driver_id"].map(_normalize_driver_id)
    results["driver_id"] = results["driver_id"].map(_normalize_driver_id)
    weather["driver_id"] = weather["driver_id"].map(_normalize_driver_id)
    pits["driver_id"] = pits["driver_id"].map(_normalize_driver_id)

    df = laps.merge(results, on=["season", "round", "driver_id"], how="left")
    df = df.merge(weather, on=["season", "round", "driver_id"], how="left")
    df = df.merge(pits, on=["season", "round", "driver_id"], how="left")

    if "driver_name_x" in df.columns and "driver_name_y" in df.columns:
        df["driver_name"] = df["driver_name_x"].fillna(df["driver_name_y"])
        df = df.drop(columns=["driver_name_x", "driver_name_y"])
    elif "driver_name_x" in df.columns:
        df = df.rename(columns={"driver_name_x": "driver_name"})
    elif "driver_name_y" in df.columns:
        df = df.rename(columns={"driver_name_y": "driver_name"})

    cols = [
        "season",
        "round",
        "driver_id",
        "driver_name",
        "lap_number",
        "lap_time",
        "sector_1",
        "sector_2",
        "sector_3",
        "compound",
        "tyre_life",
        "fresh_tyre",
        "position",
        "grid_position",
        "finish_position",
        "weather_temp",
        "track_temp",
        "humidity",
        "pit_count",
    ]

    # Faltan race_name, si lo quisieras, podés completar desde los resultados
    df = df[cols]
    validate_silver_dataset(
        df,
        key_cols=["season", "round", "driver_id", "lap_number"],
        min_rows=1000,
        min_columns=5,
        allow_all_null_cols=["weather_temp", "track_temp", "humidity", "pit_count"],
    )
    df.to_csv(SILVER / "driver_lap_features.csv", index=False)
    print(f"driver_lap_features.csv generado: {len(df)} filas")
    return df


if __name__ == "__main__":
    build_driver_race_features()
    build_driver_lap_features()
