"""Tests de la posición y el estado al corte del Silver D. No necesitan Airflow ni red:

    pytest tests/test_snapshots.py        (o `astro dev pytest`)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "include"))
from f1.silver_builder import build_lap_snapshot  # noqa: E402

NAN = np.nan


def _vuelta(piloto, n, tiempo, posicion, compuesto, edad):
    return {"season": 2025, "round": 1, "driver_code": piloto, "lap_number": n,
            "lap_time_s": tiempo, "lap_position": posicion, "compound": compuesto, "tyre_life": edad}


def carrera():
    """Una carrera de 8 vueltas con 4 autos.

    AAA larga 1° y va primero toda la carrera.
    BBB larga 2° y va segundo siempre, pero a su vuelta 3 le falta el tiempo. Para en la
        vuelta 4 (SOFT -> HARD) y en la 6 le falta el dato del neumático.
    DDD larga 3° y CCC desde boxes; los dos abandonan en la vuelta 2, sin tiempo ni posición.
    """
    v = []
    for n in range(1, 9):
        v.append(_vuelta("AAA", n, 90.0, 1, "MEDIUM", n))
        v.append(_vuelta("BBB", n, NAN if n == 3 else 90.5, 2,
                         NAN if n == 6 else ("SOFT" if n <= 4 else "HARD"),
                         NAN if n == 6 else (n if n <= 4 else n - 4)))
    for piloto, posicion in (("DDD", 3), ("CCC", 4)):
        v.append(_vuelta(piloto, 1, 95.0, posicion, "SOFT", 1))
        v.append(_vuelta(piloto, 2, NAN, NAN, "SOFT", 2))
    # Como el Silver A, ordenado por finish_position: DDD terminó adelante de CCC.
    meta = pd.DataFrame({"season": 2025, "round": 1, "driver_code": ["AAA", "BBB", "DDD", "CCC"],
                         "race_laps_total": 8, "grid_position": [1, 2, 3, NAN],
                         "pit_lane_start": [False, False, False, True], "quali_position": [1, 2, 4, 3]})
    return pd.DataFrame(v), meta


def snapshot(pct, vueltas=None, meta=None):
    v, m = carrera()
    v = v if vueltas is None else vueltas
    m = m if meta is None else meta
    return build_lap_snapshot(v, pd.DataFrame(), m, pct).set_index("driver_code").sort_index()


def test_una_vuelta_sin_tiempo_no_adelanta():
    # Ordenando por cumulative_time_s, BBB quedaba primero: le falta una vuelta en la suma.
    s = snapshot(0.75)
    assert s.loc["BBB", "cumulative_time_s"] < s.loc["AAA", "cumulative_time_s"]
    assert s["current_position"].to_dict() == {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}
    assert s.loc["BBB", "laps_without_time_so_far"] == 1


def test_antes_de_largar_es_el_orden_de_largada():
    s = snapshot(0.0)
    assert s["current_position"].to_dict() == {"AAA": 1, "BBB": 2, "DDD": 3, "CCC": 4}   # boxes al fondo
    assert (s["laps_done_at_cutoff"] == 0).all()
    assert s[["timing_position_at_cutoff", "pace_last_5_laps_s", "cumulative_time_s"]].isna().all().all()


def test_los_empates_no_dependen_del_orden_de_las_filas():
    # CCC y DDD abandonan en la misma vuelta y sin posición de timing: el empate se resuelve por
    # driver_code, no por el orden de las filas (que en el Silver A es el resultado de la carrera).
    _, meta = carrera()
    for semilla in range(5):
        s = snapshot(0.75, meta=meta.sample(frac=1, random_state=semilla))
        assert s["current_position"].to_dict() == {"AAA": 1, "BBB": 2, "CCC": 3, "DDD": 4}


def test_no_usa_vueltas_posteriores_al_corte():
    vueltas, _ = carrera()
    for pct in (0.25, 0.5, 0.75):
        corte = round(8 * pct)
        recortadas = vueltas[vueltas["lap_number"] <= corte]
        pd.testing.assert_frame_equal(snapshot(pct), snapshot(pct, vueltas=recortadas))


def test_neumaticos():
    s = snapshot(1.0)
    assert s.loc["BBB", "tyre_changes_so_far"] == 1      # el hueco de la vuelta 6 no cuenta como cambio
    assert s.loc["BBB", "dry_compounds_so_far"] == 2
    assert s.loc["BBB", "tyre_life_at_cutoff"] == 4
    assert s.loc["AAA", "tyre_changes_so_far"] == 0
    assert not s["used_wet_tyres_so_far"].any()
