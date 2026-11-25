"""Construye la aplicación del proyecto (docs/index.html) a partir de los datos del pipeline.

    python app/construir_app.py --datos <carpeta con los dos CSV del pipeline>

Qué hace, en orden:
  1. Ejecuta las celdas del notebook de modelado (include/Entrega3_Modelado_F1.ipynb) que leen los CSV,
     arman la posición en cada momento, separan entrenamiento y test y definen el pipeline. No hay una copia
     de ese código acá: si el notebook cambia, la aplicación cambia con él.
  2. Arma los modelos que eligió el notebook (DECISIONES, más abajo), con la misma semilla, y los vuelve a medir
     con los mismos 5 cortes. Si algún número no coincide con el notebook, avisa.
  3. Calcula las predicciones fuera de muestra (validación 2022-2025 y test 2026) para «Explorá», y la fórmula
     de cada modelo para «Probá el modelo».
  4. Dibuja todos los gráficos con Altair y escribe una sola página HTML, lista para GitHub Pages.

Si el pipeline produce datos nuevos, basta con volver a correr este script.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

RAIZ = Path(__file__).resolve().parents[1]
alt.data_transformers.disable_max_rows()        # «Explorá» lleva unas 7.000 filas: viajan dentro de la página
SEMILLA = 42

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 1. El notebook: se ejecutan sus celdas de datos y de pipeline, tal cual están
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
CELDAS = ["import os",                       # librerías, semilla, momentos
          "# Los dos CSV",                   # lectura de los CSV del pipeline (los baja de Drive si no están)
          "def estado_en_el_corte",          # posición al corte y estado de cada piloto
          'OBJETIVO = "finish_position"',    # fuga, entrenamiento 2022-2025, test 2026 y los 5 cortes
          "class EstadoEnCarrera",           # columnas derivadas y el envoltorio Llegada
          "def medir(",                      # medir, gana_claro
          "REF, filas"]                      # piso y regla ingenua


def ejecutar_notebook(ruta, carpeta_datos):
    nb = json.loads(Path(ruta).read_text(encoding="utf-8"))
    codigo = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "notebook_e3", "display": lambda *a, **k: None}
    antes = os.getcwd()
    os.chdir(carpeta_datos)
    try:
        for marca in CELDAS:
            fuente = next(c for c in codigo if c.lstrip().startswith(marca))
            if marca == "# Los dos CSV":
                # si el pipeline dejó una corrida más nueva en la carpeta, se usa esa (el notebook nombra la del 23/09)
                corridas = sorted(Path(".").glob("driver_race_snapshots_*.csv"))
                fecha = corridas[-1].stem.split("_")[-1] if corridas else "2026-09-23"
                if fecha != "2026-09-23" and Path(f"driver_lap_features_{fecha}.csv").exists():
                    fuente = fuente.replace("2026-09-23", fecha)
                    avisar(f"se usa la corrida del {fecha}, no la del notebook: los números van a cambiar")
            exec(compile(fuente, f"<notebook: {marca}>", "exec"), g)
    finally:
        os.chdir(antes)
    return g


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 2. Lo que decidió el notebook (secciones 4 y 5) y los números que dio, para comprobar que coinciden
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def arbol(**k):
    return DecisionTreeRegressor(criterion="absolute_error", random_state=SEMILLA, **k)


def bosque(**k):
    return RandomForestRegressor(n_estimators=200, random_state=SEMILLA, n_jobs=-1, **k)


def boosting(**k):
    return HistGradientBoostingRegressor(loss="absolute_error", random_state=SEMILLA, **k)


DECISIONES = {
    # columnas numéricas que quedaron en cada momento (ninguna categórica) y si se ordena dentro de la carrera
    "columnas": {0.0: ["posicion_al_corte", "driver_standing_before", "log_puntos"],
                 0.25: ["posicion_al_corte", "vueltas_detras", "tendencia_10", "gap_qualy_pct", "log_puntos_equipo"],
                 0.5: ["posicion_al_corte", "vueltas_detras", "neumatico_vs_carrera", "tendencia_10"],
                 0.75: ["posicion_al_corte", "neumatico_vs_carrera", "tendencia_10"]},
    "ordenar": {0.0: False, 0.25: True, 0.5: True, 0.75: True},
    # cada familia en su mejor versión: la ajustada sólo donde le ganó claro a la de fábrica
    "familias": {
        0.0: {"regresión lineal": Ridge(alpha=100), "árbol": arbol(max_depth=6, min_samples_leaf=30),
              "bosque": bosque(max_features=0.5, min_samples_leaf=10),
              "boosting": boosting(learning_rate=0.03, max_iter=100, max_leaf_nodes=7)},
        0.25: {"regresión lineal": LinearRegression(), "árbol": arbol(max_depth=5, min_samples_leaf=1),
               "bosque": bosque(max_features=1.0, min_samples_leaf=20), "boosting": boosting()},
        0.5: {"regresión lineal": LinearRegression(), "árbol": arbol(max_depth=8, min_samples_leaf=10),
              "bosque": bosque(max_features=0.5, min_samples_leaf=5), "boosting": boosting()},
        0.75: {"regresión lineal": LinearRegression(), "árbol": arbol(max_depth=5, min_samples_leaf=10),
               "bosque": bosque(max_features=0.5, min_samples_leaf=10),
               "boosting": boosting(learning_rate=0.03, max_iter=300, max_leaf_nodes=31)}},
    "ajuste": {0.0: {"regresión lineal": "alpha=100", "árbol": "max_depth=6, min_samples_leaf=30",
                     "bosque": "max_features=0.5, min_samples_leaf=10",
                     "boosting": "learning_rate=0.03, max_iter=100, max_leaf_nodes=7"},
               0.25: {"árbol": "max_depth=5, min_samples_leaf=1", "bosque": "max_features=1.0, min_samples_leaf=20"},
               0.5: {"árbol": "max_depth=8, min_samples_leaf=10", "bosque": "max_features=0.5, min_samples_leaf=5"},
               0.75: {"árbol": "max_depth=5, min_samples_leaf=10", "bosque": "max_features=0.5, min_samples_leaf=10",
                      "boosting": "learning_rate=0.03, max_iter=300, max_leaf_nodes=31"}},
    "elegido": {0.0: "boosting", 0.25: "regresión lineal", 0.5: "regresión lineal", 0.75: "regresión lineal"},
    # para «Probá el modelo» antes de largar: el boosting no se puede escribir en la página; el árbol sí
    "portable": {0.0: "árbol"},
}
DEL_NOTEBOOK = {"piso": 4.962, "regla": {0.0: 3.311, 0.25: 2.757, 0.5: 2.029, 0.75: 1.400},
                "elegido": {0.0: 2.911, 0.25: 2.475, 0.5: 1.853, 0.75: 1.334},
                "test": {0.0: 3.040, 0.25: 2.332, 0.5: 1.654, 0.75: 1.249}}
# números de la sección 4 del notebook (selección de columnas), que este script no vuelve a calcular
SELECCION_E3 = {"aporte_campeonato": 0.259, "equipo_varianza": 32.6, "ordenar": {0.25: 0.073, 0.5: 0.121, 0.75: 0.163},
                # sección 5: el modelo final con y sin equipo_top (mejora media, desvío, cortes en que mejora)
                "equipo_top": {0.0: (0.012, 0.023, 4), 0.25: (0.037, 0.062, 4), 0.5: (-0.008, 0.018, 2), 0.75: (-0.009, 0.015, 1)}}
# de qué columnas del snapshot sale cada entrada del modelo (para mezclarlas en la importancia por permutación)
ORIGEN = {"posicion_al_corte": ["posicion_al_corte"], "vueltas_detras": ["ultima_vuelta"],
          "neumatico_vs_carrera": ["vueltas_neumatico", "ultima_vuelta"], "tendencia_10": ["pos_hace_10"],
          "driver_standing_before": ["driver_standing_before"], "log_puntos": ["driver_points_before"],
          "gap_qualy_pct": ["quali_best_s"], "log_puntos_equipo": ["driver_points_before", "constructor_name"]}
FAMILIAS = ["regresión lineal", "árbol", "bosque", "boosting"]
ETIQUETA = {0.0: "Antes de largar", 0.25: "25 %", 0.5: "50 %", 0.75: "75 %"}
CORTO = {0.0: "0", 0.25: "25", 0.5: "50", 0.75: "75"}

AVISOS = []


def avisar(texto):
    AVISOS.append(texto)
    print("  AVISO:", texto)


def comparar(nombre, calculado, esperado, tol=0.0015):
    if abs(calculado - esperado) > tol:
        avisar(f"{nombre}: el notebook dio {esperado:.3f} y ahora da {calculado:.3f}")


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 3. Medición, predicciones fuera de muestra y fórmulas
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def calcular(g):
    D, MOMENTOS, armar, medir = g["DATOS"], g["MOMENTOS"], g["armar"], g["medir"]
    EstadoEnCarrera, REF = g["EstadoEnCarrera"], g["REF"]
    R = {"comparacion": {}, "explora": [], "proba": {}, "test": {}, "situacion": [], "coef": {}, "rangos": {}}

    for m in MOMENTOS:
        num, ordenar = DECISIONES["columnas"][m], DECISIONES["ordenar"][m]
        modelos = {k: armar(clone(v), num, [], ordenar=ordenar) for k, v in DECISIONES["familias"][m].items()}
        R["comparacion"][m] = {"piso": REF[m]["piso"], "regla": REF[m]["regla"],
                               **{k: medir(v, m) for k, v in modelos.items()}}
        elegido = modelos[DECISIONES["elegido"][m]]
        comparar(f"regla, {ETIQUETA[m]}", REF[m]["regla"].mean(), DEL_NOTEBOOK["regla"][m])
        comparar(f"modelo elegido, {ETIQUETA[m]}", R["comparacion"][m][DECISIONES["elegido"][m]].mean(),
                 DEL_NOTEBOOK["elegido"][m])
        comparar("piso", REF[m]["piso"].mean(), DEL_NOTEBOOK["piso"])

        X, y, Xt, yt = D[m]["X"], D[m]["y"], D[m]["X_test"], D[m]["y_test"]
        tr, te = D[m]["train"], D[m]["test"]

        def filas_explora(ajustado, idx, origen, muestra, corte=-1):
            Xp = X.iloc[idx] if idx is not None else Xt
            esperada = ajustado.pipeline_.predict(Xp) + ajustado._base(Xp)
            final = ajustado.predict(Xp)
            derivadas = EstadoEnCarrera().fit(Xp).transform(Xp)
            o = origen.iloc[idx] if idx is not None else origen
            terminó = o["status"].str.contains("Finished|Lap", regex=True).to_numpy()
            ya = (derivadas["vueltas_detras"] > 2).to_numpy()
            return pd.DataFrame({
                "c": o["carrera"].to_numpy(), "gp": o["race_name"].str.replace(" Grand Prix", "", regex=False).to_numpy(),
                "fecha": o["race_date"].dt.strftime("%Y-%m-%d").to_numpy(), "temp": o["season"].astype(str).to_numpy(),
                "m": ETIQUETA[m], "piloto": o["driver_code"].to_numpy(), "nombre": o["driver_name"].to_numpy(),
                "equipo": o["team_name"].to_numpy(), "pos": o["posicion_al_corte"].to_numpy(),
                "pred": np.round(final, 2), "esp": np.round(esperada, 2), "real": o["finish_position"].to_numpy(),
                "fin": np.where(terminó, "terminó", "abandonó"),
                "sit": np.select([ya, terminó], ["ya había abandonado en ese momento", "terminó"], "abandonó más adelante"),
                "muestra": muestra, "corte": corte})

        # validación: cada carrera validada la predice el modelo del corte que no la vio
        for i_corte, (e, w) in enumerate(D[m]["cortes"]):
            aj = clone(elegido).fit(X.iloc[e], y.iloc[e])
            R["explora"].append(filas_explora(aj, w, tr, "validación", i_corte))
        # test: el modelo entrenado con 2022-2025, una vez
        final = clone(elegido).fit(X, y)
        ft = filas_explora(final, None, te, "test 2026")
        R["explora"].append(ft)
        regla_t = Xt["posicion_al_corte"].to_numpy()
        pc = ft.assign(em=(ft["pred"] - ft["real"]).abs(), er=(ft["pos"] - ft["real"]).abs()).groupby("c")[["em", "er"]].mean()
        R["test"][m] = {"modelo": mean_absolute_error(yt, final.predict(Xt)), "regla": mean_absolute_error(yt, regla_t),
                        "piso": mean_absolute_error(yt, np.full(len(yt), float(np.median(y)))),
                        "mejor": int((pc["em"] < pc["er"]).sum()), "igual": int((pc["em"] == pc["er"]).sum()),
                        "peor": int((pc["em"] > pc["er"]).sum())}
        comparar(f"test, {ETIQUETA[m]}", R["test"][m]["modelo"], DEL_NOTEBOOK["test"][m])
        for s_, t_ in ft.assign(em=(ft["pred"] - ft["real"]).abs(), er=(ft["pos"] - ft["real"]).abs()).groupby("sit"):
            R["situacion"].append({"m": m, "sit": s_, "n": len(t_), "modelo": t_["em"].mean(), "regla": t_["er"].mean()})

        # «Probá el modelo»: la fórmula que se escribe en la página
        portable = DECISIONES["portable"].get(m)
        pm = final if portable is None else armar(clone(DECISIONES["familias"][m][portable]), num, [],
                                                    ordenar=ordenar).fit(X, y)
        R["proba"][m] = formula(pm, m, num, X, Xt, te, EstadoEnCarrera, final)
        if portable:
            R["proba"][m]["mae"] = R["comparacion"][m][portable]
        # en qué se apoya: cuánto sube el MAE del test al mezclar cada columna de entrada (como el notebook)
        rng, base = np.random.default_rng(SEMILLA), mean_absolute_error(yt, final.predict(Xt))
        R.setdefault("importancia", {})[m] = {}
        for col in sorted({c for k in num for c in ORIGEN[k]}):
            subas = []
            for _ in range(30):
                Xp = Xt.copy()
                Xp[col] = rng.permutation(Xp[col].to_numpy())
                subas.append(mean_absolute_error(yt, final.predict(Xp)) - base)
            R["importancia"][m][col] = float(np.mean(subas))
        if portable is None:
            pre = pm.pipeline_.named_steps["preparar"].named_transformers_["numericas"]
            coef = pm.pipeline_.named_steps["modelo"].coef_ / pre.named_steps["escalar"].scale_
            R["coef"][m] = dict(zip(num, coef))

    R["explora"] = pd.concat(R["explora"], ignore_index=True)
    R["situacion"] = pd.DataFrame(R["situacion"])
    return R


def formula(pm, m, num, X, Xt, te, EstadoEnCarrera, final):
    """Escribe el modelo como una expresión de Vega y la comprueba contra scikit-learn en el test."""
    pre = pm.pipeline_.named_steps["preparar"].named_transformers_["numericas"]
    imp, esc = pre.named_steps["imputar"].statistics_, pre.named_steps["escalar"]
    modelo = pm.pipeline_.named_steps["modelo"]
    entradas_tr = pd.DataFrame(EstadoEnCarrera().fit(X).transform(X)[num]).fillna(dict(zip(num, imp)))
    entradas_te = pd.DataFrame(EstadoEnCarrera().fit(Xt).transform(Xt)[num]).fillna(dict(zip(num, imp)))
    var = {c: f"datum.x_{c}" for c in num}
    if hasattr(modelo, "coef_"):
        partes = [f"{modelo.intercept_:.6f}"] + [f"{w:+.6f} * ({var[c]} - {mu:.6f}) / {s:.6f}"
                                                  for c, w, mu, s in zip(num, modelo.coef_, esc.mean_, esc.scale_)]
        expr = f"{var['posicion_al_corte']} + " + " ".join(partes)
        cambio = modelo.intercept_ + ((entradas_te - esc.mean_) / esc.scale_).to_numpy() @ modelo.coef_
        tipo = "lineal"
        terminos = {"intercepto": float(modelo.intercept_),
                    "partes": [(c, float(w), float(mu), float(s)) for c, w, mu, s in zip(num, modelo.coef_, esc.mean_, esc.scale_)]}
    else:
        t = modelo.tree_

        def nodo(i, fila=None):
            if t.children_left[i] == -1:
                return t.value[i][0][0] if fila is not None else f"{t.value[i][0][0]:.4f}"
            j = t.feature[i]
            umbral = t.threshold[i] * esc.scale_[j] + esc.mean_[j]          # en unidades de la columna
            if fila is not None:
                return nodo(t.children_left[i] if fila[j] <= umbral else t.children_right[i], fila)
            return (f"({var[num[j]]} <= {umbral:.6f} ? {nodo(t.children_left[i])} : "
                    f"{nodo(t.children_right[i])})")
        expr = f"{var['posicion_al_corte']} + {nodo(0)}"

        def a_json(i):
            if t.children_left[i] == -1:
                return {"v": round(float(t.value[i][0][0]), 6)}
            j = t.feature[i]
            return {"c": num[j], "u": float(t.threshold[i] * esc.scale_[j] + esc.mean_[j]),
                    "l": a_json(t.children_left[i]), "r": a_json(t.children_right[i])}
        cambio = np.array([nodo(0, fila) for fila in entradas_te.to_numpy()])
        tipo = "árbol"
        terminos = {"arbol": a_json(0)}
    esperada = entradas_te["posicion_al_corte"].to_numpy() + cambio
    sk = pm.pipeline_.predict(Xt) + pm._base(Xt)
    if not np.allclose(esperada, sk, atol=1e-6):
        avisar(f"la fórmula de {ETIQUETA[m]} no reproduce a scikit-learn (máx. dif. {np.abs(esperada - sk).max():.4f})")
    if tipo == "lineal" and pm.ordenar:
        orden = pd.DataFrame({"e": esperada, "p": Xt["posicion_al_corte"].to_numpy(), "c": Xt["carrera"].to_numpy()})
        orden = orden.sort_values(["c", "e", "p"]).assign(o=lambda d: d.groupby("c").cumcount() + 1)["o"].sort_index()
        if not np.array_equal(orden.to_numpy(), final.predict(Xt)):
            avisar(f"el orden que arma la página al {ETIQUETA[m]} no coincide con el del modelo")
    # rangos que vio el modelo al entrenar (2022-2025): los deslizadores no pasan de ahí
    rangos = {c: (float(entradas_tr[c].min()), float(entradas_tr[c].max())) for c in num}
    # para los deslizadores: del percentil 1 al 99 en las columnas con colas largas (siguen siendo valores vistos)
    colas = {"tendencia_10", "gap_qualy_pct", "neumatico_vs_carrera"}
    rangos_p = {c: ((float(entradas_tr[c].quantile(0.01)), float(entradas_tr[c].quantile(0.99))) if c in colas else rangos[c])
                for c in num}
    filas = pd.DataFrame({"c": te["carrera"].to_numpy(),
                          "gp": te["race_name"].str.replace(" Grand Prix", "", regex=False).to_numpy(),
                          "piloto": te["driver_code"].to_numpy(), "nombre": te["driver_name"].to_numpy(),
                          "equipo": te["team_name"].to_numpy(), "real": te["finish_position"].to_numpy(),
                          "esp_sk": np.round(sk, 6), **{c: entradas_te[c].round(6).to_numpy() for c in num}})
    return {"tipo": tipo, "expr": expr, "rangos": rangos, "rangos_p": rangos_p, "filas": filas, "columnas": num, "terminos": terminos}




# la tabla completa de cada carrera (notebook, sección 6): Spearman y aciertos del podio y del top 10
DEL_NOTEBOOK_TABLA = {"spearman": {"validación": {"modelo": [0.688, 0.783, 0.859, 0.906], "regla": [0.653, 0.747, 0.844, 0.899]},
                                   "test": {"modelo": [0.690, 0.823, 0.880, 0.914], "regla": [0.682, 0.821, 0.882, 0.918]}}}


def tabla_de_carrera(t, col):
    """Ordena a los pilotos de una carrera por `col` (si empatan, por dónde van) y la compara con la llegada."""
    from scipy.stats import spearmanr
    t = t.sort_values([col, "pos"])
    return pd.Series({"spearman": spearmanr(np.arange(1, len(t) + 1), t["real"]).statistic,
                      "podio": (t["real"].head(3) <= 3).mean(), "top10": (t["real"].head(10) <= 10).mean()})


def metricas_tabla(e):
    """Una fila por momento y carrera, con las tres métricas del modelo y de la regla; y los resúmenes por momento."""
    filas = []
    for (m, c), t in e.groupby(["m", "c"]):
        mo, rg = tabla_de_carrera(t, "pred"), tabla_de_carrera(t.assign(regla=t["pos"]), "regla")
        filas.append({"m": m, "c": c, "corte": t["corte"].iloc[0], "muestra": t["muestra"].iloc[0],
                      **{f"{k}_m": v for k, v in mo.items()}, **{f"{k}_r": v for k, v in rg.items()}})
    pc = pd.DataFrame(filas)
    resumen = []
    for i, mm in enumerate(MOMS):
        val = pc[(pc["m"] == mm) & (pc["muestra"] == "validación")].groupby("corte").mean(numeric_only=True)
        te = pc[(pc["m"] == mm) & (pc["muestra"] == "test 2026")]
        fila = {"momento": mm, "mejor_test": int((te["spearman_m"] > te["spearman_r"]).sum()),
                "igual_test": int((te["spearman_m"] == te["spearman_r"]).sum()),
                "peor_test": int((te["spearman_m"] < te["spearman_r"]).sum())}
        for k in ("spearman", "podio", "top10"):
            for q in ("m", "r"):
                fila[f"{k}_{q}_val"], fila[f"{k}_{q}_sd"] = val[f"{k}_{q}"].mean(), val[f"{k}_{q}"].std()
                fila[f"{k}_{q}_test"] = te[f"{k}_{q}"].mean()
            fila[f"{k}_cortes"] = int((val[f"{k}_m"] > val[f"{k}_r"]).sum())
        for q, nombre in (("m", "modelo"), ("r", "regla")):
            comparar(f"Spearman {nombre} en validación, {mm}", fila[f"spearman_{q}_val"],
                     DEL_NOTEBOOK_TABLA["spearman"]["validación"][nombre][i], 0.0006)
            comparar(f"Spearman {nombre} en test, {mm}", fila[f"spearman_{q}_test"],
                     DEL_NOTEBOOK_TABLA["spearman"]["test"][nombre][i], 0.0006)
        resumen.append(fila)
    return pc, pd.DataFrame(resumen)

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 4. Los números del tallo que no son del modelo: los datos y el hallazgo
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
GP = {"Abu Dhabi": "Abu Dabi", "Australian": "Australia", "Austrian": "Austria", "Azerbaijan": "Azerbaiyán",
      "Bahrain": "Baréin", "Barcelona": "Barcelona", "Belgian": "Bélgica", "British": "Gran Bretaña",
      "Canadian": "Canadá", "Chinese": "China", "Dutch": "Países Bajos", "Emilia Romagna": "Emilia-Romaña",
      "French": "Francia", "Hungarian": "Hungría", "Italian": "Italia", "Japanese": "Japón", "Las Vegas": "Las Vegas",
      "Mexico City": "Ciudad de México", "Miami": "Miami", "Monaco": "Mónaco", "Qatar": "Catar",
      "Saudi Arabian": "Arabia Saudita", "Singapore": "Singapur", "Spanish": "España", "São Paulo": "São Paulo",
      "United States": "Estados Unidos"}


def datos_y_hallazgo(g):
    D, snapshots, vueltas = g["DATOS"], g["snapshots"], g["vueltas"]
    todo = pd.concat([D[0.0]["train"], D[0.0]["test"]], ignore_index=True)
    terminó = todo["status"].str.contains("Finished|Lap", regex=True)
    cambio = todo["posicion_al_corte"] - todo["finish_position"]      # lugares ganados: largada − llegada
    N = {"snap": len(snapshots), "pcts": int(snapshots["pct_race_complete"].nunique()), "vueltas": len(vueltas),
         "por_momento": len(todo), "carreras": todo["carrera"].nunique(), "pilotos": todo["driver_code"].nunique(),
         "carreras_train": D[0.0]["train"]["carrera"].nunique(), "filas_train": len(D[0.0]["train"]),
         "carreras_test": D[0.0]["test"]["carrera"].nunique(), "filas_test": len(D[0.0]["test"]),
         "abandonos": int((~terminó).sum()), "abandonos_pct": float((~terminó).mean()),
         "cerca": float((cambio.abs() <= 2).mean()), "caida_abandono": float(-cambio[~terminó].mean()),
         "grandes_abandono": float(((cambio <= -5) & ~terminó).sum() / (cambio <= -5).sum()),
         "fuga": list(g["FUGA"]), "coincide": {m: float(g["POSICION"].loc[g["NOMBRE"][m], "current_position coincide"])
                                              for m in (0.25, 0.5, 0.75)}}
    N["solo_entrenan"] = D[0.0]["X"].iloc[D[0.0]["cortes"][0][0]]["carrera"].nunique()
    N["validadas"] = N["carreras_train"] - N["solo_entrenan"]
    hist = pd.DataFrame({"cambio": cambio.to_numpy(), "fin": np.where(terminó, "terminó", "abandonó")})
    filas = []
    for m in (0.0, 0.25, 0.5, 0.75):
        t = D[m]["train"].copy()
        puntos_eq = t.groupby(["carrera", "constructor_name"])["driver_points_before"].transform("sum")
        t["puesto_equipo"] = puntos_eq.groupby(t["carrera"]).rank(ascending=False, method="dense")
        for col, serie in [("posicion_al_corte", "dónde va en ese momento"),
                           ("driver_standing_before", "puesto en el campeonato"),
                           ("puesto_equipo", "puesto del equipo")]:
            filas.append({"momento": ETIQUETA[m], "serie": serie,
                          "rho": round(float(t["finish_position"].corr(t[col], method="spearman")), 3)})
    return N, hist, pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 5. Los gráficos (Altair). Colores fijos para toda la aplicación
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
AZUL = "#2563c9"          # lo que dice el modelo
NARANJA = "#e2711d"       # dónde va el piloto = lo que dice «termina donde está»
PISO = "#9c9a92"          # el piso
TINTA = "#22211e"         # lo que pasó de verdad (llegada real)
VINO = "#a3326f"          # abandonó
CLARO = "#c4c1b6"         # terminó (en el histograma) y elementos de apoyo
FUENTE = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MOMS = [ETIQUETA[m] for m in (0.0, 0.25, 0.5, 0.75)]
X_PUESTO = alt.Scale(domain=[0.5, 22.5], nice=False, zero=False)


def estilo(ch):
    ch = ch.properties(autosize=alt.AutoSizeParams(type="pad", resize=True))
    return (ch.configure(background="#ffffff", font=FUENTE, padding={"left": 4, "right": 12, "top": 6, "bottom": 4})
              .configure_view(stroke=None)
              .configure_axis(labelColor="#5f5d57", titleColor="#5f5d57", labelFontSize=12, titleFontSize=12,
                              titleFontWeight="normal", gridColor="#eeede8", domainColor="#cfccc3", tickColor="#cfccc3")
              .configure_legend(labelFontSize=12, labelColor="#2b2a26", titleFontSize=12, symbolSize=110)
              .configure_title(anchor="start", fontSize=14, fontWeight=600, color=TINTA, subtitleColor="#5f5d57",
                               subtitleFontSize=12, offset=10))


def g_datos(hist, N):
    base = (alt.Chart(hist)
              .transform_aggregate(n="count()", groupby=["cambio", "fin"])
              .transform_calculate(orden="datum.fin == 'terminó' ? 0 : 1"))
    banda = alt.Chart(pd.DataFrame({"a": [-2.5], "b": [2.5]})).mark_rect(color="#f2efe6").encode(
        x=alt.X("a:Q", scale=alt.Scale(domain=[-21, 19], nice=False)), x2="b:Q")
    barras = base.mark_bar(size=13).encode(
        x=alt.X("cambio:Q", scale=alt.Scale(domain=[-21, 19], nice=False),
                axis=alt.Axis(values=list(range(-20, 16, 5)), grid=False),
                title="← perdió lugares          ganó lugares →"),
        y=alt.Y("n:Q", stack="zero", title="Cantidad de casos"),
        color=alt.Color("fin:N", scale=alt.Scale(domain=["terminó", "abandonó"], range=[CLARO, VINO]),
                        legend=alt.Legend(title=None, orient="top-left", direction="vertical",
                                          labelExpr="datum.label == 'terminó' ? 'terminó la carrera' : 'abandonó'")),
        order=alt.Order("orden:Q"),
        tooltip=[alt.Tooltip("cambio:Q", title="Lugares ganados (negativo: perdidos)"), alt.Tooltip("fin:N", title="Situación"),
                 alt.Tooltip("n:Q", title="Casos")])
    nota = alt.Chart(pd.DataFrame({"x": [0], "t": [f"{N['cerca']:.0%} queda a 2 lugares o menos".replace("%", " %")]})).mark_text(
        dy=-6, fontSize=12, fontWeight=600, color="#5f5d57", baseline="bottom").encode(
        x="x:Q", y=alt.value(0), text="t:N")
    return (banda + barras + nota).properties(
        width=700, height=300,
        title=alt.Title("Cuántos lugares gana o pierde cada piloto entre la largada y la llegada",
                        subtitle=[f"Cada barra cuenta cuántos pilotos ganaron o perdieron esa cantidad de lugares. "
                                  f"{N['por_momento']:,} casos (un piloto en una carrera), {N['carreras']} carreras de 2022 a 2026.".replace(",", ".", 1)]))


def g_hallazgo(corr):
    series = ["dónde va en ese momento", "puesto en el campeonato", "puesto del equipo"]
    nombres = {"dónde va en ese momento": "el lugar en el que va", "puesto en el campeonato": "su puesto en el campeonato",
               "puesto del equipo": "el puesto de su equipo"}
    corr = corr.assign(nombre=corr["serie"].map(nombres))
    color = alt.Color("serie:N", scale=alt.Scale(domain=series, range=[NARANJA, "#5f5d57", "#aeaba1"]), legend=None)
    base = alt.Chart(corr).encode(
        x=alt.X("momento:N", sort=MOMS, title="Cuánto de la carrera se había corrido", axis=alt.Axis(labelAngle=0, labelPadding=6)),
        y=alt.Y("rho:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="Qué tan bien anticipa el orden de llegada",
                axis=alt.Axis(format=".1f")),
        color=color)
    tip = [alt.Tooltip("nombre:N", title="Dato"), alt.Tooltip("momento:N", title="Momento"),
           alt.Tooltip("rho:Q", title="Qué tan bien anticipa (0 a 1)", format=".2f")]
    lineas = base.mark_line(strokeWidth=3)
    puntos = base.mark_circle(size=95, opacity=1).encode(tooltip=tip)
    valores = base.transform_filter("datum.serie == 'dónde va en ese momento'").mark_text(
        dy=-14, fontSize=12, fontWeight=600).encode(text=alt.Text("rho:Q", format=".2f"))
    rotulos = base.transform_filter("datum.momento == '75 %'").transform_calculate(
        t="datum.nombre + ' · ' + format(datum.rho, '.2f')").mark_text(align="left", dx=12, fontSize=12.5).encode(
        text="t:N")
    return (lineas + puntos + valores + rotulos).properties(
        width=520, height=320, padding={"left": 4, "right": 210, "top": 6, "bottom": 4},
        title=alt.Title("Qué tan bien anticipa cada dato el orden de llegada",
                        subtitle=["1 = acierta el orden exacto; 0 = no tiene nada que ver. Carreras de 2022 a 2025.",
                                  "Antes de la largada, «el lugar en el que va» es su lugar en la grilla."]))


def g_modelo(R):
    regla_n, modelo_n = "«termina donde está» (cada uno llega donde va)", "nuestro modelo"
    filas = []
    for m in (0.0, 0.25, 0.5, 0.75):
        c, el = R["comparacion"][m], DECISIONES["elegido"][m]
        for quien, v, t in [(regla_n, c["regla"], R["test"][m]["regla"]), (modelo_n, c[el], R["test"][m]["modelo"])]:
            filas.append({"momento": ETIQUETA[m], "quien": quien, "muestra": "carreras de 2022 a 2025", "mae": v.mean()})
            filas.append({"momento": ETIQUETA[m], "quien": quien, "muestra": "carreras de 2026 (nunca vistas)", "mae": t})
    df = pd.DataFrame(filas)
    quienes = [regla_n, modelo_n]
    X_M = alt.X("momento:N", sort=MOMS, title="Cuánto de la carrera se había corrido",
                axis=alt.Axis(labelAngle=0, labelPadding=6))
    XO = alt.XOffset("quien:N", sort=quienes, scale=alt.Scale(paddingOuter=0.6))
    color = alt.Color("quien:N", scale=alt.Scale(domain=quienes, range=[NARANJA, AZUL]),
                      legend=alt.Legend(title=None, orient="bottom", direction="vertical", labelLimit=400, rowPadding=6))
    tip = [alt.Tooltip("quien:N", title="Quién predice"), alt.Tooltip("momento:N", title="Momento"),
           alt.Tooltip("muestra:N", title="Carreras"), alt.Tooltip("mae:Q", title="Se equivoca en promedio (puestos)", format=".2f")]
    base = alt.Chart(df).encode(x=X_M, xOffset=XO, color=color)
    val = base.transform_filter("datum.muestra == 'carreras de 2022 a 2025'")
    Y = alt.Y("mae:Q", scale=alt.Scale(domain=[0, 5.5]), title="Puestos que se equivoca, en promedio")
    puntos = val.mark_circle(size=130, opacity=1).encode(y=Y, tooltip=tip)
    valores = val.mark_text(dy=-12, fontSize=11.5, fontWeight=600, baseline="bottom").encode(
        y="mae:Q", text=alt.Text("mae:Q", format=".1f"))
    test = base.transform_filter("datum.muestra == 'carreras de 2026 (nunca vistas)'").mark_point(
        shape="diamond", filled=False, size=120, strokeWidth=2).encode(y="mae:Q", tooltip=tip)
    piso = alt.Chart(pd.DataFrame({"y": [R["comparacion"][0.0]["piso"].mean()]}))
    linea_piso = piso.mark_rule(strokeDash=[6, 4], color=PISO, strokeWidth=1.5).encode(y="y:Q")
    texto_piso = piso.transform_calculate(t="'adivinar a ciegas: todos terminan en el medio del pelotón (' + format(datum.y, '.1f') + ')'")\
        .mark_text(align="left", baseline="bottom", dy=-4, x=4, color="#7a776f", fontSize=11.5).encode(y="y:Q", text="t:N")
    return (linea_piso + texto_piso + puntos + valores + test).properties(
        width=640, height=330,
        title=alt.Title("Cuántos puestos se equivoca cada forma de predecir, en promedio (más abajo es mejor)",
                        subtitle=["Puntos: carreras de 2022 a 2025. Rombos: las 14 carreras de 2026, que el modelo nunca había visto."]))


TRAMOS_ERR = ["exacto", "a 1 puesto", "a 2 o 3", "a 4 o más"]
AZULES = [AZUL, "#7fa4e3", "#c3d4f1", "#e3e0d7"]   # del acierto exacto (azul del modelo) al error grande (gris)


def dist_error(e):
    """Qué parte de los pilotos queda a cada distancia de lo que predijo el modelo, por momento (validación 2022-2025)."""
    v = e[e["muestra"] == "validación"].copy()
    err = (v["pred"] - v["real"]).abs()
    v["tramo"] = np.select([err < 0.5, err < 1.5, err < 3.5], TRAMOS_ERR[:3], TRAMOS_ERR[3])
    t = v.groupby(["m", "tramo"]).size().rename("n").reset_index()
    t["pct"] = t["n"] / t.groupby("m")["n"].transform("sum")
    return [{"m": r.m, "tramo": r.tramo, "orden": TRAMOS_ERR.index(r.tramo), "pct": round(float(r.pct), 3)}
            for r in t.itertuples()]


def g_kpi_error(dist):
    """Portada, tarjeta 1: a qué distancia del puesto real queda cada predicción, en cada momento (0 % abajo, 75 % arriba)."""
    df = pd.DataFrame(dist)
    base = alt.Chart(df).encode(
        y=alt.Y("m:N", sort=MOMS[::-1], title="Momento de la carrera",
                axis=alt.Axis(domain=False, ticks=False, labelFontSize=11, labelPadding=6, titleFontSize=11,
                              labelExpr="datum.label == 'Antes de largar' ? '0 %' : datum.label")),
        x=alt.X("pct:Q", stack="normalize", title="Parte de los pilotos",
                axis=alt.Axis(format="%", values=[0, 0.5, 1], grid=False, domain=False, ticks=False,
                              labelFontSize=11, titleFontSize=11)),
        order=alt.Order("orden:Q"))
    barras = base.mark_bar(height=18).encode(
        color=alt.Color("tramo:N", scale=alt.Scale(domain=TRAMOS_ERR, range=AZULES), sort=TRAMOS_ERR,
                        legend=alt.Legend(title="Lejos del puesto real:", orient="top", direction="horizontal",
                                          labelFontSize=11, titleFontSize=11, symbolType="square", symbolSize=90,
                                          columnPadding=8, titleOrient="top")),
        tooltip=[alt.Tooltip("m:N", title="Momento"), alt.Tooltip("tramo:N", title="Se equivocó"),
                 alt.Tooltip("pct:Q", title="Parte de los pilotos", format=".0%")])
    exacto = base.transform_filter("datum.tramo == 'exacto'").mark_text(
        align="left", color="#ffffff", fontSize=11, fontWeight=700).encode(
        x=alt.value(5), text=alt.Text("pct:Q", format=".0%"))
    return (barras + exacto).properties(width=280, height=120)


def g_kpi_puntos(aciertos):
    """Portada, tarjeta 2: de los diez que el modelo pone en la zona de puntos, cuántos terminan ahí."""
    df = pd.DataFrame({"i": range(1, 11), "res": ["terminó entre los 10 primeros" if i <= aciertos else "terminó más atrás"
                                                  for i in range(1, 11)]})
    return alt.Chart(df).mark_square(size=420, opacity=1).encode(
        x=alt.X("i:O", title="Los 10 pilotos que el modelo pone en la zona de puntos",
                axis=alt.Axis(labels=False, ticks=False, domain=False, titleFontSize=11, titlePadding=8)),
        color=alt.Color("res:N", scale=alt.Scale(domain=["terminó entre los 10 primeros", "terminó más atrás"],
                                                  range=[AZUL, "#e3e0d7"]),
                        legend=alt.Legend(title=None, orient="bottom", direction="vertical", labelFontSize=11,
                                          symbolType="square", symbolSize=120, rowPadding=4)),
        tooltip=[alt.Tooltip("res:N", title="Qué pasó")]).properties(width=280, height=34)


def g_kpi_abandono(p_pilotos, p_error):
    """Portada, tarjeta 3: qué parte de los pilotos abandona y qué parte del error total explican (antes de la largada)."""
    filas = ["Parte de los pilotos", "Parte del error"]
    df = pd.DataFrame([{"fila": filas[0], "fin": "abandonaron", "o": 0, "pct": round(float(p_pilotos), 3)},
                       {"fila": filas[0], "fin": "terminaron", "o": 1, "pct": round(1 - float(p_pilotos), 3)},
                       {"fila": filas[1], "fin": "abandonaron", "o": 0, "pct": round(float(p_error), 3)},
                       {"fila": filas[1], "fin": "terminaron", "o": 1, "pct": round(1 - float(p_error), 3)}])
    y = alt.Y("fila:N", sort=filas, title=None,
              axis=alt.Axis(domain=False, ticks=False, labelFontSize=12, labelPadding=8, labelColor="#2b2a26"))
    barras = alt.Chart(df).mark_bar(height=22).encode(
        y=y,
        x=alt.X("pct:Q", stack="zero", scale=alt.Scale(domain=[0, 1]), title="Porcentaje del total",
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], grid=False, domain=False,
                              labelFontSize=11, titleFontSize=11)),
        order=alt.Order("o:Q"),
        color=alt.Color("fin:N", scale=alt.Scale(domain=["abandonaron", "terminaron"], range=[VINO, CLARO]),
                        legend=alt.Legend(title=None, orient="top", direction="horizontal", labelFontSize=11,
                                          symbolType="square", symbolSize=100, offset=6)),
        tooltip=[alt.Tooltip("fila:N", title=" "), alt.Tooltip("fin:N", title="Pilotos que"),
                 alt.Tooltip("pct:Q", title="Porcentaje", format=".0%")])
    valores = alt.Chart(df).transform_filter("datum.fin == 'abandonaron'").mark_text(
        align="left", dx=6, fontSize=12, fontWeight=700, color=TINTA).encode(
        y=y, x=alt.X("pct:Q"), text=alt.Text("pct:Q", format=".0%"))
    return (barras + valores).properties(width=210, height=80)


def g_explora(e, pc, inicial):
    temps = sorted(e["temp"].unique())
    pil = (e.drop_duplicates("piloto").sort_values("nombre"))
    momento = alt.param(name="momento_e", value=inicial["m"],
                        bind=alt.binding_radio(options=MOMS, name=" ", element="#c-momento"))
    temporada = alt.param(name="temporada_e", value="Todas",
                          bind=alt.binding_select(options=["Todas"] + temps, name=" ", element="#c-temporada"))
    piloto = alt.param(name="piloto_e", value="Todos",
                       bind=alt.binding_select(options=["Todos"] + pil["piloto"].tolist(),
                                               labels=["Todos"] + [f"{n} ({c})" for n, c in zip(pil["nombre"], pil["piloto"])],
                                               name=" ", element="#c-piloto"))
    medida = alt.param(name="medida_e", value="Cuántos puestos se equivoca",
                       bind=alt.binding_radio(options=["Cuántos puestos se equivoca", "Qué tan parecida queda la tabla"], name=" ",
                                              element="#c-medida"))
    # sólo cuenta el clic sobre una carrera: un clic (o mantener apretado) en el vacío no borra la elección
    carrera = alt.selection_point(name="carrera_e", fields=["c"], value=[{"c": inicial["c"]}],
                                  on="click[event.item && event.item.datum && event.item.datum.c]", clear=False, toggle=False, empty=False)
    filtro = ("datum.m == momento_e && (temporada_e == 'Todas' || datum.temp == temporada_e) && "
              "(piloto_e == 'Todos' || datum.piloto == piloto_e)")
    errores = dict(em="abs(datum.pred - datum.real)", er="abs(datum.pos - datum.real)")
    pano = (alt.Chart(e).transform_filter(filtro).transform_calculate(**errores)
              .transform_aggregate(mae_m="mean(em)", mae_r="mean(er)", n="count()",
                                   groupby=["c", "m", "gp", "fecha", "temp", "muestra"])
              .transform_calculate(k="datum.c + '|' + datum.m")
              .transform_lookup(lookup="k", from_=alt.LookupData(pc, key="k", fields=TABLA_CAMPOS))
              .transform_calculate(mae="medida_e == 'Cuántos puestos se equivoca'",
                                   modelo="datum.mae ? datum.mae_m : datum.spearman_m",
                                   regla="datum.mae ? datum.mae_r : datum.spearman_r"))
    fin_test = (pd.to_datetime(e["fecha"].max()) + pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    test = pd.DataFrame({"a": ["2026-01-01"], "b": [fin_test], "t": ["2026: carreras nuevas"]})
    X_F = alt.X("fecha:T", title=None, axis=alt.Axis(format="%Y", tickCount="year", grid=False), scale=alt.Scale(padding=12))
    banda = alt.Chart(test).mark_rect(color="#f2efe6", clip=True).encode(x=alt.X("a:T"), x2="b:T")
    # el rótulo de 2026 va abajo de la franja, donde casi no hay puntos
    banda_t = alt.Chart(test).mark_text(align="left", baseline="bottom", dx=6, y=276, color="#7a776f", fontSize=11.5).encode(
        x="a:T", text="t:N")
    tip_p = [alt.Tooltip("gp:N", title="Carrera"), alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"),
             alt.Tooltip("mae_m:Q", title="El modelo se equivoca (puestos)", format=".1f"),
             alt.Tooltip("mae_r:Q", title="«Termina donde está» se equivoca (puestos)", format=".1f"),
             alt.Tooltip("spearman_m:Q", title="Parecido de la tabla del modelo (0 a 1)", format=".2f"),
             alt.Tooltip("spearman_r:Q", title="Parecido de la tabla de la regla (0 a 1)", format=".2f"),
             alt.Tooltip("podio_m:Q", title="Podio que acertó el modelo", format=".0%"),
             alt.Tooltip("top10_m:Q", title="Zona de puntos que acertó el modelo", format=".0%")]
    # acercar y alejar: rueda del mouse o pellizco; arrastrar mueve; doble clic vuelve a la vista completa
    zoom = alt.selection_interval(name="zoom_e", bind="scales", encodings=["x", "y"])
    une = pano.mark_rule(color="#d9d6cd", clip=True).encode(x=X_F, y="regla:Q", y2="modelo:Q")
    regla = pano.mark_tick(color=NARANJA, thickness=2.5, size=11, opacity=0.9, clip=True).encode(x=X_F, y="regla:Q", tooltip=tip_p)
    modelo = pano.mark_circle(color=AZUL, opacity=1, cursor="pointer", clip=True).encode(
        x=X_F, y=alt.Y("modelo:Q", scale=alt.Scale(zero=False, padding=12), title=None),
        size=alt.condition(carrera, alt.value(190), alt.value(48)),
        stroke=alt.condition(carrera, alt.value(TINTA), alt.value(None)), strokeWidth=alt.value(2),
        tooltip=tip_p).add_params(carrera, zoom)
    panorama = (banda + banda_t + une + regla + modelo).properties(
        width=830, height=280,
        title=alt.Title("Todas las carreras", offset=22,
                        subtitle=["Cada punto azul es una carrera: cómo le fue al modelo. La raya naranja: cómo le fue a «termina donde está».",
                                  "Hacé clic en una carrera para abrirla abajo."]))

    sel = alt.Chart(e).transform_filter("datum.m == momento_e").transform_filter(carrera)
    cab_base = (sel.transform_calculate(**errores)
                   .transform_aggregate(modelo="mean(em)", regla="mean(er)", groupby=["c", "m", "gp", "temp", "muestra"])
                   .transform_calculate(k="datum.c + '|' + datum.m")
                   .transform_lookup(lookup="k", from_=alt.LookupData(pc, key="k", fields=TABLA_CAMPOS))
                   .transform_calculate(
                       t="datum.gp + ' ' + datum.temp + ' · ' + (momento_e == 'Antes de largar' ? 'antes de la largada' : "
                         "'con el ' + momento_e + ' de la carrera corrido')",
                       t2="'Nuestro modelo: se equivocó ' + format(datum.modelo, '.1f') + ' puestos en promedio · acertó ' + "
                          "round(datum.podio_m * 3) + ' de 3 del podio y ' + round(datum.top10_m * 10) + ' de 10 de la zona de puntos'",
                       t3="'«Termina donde está»: se equivocó ' + format(datum.regla, '.1f') + ' puestos en promedio · acertó ' + "
                          "round(datum.podio_r * 3) + ' de 3 del podio y ' + round(datum.top10_r * 10) + ' de 10 de la zona de puntos'"))
    cab = (cab_base.mark_text(align="left", baseline="top", x=0, y=2, fontSize=14, fontWeight=600, color=TINTA).encode(text="t:N")
           + cab_base.mark_text(align="left", baseline="top", x=0, y=24, fontSize=12.5, color=AZUL).encode(text="t2:N")
           + cab_base.mark_text(align="left", baseline="top", x=0, y=42, fontSize=12.5, color="#b35a16").encode(text="t3:N")
           ).properties(width=760, height=60)
    # el detalle va en dos vistas aparte (se elige con botones): reciben momento, piloto y carrera desde el panorama
    sub = alt.Chart(e).transform_filter("datum.m == momento_e && datum.c == carrera_c")
    libres = [alt.param(name="momento_e", value=inicial["m"]), alt.param(name="piloto_e", value="Todos"),
              alt.param(name="carrera_c", value=inicial["c"])]
    det = sub.transform_calculate(etq="datum.piloto + ' · ' + datum.equipo",
                                  tuyo="piloto_e != 'Todos' && datum.piloto == piloto_e",
                                  error="abs(datum.pred - datum.real)")
    Y_D = alt.Y("etq:N", sort=alt.EncodingSortField("real", order="ascending"), title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=11.5, ticks=False, domain=False))
    X_D = dict(scale=X_PUESTO, axis=alt.Axis(values=list(range(1, 23)), orient="top", grid=True),
               title="Puesto (1 = el primero)")
    tip_d = [alt.Tooltip("nombre:N", title="Piloto"), alt.Tooltip("equipo:N", title="Equipo"),
             alt.Tooltip("m:N", title="Momento"), alt.Tooltip("pos:Q", title="Dónde iba"),
             alt.Tooltip("pred:Q", title="Dónde dijo el modelo que terminaba", format=".1f"),
             alt.Tooltip("real:Q", title="Dónde terminó"), alt.Tooltip("sit:N", title="Qué pasó"),
             alt.Tooltip("error:Q", title="Puestos que se equivocó el modelo", format=".1f")]
    resalte = det.transform_filter("datum.tuyo").mark_bar(color="#fdf0c4", opacity=1).encode(
        y=Y_D, x=alt.X(datum=0.5, scale=X_PUESTO), x2=alt.X2(datum=22.5))
    mov = det.mark_rule(color="#d9d6cd", strokeWidth=2).encode(y=Y_D, x=alt.X("pos:Q", **X_D), x2="pred:Q")
    iba = det.mark_circle(color=NARANJA, size=80, opacity=1).encode(y=Y_D, x=alt.X("pos:Q", **X_D), tooltip=tip_d)
    dijo = det.mark_circle(color=AZUL, size=95, opacity=1).encode(y=Y_D, x=alt.X("pred:Q", **X_D), tooltip=tip_d)
    real = det.mark_point(filled=True, size=90, opacity=1).encode(
        y=Y_D, x=alt.X("real:Q", **X_D), tooltip=tip_d,
        shape=alt.Shape("fin:N", scale=alt.Scale(domain=["terminó", "abandonó"], range=["diamond", "cross"]), legend=None),
        color=alt.Color("fin:N", scale=alt.Scale(domain=["terminó", "abandonó"], range=[TINTA, VINO]), legend=None))
    detalle = (resalte + mov + iba + dijo + real).properties(width=760, height=alt.Step(19))
    tablas = g_dos_tablas(sub, "pred", "piloto_e != 'Todos' && datum.piloto == piloto_e",
                          tip_d, "Lo que dijo el modelo", "Cómo terminó")
    return {"panorama": alt.vconcat(panorama, cab, spacing=14).add_params(momento, temporada, piloto, medida),
            "tablas": tablas.add_params(*libres), "detalle": alt.vconcat(detalle).add_params(*libres)}


TABLA_CAMPOS = ["spearman_m", "spearman_r", "podio_m", "podio_r", "top10_m", "top10_r"]


def g_dos_tablas(sel, col_pred, tuyo, tip, izq, der, real="real", alto=21):
    """Dos columnas de puestos unidas por una línea por piloto: la tabla que arma una predicción y la de llegada."""
    t = (sel.transform_window(pp="row_number()", sort=[alt.SortField(col_pred), alt.SortField("pos")])
            .transform_calculate(dif=f"abs(datum.pp - datum.{real})", error=f"abs(datum.{col_pred} - datum.{real})", tuyo=tuyo,
                                 grupo=("datum.fin == 'abandonó' ? 'abandonó' : datum.dif <= 1 ? 'acertó (o le erró por 1)' : "
                                        "datum.dif <= 4 ? 'le erró por 2 a 4' : 'le erró por 5 o más'"),
                                 izq="'P' + datum.pp + '  ' + datum.piloto", der=f"datum.piloto + '  P' + datum.{real}"))
    tip = tip[:5] + [alt.Tooltip("pp:Q", title="Puesto en la tabla que armó el modelo")] + tip[5:]
    Y = dict(scale=alt.Scale(domain=[0.5, 22.5], reverse=True, nice=False, zero=False), axis=None, title=None)
    X = alt.Scale(domain=[-0.42, 1.42], nice=False, zero=False)
    grupos = ["acertó (o le erró por 1)", "le erró por 2 a 4", "le erró por 5 o más", "abandonó"]
    zonas = alt.Chart(pd.DataFrame({"a": [0.5, 3.5], "b": [3.5, 10.5], "t": ["podio", "zona de puntos (top 10)"],
                                    "c": ["#fdf0c4", "#f4f2ec"], "y": [2, 7]}))
    capas = [*[zonas.transform_filter(f"datum.c == '{col}'").mark_rect(color=col).encode(
                 y=alt.Y("a:Q", **Y), y2="b:Q", x=alt.X(datum=-0.42, scale=X, axis=None, title=None), x2=alt.X2(datum=1.42))
               for col in ("#fdf0c4", "#f4f2ec")],
             zonas.mark_text(fontSize=10.5, color="#8a877f").encode(y=alt.Y("y:Q", **Y), x=alt.X(datum=0.5, scale=X, axis=None, title=None), text="t:N"),
             t.mark_rule(strokeCap="round").encode(
                 x=alt.X(datum=0, scale=X, axis=None, title=None), x2=alt.X2(datum=1), y=alt.Y("pp:Q", **Y), y2=f"{real}:Q",
                 color=alt.Color("grupo:N", scale=alt.Scale(domain=grupos, range=["#b9b6ab", "#7b8db0", TINTA, VINO]),
                                 legend=alt.Legend(title=None, orient="bottom", direction="horizontal")),
                 strokeWidth=alt.condition("datum.tuyo", alt.value(5), alt.value(2.2)),
                 strokeDash=alt.condition("datum.fin == 'abandonó'", alt.value([5, 3]), alt.value([1, 0])), tooltip=tip),
             *[t.transform_filter(fl).mark_text(align="right", dx=-8, fontSize=12, fontWeight=w).encode(
                 x=alt.X(datum=0, scale=X, axis=None, title=None), y=alt.Y("pp:Q", **Y), text="izq:N", tooltip=tip)
                   for fl, w in (("datum.tuyo", 800), ("!datum.tuyo", 400))],
             *[t.transform_filter(fl).mark_text(align="left", dx=8, fontSize=12, fontWeight=w).encode(
                 x=alt.X(datum=1, scale=X, axis=None, title=None), y=alt.Y(f"{real}:Q", **Y), text="der:N", tooltip=tip)
                   for fl, w in (("datum.tuyo", 800), ("!datum.tuyo", 400))]]
    tit = alt.Chart(pd.DataFrame({"x": [0, 1], "t": [izq, der]})).mark_text(fontSize=12.5, fontWeight=700, dy=-12).encode(
        x=alt.X("x:Q", scale=X, axis=None, title=None), y=alt.value(0), text="t:N")
    return alt.layer(*capas, tit).properties(width=560, height=22 * alto)



def g_tabla(res):
    """Spearman, podio y top 10 por momento: modelo contra regla, validación (± un desvío) y test."""
    filas = []
    for _, r in res.iterrows():
        for k, tit in [("spearman", "Spearman"), ("podio", "Aciertos del podio"), ("top10", "Aciertos del top 10")]:
            for q, quien in [("r", "«termina donde está»"), ("m", "modelo")]:
                v, sd = r[f"{k}_{q}_val"], r[f"{k}_{q}_sd"]
                filas.append({"panel": tit, "momento": r["momento"], "quien": quien, "muestra": "validación 2022–2025",
                              "v": v, "lo": v - sd, "hi": v + sd})
                filas.append({"panel": tit, "momento": r["momento"], "quien": quien, "muestra": "test 2026",
                              "v": r[f"{k}_{q}_test"], "lo": None, "hi": None})
    df = pd.DataFrame(filas)
    quienes = ["«termina donde está»", "modelo"]
    paneles = []
    for i, tit in enumerate(["Spearman", "Aciertos del podio", "Aciertos del top 10"]):
        base = alt.Chart(df).transform_filter(f"datum.panel == '{tit}'").encode(
            x=alt.X("momento:N", sort=MOMS, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=11)),
            xOffset=alt.XOffset("quien:N", sort=quienes),
            color=alt.Color("quien:N", scale=alt.Scale(domain=quienes, range=[NARANJA, AZUL]),
                            legend=alt.Legend(title=None, orient="top") if i == 0 else None))
        tip = [alt.Tooltip("quien:N", title="Quién"), alt.Tooltip("momento:N", title="Momento"),
               alt.Tooltip("muestra:N", title="Dónde"), alt.Tooltip("v:Q", title=tit, format=".3f")]
        val = base.transform_filter("datum.muestra == 'validación 2022–2025'")
        Y = alt.Y("v:Q", scale=alt.Scale(domain=[0.5, 1.0]), title=None, axis=alt.Axis(format=".2f"))
        paneles.append((val.mark_rule(strokeWidth=2).encode(y=alt.Y("lo:Q", scale=alt.Scale(domain=[0.5, 1.0])), y2="hi:Q")
                        + val.mark_circle(size=80, opacity=1).encode(y=Y, tooltip=tip)
                        + base.transform_filter("datum.muestra == 'test 2026'").mark_point(
                            shape="diamond", filled=False, size=80, strokeWidth=2).encode(y=Y, tooltip=tip))
                       .properties(width=230, height=230, title=tit))
    return alt.hconcat(*paneles, spacing=18).properties(
        title=alt.Title("La tabla completa de cada carrera, comparada con la final (más arriba, mejor)",
                        subtitle=["Punto: media de los 5 cortes de la validación, con un desvío. Rombo: promedio de las 14 carreras de 2026."]))


ENTRADA = {  # cómo se muestra cada entrada del modelo en los deslizadores de «Probá el modelo»
    "posicion_al_corte": ("Lugar en el que va", 1, None),
    "vueltas_detras": ("Vueltas que le lleva el líder (3 o más: ya abandonó)", 1, None),
    "tendencia_10": ("Lugares ganados en las últimas 10 vueltas (negativo: perdidos)", 1, None),
    "gap_qualy_pct": ("Cuánto más lento que el más rápido fue en la clasificación (%)", 0.1, None),
    "log_puntos_equipo": ("Puntos de su equipo en el campeonato", 1, "log"),
    "neumatico_vs_carrera": ("Vueltas de uso de sus neumáticos, comparado con el resto (más = más gastados)", 0.5, None),
    "driver_standing_before": ("Puesto en el campeonato", 1, None),
    "log_puntos": ("Puntos del piloto en el campeonato", 0.5, "log"),
}


def a_pantalla(col, v):
    """De la unidad del modelo a la del deslizador (los puntos entran al modelo como log(1 + puntos)),
    redondeado al paso del deslizador."""
    x = np.expm1(v) if ENTRADA[col][2] == "log" else v
    paso = ENTRADA[col][1]
    return round(round(float(x) / paso) * paso, 2)



def g_proba_tabla():
    """La tabla de «Probá»: dónde va cada piloto (izquierda) y dónde lo hace terminar el modelo (derecha).
    Los datos los pone la página cada vez que se mueve algo (conjunto «tabla»)."""
    t = (alt.Chart(alt.NamedData(name="tabla"))
            .transform_calculate(cambio="datum.final < datum.pos ? 'gana lugares' : datum.final > datum.pos ? 'pierde lugares' : 'queda igual'",
                                 izq="'P' + datum.pos + '  ' + datum.piloto",
                                 der="datum.lineal ? datum.piloto + '  P' + datum.final : datum.piloto + '  ~' + format(datum.esp, '.0f')"))
    Y = dict(scale=alt.Scale(domain=[0.5, 22.5], reverse=True, nice=False, zero=False), axis=None, title=None)
    X = alt.Scale(domain=[-0.42, 1.42], nice=False, zero=False)
    tip = [alt.Tooltip("nombre:N", title="Piloto"), alt.Tooltip("equipo:N", title="Equipo"),
           alt.Tooltip("pos:Q", title="Dónde va"), alt.Tooltip("esp:Q", title="Puesto que calcula el modelo (con decimales)", format=".1f"),
           alt.Tooltip("final:Q", title="Dónde termina, según el modelo")]
    zonas = alt.Chart(pd.DataFrame({"a": [0.5, 3.5], "b": [3.5, 10.5], "c": ["#fdf0c4", "#f4f2ec"]}))
    capas = [*[zonas.transform_filter(f"datum.c == '{col}'").mark_rect(color=col).encode(
                 y=alt.Y("a:Q", **Y), y2="b:Q", x=alt.X(datum=-0.42, scale=X, axis=None, title=None), x2=alt.X2(datum=1.42))
               for col in ("#fdf0c4", "#f4f2ec")],
             t.mark_rule(strokeCap="round").encode(
                 x=alt.X(datum=0, scale=X, axis=None, title=None), x2=alt.X2(datum=1), y=alt.Y("pos:Q", **Y), y2="final:Q",
                 color=alt.Color("cambio:N", scale=alt.Scale(domain=["gana lugares", "queda igual", "pierde lugares"],
                                                             range=[AZUL, "#c4c1b6", TINTA]),
                                 legend=alt.Legend(title=None, orient="bottom", direction="horizontal")),
                 strokeWidth=alt.condition("datum.tuyo", alt.value(5), alt.value(2.2)), tooltip=tip),
             *[t.transform_filter(fl).mark_text(align="right", dx=-8, fontSize=12, fontWeight=w).encode(
                 x=alt.X(datum=0, scale=X, axis=None, title=None), y=alt.Y("pos:Q", **Y), text="izq:N", tooltip=tip)
                   for fl, w in (("datum.tuyo", 800), ("!datum.tuyo", 400))],
             *[t.transform_filter(fl).mark_text(align="left", dx=8, fontSize=12, color=AZUL, fontWeight=w).encode(
                 x=alt.X(datum=1, scale=X, axis=None, title=None), y=alt.Y("final:Q", **Y), text="der:N", tooltip=tip)
                   for fl, w in (("datum.tuyo", 800), ("!datum.tuyo", 500))]]
    tit = alt.Chart(pd.DataFrame({"x": [0, 1], "t": ["Ahora", "Al final, según el modelo"]})).mark_text(
        fontSize=12.5, fontWeight=700, dy=-12).encode(x=alt.X("x:Q", scale=X, axis=None, title=None), y=alt.value(0), text="t:N")
    return alt.layer(*capas, tit).properties(width=380, height=22 * 21)


def g_proba_aportes():
    a = alt.Chart(alt.NamedData(name="aportes"))
    Y = alt.Y("etq:N", sort=alt.EncodingSortField("v", order="ascending"), title=None,
              axis=alt.Axis(labelLimit=340, ticks=False, domain=False))
    barras = a.mark_bar(height=16, color=AZUL).encode(
        y=Y, x=alt.X("v:Q", title="← lo adelanta          lo atrasa → (lugares)"),
        opacity=alt.condition("datum.v < 0", alt.value(1), alt.value(0.45)),
        tooltip=[alt.Tooltip("etq:N", title="Dato"), alt.Tooltip("v:Q", title="Lugares que lo mueve", format="+.1f")])
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a877f").encode(x="x:Q")
    valores = a.mark_text(fontSize=11.5, fontWeight=600, color=TINTA, align="left").encode(
        y=Y, x=alt.value(432), text=alt.Text("v:Q", format="+.1f"))
    return (cero + barras + valores).properties(width=420, height=alt.Step(26),
                                      title="Qué lo empuja hacia adelante y qué hacia atrás")


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>¿Dónde termina un piloto?</title>
<meta name="description" content="Proyecto integrador de Ciencia de Datos (UTN FRM, 2026): predecir el puesto de llegada de cada piloto de Fórmula 1 en cuatro momentos de la carrera.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Titillium+Web:wght@400;600;700;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.20.1"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0"></script>
<style>
:root{--fondo:#ffffff;--papel:#ffffff;--tinta:#1d1c1a;--suave:#5e5e66;--tenue:#82828a;--borde:#e4e4e9;
--rojo:#e10600;--azul:#2563c9;--naranja:#e2711d;--vino:#a3326f;--amarillo:#fdf0c4;
--carbono:#15151e;--gris-claro:#f3f3f5;--carbono2:#2a2a33;--gris:#38383f;--display:"Titillium Web","Segoe UI",Inter,sans-serif;
--logo-f1:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 30'%3E%3Cpath d='M101.086812,30 L101.711812,30 L101.711812,27.106875 L101.722437,27.106875 L102.761812,30 L103.302437,30 L104.341812,27.106875 L104.352437,27.106875 L104.352437,30 L104.977437,30 L104.977437,26.25125 L104.063687,26.25125 L103.055562,29.18625 L103.044937,29.18625 L102.011187,26.25125 L101.086812,26.25125 L101.086812,30 Z M97.6274375,26.818125 L98.8136875,26.818125 L98.8136875,30 L99.4699375,30 L99.4699375,26.818125 L100.661812,26.818125 L100.661812,26.25125 L97.6274375,26.25125 L97.6274375,26.818125 Z M89.9999375,30 L119.999937,0 L101.943687,0 L71.9443125,30 L89.9999375,30 Z M85.6986875,13.065 L49.3818125,13.065 C38.3136875,13.065 36.3768125,13.651875 31.6361875,18.3925 C27.2024375,22.82625 20.0005625,30 20.0005625,30 L35.7324375,30 L39.4855625,26.246875 C41.9530625,23.779375 43.2255625,23.52375 48.4068125,23.52375 L75.2405625,23.52375 L85.6986875,13.065 Z M31.1518125,16.253125 C27.8774375,19.3425 20.7530625,26.263125 16.9130625,30 L-6.25e-05,30 C-6.25e-05,30 13.5524375,16.486875 21.0849375,9.0725 C28.8455625,1.685 32.7143125,0 46.9486875,0 L98.7643125,0 L87.5449375,11.21875 L48.0011875,11.21875 C37.9993125,11.21875 35.7518125,11.911875 31.1518125,16.253125 Z' fill='%23e10600'/%3E%3C/svg%3E");
--corte:polygon(8px 0,100% 0,calc(100% - 8px) 100%,0 100%);--salida:cubic-bezier(.16,1,.3,1)}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;background:var(--fondo);color:var(--tinta);font:16px/1.6 Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
::selection{background:var(--rojo);color:#fff}
:focus-visible{outline:2px solid var(--rojo);outline-offset:2px}
a{color:var(--tinta);text-decoration-color:var(--rojo);text-decoration-thickness:2px;text-underline-offset:3px;transition:color .2s}
a:hover{color:var(--rojo)}
code{font-size:.84em;background:#efeff2;padding:1px 5px;border-radius:2px}
header.barra{position:sticky;top:0;z-index:10;background:var(--carbono);color:#fff}
header.barra::after{content:"";display:block;height:3px;background:var(--rojo);transform-origin:0 50%}
.barra .dentro{max-width:1040px;margin:0 auto;padding:10px 20px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.marca{font-family:var(--display);font-size:14px;font-weight:700;letter-spacing:.02em;text-transform:uppercase;display:flex;align-items:center;gap:10px;white-space:nowrap}
.marca::before{content:"";flex:none;width:40px;height:10px;background:var(--logo-f1) left center/contain no-repeat}
.marca .bandera{display:none}
nav{display:flex;align-items:center;gap:2px;flex-wrap:wrap;font-size:14px}
nav .grupo{font-family:var(--display);color:#80808a;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin:0 4px 0 12px}
nav a{color:#cfcfd6;text-decoration:none;padding:4px 8px;font-weight:500;clip-path:var(--corte);transition:background-color .25s,color .25s}
nav a:hover{background:var(--carbono2);color:#fff}
nav a.activo{background:var(--rojo);color:#fff}
main{max-width:1040px;margin:0 auto;padding:0 20px 40px}
section{padding:68px 0 24px;border-bottom:1px solid var(--borde)}
section:last-of-type{border-bottom:0}
#explora,#proba{background:var(--gris-claro);box-shadow:0 0 0 100vmax var(--gris-claro);clip-path:inset(0 -100vmax)}
#explora .controles,#proba .controles{background:#fff}
#datos,#hallazgo,#modelo{position:relative}
#datos::before,#hallazgo::before,#modelo::before{content:"01";content:"01"/"";position:absolute;top:36px;right:0;z-index:-1;font-family:var(--display);font-size:clamp(88px,13vw,170px);font-weight:900;line-height:1;letter-spacing:-.03em;color:transparent;-webkit-text-stroke:1.5px #d2d2d8;pointer-events:none}
#hallazgo::before{content:"02";content:"02"/""}#modelo::before{content:"03";content:"03"/""}
#portada{position:relative;background:var(--carbono) var(--logo-f1) no-repeat right 0 top 22px/96px 24px;box-shadow:0 0 0 100vmax var(--carbono);clip-path:inset(0 -100vmax);color:#fff;border-bottom:0;padding-bottom:48px}
#portada::before,#portada::after{content:"";position:absolute;top:26px;left:0;height:14px;width:100px;background:radial-gradient(circle at 7px 7px,var(--gris) 6px,transparent 6.5px) 0 0/20px 14px repeat-x}
#portada::after{width:0;background-image:radial-gradient(circle at 7px 7px,var(--rojo) 6px,transparent 6.5px)}
#portada .parte{color:#9d9da5}
#portada h1{color:#fff}
#portada .idea{color:#c9c9d0}
#portada .idea a{color:#fff}
#portada .kpi,#portada .pista{color:var(--tinta)}
.parte{display:inline-flex;align-items:center;gap:12px;font-family:var(--display);font-size:12.5px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--suave)}
.parte::before{content:"";width:28px;height:6px;background:var(--rojo);transform:skewX(-28deg);transform-origin:0 50%}
h1{font-family:var(--display);font-size:clamp(36px,6vw,70px);font-weight:900;line-height:1;letter-spacing:-.01em;text-transform:uppercase;margin:16px 0 24px;max-width:20ch}
h2{font-family:var(--display);font-size:clamp(28px,3.6vw,42px);font-weight:700;line-height:1.1;letter-spacing:-.01em;margin:12px 0 14px;max-width:none}
h4{font-family:var(--display);font-size:16px;font-weight:700;margin:22px 0 6px}
section>h3{font-family:var(--display);font-size:clamp(21px,2.5vw,26px);font-weight:700;line-height:1.2;letter-spacing:-.005em;margin:40px 0 10px}
.idea{font-size:clamp(18px,2.1vw,21px);line-height:1.55;max-width:46ch;color:#2b2a26;margin:0 0 32px}
#portada h1{max-width:none}
#portada .idea{max-width:62ch}
main p:not(.aviso),main li{max-width:72ch}
.kpis{display:grid;grid-template-columns:1fr;gap:12px;margin:12px 0 8px}
.tip-corto{position:fixed;z-index:60;pointer-events:none;background:var(--carbono);color:#fff;border-radius:2px;border-left:3px solid var(--rojo);padding:7px 10px;font-size:13px;line-height:1.4;max-width:280px;box-shadow:0 6px 18px rgba(0,0,0,.24)}.tip-corto span{opacity:.75}.tip-corto .clic{opacity:.7;font-size:11.5px;margin-top:3px}.panel-e{position:fixed;z-index:55;top:0;right:0;height:100vh;width:min(360px,92vw);box-sizing:border-box;overflow:auto;background:var(--papel);border-left:3px solid var(--rojo);box-shadow:-8px 0 24px rgba(0,0,0,.12);padding:22px 22px 28px}.panel-e h3{font-family:var(--display);margin:0 34px 4px 0;font-size:22px;font-weight:700;line-height:1.2}.panel-e .sub{color:var(--suave);font-size:13px;margin:0 0 14px}.panel-e dl{margin:0;display:grid;grid-template-columns:1fr;gap:0}.panel-e dt{font-family:var(--display);color:var(--suave);font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-top:12px}.panel-e dd{margin:2px 0 0;font-size:16px;font-weight:600;padding-bottom:8px;border-bottom:1px solid var(--borde)}.panel-e .cerrar{position:absolute;top:14px;right:14px;width:32px;height:32px;border-radius:2px;border:1px solid var(--borde);background:#fff;font-size:18px;line-height:1;cursor:pointer;transition:background-color .2s,color .2s}.panel-e .cerrar:hover{background:var(--carbono);color:#fff}@media (max-width:760px){.panel-e{top:auto;bottom:0;height:auto;max-height:65vh;width:100%;border-left:none;border-top:3px solid var(--rojo);border-radius:4px 4px 0 0;box-shadow:0 -8px 24px rgba(0,0,0,.14)}}
.pista{margin:4px 0 24px;background:var(--papel);border:0;border-radius:2px;padding:18px 20px 10px}.pista figcaption{font-size:15px;color:var(--suave);margin-bottom:8px}.pista-svg{overflow-x:auto}.pista svg{display:block;width:100%;min-width:600px;height:auto}
.kpi{background:var(--papel);border:0;border-radius:2px;padding:24px 28px;border-left:5px solid var(--azul);display:grid;grid-template-columns:minmax(210px,250px) minmax(0,1fr) 370px;grid-template-areas:"etq que graf" "num que graf" "tit ctx graf";column-gap:32px;align-items:start}
.kpi .graf{grid-area:graf;align-self:center;justify-self:end;min-height:120px;max-width:370px}
.kpi .graf .vega-embed{display:block}
.kpi:nth-child(3){border-left-color:var(--vino)}
.kpi .etq{grid-area:etq;font-family:var(--display);font-size:12.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--tenue)}
.kpi .num{grid-area:num;color:var(--rojo);font-family:var(--display);font-size:clamp(46px,5.4vw,64px);font-weight:900;letter-spacing:-.02em;line-height:1;font-variant-numeric:tabular-nums;margin-top:8px}
.kpi .tit{grid-area:tit;font-size:17px;font-weight:700;line-height:1.3;color:var(--tinta);margin-top:6px}
.kpi .que{grid-area:que;color:var(--suave);font-size:16px;line-height:1.6;margin:0;align-self:center}
.kpi .ctx{grid-area:ctx;align-self:end;margin-top:14px;padding-top:12px;border-top:1px solid var(--borde);font-size:14.5px;line-height:1.45;color:var(--suave)}
.kpi .ctx b{color:var(--tinta)}
.kpi:nth-child(3) .num,.kpi:nth-child(3) .tit{text-align:center}
.kpi .num.doble{display:grid;grid-template-columns:1fr auto 1fr;align-items:start;column-gap:4px;width:100%;max-width:250px}
.kpi .num.doble .v{display:flex;flex-direction:column;align-items:center;text-align:center;line-height:1}
.kpi .num.doble .fl{color:var(--tenue);font-weight:400;font-size:.62em;line-height:1.7;text-align:center}
.kpi .num.doble small{font-family:Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:13px;font-weight:600;letter-spacing:0;line-height:1.3;color:var(--suave);margin-top:8px}
@media (max-width:1080px){.kpi{grid-template-columns:minmax(200px,250px) 1fr;grid-template-areas:"etq que" "num que" "tit ctx" "graf graf"}.kpi .graf{justify-self:start;margin-top:16px}}
@media (max-width:760px){.kpi{grid-template-columns:1fr;grid-template-areas:"etq" "num" "tit" "que" "graf" "ctx";padding:20px 20px}.kpi .que{margin-top:12px}.kpi .graf{max-width:100%;overflow-x:auto}}
.guia{color:var(--suave);font-size:15px;max-width:72ch}
.lectura{max-width:72ch;color:#2b2a26;margin:0 0 16px}
.grafico{max-width:100%;background:var(--papel);border:1px solid var(--borde);border-radius:2px;padding:14px;overflow-x:auto;margin:6px 0 14px;transition:border-color .3s}
.grafico:hover{border-color:#c4c4cc}
.grafico .vega-embed{display:block}
.leyenda{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--suave);margin:4px 0 8px}
.leyenda span{display:inline-flex;align-items:center;gap:6px}
.pt{width:11px;height:11px;border-radius:50%;display:inline-block}
.rombo{width:10px;height:10px;transform:rotate(45deg);display:inline-block;background:var(--tinta)}
.cruz{color:var(--vino);font-weight:800;font-size:15px;line-height:1}
.franja{width:22px;height:11px;background:var(--amarillo);display:inline-block;border-radius:2px}
details.como{background:var(--papel);border:1px solid var(--borde);border-radius:2px;margin:6px 0 10px;transition:border-color .3s}
details.como:hover,details.como[open]{border-color:#c4c4cc}
details.como>summary{cursor:pointer;list-style:none;padding:13px 18px;font-weight:700;display:flex;align-items:center;gap:12px}
details.como>summary::-webkit-details-marker{display:none}
details.como>summary::before{content:"+";flex:none;width:26px;height:22px;background:var(--carbono);color:#fff;display:inline-grid;place-items:center;font-size:16px;line-height:1;clip-path:polygon(5px 0,100% 0,calc(100% - 5px) 100%,0 100%);transition:background-color .25s}
details.como>summary:hover::before,details.como[open]>summary::before{background:var(--rojo)}
details.como[open]>summary::before{content:"–"}
details.como>summary small{font-weight:400;color:var(--tenue)}
details.como .cuerpo{padding:0 20px 16px;font-size:15px;color:#2b2a26}
details.como .cuerpo p{max-width:72ch}
.tabla{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;font-size:13.5px;min-width:100%}
th,td{padding:8px 10px;border-bottom:1px solid #ececf0;text-align:left;vertical-align:top}
th{font-family:var(--display);font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#fff;background:var(--carbono);border-bottom:0}
tbody tr:nth-child(even){background:#f8f8fa}
table.num td:not(:first-child),table.num th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td .n{color:var(--tenue);font-size:12px}
.nota{font-size:14px;color:var(--suave)}
.controles{display:flex;flex-wrap:wrap;gap:12px 26px;align-items:flex-end;background:#f6f6f8;border:0;border-left:3px solid var(--carbono);border-radius:2px;padding:14px 16px;margin:8px 0}
.control{display:flex;flex-direction:column;gap:6px;font-size:13px}
.control>.titulo{font-family:var(--display);font-size:11.5px;text-transform:uppercase;letter-spacing:.12em;color:var(--tenue);font-weight:700}
.vega-bind-name{display:none}
.vega-bind label,.vega-bind{font-size:14px}
.vega-bind select,.control select{font:inherit;font-size:14px;padding:6px 8px;border:1px solid var(--borde);border-radius:2px;background:#fff;max-width:280px}
.vega-bind-radio{display:inline-flex;gap:4px;flex-wrap:wrap}
.vega-bind-radio input{display:none}
.vega-bind-radio label{padding:5px 12px;border:1px solid var(--borde);border-radius:2px;cursor:pointer;background:#fff;transition:background-color .2s,color .2s,border-color .2s}
.vega-bind-radio label:hover{border-color:var(--carbono)}
.vega-bind-radio label:has(input:checked){background:var(--carbono);color:#fff;border-color:var(--carbono);box-shadow:inset 0 -3px 0 var(--rojo)}
.pastillas{display:inline-flex;gap:4px;flex-wrap:wrap}
.pastillas button{font:inherit;font-size:14px;padding:5px 12px;border:1px solid var(--borde);border-radius:2px;background:#fff;cursor:pointer;transition:background-color .2s,color .2s,border-color .2s}
.pastillas button:hover{border-color:var(--carbono)}
.pastillas button.on{background:var(--carbono);color:#fff;border-color:var(--carbono);box-shadow:inset 0 -3px 0 var(--rojo)}
button.volver{font:inherit;font-size:14px;padding:7px 12px;border-radius:2px;border:1px solid var(--carbono);background:#fff;cursor:pointer;transition:background-color .2s,color .2s}
button.volver:hover{background:var(--rojo);border-color:var(--rojo);color:#fff}
.deslizadores{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:10px;margin:10px 0}
.desl{background:var(--papel);border:1px solid var(--borde);border-radius:2px;padding:10px 12px}
.desl .etq{display:block;font-size:13.5px;font-weight:600;margin-bottom:2px}
.desl .rango{display:block;font-size:12px;color:var(--tenue)}
.desl input[type=range]{width:72%;accent-color:var(--rojo);vertical-align:middle}
.desl .vega-bind-value{font-weight:700;font-variant-numeric:tabular-nums;margin-left:8px}
.modelo-usado{font-size:13.5px;color:var(--suave);margin:8px 0 0}
.aviso{background:#fff6e0;border:0;border-left:3px solid #e0a800;color:#6b4e00;border-radius:2px;padding:10px 14px;font-size:14px;margin:8px 0}
.queToca{font-size:14px;color:var(--suave);max-width:none}
.controles.chico{padding:10px 16px}
.eje-e{font-size:13.5px;color:var(--suave);margin:10px 0 4px;display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center}.eje-e .volver{margin-left:auto;padding:5px 11px;font-size:13px}.eje-e strong{color:var(--tinta);font-weight:600}
.cuenta{font-size:15px;margin:4px 0 8px}.cuenta b{color:var(--azul)}
.camino{margin:6px 0 10px;padding-left:22px;font-size:14.5px}.camino li{margin:3px 0}.camino .si{color:var(--azul);font-weight:700}.camino .no{color:#b35a16;font-weight:700}
.titular{font-family:var(--display);font-size:22px;font-weight:700;line-height:1.25;color:var(--azul);margin:10px 0 4px;min-height:1.5em}
.proba{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:16px;align-items:start}.proba>*{min-width:0}
@media (max-width:900px){.proba{grid-template-columns:1fr}}
.col-tit{font-family:var(--display);font-size:12.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;margin:0 0 8px}.col-tit small{font-family:Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-weight:400;letter-spacing:0;text-transform:none;font-size:13px;color:var(--tenue)}
.lista{list-style:none;margin:0;padding:0;background:var(--papel);border:1px solid var(--borde);border-top:3px solid var(--carbono);border-radius:2px;overflow:hidden}
.lista li{border-bottom:1px solid #efeff2}.lista li:last-child{border-bottom:0}
.fila{display:grid;grid-template-columns:18px 34px minmax(0,1fr) auto auto;gap:8px;align-items:center;padding:6px 10px;cursor:pointer;user-select:none;transition:background-color .15s}
.fila:hover{background:#f6f6f8}
.lista li.abierto .fila{background:var(--amarillo)}
.lista li.arrastrando{opacity:.4}.lista li.sobre{box-shadow:inset 0 3px 0 var(--azul)}
.asa{color:#b4b4bc;cursor:grab;font-size:14px;letter-spacing:-2px;touch-action:none}
.pos{font-family:var(--display);font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}
.quien{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}.quien b{font-weight:700}.quien span{color:var(--tenue);font-size:12.5px;margin-left:6px}
.va{font-size:13px;font-weight:700;color:var(--azul);white-space:nowrap;font-variant-numeric:tabular-nums}
.flechas button{border:1px solid var(--borde);background:#fff;border-radius:2px;width:24px;height:22px;cursor:pointer;font-size:11px;padding:0;margin-left:2px;transition:background-color .2s,color .2s}
.flechas button:hover{background:var(--carbono);color:#fff;border-color:var(--carbono)}
.vars{padding:8px 12px 12px;background:#fffbea;border-top:1px dashed #ecd99a;display:grid;gap:8px}
.var label{display:flex;justify-content:space-between;font-size:13px;font-weight:600}
.var label output{font-variant-numeric:tabular-nums}
.var input{width:100%;accent-color:var(--rojo)}
.var .rango{font-size:11.5px;color:var(--tenue)}
.var .fuera{font-size:12px;color:#6b4e00;background:#fff6e0;border-radius:2px;padding:4px 8px;margin-top:3px}
footer{max-width:1040px;margin:0 auto;padding:30px 20px 50px;color:#9d9da5;font-size:13px;background:var(--carbono) var(--logo-f1) no-repeat right 20px top 26px/72px 18px;box-shadow:0 0 0 100vmax var(--carbono);clip-path:inset(0 -100vmax)}
footer::before{content:"";display:block;width:28px;height:6px;background:var(--rojo);transform:skewX(-28deg);margin:0 0 18px}
footer code{background:var(--carbono2);color:#e6e6ea}
footer a{color:#fff}
@media (max-width:760px){.kpis{grid-template-columns:1fr}nav .grupo{display:none}section{padding-top:46px}#portada{padding-top:58px}#portada::before,#portada::after{top:20px}.barra .dentro{gap:6px;padding:8px 16px 6px}nav{flex-wrap:nowrap;overflow-x:auto;width:calc(100% + 32px);margin:0 -16px;padding:0 16px 2px;scrollbar-width:none;-webkit-mask-image:linear-gradient(90deg,#000 82%,transparent);mask-image:linear-gradient(90deg,#000 82%,transparent)}nav::-webkit-scrollbar{display:none}nav a{flex:none;white-space:nowrap}#datos::before,#hallazgo::before,#modelo::before{top:24px;font-size:84px}}
@keyframes aparecer{from{opacity:0}to{opacity:1}}
@keyframes subir{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
@keyframes barrido{from{transform:skewX(-28deg) scaleX(0)}to{transform:skewX(-28deg) scaleX(1)}}
@keyframes arranque{from{opacity:0;transform:translateX(-80px)}to{opacity:1;transform:none}}
@keyframes luces{to{width:100px}}
@keyframes apagar{to{opacity:0}}
@keyframes abrir{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:none}}
@keyframes numero{from{opacity:0;transform:translateX(40px)}to{opacity:1;transform:none}}
@keyframes progreso{from{transform:scaleX(0)}to{transform:scaleX(1)}}
#portada::after{animation:luces 2s steps(5,end) .3s forwards,apagar .15s ease-out 2.9s forwards}
#portada>*{animation:aparecer .9s var(--salida) both}
#portada>:nth-child(2){animation-delay:.12s}#portada>:nth-child(3){animation-delay:.24s}#portada>:nth-child(4){animation-delay:.36s}#portada>:nth-child(5){animation-delay:.48s}
@supports (animation-timeline:view()){
main section:not(#portada)>*{animation:aparecer var(--salida) both;animation-timeline:view();animation-range:entry 0% cover 20%}
}
@supports (animation-timeline:scroll()){
header.barra::after{animation:progreso linear both;animation-timeline:scroll(root)}
}
@media (prefers-reduced-motion:no-preference){
#portada>*{animation-name:subir}
.pista svg>g{animation:arranque 1.3s var(--salida) .9s both}.pista svg>g+g{animation-delay:1.05s}
details.como[open]>.cuerpo{animation:abrir .35s var(--salida) both}
@supports (animation-timeline:view()){
main section:not(#portada)>*{animation-name:subir}
#datos::before,#hallazgo::before,#modelo::before{animation:numero var(--salida) both;animation-timeline:view();animation-range:entry 0% cover 30%}
main section:not(#portada)>.parte::before{animation:barrido var(--salida) both;animation-timeline:view();animation-range:entry 10% cover 30%}
}
}
</style>
</head>
<body>
<header class="barra"><div class="dentro">
  <div class="marca"><span class="bandera" aria-hidden="true"></span>¿Dónde termina un piloto?</div>
  <nav aria-label="Secciones">
    <span class="grupo">El recorrido</span>
    <a href="#portada">Inicio</a><a href="#datos">Los datos</a><a href="#hallazgo">El hallazgo</a><a href="#modelo">El modelo</a>
    <span class="grupo">En tus manos</span>
    <a href="#explora">Explorá</a><a href="#proba">Probá el modelo</a>
  </nav>
</div></header>
<main>

<section id="portada">
  <span class="parte">Proyecto integrador · Ciencia de Datos · UTN FRM 2026 · Grupo 8 (5K09)</span>
  <h1>¿En qué puesto va a terminar cada piloto en una carrera de Fórmula 1?</h1>
  <p class="idea">__IDEA__</p>
  <figure class="pista"><figcaption>El modelo predice la llegada desde cuatro momentos de la carrera. En cada uno sólo usa lo que ya pasó, a la izquierda de la línea; debajo, los datos que mira.</figcaption><div class="pista-svg"><svg viewBox="0 0 760 200" role="img" aria-label="Los cuatro momentos en que el modelo predice la llegada: antes de largar, al 25, al 50 y al 75 % de la carrera" font-size="12.5"><rect x="24" y="68" width="668" height="44" rx="6" fill="#ecebe5" stroke="var(--borde)"/><line x1="32" x2="684" y1="90" y2="90" stroke="#d6d3ca" stroke-dasharray="10 8"/><g><rect x="52" y="76" width="34" height="8" rx="3" fill="var(--rojo)"/><path d="M86 77L96 80L86 83z" fill="var(--rojo)"/><rect x="49" y="73" width="4" height="14" rx="1" fill="var(--tinta)"/><circle cx="59" cy="74" r="3.5" fill="var(--tinta)"/><circle cx="59" cy="86" r="3.5" fill="var(--tinta)"/><circle cx="81" cy="74" r="3.5" fill="var(--tinta)"/><circle cx="81" cy="86" r="3.5" fill="var(--tinta)"/></g><g><rect x="80" y="96" width="34" height="8" rx="3" fill="var(--rojo)"/><path d="M114 97L124 100L114 103z" fill="var(--rojo)"/><rect x="77" y="93" width="4" height="14" rx="1" fill="var(--tinta)"/><circle cx="87" cy="94" r="3.5" fill="var(--tinta)"/><circle cx="87" cy="106" r="3.5" fill="var(--tinta)"/><circle cx="109" cy="94" r="3.5" fill="var(--tinta)"/><circle cx="109" cy="106" r="3.5" fill="var(--tinta)"/></g><line x1="136" x2="136" y1="40" y2="136" stroke="var(--azul)" stroke-width="2" stroke-dasharray="5 4"/><text x="136" y="30" text-anchor="middle" font-size="14" font-weight="600" fill="var(--tinta)">Antes de largar</text><text x="136" y="156" text-anchor="middle" fill="var(--suave)">grilla y</text><text x="136" y="173" text-anchor="middle" fill="var(--suave)">campeonato</text><line x1="276" x2="276" y1="40" y2="136" stroke="var(--azul)" stroke-width="2" stroke-dasharray="5 4"/><text x="276" y="30" text-anchor="middle" font-size="14" font-weight="600" fill="var(--tinta)">25 %</text><text x="276" y="156" text-anchor="middle" fill="var(--suave)">posición, vueltas,</text><text x="276" y="173" text-anchor="middle" fill="var(--suave)">qualy y equipo</text><line x1="416" x2="416" y1="40" y2="136" stroke="var(--azul)" stroke-width="2" stroke-dasharray="5 4"/><text x="416" y="30" text-anchor="middle" font-size="14" font-weight="600" fill="var(--tinta)">50 %</text><text x="416" y="156" text-anchor="middle" fill="var(--suave)">posición, vueltas</text><text x="416" y="173" text-anchor="middle" fill="var(--suave)">y neumáticos</text><line x1="556" x2="556" y1="40" y2="136" stroke="var(--azul)" stroke-width="2" stroke-dasharray="5 4"/><text x="556" y="30" text-anchor="middle" font-size="14" font-weight="600" fill="var(--tinta)">75 %</text><text x="556" y="156" text-anchor="middle" fill="var(--suave)">posición y</text><text x="556" y="173" text-anchor="middle" fill="var(--suave)">neumáticos</text><line x1="684" x2="684" y1="36" y2="112" stroke="var(--tinta)" stroke-width="2"/><rect x="686" y="38" width="8" height="8" fill="var(--tinta)" stroke="var(--tinta)" stroke-width=".5"/><rect x="694" y="38" width="8" height="8" fill="#fff" stroke="var(--tinta)" stroke-width=".5"/><rect x="702" y="38" width="8" height="8" fill="var(--tinta)" stroke="var(--tinta)" stroke-width=".5"/><rect x="710" y="38" width="8" height="8" fill="#fff" stroke="var(--tinta)" stroke-width=".5"/><rect x="686" y="46" width="8" height="8" fill="#fff" stroke="var(--tinta)" stroke-width=".5"/><rect x="694" y="46" width="8" height="8" fill="var(--tinta)" stroke="var(--tinta)" stroke-width=".5"/><rect x="702" y="46" width="8" height="8" fill="#fff" stroke="var(--tinta)" stroke-width=".5"/><rect x="710" y="46" width="8" height="8" fill="var(--tinta)" stroke="var(--tinta)" stroke-width=".5"/><rect x="686" y="54" width="8" height="8" fill="var(--tinta)" stroke="var(--tinta)" stroke-width=".5"/><rect x="694" y="54" width="8" height="8" fill="#fff" stroke="var(--tinta)" stroke-width=".5"/><rect x="702" y="54" width="8" height="8" fill="var(--tinta)" stroke="var(--tinta)" stroke-width=".5"/><rect x="710" y="54" width="8" height="8" fill="#fff" stroke="var(--tinta)" stroke-width=".5"/><text x="684" y="156" text-anchor="middle" font-weight="600" fill="var(--tinta)">Llegada:</text><text x="684" y="173" text-anchor="middle" fill="var(--tinta)">lo que predice</text></svg></div></figure>
  <div class="kpis">__KPIS__</div>
</section>

<section id="datos">
  <span class="parte">1 · Los datos</span>
  <h2>__T_DATOS__</h2>
  <p class="lectura">__L_DATOS__</p>
  <div class="grafico" id="g-datos"></div>
  <details class="como"><summary>Cómo lo sabemos <small>de dónde salen los datos, qué filtramos y qué dejamos afuera</small></summary><div class="cuerpo">__C_DATOS__</div></details>
</section>

<section id="hallazgo">
  <span class="parte">2 · Lo que encontramos</span>
  <h2>__T_HALLAZGO__</h2>
  <p class="lectura">__L_HALLAZGO__</p>
  <div class="grafico" id="g-hallazgo"></div>
  <details class="como"><summary>Cómo lo sabemos <small>una hipótesis que se sostuvo, una que no y cómo se midió</small></summary><div class="cuerpo">__C_HALLAZGO__</div></details>
</section>

<section id="modelo">
  <span class="parte">3 · El modelo</span>
  <h2>__T_MODELO__</h2>
  <p class="lectura">__L_MODELO__</p>
  <div class="grafico" id="g-modelo"></div>
  <p class="lectura">__L_TABLA__</p>
  <details class="como"><summary>Cómo lo sabemos <small>métrica, validación, comparación de modelos, test, errores y en qué se apoya</small></summary><div class="cuerpo">__C_MODELO__</div></details>
  <h3>La tabla completa de cada carrera</h3>
  <p class="lectura">El MAE mide al modelo piloto por piloto. Pero lo que se predice, en el fondo, es una tabla de posiciones, así que también
  la comparamos entera con la tabla final, carrera por carrera. Para cada carrera se ordena a los pilotos según la predicción
  (si dos empatan, adelante el que va adelante) y se calcula:</p>
  <p class="lectura"><strong>Spearman</strong>: la correlación entre el puesto predicho y el puesto final de cada piloto de esa carrera. Compara
  órdenes, no distancias: da 1 si las dos tablas son idénticas y 0 si no tienen nada que ver. <strong>Podio</strong>: de los 3
  primeros de la tabla predicha, qué parte terminó en el podio. <strong>Top 10</strong>: de los 10 primeros predichos, qué
  parte sumó puntos. Después se promedia sobre las carreras. La regla «termina donde está» también arma una tabla (la grilla, o
  el orden en ese momento), y se mide igual. Estos números no cambiaron ninguna decisión: el modelo ya estaba elegido con el MAE.</p>
  <div class="grafico" id="g-tabla"></div>
  <details class="como"><summary>Cómo lo sabemos <small>los números de la validación y del test, carrera por carrera</small></summary><div class="cuerpo">__C_TABLA__</div></details>
</section>

<section id="explora">
  <span class="parte">En tus manos · Explorá</span>
  <h2>Recorré las __CARRERAS_EXPLORA__ carreras: qué dijo el modelo de cada piloto y qué pasó de verdad</h2>
  <p class="lectura">Arriba, cada punto es una carrera. Elegí el momento de la carrera (25 % quiere decir que ya se corrió
  un cuarto de la carrera) y, si querés, una temporada o un piloto. Hacé clic en una carrera para abrirla abajo y comparar el orden que predijo el modelo con cómo terminó. Pasá el
  mouse por los puntos y las líneas para ver el detalle de cada piloto. Todas estas predicciones las hizo el modelo sin
  conocer el resultado de esa carrera.</p>
  <div class="controles">
    <div class="control"><span class="titulo">Momento de la carrera</span><div id="c-momento"></div></div>
    <div class="control"><span class="titulo">Temporada</span><div id="c-temporada"></div></div>
    <div class="control"><span class="titulo">Seguir a un piloto</span><div id="c-piloto"></div></div>
    <div class="control"><span class="titulo">Medir cada carrera por</span><div id="c-medida"></div></div>
  </div>
  <p class="eje-e"><strong id="e-eje">Eje vertical: puestos que se equivoca, en promedio (más abajo es mejor).</strong>
  <span>Rueda del mouse o pellizco: acercar y alejar · arrastrar: moverse · doble clic: volver a ver todo.</span>
  <button type="button" class="volver" id="e-restablecer">Restablecer la vista</button></p>
  <div class="grafico" id="g-explora"></div>
  <div class="controles chico">
    <div class="control"><span class="titulo">Ver la carrera elegida</span><div class="pastillas" id="e-vista">
      <button data-v="tablas" class="on">Las dos tablas</button><button data-v="det">Piloto por piloto</button></div></div>
  </div>
  <div class="vista-e" data-v="tablas">
    <p class="nota">A la izquierda, el orden que predijo el modelo; a la derecha, cómo terminó la carrera. Cada línea une a un
    piloto: si queda horizontal, el modelo acertó su puesto. Las líneas punteadas son pilotos que abandonaron.</p>
    <div class="grafico" id="g-explora-tablas"></div></div>
  <div class="vista-e" data-v="det" hidden>
    <div class="leyenda">
      <span><i class="pt" style="background:var(--naranja)"></i>dónde iba en ese momento (lo que dice «termina donde está»)</span>
      <span><i class="pt" style="background:var(--azul)"></i>dónde dijo el modelo que terminaba</span>
      <span><i class="rombo"></i>dónde terminó</span>
      <span><b class="cruz">✕</b>abandonó</span>
      <span><i class="franja"></i>el piloto que seguís</span>
    </div>
    <div class="grafico" id="g-explora-det"></div></div>
  <details class="como"><summary>Cómo lo sabemos <small>qué predicciones son, cómo se mide cada carrera y qué toca cada control</small></summary><div class="cuerpo">__C_EXPLORA__</div></details>
</section>

<section id="proba">
  <span class="parte">En tus manos · Probá el modelo</span>
  <h2>Armá tu carrera: ordená a los pilotos y mirá dónde los hace terminar el modelo</h2>
  <p class="lectura">Elegí una carrera de 2026 y un momento (25 % quiere decir que ya se corrió un cuarto de la carrera). A la izquierda está el orden de la carrera en ese momento:
  <strong>arrastrá a los pilotos</strong> (o usá las flechas) para cambiarlo, y <strong>hacé clic en un piloto</strong> para
  cambiar sus datos. Los datos que se pueden cambiar dependen del momento, porque en cada uno el modelo usa los que le
  sirven: por ejemplo, al 25 % cuentan los puntos de su equipo, y al 50 % y al 75 %, qué tan gastados tiene los neumáticos. A la derecha, el
  modelo te muestra en qué puesto terminaría cada uno. Si probás algo que nunca pasó en las carreras de las que aprendió el
  modelo, la página te avisa.</p>
  <div class="controles">
    <div class="control"><span class="titulo">Momento de la carrera</span><div class="pastillas" id="p-momento">
      <button data-k="0">Antes de largar</button><button data-k="25" class="on">25 %</button><button data-k="50">50 %</button><button data-k="75">75 %</button></div></div>
    <div class="control"><span class="titulo">Carrera de 2026</span><select id="p-carrera"></select></div>
    <div class="control"><button type="button" class="volver" id="p-volver">Volver a la carrera real</button></div>
  </div>
  <p class="modelo-usado" id="p-modelo"></p>
  <p class="titular" id="p-titular"></p>
  <p class="nota" id="p-nota"></p>
  <div class="proba">
    <div class="col-lista"><div class="col-tit">Cómo va la carrera ahora <small id="p-ayuda">arrastrá para cambiar el orden</small><br><small>→ dónde termina según el modelo · ▲ gana lugares · ▼ pierde lugares</small></div>
      <ol class="lista" id="p-lista"></ol></div>
    <div class="col-res"><div class="col-tit">Cómo terminaría, según el modelo</div>
      <div class="grafico" id="g-proba"></div></div>
  </div>
  <p class="aviso" id="p-aviso" hidden></p>
  <details class="como"><summary>Ver más <small>por qué el modelo pone al piloto elegido en ese lugar</small></summary>
    <div class="cuerpo"><p class="cuenta" id="p-apo-cuenta"></p><div class="grafico" id="g-proba-apo-caja"><div id="g-proba-apo"></div></div>
    <ol class="camino" id="p-camino" hidden></ol><p class="nota" id="p-apo-nota"></p></div></details>
  <details class="como"><summary>Cómo lo sabemos <small>la fórmula, el modelo que se usa antes de largar y las reglas de esta sección</small></summary><div class="cuerpo">__C_PROBA__</div></details>
</section>
</main>
<aside class="panel-e" id="panel-e" hidden aria-live="polite"><button type="button" class="cerrar" id="panel-e-cerrar" aria-label="Cerrar el panel">×</button><h3 id="panel-e-tit"></h3><p class="sub" id="panel-e-sub"></p><dl id="panel-e-dl"></dl></aside>
<footer>
  Grupo 8 · 5K09 · Luconi Martina, Olea Facundo, Miranda Mauricio, Mezzabota Tomás, De Coninck Ramiro, Acosta Victoria.<br>
  Datos: corrida del pipeline del 23/09/2026 (Jolpica-F1, FastF1, OpenF1). Página generada con <code>app/construir_app.py</code>
  el __FECHA__ · <a href="https://github.com/Maaurimiranda/f1-predictpositions">repositorio</a>.
</footer>
<script>
const SPECS = __SPECS__;
const PR = __PROBA__;
const OPC = {actions: false, renderer: "svg",
  formatLocale: {decimal: ",", thousands: ".", grouping: [3], currency: ["$", ""]}};
const vistas = {};
// Explorá: al pasar el mouse, un cartel corto; al hacer clic, todos los datos en un panel a la derecha
const escH = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const TIP = document.createElement("div"); TIP.className = "tip-corto"; TIP.hidden = true; document.body.appendChild(TIP);
const CORTO = ["Carrera", "El modelo se equivoca (puestos)", "Piloto", "Dónde terminó"];
function tipCorto(handler, ev, item, value) {
  if (value == null || value === "" || !ev) { TIP.hidden = true; return; }
  const o = typeof value === "object" ? value : {"": value};
  const ks = CORTO.filter(k => k in o);
  TIP.innerHTML = (ks.length ? ks : Object.keys(o).slice(0, 2)).slice(0, 2)
    .map(k => `<div>${k ? `<span>${escH(k)}:</span> ` : ""}<b>${escH(o[k])}</b></div>`).join("")
    + '<div class="clic">Hacé clic para ver todos los datos</div>';
  TIP.hidden = false;
  const w = TIP.offsetWidth, h = TIP.offsetHeight;
  TIP.style.left = Math.max(8, Math.min(ev.clientX + 14, innerWidth - w - 8)) + "px";
  TIP.style.top = Math.max(8, Math.min(ev.clientY + 14, innerHeight - h - 8)) + "px";
}
const OPC_E = {...OPC, tooltip: tipCorto};
const PANEL = document.getElementById("panel-e");
function abrirPanel(o) {
  const tit = o["Carrera"] || o["Piloto"] || "Detalle";
  document.getElementById("panel-e-tit").textContent = tit;
  document.getElementById("panel-e-sub").textContent = o["Carrera"] ? [o["Fecha"]].filter(Boolean).join(" · ")
    : [o["Equipo"], o["Momento"]].filter(Boolean).join(" · ");
  const fuera = new Set(["Carrera", "Piloto", "Fecha", "Equipo", "Momento"]);
  document.getElementById("panel-e-dl").innerHTML = Object.entries(o).filter(([k]) => !fuera.has(k))
    .map(([k, v]) => `<dt>${escH(k)}</dt><dd>${escH(v)}</dd>`).join("");
  TIP.hidden = true; PANEL.hidden = false;
}
const cerrarPanel = () => { PANEL.hidden = true; };
document.getElementById("panel-e-cerrar").addEventListener("click", cerrarPanel);
document.addEventListener("keydown", e => { if (e.key === "Escape") cerrarPanel(); });
const clicPanel = v => v.addEventListener("click", (ev, item) => {
  const t = item && item.tooltip; if (t && typeof t === "object") abrirPanel(t); });
const listo = Promise.all(Object.entries(SPECS).map(([id, spec]) =>
  vegaEmbed("#" + id, spec, id.startsWith("g-explora") ? OPC_E : OPC).then(r => { vistas[id] = r.view; })));
listo.then(() => ["g-explora-tablas", "g-explora-det"].forEach(id => clicPanel(vistas[id])));

// Explorá: el panorama le pasa momento, piloto y carrera elegida a las dos vistas del detalle
const EJE = x => { document.getElementById("e-eje").textContent = x === "Cuántos puestos se equivoca"
  ? "Eje vertical: puestos que se equivoca, en promedio (más abajo es mejor)."
  : "Eje vertical: qué tan parecida es la tabla predicha a la final (1 = idénticas; más arriba es mejor)."; };
function conectarExplora() {
  const pv = vistas["g-explora"], subs = [vistas["g-explora-tablas"], vistas["g-explora-det"]];
  for (const nm of ["momento_e", "piloto_e"]) {
    pv.addSignalListener(nm, (_, x) => subs.forEach(v => v.signal(nm, x).runAsync()));
    subs.forEach(v => v.signal(nm, pv.signal(nm)));
  }
  clicPanel(pv);
  pv.addSignalListener("medida_e", (_, x) => EJE(x));
  EJE(pv.signal("medida_e"));
  const pasar = v => { if (v && v.length) subs.forEach(s => s.signal("carrera_c", v[0].values[0]).runAsync()); };
  pv.addDataListener("carrera_e_store", (_, v) => pasar(v));
  pasar(pv.data("carrera_e_store"));
  subs.forEach(v => v.runAsync());
}
listo.then(conectarExplora);
// «Restablecer la vista»: vuelve a dibujar el panorama como al abrir la página (sin zoom, filtros ni carrera elegida)
document.getElementById("e-restablecer").addEventListener("click", async () => {
  vistas["g-explora"].finalize();
  ["c-momento", "c-temporada", "c-piloto", "c-medida"].forEach(id => { document.getElementById(id).innerHTML = ""; });
  vistas["g-explora"] = (await vegaEmbed("#g-explora", SPECS["g-explora"], OPC_E)).view;
  conectarExplora();
  document.querySelector('#e-vista button[data-v="tablas"]').click();
});
document.querySelectorAll("#e-vista button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#e-vista button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".vista-e").forEach(d => { d.hidden = d.dataset.v !== b.dataset.v; });
}));

// Probá el modelo. La fórmula de cada momento viaja en PR: los coeficientes de la regresión lineal (con la media y el
// desvío del escalado) o, antes de largar, el árbol. La posición de cada piloto es su lugar en la lista.
const num = (x, d = 1) => x.toFixed(d).replace(".", ",").replace(/,0+$/, "");
const est = {k: "25", c: PR.inicial, orden: [], vals: {}, reales: {}, abierto: null};
const selC = document.getElementById("p-carrera"), lista = document.getElementById("p-lista");
for (const [c, nombre] of Object.entries(PR.carreras)) selC.add(new Option(nombre, c));
selC.value = est.c;

function predecir(k, orden, vals) {
  const M = PR.momentos[k];
  const res = orden.map((p, i) => {
    const x = Object.assign({}, vals[p], {posicion_al_corte: i + 1});
    let cambio, partes = [];
    if (M.lineal) {
      cambio = M.lineal.intercepto;
      partes.push(["Ajuste general", M.lineal.intercepto]);
      for (const [c, w, mu, sd] of M.lineal.partes) {
        const a = w * (x[c] - mu) / sd; cambio += a;
        const v = pantalla(k, c, x[c]), d = M.rangos[c][2] < 1 ? 1 : 0;
        partes.push([`${M.etq[c].replace(/ \([^)]*\)$/, "")}: ${num(v, d)}`, a]);
      }
    } else {
      let n = M.arbol;
      while (n.v === undefined) {
        const si = x[n.c] <= n.u;
        partes.push([n.c, n.u, si, x[n.c]]);
        n = si ? n.l : n.r;
      }
      cambio = n.v;
    }
    return {p, pos: i + 1, esp: i + 1 + cambio, partes};
  });
  [...res].sort((a, b) => a.esp - b.esp || a.pos - b.pos).forEach((r, i) => { r.final = i + 1; });
  return res;
}
window.predecir = predecir;

const pantalla = (k, c, v) => PR.momentos[k].log.includes(c) ? Math.expm1(v) : v;
const modelo = (k, c, v) => PR.momentos[k].log.includes(c) ? Math.log1p(v) : v;

function cargar() {
  nota("");
  const filas = PR.momentos[est.k].carreras[est.c];
  est.orden = filas.map(f => f.p);
  est.vals = Object.fromEntries(filas.map(f => [f.p, Object.assign({}, f.x)]));
  est.reales = Object.fromEntries(filas.map(f => [f.p, f]));
  if (!est.reales[est.abierto]) est.abierto = null;
  const M = PR.momentos[est.k];
  document.getElementById("p-modelo").textContent = M.tipo === "árbol"
    ? "Antes de la largada, el modelo responde preguntas de sí o no sobre cada piloto (por ejemplo, «¿larga entre los 8 primeros?»). Es una versión simplificada del modelo principal, que da casi lo mismo y se puede calcular en esta página."
    : "El modelo le suma o le resta lugares a cada piloto según sus datos y después los ordena del primero al último.";
  document.getElementById("p-ayuda").textContent = est.k === "0" ? "arrastrá para cambiar la grilla de largada" : "arrastrá para cambiar el orden";
  dibujarLista(); actualizar();
}

function dibujarLista() {
  const M = PR.momentos[est.k], filas = est.reales;
  lista.innerHTML = "";
  est.orden.forEach((p, i) => {
    const f = filas[p], li = document.createElement("li");
    li.dataset.p = p;
    if (p === est.abierto) li.classList.add("abierto");
    li.innerHTML = `<div class="fila"><span class="asa" aria-hidden="true">⋮⋮</span><span class="pos">P${i + 1}</span>
      <span class="quien"><b>${p}</b><span>${f.nombre} · ${f.equipo}</span></span><span class="va" data-va></span>
      <span class="flechas"><button type="button" data-m="-1" aria-label="Subir">▲</button><button type="button" data-m="1" aria-label="Bajar">▼</button></span></div>`;
    if (p === est.abierto) li.appendChild(editor(p));
    lista.appendChild(li);
  });
}

function editor(p) {
  const M = PR.momentos[est.k], caja = document.createElement("div");
  caja.className = "vars";
  const cols = M.cols.filter(c => c !== "posicion_al_corte" && c !== "driver_standing_before");
  if (M.cols.includes("driver_standing_before")) {
    const info = document.createElement("div");
    info.className = "var";
    info.innerHTML = `<label>Puesto en el campeonato<output>${Math.round(est.vals[p].driver_standing_before)}.º</output></label>` +
      `<span class="rango">Sale de los puntos: si los cambiás, se reordena el campeonato de todos.</span>`;
    caja.appendChild(info);
  }
  if (!cols.length) caja.innerHTML = "<span class='rango'>En este momento el modelo sólo mira el lugar en el que va: cambialo arrastrando.</span>";
  for (const c of cols) {
    const [lo, hi, paso] = M.rangos[c], real = pantalla(est.k, c, est.reales[p].x[c]);
    const v = Math.min(hi, Math.max(lo, pantalla(est.k, c, est.vals[p][c])));
    const d = paso < 1 ? 1 : 0, div = document.createElement("div");
    div.className = "var";
    div.innerHTML = `<label>${M.etq[c]}<output>${num(v, d)}</output></label>
      <input type="range" min="${lo}" max="${hi}" step="${paso}" value="${v}"><span class="rango">valor real: ${num(real, d)} · se puede mover entre ${num(lo, d)} y ${num(hi, d)}</span>` +
      (real < lo || real > hi ? `<div class="fuera">⚠ El valor real (${num(real, d)}) es más extremo que lo que vio el modelo cuando aprendió; el control llega hasta ${num(real < lo ? lo : hi, d)}.</div>` : "");
    const inp = div.querySelector("input"), out = div.querySelector("output");
    inp.addEventListener("input", () => { out.textContent = num(+inp.value, d); cambiar(p, c, modelo(est.k, c, +inp.value)); });
    inp.addEventListener("change", () => { if (c === "vueltas_detras" || c === "log_puntos") { dibujarLista(); actualizar(); } });
    inp.addEventListener("click", e => e.stopPropagation());
    caja.appendChild(div);
  }
  caja.addEventListener("click", e => e.stopPropagation());
  return caja;
}

function actualizar() {
  const M = PR.momentos[est.k], res = predecir(est.k, est.orden, est.vals), por = {};
  res.forEach(r => { por[r.p] = r; });
  lista.querySelectorAll("li").forEach(li => {
    const r = por[li.dataset.p], d = r.final - r.pos;
    li.querySelector("[data-va]").textContent = M.lineal
      ? `→ P${r.final}` + (d < 0 ? ` ▲${-d}` : d > 0 ? ` ▼${d}` : "") : `→ ~${Math.round(r.esp)}`;
  });
  const filas = res.map(r => ({piloto: r.p, nombre: est.reales[r.p].nombre, equipo: est.reales[r.p].equipo,
    pos: r.pos, final: r.final, esp: Math.round(r.esp * 100) / 100, tuyo: r.p === est.abierto, lineal: !!M.lineal}));
  const v = vistas["g-proba"];
  if (v) v.change("tabla", vega.changeset().remove(() => true).insert(filas)).runAsync();
  const t = document.getElementById("p-titular");
  if (est.abierto) {
    const r = por[est.abierto];
    t.textContent = M.lineal
      ? `${r.p} va ${r.pos}.º y, según el modelo, termina ${r.final}.º`
      : `${r.p} larga ${r.pos}.º y, según el modelo, termina cerca del ${Math.round(r.esp)}.º puesto`;
  } else t.textContent = "Hacé clic en un piloto para ver y cambiar sus datos.";
  verMas(M, por);
  const aviso = document.getElementById("p-aviso"), n = est.orden.length;
  aviso.hidden = n <= M.max_pos;
  aviso.textContent = `⚠ En 2026 hay ${n} autos, pero el modelo aprendió con carreras de ${M.max_pos}: para los puestos ${M.max_pos + 1}${n - M.max_pos === 2 ? " y " : " a "}${n} su predicción es menos confiable.`;
}

// Reglas para que la carrera siga siendo posible:
// · la posición se ordena primero por vueltas completadas: nadie puede ir adelante de alguien con menos vueltas detrás;
// · moverse en la lista son puestos ganados o perdidos ahora, y eso entra en «puestos ganados en las últimas 10 vueltas»;
// · los puntos del equipo son del equipo: cambian para los dos pilotos;
// · el puesto en el campeonato sale de ordenar a todos por sus puntos.
const tiene = c => PR.momentos[est.k].cols.includes(c);
const nota = t => { document.getElementById("p-nota").textContent = t || ""; };

function reordenar(nuevo) {
  const antes = est.orden;
  if (tiene("tendencia_10")) nuevo.forEach((q, i) => { est.vals[q].tendencia_10 += antes.indexOf(q) - i; });
  est.orden = nuevo;
}

// «Ver más»: cómo llega el modelo al número del piloto elegido
function verMas(M, por) {
  const r = est.abierto ? por[est.abierto] : null, a = vistas["g-proba-apo"];
  const cuenta = document.getElementById("p-apo-cuenta"), caja = document.getElementById("g-proba-apo-caja");
  const camino = document.getElementById("p-camino"), notaV = document.getElementById("p-apo-nota");
  const filas = r && M.lineal ? r.partes.map(([etq, v]) => ({etq, v: Math.round(v * 1000) / 1000})) : [];
  if (a) a.change("aportes", vega.changeset().remove(() => true).insert(filas)).runAsync();
  caja.hidden = !(r && M.lineal); camino.hidden = !(r && !M.lineal);
  if (!r) { cuenta.innerHTML = ""; notaV.textContent = "Hacé clic en un piloto de la lista para ver por qué el modelo lo pone ahí."; return; }
  const suma = r.esp - r.pos, mueve = Math.abs(suma) < 0.05 ? "no lo mueve" : (suma < 0 ? "lo adelanta " : "lo atrasa ") + num(Math.abs(suma), 1) + " lugares";
  if (M.lineal) {
    cuenta.innerHTML = `<strong>${r.p}</strong> va ${r.pos}.º. Sumando lo que aporta cada dato, el modelo ${mueve}: le calcula un puesto ` +
      `de <b>${num(r.esp, 1)}</b>. Después ordena a todos los pilotos según ese número, y ${r.p} queda <b>${r.final}.º</b>.`;
    notaV.textContent = "Cada barra muestra cuántos lugares lo adelanta o lo atrasa ese dato, comparado con un piloto típico de " +
      "las carreras de 2022 a 2025. «Ajuste general» es una corrección fija que el modelo le hace a todos.";
  } else {
    cuenta.innerHTML = `<strong>${r.p}</strong> larga ${r.pos}.º. Siguiendo las preguntas, el modelo ${mueve}: termina cerca del ` +
      `<b>${Math.round(r.esp)}.º</b> puesto.`;
    const pregunta = ([c, u, si, v]) => {
      const ent = Math.floor(pantalla(est.k, c, u)), tiene = num(pantalla(est.k, c, v), c === "log_puntos" ? 1 : 0);
      const q = c === "posicion_al_corte" ? `¿Larga ${ent}.º o más adelante?`
        : c === "driver_standing_before" ? `¿Está ${ent}.º o mejor en el campeonato?`
        : `¿Tiene ${ent} puntos o menos en el campeonato?`;
      const dato = c === "posicion_al_corte" ? `larga ${tiene}.º` : c === "driver_standing_before" ? `está ${tiene}.º` : `tiene ${tiene}`;
      return `${q} <span class="${si ? "si" : "no"}">${si ? "Sí" : "No"}</span> (${dato})`;
    };
    camino.innerHTML = r.partes.map(q => `<li>${pregunta(q)}</li>`).join("") +
      `<li>Resultado: ${suma <= -0.05 ? "gana " + num(-suma, 1) + " lugares" : suma >= 0.05 ? "pierde " + num(suma, 1) + " lugares" : "termina en el mismo lugar en el que larga"}.</li>`;
    notaV.textContent = "Antes de la largada, el modelo de esta página es un árbol de preguntas: va respondiendo sí o no sobre el " +
      "piloto, y cada camino termina en cuántos lugares gana o pierde. Estas son las preguntas que respondió este piloto.";
  }
}

function mover(p, destino) {
  const i = est.orden.indexOf(p);
  if (destino < 0 || destino >= est.orden.length || destino === i) return;
  const nuevo = est.orden.filter(q => q !== p);
  nuevo.splice(destino, 0, p);
  reordenar(nuevo);
  let t = "";
  if (tiene("vueltas_detras")) {
    const vd = q => est.vals[q].vueltas_detras, arriba = nuevo[destino - 1], abajo = nuevo[destino + 1];
    const lo = arriba ? vd(arriba) : 0, hi = abajo ? vd(abajo) : Infinity, v = vd(p);
    if (v < lo || v > hi) {
      est.vals[p].vueltas_detras = Math.min(hi, Math.max(lo, v));
      t = est.vals[p].vueltas_detras
        ? `Para que ${p} pueda ir ${destino + 1}.º, el líder le lleva ahora ${est.vals[p].vueltas_detras} vuelta(s), como a los que tiene alrededor: nadie puede ir adelante de un auto que dio más vueltas. `
        : `Para que ${p} pueda ir ${destino + 1}.º, lo pusimos en la misma vuelta que el líder: nadie puede ir adelante de un auto que dio más vueltas. `;
    }
  }
  if (tiene("tendencia_10")) t += `Los lugares que ganó o perdió cada piloto con el cambio cuentan como lugares ganados o perdidos en las últimas vueltas.`;
  nota(t);
  dibujarLista(); actualizar();
}

function cambiar(p, c, v) {
  if (c === "log_puntos_equipo") {           // es del equipo: los dos pilotos
    const eq = est.reales[p].equipo;
    est.orden.forEach(q => { if (est.reales[q].equipo === eq) est.vals[q][c] = v; });
    nota(`Los puntos son del equipo: cambiaron para los dos pilotos de ${eq}.`);
  } else est.vals[p][c] = v;
  if (c === "log_puntos") {                  // el campeonato se reordena con los puntos
    const orden = [...est.orden].sort((a, b) => est.vals[b].log_puntos - est.vals[a].log_puntos);
    orden.forEach(q => { est.vals[q].driver_standing_before = 1 + orden.findIndex(r => est.vals[r].log_puntos === est.vals[q].log_puntos); });
    nota(`Con esos puntos, ${p} queda ${est.vals[p].driver_standing_before}.º en el campeonato.`);
  }
  if (c === "vueltas_detras") {              // lo ubica donde puede ir con esas vueltas
    const otros = est.orden.filter(q => q !== p), vd = q => est.vals[q].vueltas_detras;
    const a = otros.filter(q => vd(q) < v).length, b = otros.filter(q => vd(q) <= v).length;
    const i = est.orden.indexOf(p), j = Math.min(b, Math.max(a, i));
    if (j !== i) {
      otros.splice(j, 0, p); reordenar(otros);
      nota(`Si el líder le lleva ${v} vuelta(s), ${p} no puede ir adelante de autos que dieron más vueltas: pasa al ${j + 1}.º lugar.`);
    }
  }
  actualizar();
  lista.querySelectorAll("li").forEach((li, i) => { li.querySelector(".pos").textContent = "P" + (i + 1); });
}
// arrastrar con el mouse o el dedo (en pantallas táctiles, desde el asa ⋮⋮); un clic sin arrastrar abre el piloto
let arr = null;
lista.addEventListener("pointerdown", e => {
  const li = e.target.closest("li");
  if (!li || e.target.closest("button, input, .vars")) return;
  if (e.pointerType === "touch" && !e.target.closest(".asa")) return;
  arr = {p: li.dataset.p, y: e.clientY, mueve: false, li};
  li.setPointerCapture(e.pointerId);
});
lista.addEventListener("pointermove", e => {
  if (!arr) return;
  if (!arr.mueve && Math.abs(e.clientY - arr.y) < 6) return;
  arr.mueve = true; arr.li.classList.add("arrastrando");
  if (e.clientY < 70) window.scrollBy(0, -14); else if (e.clientY > innerHeight - 50) window.scrollBy(0, 14);
  const sobre = document.elementFromPoint(e.clientX, e.clientY)?.closest("#p-lista li");
  lista.querySelectorAll("li").forEach(l => l.classList.toggle("sobre", l === sobre));
  arr.destino = sobre ? sobre.dataset.p : arr.destino;
});
lista.addEventListener("pointerup", e => {
  if (!arr) return;
  const a = arr; arr = null;
  lista.querySelectorAll("li").forEach(l => l.classList.remove("arrastrando", "sobre"));
  if (a.mueve) { if (a.destino) mover(a.p, est.orden.indexOf(a.destino)); return; }
  est.abierto = est.abierto === a.p ? null : a.p;
  dibujarLista(); actualizar();
});
lista.addEventListener("click", e => {
  const b = e.target.closest("button[data-m]"), li = e.target.closest("li");
  if (b && li) mover(li.dataset.p, est.orden.indexOf(li.dataset.p) + (+b.dataset.m));
});
document.querySelectorAll("#p-momento button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#p-momento button").forEach(x => x.classList.toggle("on", x === b));
  est.k = b.dataset.k; cargar();
}));
selC.addEventListener("change", () => { est.c = selC.value; cargar(); });
document.getElementById("p-volver").addEventListener("click", cargar);
est.abierto = PR.piloto_inicial;
listo.then(cargar);

// Navegación: marca la sección que se está leyendo
const enlaces = [...document.querySelectorAll("nav a")];
const secciones = [...document.querySelectorAll("main section")];
function marcar() {
  const y = window.scrollY + 120;
  let actual = secciones[0];
  for (const s of secciones) if (s.offsetTop <= y) actual = s;
  enlaces.forEach(a => a.classList.toggle("activo", a.getAttribute("href") === "#" + actual.id));
}
window.addEventListener("scroll", marcar, {passive: true});
marcar();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
# 6. La página
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def f(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


def n(x):
    return f"{int(x):,}".replace(",", ".")


def pct(x, d=0):
    return f(100 * x, d) + " %"


def tabla(cab, filas, clase=""):
    th = "".join(f"<th>{c}</th>" for c in cab)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in fila) + "</tr>" for fila in filas)
    return f'<div class="tabla"><table class="{clase}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


NOMBRE_COL = {"posicion_al_corte": "dónde va (puesto)", "vueltas_detras": "vueltas detrás del líder",
              "tendencia_10": "puestos ganados en las últimas 10 vueltas", "gap_qualy_pct": "% más lento que la pole en la qualy",
              "log_puntos_equipo": "puntos del equipo (log)", "neumatico_vs_carrera": "vueltas del neumático contra la carrera",
              "driver_standing_before": "puesto en el campeonato", "log_puntos": "puntos del piloto (log)",
              "driver_points_before": "puntos del piloto", "ultima_vuelta": "vueltas completadas",
              "vueltas_neumatico": "edad del neumático", "pos_hace_10": "posición 10 vueltas antes",
              "quali_best_s": "mejor tiempo de qualy", "constructor_name": "equipo"}


def textos(R, N, corr):
    """Lo que se lee fuera de «Cómo lo sabemos»: pensado para alguien que no cursó la materia."""
    C, T, el = R["comparacion"], R["test"], DECISIONES["elegido"]
    M = (0.0, 0.25, 0.5, 0.75)
    mod = {m: C[m][el[m]].mean() for m in M}
    regla = {m: C[m]["regla"].mean() for m in M}
    ventaja = {m: regla[m] - mod[m] for m in M}
    gana_test = [m for m in M if T[m]["modelo"] < T[m]["regla"]]
    pierde_test = [m for m in M if T[m]["modelo"] >= T[m]["regla"]]
    rho = corr.pivot(index="momento", columns="serie", values="rho")
    rv = lambda mm: rho.loc[ETIQUETA[mm], "dónde va en ese momento"]
    tr = R["tabla_res"].set_index("momento")
    top10 = f(10 * tr.loc["Antes de largar", "top10_m_val"], 0)
    cuando = {0.0: "antes de la largada", 0.25: "a un cuarto de carrera", 0.5: "a mitad de carrera",
              0.75: "a tres cuartos de carrera"}
    z = {}
    ex = R["explora"][R["explora"]["m"] == ETIQUETA[0.0]]
    err_ex, ab_ex = (ex["pred"] - ex["real"]).abs(), ex["fin"] != "terminó"
    z["idea"] = ("El lugar en el que va un piloto dice casi todo sobre dónde va a terminar, y nuestro modelo lo afina con lo que "
                 "pasa en la pista prediciendo en qué puesto termina cada piloto en cuatro momentos: antes de la largada y "
                 "cuando ya se corrió un cuarto, la mitad y tres cuartos de la carrera, usando sólo lo que se sabe hasta ese "
                 "momento. Abajo contamos qué datos usamos, qué descubrimos y qué tan bien funciona; si querés ver las pruebas, "
                 "abrí «Cómo lo sabemos» en cada parte. Al final podés recorrer todas las carreras y probar el modelo vos mismo.")
    top10_r = f(10 * tr.loc["Antes de largar", "top10_r_val"], 1)
    e_ab, e_ok = f(err_ex[ab_ex].mean(), 1), f(err_ex[~ab_ex].mean(), 1)
    z["kpi_datos"] = {"aciertos": int(round(10 * tr.loc["Antes de largar", "top10_m_val"])),
                      "dist": dist_error(R["explora"]),
                      "e_ab": err_ex[ab_ex].mean(), "e_ok": err_ex[~ab_ex].mean(),
                      "p_pilotos": ab_ex.mean(), "p_error": err_ex[ab_ex].sum() / err_ex.sum()}
    z["kpis"] = [
        ("¿Cuántos puestos se equivoca?",
         f'<span class="v">{f(mod[0.0], 0)}<small>antes de la largada</small></span><span class="fl">→</span>'
         f'<span class="v">{f(mod[0.75], 0)}<small>al 75&nbsp;% de la carrera</small></span>', "",
         "Si un piloto termina 5.º y el modelo había dicho 3.º, se equivocó por 2 puestos. Los números son el promedio "
         "de todos los pilotos: cuanto más avanza la carrera, menos queda por pasar y menos se equivoca.",
         f"En cifras exactas, el modelo pasa de <b>{f(mod[0.0], 1)}</b> a <b>{f(mod[0.75], 1)}</b> puestos de error."),
        ("Zona de puntos", f"{top10} de 10", "aciertos entre los diez primeros, antes de la largada",
         "Sólo los diez primeros de cada carrera suman puntos. Antes de que se apague el semáforo, de los diez pilotos "
         "que el modelo pone en esa zona, en promedio ocho terminan efectivamente entre los diez primeros."
         if top10 == "8" else
         f"Sólo los diez primeros de cada carrera suman puntos. Antes de que se apague el semáforo, de los diez pilotos "
         f"que el modelo pone en esa zona, en promedio {top10} terminan efectivamente entre los diez primeros.",
         f"Usando sólo la grilla de largada se aciertan <b>{top10_r} de 10</b>."),
        ("La mayor causa de error", e_ab, "puestos de error con los pilotos que abandonan",
         f"Un auto que se rompe o choca termina al fondo, y eso no se puede saber antes de que pase. Antes de la largada, "
         f"con esos pilotos el modelo se equivoca {e_ab} puestos en promedio; con los que terminan la carrera, {e_ok}.",
         "Por eso el error no llega a cero, ni siquiera al final de la carrera.")]
    # 1 · Los datos
    z["t_datos"] = ("Más de la mitad de los pilotos termina a dos lugares o menos de donde largó; los que pierden muchos "
                    "lugares son, casi siempre, los que abandonan")
    z["l_datos"] = (f"Juntamos los datos de {N['carreras']} carreras de Fórmula 1, de 2022 a 2026: la grilla de largada, "
                    f"el campeonato y lo que hizo cada piloto vuelta por vuelta. El gráfico cuenta cuántos lugares ganó o "
                    f"perdió cada piloto entre la largada y la llegada. La franja del medio junta al {pct(N['cerca'])} de "
                    f"los casos: terminaron a dos lugares o menos de donde largaron. A la izquierda quedan los que "
                    f"perdieron muchos lugares, y casi todos son abandonos: un auto que se rompe o choca termina al fondo. "
                    f"Eso no lo puede anticipar nadie, y va a ser la mayor fuente de error de cualquier predicción.")
    # 2 · Lo que encontramos
    z["t_hallazgo"] = ("Desde el primer cuarto de carrera, el lugar en el que va un piloto anticipa su llegada mejor que "
                       "el campeonato y que su equipo")
    z["l_hallazgo"] = (f"Para cada dato medimos qué tan bien anticipa el orden de llegada: 1 sería acertar el orden exacto y "
                       f"0, nada que ver. Antes de la largada, el lugar en la grilla ({f(rv(0.0))}) y el puesto en el "
                       f"campeonato ({f(rho.loc[ETIQUETA[0.0], 'puesto en el campeonato'])}) anticipan más o menos lo "
                       f"mismo. Pero a medida que avanza la carrera, el lugar en el que va cada piloto (la línea naranja) "
                       f"se parece cada vez más al resultado final: llega a {f(rv(0.75))} a tres cuartos de carrera, "
                       f"mientras que lo que se sabía antes de largar se queda igual. Por eso armamos un modelo distinto "
                       f"para cada momento de la carrera. Y por eso nuestro rival es la regla «termina donde está»: "
                       f"suponer que cada piloto llega en el mismo lugar en el que va. Es simple, pero difícil de superar.")
    # 3 · El modelo
    def lista_momentos(ms):
        return " y ".join(cuando[m] for m in ms)
    perdidas = "; ".join(
        (f"{cuando[m]} empataron prácticamente" if T[m]["modelo"] - T[m]["regla"] < 0.05 else
         f"{cuando[m]} la regla fue mejor, por {f(T[m]['modelo'] - T[m]['regla'], 1)} puestos")
        for m in pierde_test)
    z["t_modelo"] = (f"El modelo se equivoca menos que suponer que nadie cambia de lugar, sobre todo al principio de la "
                     f"carrera; en las carreras nuevas de 2026, esa ventaja se mantuvo en {len(gana_test)} de los 4 momentos")
    z["l_modelo"] = (f"Comparamos tres formas de predecir. Adivinar a ciegas (decir que todos terminan en el medio del "
                     f"pelotón) se equivoca en {f(C[0.0]['piso'].mean(), 0)} puestos. La regla «termina donde está» (cada "
                     f"piloto llega en el lugar en el que va o, antes de la largada, en el que larga) ya es mucho mejor: "
                     f"se equivoca {f(regla[0.0], 1)} puestos antes de largar y {f(regla[0.75], 1)} a tres cuartos de "
                     f"carrera. Nuestro modelo (los puntos azules) parte de esa misma idea y la corrige con otros datos: "
                     f"el campeonato, cuántas vueltas detrás del líder va cada uno, si viene ganando o perdiendo lugares "
                     f"y qué tan gastados tiene los neumáticos. Así baja a {f(mod[0.0], 1)} y {f(mod[0.75], 1)}. La "
                     f"ventaja es más grande al principio y se achica al final, cuando queda poca carrera para cambiar de "
                     f"lugar. Los rombos son las {N['carreras_test']} carreras de 2026, que el modelo nunca había visto: "
                     f"ganó {lista_momentos(gana_test)}; {perdidas}.")
    z["l_tabla"] = (f"También miramos la tabla completa de cada carrera, para ver si el orden que arma el modelo se parece "
                    f"al orden final. Antes de la largada, de los 10 pilotos que el modelo pone en la zona de puntos, "
                    f"{top10} terminan ahí. El podio, en cambio, casi no lo mejora: los que van adelante suelen terminar "
                    f"adelante, y eso ya lo sabe cualquiera.")
    return z, mod, regla, ventaja


def como_datos(N):
    pasos = [["Corrida del pipeline del 23/09/2026: el snapshot (Silver D)",
              f"{n(N['snap'])} filas", f"Un piloto en una carrera en {N['pcts']} momentos: 0, 25, 50, 75 y 100 % de las vueltas del líder."],
             ["Sólo los cuatro momentos en que se predice", f"{n(N['por_momento'] * 4)} filas",
              f"Sale el 100 %: es la carrera terminada. Quedan {n(N['por_momento'])} filas por momento, de {N['carreras']} carreras."],
             ["Entrenamiento y validación: 2022–2025", f"{n(N['filas_train'])} por momento",
              f"{N['carreras_train']} carreras. Con ellas se eligió todo: columnas, familia y ajuste."],
             ["Test: 2026", f"{n(N['filas_test'])} por momento",
              f"{N['carreras_test']} carreras, medidas una sola vez al final, con 22 autos por carrera."],
             ["En «Explorá»: las que tienen predicción fuera de muestra", f"{n(N['filas_test'] + R_EXPLORA_TRAIN[0])} por momento",
              f"Las primeras {N['solo_entrenan']} carreras de 2022 sólo entrenan (la validación por fecha nunca las valida); "
              f"quedan {N['validadas']} validadas y las {N['carreras_test']} del test."]]
    c = N["coincide"]
    return f"""
<h4>De dónde salen</h4>
<p>Del pipeline del repositorio: un DAG de Airflow (<code>dags/f1_ingest.py</code>) baja resultados, grilla y campeonato de
Jolpica-F1, vueltas y qualy de FastF1, y clima y paradas de OpenF1; los guarda crudos (bronze) y arma las tablas limpias
(silver). Usamos dos archivos de la corrida del <strong>23/09/2026</strong>: <code>driver_race_snapshots_2026-09-23.csv</code>
({n(N['snap'])} filas) y <code>driver_lap_features_2026-09-23.csv</code>, una fila por piloto y vuelta ({n(N['vueltas'])} filas).</p>
<h4>Qué es una fila</h4>
<p>Un piloto en una carrera en un momento: antes de largar o al cerrar la vuelta del 25, 50 o 75 % de la carrera.
Cada momento sólo usa vueltas hasta ese corte; lo que pasa después no entra.</p>
<h4>Qué se filtró, paso por paso</h4>
{tabla(["Paso", "Quedan", "Por qué"], pasos)}
<h4>La posición en cada momento, rehecha</h4>
<p>La columna <code>current_position</code> ordenaba por tiempo acumulado, que no suma las vueltas sin tiempo registrado:
un piloto con una vuelta sin tiempo aparecía adelante. La rehicimos como la clasificación oficial (primero los que completaron
más vueltas y, en la misma vuelta, el orden en que cruzaron la línea). La vieja coincidía sólo en el {pct(c[0.25], 1)} de las
filas al 25 %, el {pct(c[0.5], 1)} al 50 % y el {pct(c[0.75], 1)} al 75 %. Antes de largar, la posición es la grilla, y quien
larga desde boxes va último.</p>
<h4>Columnas que no entran nunca (fuga)</h4>
<p>Todo lo que se completa al terminar la carrera: {", ".join(f"<code>{x}</code>" for x in N['fuga'])}.
<code>laps_total</code> parecía el largo de la carrera, pero son las vueltas que completó cada piloto (coincide en el 100 % de
las filas): también es fuga.</p>
<h4>Cómo se armó el gráfico</h4>
<p>Datos: las {n(N['por_momento'])} filas de antes de largar (en el texto, «casos»: un piloto en una carrera).
Transformación: lugares ganados = lugar de largada − puesto de llegada, contados por valor y por situación; positivo es
ganar lugares. Marca: barras apiladas. Color: abandonó (estado distinto de «Finished» o «+N Lap») contra terminó; el mismo
color marca a los que abandonaron en toda la página. La franja del medio son los casos con un cambio de 2 lugares o menos
({pct(N['cerca'])}). Los abandonos pierden {f(N['caida_abandono'], 1)} lugares de media y son {pct(N['grandes_abandono'])}
de los que pierden 5 o más.</p>"""


def como_hallazgo(corr):
    s = SELECCION_E3
    return f"""
<h4>Una hipótesis que se sostuvo: el campeonato (H2), pero sólo antes de largar</h4>
<p>En la Entrega 2 quedó inconclusa: el puesto en el campeonato se mezclaba con el auto y con la qualy. En el modelo de antes
de largar, sacar el campeonato (puesto y puntos) empeora el error en <strong>{f(s['aporte_campeonato'], 3)} puestos</strong>,
y gana claro en los cinco cortes. Desde el 25 % ya no aporta: lo que decía, la posición en pista lo dice mejor.</p>
<h4>Una que no: el equipo (H3)</h4>
<p>El equipo explicaba el <strong>{f(s['equipo_varianza'], 1)} %</strong> de la varianza de la llegada, y parecía
imprescindible. Pero con la posición y el campeonato adentro no le gana claro al modelo sin él en ningún momento. Una
explicación posible: lo que vale cada equipo cambia de temporada en temporada, y al 25 % la fuerza del auto entra mejor por
los puntos del equipo y el tiempo de qualy.</p>
<p>También probamos una versión más gruesa: una sola columna <code>equipo_top</code> (1 si el equipo está entre los 4 con
más puntos antes de esa carrera; en la primera fecha, el orden final de la temporada anterior). Separa bien a los pilotos
(los de equipo top llegan, en promedio, 6,5.º; el resto, 13,1.º), pero tampoco gana claro en ningún momento:
{tabla(["Momento", "Mejora media (puestos)", "Desvío de la mejora", "Cortes en que mejora"],
       [[ETIQUETA[m], f(a, 3), f(b, 3), f"{c} de 5"] for m, (a, b, c) in s["equipo_top"].items()], "num")}
Lo que dice el equipo ya lo dicen el campeonato, los puntos del equipo, la qualy y, sobre todo, la posición en pista.</p>
<h4>Cómo se construyó el gráfico</h4>
<p>Lo que el gráfico llama «qué tan bien anticipa el orden de llegada» es la correlación de Spearman entre el puesto de
llegada y cada dato: compara órdenes, no distancias (1 = mismo orden; 0 = sin relación). Datos: 2022–2025 (el test no se
mira), 1.826 pilotos-carrera por momento, un valor por momento. «Dónde va» es la posición rehecha en cada corte (antes de largar,
la grilla); el puesto del equipo sale de ordenar a los equipos de cada carrera por los puntos que sumaban sus pilotos antes
de esa carrera. Las dos líneas grises son planas porque son datos de antes de largar: valen lo mismo en los cuatro
momentos. El eje empieza en 0,4 porque las tres series viven entre 0,5 y 0,9; es un gráfico de líneas, no de barras.</p>"""



def como_tabla(R):
    tr = R["tabla_res"]
    fmt = lambda r, k, q: f"{f(r[f'{k}_{q}_val'], 3)} ± {f(r[f'{k}_{q}_sd'], 3)}"
    val = [[r["momento"], fmt(r, "spearman", "m"), fmt(r, "spearman", "r"), f"{r['spearman_cortes']} de 5",
            f"{f(r['podio_m_val'])} / {f(r['podio_r_val'])}", f"{f(r['top10_m_val'])} / {f(r['top10_r_val'])}"]
           for _, r in tr.iterrows()]
    test = [[r["momento"], f(r["spearman_m_test"], 3), f(r["spearman_r_test"], 3),
             f"{r['mejor_test']} / {r['igual_test']} / {r['peor_test']}",
             f"{f(r['podio_m_test'])} / {f(r['podio_r_test'])}", f"{f(r['top10_m_test'])} / {f(r['top10_r_test'])}"]
            for _, r in tr.iterrows()]
    return f"""
{tabla(["Validación 2022–2025", "Spearman, modelo", "Spearman, regla", "Cortes en que gana el modelo",
        "Podio, modelo / regla", "Top 10, modelo / regla"], val, "num")}
{tabla(["Test 2026", "Spearman, modelo", "Spearman, regla", "Carreras mejor / igual / peor",
        "Podio, modelo / regla", "Top 10, modelo / regla"], test, "num")}
<p class="nota">La historia es la misma que con el MAE. La tabla del modelo se parece más a la final que la de la regla,
sobre todo antes de largar y al 25 %, y ordena mejor la zona de puntos. El podio casi no cambia: con los que van adelante el
modelo tiene poco para agregar. En 2026 las dos tablas quedan prácticamente iguales. En «Explorá» se puede ver cada carrera
como dos tablas lado a lado.</p>"""


def como_modelo(R, mod, regla):
    C, T, el, M = R["comparacion"], R["test"], DECISIONES["elegido"], (0.0, 0.25, 0.5, 0.75)
    filas = []
    for nombre in ["piso", "regla"] + FAMILIAS:
        etq = {"piso": "piso (siempre la mediana)", "regla": "«termina donde está»"}.get(nombre, nombre)
        celdas = [etq]
        for m in M:
            v = C[m][nombre]
            txt = f"{f(v.mean(), 3)} ± {f(v.std(), 3)}"
            celdas.append(f"<strong>{txt}</strong>" if nombre == el[m] else txt)
        filas.append(celdas)
    comp = tabla(["", *MOMS], filas, "num")
    ajuste = [[ETIQUETA[m], el[m] + (" ajustado" if el[m] in DECISIONES["ajuste"].get(m, {}) else " (sin penalización)"),
               DECISIONES["ajuste"].get(m, {}).get(el[m], "por defecto; Ridge con penalización no le ganó claro"),
               "posición esperada" if not DECISIONES["ordenar"][m] else "orden de llegada 1, 2, 3…",
               ", ".join(NOMBRE_COL[c] for c in DECISIONES["columnas"][m])] for m in M]
    test = [[ETIQUETA[m], f(T[m]["modelo"], 3), f(T[m]["regla"], 3), f(T[m]["piso"], 3), f(mod[m], 3),
             f"{T[m]['mejor']} / {T[m]['igual']} / {T[m]['peor']}"] for m in M]
    S = R["situacion"]
    sit = []
    for s_ in ["terminó", "abandonó más adelante", "ya había abandonado en ese momento"]:
        fila = [{"terminó": "terminó la carrera", "abandonó más adelante": "abandonó después del corte",
                 "ya había abandonado en ese momento": "ya había abandonado al corte"}[s_]]
        for m in M:
            r = S[(S["m"] == m) & (S["sit"] == s_)]
            fila.append("—" if r.empty else f"{f(r['modelo'].iloc[0])} / {f(r['regla'].iloc[0])} <span class='n'>({int(r['n'].iloc[0])})</span>")
        sit.append(fila)
    coef = []
    for c in sorted({c for m in (0.25, 0.5, 0.75) for c in R["coef"][m]}, key=lambda c: DECISIONES["columnas"][0.25].index(c)
                    if c in DECISIONES["columnas"][0.25] else 9):
        coef.append([NOMBRE_COL[c]] + [f(R["coef"][m][c], 3) if c in R["coef"][m] else "—" for m in (0.25, 0.5, 0.75)])
    imp = R["importancia"]
    cols_imp = sorted({c for m in M for c in imp[m]}, key=lambda c: -max(imp[m].get(c, 0) for m in M))
    imp_t = [[NOMBRE_COL[c]] + [f(imp[m][c], 3) if c in imp[m] else "—" for m in M] for c in cols_imp]
    arbol = R["proba"][0.0]["mae"]
    bo = C[0.0]["boosting"]
    sit_d = S[S["sit"] == "abandonó más adelante"]
    ya25 = S[(S["m"] == 0.25) & (S["sit"] == "ya había abandonado en ese momento")]["modelo"].iloc[0]
    return f"""
<h4>La métrica, y por qué</h4>
<p>El MAE en puestos: si un piloto llega 5.º y el modelo dijo 3.º, el error es 2. Se lee directo y cada puesto pesa lo mismo.
La elegimos antes de probar modelos: los abandonos producen errores de cinco o diez puestos que nadie puede prever, y con el
error cuadrático esos pocos casos dominarían el número.</p>
<h4>El piso, y un rival más exigente</h4>
<p>El piso es decir siempre el puesto del medio: erra {f(C[0.0]['piso'].mean())} puestos (en 2026, con 22 autos,
{f(T[0.0]['piso'])}). El rival de verdad es «termina donde está»: cada piloto llega en el puesto que tiene en ese momento
(antes de largar, el de la grilla). Erra {", ".join(f"{f(regla[m])} {('antes de largar' if m == 0 else 'al ' + ETIQUETA[m])}" for m in M)}.</p>
<h4>Qué muestra el gráfico de arriba</h4>
<p>Cada punto es el MAE medio de los 5 cortes de la validación (2022–2025); cada rombo, el MAE en el test (2026). «Adivinar a
ciegas» es el piso del notebook (predecir siempre la mediana del entrenamiento); «termina donde está» es la regla ingenua
(predecir el puesto al corte; antes de largar, la grilla). El desvío entre cortes de cada número está en la tabla de
comparación de más abajo. En el texto se redondea a un decimal.</p>
<h4>Cómo se armó la validación</h4>
<p>Las filas no son independientes: unos 20 pilotos por carrera forman un orden, y el modelo sirve para carreras que todavía no
se corrieron. Por eso el test son las 14 carreras de 2026, y la validación sobre 2022–2025 usa <code>TimeSeriesSplit</code>
sobre la lista de carreras ordenada por fecha: una carrera nunca queda partida y siempre se valida con carreras posteriores.
Son 5 cortes; el entrenamiento crece de 17 a 77 carreras y cada corte valida con las 15 siguientes. Sorteando carreras la
ventaja sobre la regla da parecida; sorteando filas, el modelo ni siquiera se puede calcular bien.</p>
<h4>La comparación: media ± desvío de los 5 cortes, cada familia en su mejor versión</h4>
{comp}
<p class="nota">En negrita, el elegido. Una opción reemplaza a la más simple sólo si «gana claro»: baja el error en al menos 4
de los 5 cortes y la mejora media supera su propio desvío. Antes de largar el boosting le ganó a la regresión lineal en los 5
cortes (por {f(C[0.0]['regresión lineal'].mean() - bo.mean(), 3)}); en los otros momentos ninguna familia le ganó claro a la
lineal (al 75 % el bosque da {f(C[0.75]['bosque'].mean(), 3)}, pero mejora en sólo 2 de 5 cortes), y quedó lo simple.</p>
<h4>El ajuste del elegido</h4>
{tabla(["Momento", "Modelo", "Hiperparámetros", "Formato", "Columnas"], ajuste)}
<p class="nota">El modelo aprende cuántos puestos gana o pierde cada piloto desde donde va. Del 25 % en adelante, las
predicciones de cada carrera se ordenan 1, 2, 3…, porque no puede haber dos pilotos en el mismo puesto: ordenar ganó claro
(por {", ".join(f(v, 3) for v in SELECCION_E3['ordenar'].values())} puestos). Antes de largar, no.</p>
<h4>El test: las 14 carreras de 2026, una sola vez</h4>
{tabla(["Momento", "Modelo", "«Termina donde está»", "Piso", "Modelo en validación", "Carreras mejor / igual / peor que la regla"], test, "num")}
<p class="nota">La ventaja medida en 2022–2025 no se sostuvo entera. La lluvia no lo explica (2026 tuvo 1 carrera con lluvia de 14;
la validación, 9 de 75). Quedan el reglamento nuevo o una racha, y con 14 carreras no alcanza para distinguirlos.</p>
<h4>Dónde se equivoca (test 2026: error del modelo / de la regla, y cuántos pilotos)</h4>
{tabla(["Situación", *MOMS], sit, "num")}
<p class="nota">Los errores grandes son los abandonos después del corte: erran entre {f(sit_d['modelo'].min(), 1)} y
{f(sit_d['modelo'].max(), 1)} puestos, igual que la regla, porque nadie podía saber que ese auto iba a romperse o chocar. Donde
el modelo gana es entre los que terminan, hasta el 50 %. Y una confusión concreta: al 25 % movió a autos que ya habían
abandonado, que no pueden ganar puestos (el error de {f(ya25)} de esa fila, donde la regla erra 0).</p>
<h4>En qué se apoya</h4>
<p>Coeficientes de las regresiones lineales, en puestos de cambio por unidad (positivo: el piloto pierde puestos):</p>
{tabla(["Entrada", "25 %", "50 %", "75 %"], coef, "num")}
<p>Cuánto sube el error del test si se mezcla al azar cada columna de entrada (30 repeticiones):</p>
{tabla(["Columna", *MOMS], imp_t, "num")}
<p class="nota">En los cuatro momentos manda la posición, y pesa más cuanto más avanza la carrera. No es fuga: es la grilla o
la posición de la pantalla de tiempos al cerrar la vuelta de corte, se arma sólo con vueltas hasta el corte y, si tuviera la
respuesta adentro, la regla no erraría entre {f(regla[0.75])} y {f(regla[0.0])} puestos.</p>
<h4>El modelo que se prueba en esta página</h4>
<p>Antes de largar, «Probá el modelo» usa un árbol en lugar del boosting elegido, porque el boosting no se puede escribir
en la página. Los números y la justificación están en el «Cómo lo sabemos» de esa sección. En «Explorá» y en todo el
tallo, los números son los del boosting.</p>"""



def como_explora(R, N, e):
    pc = R["tabla_pc"]
    return f"""
<h4>Qué predicciones se muestran</h4>
<p>Todas son <strong>fuera de muestra</strong>: ningún modelo vio la carrera que predice. Para 2022–2025 son las predicciones
de la validación: cada carrera validada la predice el modelo del corte de <code>TimeSeriesSplit</code> que entrenó sólo con
carreras anteriores ({N['validadas']} carreras). Para 2026 son las del test: el modelo entrenado con 2022–2025 predice las
{N['carreras_test']} carreras una sola vez. Las primeras {N['solo_entrenan']} carreras de 2022 no aparecen porque la
validación por fecha nunca las valida (siempre entrenan). En total, {n(len(e))} predicciones: {e['c'].nunique()} carreras
× 4 momentos × los pilotos de cada una. Antes de largar el modelo es el boosting y la predicción es una posición esperada
(con decimales); desde el 25 % es la regresión lineal y la predicción es el orden de llegada 1, 2, 3…</p>
<h4>Cómo se mide cada carrera</h4>
<p>«Cuántos puestos se equivoca» es el MAE de esa carrera: el promedio de |predicción − puesto final| sobre sus pilotos (con un
piloto elegido, es el error de ese piloto en esa carrera). «Qué tan parecida queda la tabla» es la correlación de Spearman
entre el puesto que le da la tabla predicha a cada piloto (ordenando por la predicción y, si empatan, por dónde va) y su
puesto final; con un piloto elegido sigue siendo la de la carrera entera. El podio y la zona de puntos del encabezado cuentan
cuántos de los 3 y de los 10 primeros predichos terminaron ahí. A la regla «termina donde está» se le calcula lo mismo,
usando como predicción la posición en ese momento (antes de largar, la grilla). Son {len(pc)} combinaciones de carrera y
momento; sus promedios por momento son los de «La tabla completa de cada carrera», en la parte del modelo.</p>
<h4>Las dos vistas de la carrera</h4>
<p>«Las dos tablas»: a la izquierda, los pilotos ordenados por la predicción; a la derecha, por el puesto final. El color de
cada línea es la diferencia entre el puesto predicho y el real (1 o menos, 2 a 4, 5 o más); punteada si abandonó (estado
distinto de «Finished» o «+N Lap»). «Piloto por piloto»: naranja, la posición en ese momento (lo que predice la regla); azul,
lo que dijo el modelo; rombo, el puesto final; cruz, abandono.</p>
<h4>Qué toca cada control</h4>
<p>El momento filtra las filas de todos los gráficos. La temporada y el piloto filtran las filas antes de agrupar por carrera,
así que cambian los puntos del panorama (y el piloto se resalta abajo). «Medir cada carrera por» cambia el campo del eje
vertical del panorama: el MAE sale de promediar los pilotos; Spearman, de un cálculo por carrera hecho en Python y unido por
carrera y momento. El clic es una selección de Vega-Lite: elige la carrera que se abre abajo y agranda su punto.</p>"""


def como_proba(R):
    C = R["comparacion"]
    arbol, bo = R["proba"][0.0]["mae"], C[0.0]["boosting"]
    filas = []
    for m in (0.25, 0.5, 0.75):
        T_ = R["proba"][m]["terminos"]
        filas.append([ETIQUETA[m], "constante (ajuste general)", f(T_["intercepto"], 3), "—", "—"])
        for c, w, mu, sd in T_["partes"]:
            filas.append(["", NOMBRE_COL[c], f(w, 3), f(mu, 2), f(sd, 2)])
    return f"""
<h4>De dónde salen los pilotos y sus datos</h4>
<p>Cada carrera es una de las {R['proba'][0.25]['filas']['c'].nunique()} de 2026 (el test), con los datos de cada piloto tal
como los arma el pipeline en ese momento: la posición rehecha, las columnas derivadas (vueltas detrás del líder, tendencia de
10 vueltas, neumático contra la mediana de la carrera, puntos del equipo en log) y, si falta algún valor, la mediana del
entrenamiento, igual que el imputador del pipeline.</p>
<h4>La fórmula, del 25 % en adelante</h4>
<p>Es la regresión lineal elegida, entrenada con 2022–2025 (la misma que dio el test). Para cada piloto:
posición esperada = posición + constante + Σ coeficiente × (valor − media) / desvío, con la media y el desvío del
<code>StandardScaler</code>. Después se ordena a todos los pilotos de la carrera por la posición esperada (si empatan, por
posición) y eso da el puesto 1, 2, 3… Cada término de la suma es una barra de «Ver más»; la constante es el «ajuste
general». Los números que usa la página:</p>
{tabla(["Momento", "Entrada", "Coeficiente", "Media", "Desvío"], filas, "num")}
<p class="nota">Los puntos del equipo entran como log(1 + puntos), igual que en el notebook; el control los muestra en
puntos. Al construir la página se comprueba que esta fórmula reproduce a scikit-learn en todas las filas de 2026 (diferencia
menor a 0,000001) y que el orden coincide con el del modelo.</p>
<h4>Antes de largar: un árbol en lugar del boosting</h4>
<p>En el navegador no corre Python, y el boosting elegido (100 árboles) no se puede escribir en la página. El instructivo
permite usar un modelo que sí se pueda llevar si queda a menos de un desvío del mejor. Usamos el <strong>árbol
ajustado</strong> de la Entrega 3 (max_depth=6, min_samples_leaf=30): erra {f(arbol.mean(), 3)} ± {f(arbol.std(), 3)} contra
{f(bo.mean(), 3)} ± {f(bo.std(), 3)} del boosting. La diferencia, {f(arbol.mean() - bo.mean(), 3)} puestos, es menor que un
desvío del boosting ({f(bo.std(), 3)}). El árbol viaja en la página como preguntas encadenadas (con los umbrales pasados de
la escala estandarizada a las unidades originales), y es el camino que muestra «Ver más». También se comprueba contra
scikit-learn al construir la página.</p>
<h4>Los límites de los controles</h4>
<p>Los controles sólo llegan hasta valores que el modelo vio al entrenar. En las tres entradas con colas largas (lugares
ganados en 10 vueltas, distancia a la pole y neumático contra la carrera) van del percentil 1 al 99 de 2022–2025, para que
los extremos raros no inutilicen el control; en el resto, del mínimo al máximo. Si el valor real de un piloto queda afuera, la
página lo avisa y lo lleva al borde. La posición llega hasta 20 porque 2022–2025 tenía 20 autos; en 2026 hay 22, y para los
puestos 21 y 22 la fórmula extrapola, con un aviso.</p>
<h4>Las reglas que mantienen la carrera posible</h4>
<p>La posición en carrera se ordena primero por vueltas completadas, así que nadie puede ir adelante de un auto con menos
vueltas detrás del líder: al arrastrar se ajustan sus vueltas detrás, y al cambiarlas se lo reubica. Moverse en la lista son
lugares ganados o perdidos ahora, y se suman a la tendencia de 10 vueltas de cada piloto afectado. Los puntos del equipo
cambian para los dos pilotos. Antes de largar, el puesto en el campeonato sale de ordenar a los pilotos de la carrera por sus
puntos.</p>"""


R_EXPLORA_TRAIN = [0]


def pagina(R, N, hist, corr, salida):
    z, mod, regla, ventaja = textos(R, N, corr)
    e = R["explora"].copy()
    e["gp"] = e["gp"].map(lambda g_: "GP de " + GP.get(g_, g_))
    R_EXPLORA_TRAIN[0] = int((e["m"] == "Antes de largar").sum()) - N["filas_test"]
    inicial_e = {"m": "Antes de largar", "c": "2026-06"}
    t25 = R["proba"][0.25]["filas"]
    inicial_p = {"c": "2026-14", "piloto": t25[(t25["c"] == "2026-14") & (t25["posicion_al_corte"] == 6)]["piloto"].iloc[0]}
    pc, res = R["tabla_pc"], R["tabla_res"]
    pc = pc.assign(k=pc["c"] + "|" + pc["m"])
    ex = g_explora(e, pc, inicial_e)
    specs = {"g-datos": estilo(g_datos(hist, N)), "g-hallazgo": estilo(g_hallazgo(corr)),
             "g-modelo": estilo(g_modelo(R)), "g-tabla": estilo(g_tabla(res)),
             "g-explora": estilo(ex["panorama"]), "g-explora-tablas": estilo(ex["tablas"]),
             "g-explora-det": estilo(ex["detalle"]),
             "g-proba": estilo(g_proba_tabla()), "g-proba-apo": estilo(g_proba_aportes()),
             "g-kpi-1": estilo(g_kpi_error(z["kpi_datos"]["dist"])), "g-kpi-2": estilo(g_kpi_puntos(z["kpi_datos"]["aciertos"])),
             "g-kpi-3": estilo(g_kpi_abandono(z["kpi_datos"]["p_pilotos"], z["kpi_datos"]["p_error"]))}
    proba = {"carreras": {}, "momentos": {}, "inicial": inicial_p["c"], "piloto_inicial": inicial_p["piloto"]}
    for m in (0.0, 0.25, 0.5, 0.75):
        P, k = R["proba"][m], CORTO[m]
        rangos = {}
        for col in P["columnas"]:
            paso = ENTRADA[col][1]
            lo, hi = ((np.expm1(v) if ENTRADA[col][2] == "log" else v) for v in P["rangos_p"][col])
            rangos[col] = [round(float(np.floor(lo / paso + 1e-9) * paso), 2), round(float(np.ceil(hi / paso - 1e-9) * paso), 2), paso]
        carreras = {}
        for c, t in P["filas"].groupby("c"):
            t = t.sort_values("posicion_al_corte")
            if not np.array_equal(t["posicion_al_corte"].to_numpy(), np.arange(1, len(t) + 1)):
                avisar(f"en {c} ({ETIQUETA[m]}) las posiciones no van de 1 a N: la lista de «Probá» no las reproduce")
            carreras[c] = [{"p": r["piloto"], "nombre": r["nombre"], "equipo": r["equipo"], "esp_sk": float(r["esp_sk"]),
                            "x": {col: float(r[col]) for col in P["columnas"]}} for _, r in t.iterrows()]
            proba["carreras"][c] = "GP de " + GP.get(t["gp"].iloc[0], t["gp"].iloc[0]) + " " + c[:4]
        proba["momentos"][k] = {
            "tipo": P["tipo"], "cols": P["columnas"], "rangos": rangos, "carreras": carreras,
            "etq": {c: ("Lugar en la grilla de largada" if (m == 0 and c == "posicion_al_corte") else ENTRADA[c][0]) for c in P["columnas"]},
            "log": [c for c in P["columnas"] if ENTRADA[c][2] == "log"],
            "max_pos": int(round(P["rangos"]["posicion_al_corte"][1])),
            "lineal": P["terminos"] if P["tipo"] == "lineal" else None,
            "arbol": P["terminos"]["arbol"] if P["tipo"] == "árbol" else None}
    specs_json = json.dumps({k: v.to_dict() for k, v in specs.items()}, ensure_ascii=False, separators=(",", ":"))
    # Sin frase debajo del número (tarjeta 1): el número lleva sus propios rótulos y va en formato doble
    kpis = "".join(f'<div class="kpi"><div class="etq">{e}</div>'
                   + (f'<div class="num">{a}</div><div class="tit">{t}</div>' if t else f'<div class="num doble">{a}</div>')
                   + f'<div class="que">{b}</div><div class="ctx">{c}</div><div class="graf" id="g-kpi-{i}"></div></div>'
                   for i, (e, a, t, b, c) in enumerate(z["kpis"], 1))
    html = (PLANTILLA.replace("__IDEA__", z["idea"]).replace("__KPIS__", kpis)
            .replace("__T_DATOS__", z["t_datos"]).replace("__L_DATOS__", z["l_datos"]).replace("__C_DATOS__", como_datos(N))
            .replace("__T_HALLAZGO__", z["t_hallazgo"]).replace("__L_HALLAZGO__", z["l_hallazgo"])
            .replace("__C_HALLAZGO__", como_hallazgo(corr))
            .replace("__T_MODELO__", z["t_modelo"]).replace("__L_MODELO__", z["l_modelo"]).replace("__L_TABLA__", z["l_tabla"])
            .replace("__C_MODELO__", como_modelo(R, mod, regla)).replace("__C_TABLA__", como_tabla(R))
            .replace("__N_EXPLORA__", n(len(e))).replace("__CARRERAS_EXPLORA__", str(e["c"].nunique()))
            .replace("__C_EXPLORA__", como_explora(R, N, e)).replace("__C_PROBA__", como_proba(R))
            .replace("__FECHA__", time.strftime("%d/%m/%Y"))
            .replace("__SPECS__", specs_json).replace("__PROBA__", json.dumps(proba, ensure_ascii=False)))
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    Path(salida).write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=str(RAIZ / "include" / "output" / "silver"),
                    help="carpeta con los dos CSV del pipeline (si no están, se bajan de Drive como en el notebook)")
    ap.add_argument("--notebook", default=str(RAIZ / "include" / "Entrega3_Modelado_F1.ipynb"))
    ap.add_argument("--salida", default=str(RAIZ / "docs" / "index.html"))
    a = ap.parse_args()
    Path(a.datos).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("1. Celdas de datos y pipeline del notebook…")
    g = ejecutar_notebook(a.notebook, a.datos)
    print(f"2-3. Modelos, validación, test y fórmulas…  ({time.time() - t0:.0f} s)")
    R = calcular(g)
    R["tabla_pc"], R["tabla_res"] = metricas_tabla(R["explora"])
    N, hist, corr = datos_y_hallazgo(g)
    print(f"4. Gráficos y página…  ({time.time() - t0:.0f} s)")
    html = pagina(R, N, hist, corr, a.salida)
    print(f"Listo: {a.salida} ({len(html) / 1e6:.1f} MB) en {time.time() - t0:.0f} s.")
    if AVISOS:
        print(f"\n{len(AVISOS)} aviso(s): la aplicación ya no coincide del todo con el notebook. Revisá los textos.")
