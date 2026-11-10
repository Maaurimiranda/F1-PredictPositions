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
SELECCION_E3 = {"aporte_campeonato": 0.259, "equipo_varianza": 32.6, "ordenar": {0.25: 0.073, 0.5: 0.121, 0.75: 0.163}}
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

        def filas_explora(ajustado, idx, origen, muestra):
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
                "sit": np.select([ya, terminó], ["ya había abandonado", "terminó"], "abandonó después"),
                "muestra": muestra})

        # validación: cada carrera validada la predice el modelo del corte que no la vio
        for e, w in D[m]["cortes"]:
            aj = clone(elegido).fit(X.iloc[e], y.iloc[e])
            R["explora"].append(filas_explora(aj, w, tr, "validación"))
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
        cambio = np.array([nodo(0, fila) for fila in entradas_te.to_numpy()])
        tipo = "árbol"
        terminos = None
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
    filas = pd.DataFrame({"c": te["carrera"].to_numpy(),
                          "gp": te["race_name"].str.replace(" Grand Prix", "", regex=False).to_numpy(),
                          "piloto": te["driver_code"].to_numpy(), "nombre": te["driver_name"].to_numpy(),
                          "equipo": te["team_name"].to_numpy(), "real": te["finish_position"].to_numpy(),
                          **{c: entradas_te[c].round(4).to_numpy() for c in num}})
    return {"tipo": tipo, "expr": expr, "rangos": rangos, "filas": filas, "columnas": num, "terminos": terminos}



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
    cambio = todo["finish_position"] - todo["posicion_al_corte"]
    N = {"snap": len(snapshots), "pcts": int(snapshots["pct_race_complete"].nunique()), "vueltas": len(vueltas),
         "por_momento": len(todo), "carreras": todo["carrera"].nunique(), "pilotos": todo["driver_code"].nunique(),
         "carreras_train": D[0.0]["train"]["carrera"].nunique(), "filas_train": len(D[0.0]["train"]),
         "carreras_test": D[0.0]["test"]["carrera"].nunique(), "filas_test": len(D[0.0]["test"]),
         "abandonos": int((~terminó).sum()), "abandonos_pct": float((~terminó).mean()),
         "cerca": float((cambio.abs() <= 2).mean()), "caida_abandono": float(cambio[~terminó].mean()),
         "grandes_abandono": float(((cambio >= 5) & ~terminó).sum() / (cambio >= 5).sum()),
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
        x=alt.X("a:Q", scale=alt.Scale(domain=[-19, 21], nice=False)), x2="b:Q")
    barras = base.mark_bar(size=13).encode(
        x=alt.X("cambio:Q", scale=alt.Scale(domain=[-19, 21], nice=False),
                axis=alt.Axis(values=list(range(-15, 21, 5)), grid=False),
                title="Puesto de llegada menos lugar de largada (negativo: ganó puestos · positivo: perdió)"),
        y=alt.Y("n:Q", stack="zero", title="Pilotos-carrera"),
        color=alt.Color("fin:N", scale=alt.Scale(domain=["terminó", "abandonó"], range=[CLARO, VINO]),
                        legend=alt.Legend(title=None, orient="top-right", direction="vertical")),
        order=alt.Order("orden:Q"),
        tooltip=[alt.Tooltip("cambio:Q", title="Llegada − largada"), alt.Tooltip("fin:N", title="Situación"),
                 alt.Tooltip("n:Q", title="Pilotos-carrera")])
    nota = alt.Chart(pd.DataFrame({"x": [0], "t": [f"{N['cerca']:.0%} termina a ±2 puestos".replace("%", " %")]})).mark_text(
        dy=-6, fontSize=12, fontWeight=600, color="#5f5d57", baseline="bottom").encode(
        x="x:Q", y=alt.value(0), text="t:N")
    return (banda + barras + nota).properties(
        width=700, height=300,
        title=alt.Title("Cuántos puestos se mueve cada piloto entre la largada y la llegada",
                        subtitle=[f"{N['por_momento']:,} pilotos-carrera de {N['carreras']} carreras, 2022–2026.".replace(",", ".")]))


def g_hallazgo(corr):
    series = ["dónde va en ese momento", "puesto en el campeonato", "puesto del equipo"]
    color = alt.Color("serie:N", scale=alt.Scale(domain=series, range=[NARANJA, "#5f5d57", "#aeaba1"]), legend=None)
    base = alt.Chart(corr).encode(
        x=alt.X("momento:N", sort=MOMS, title="Momento de la carrera", axis=alt.Axis(labelAngle=0, labelPadding=6)),
        y=alt.Y("rho:Q", scale=alt.Scale(domain=[0.4, 1.0]), title="Correlación con el puesto de llegada"),
        color=color)
    tip = [alt.Tooltip("serie:N", title="Qué se mira"), alt.Tooltip("momento:N", title="Momento"),
           alt.Tooltip("rho:Q", title="Correlación (Spearman)", format=".2f")]
    lineas = base.mark_line(strokeWidth=3)
    puntos = base.mark_circle(size=95, opacity=1).encode(tooltip=tip)
    valores = base.transform_filter("datum.serie == 'dónde va en ese momento'").mark_text(
        dy=-14, fontSize=12, fontWeight=600).encode(text=alt.Text("rho:Q", format=".2f"))
    rotulos = base.transform_filter("datum.momento == '75 %'").transform_calculate(
        t="datum.serie + ' · ' + format(datum.rho, '.2f')").mark_text(align="left", dx=12, fontSize=12.5).encode(
        text="t:N")
    return (lineas + puntos + valores + rotulos).properties(
        width=520, height=320, padding={"left": 4, "right": 210, "top": 6, "bottom": 4},
        title=alt.Title("Qué tanto se parece cada dato al puesto final, en cada momento",
                        subtitle=["2022–2025, 1.826 pilotos-carrera por momento (el test no se mira).",
                                  "1 = mismo orden que la llegada; 0 = nada que ver."]))


def g_modelo(R):
    filas = []
    for m in (0.0, 0.25, 0.5, 0.75):
        c, el = R["comparacion"][m], DECISIONES["elegido"][m]
        for quien, v, t in [("«termina donde está»", c["regla"], R["test"][m]["regla"]),
                            ("modelo", c[el], R["test"][m]["modelo"])]:
            filas.append({"momento": ETIQUETA[m], "quien": quien, "muestra": "validación 2022–2025", "mae": v.mean(),
                          "lo": v.mean() - v.std(), "hi": v.mean() + v.std(), "desvio": v.std()})
            filas.append({"momento": ETIQUETA[m], "quien": quien, "muestra": "test 2026", "mae": t,
                          "lo": t, "hi": t, "desvio": None})
    df = pd.DataFrame(filas)
    quienes = ["«termina donde está»", "modelo"]
    X_M = alt.X("momento:N", sort=MOMS, title="Momento de la carrera", axis=alt.Axis(labelAngle=0, labelPadding=6))
    XO = alt.XOffset("quien:N", sort=quienes, scale=alt.Scale(paddingOuter=0.6))
    color = alt.Color("quien:N", scale=alt.Scale(domain=quienes, range=[NARANJA, AZUL]),
                      legend=alt.Legend(title=None, orient="top-right", direction="vertical"))
    tip = [alt.Tooltip("quien:N", title="Quién"), alt.Tooltip("momento:N", title="Momento"),
           alt.Tooltip("muestra:N", title="Dónde se midió"), alt.Tooltip("mae:Q", title="Error medio (puestos)", format=".3f"),
           alt.Tooltip("desvio:Q", title="Desvío entre cortes", format=".3f")]
    base = alt.Chart(df).encode(x=X_M, xOffset=XO, color=color)
    val = base.transform_filter("datum.muestra == 'validación 2022–2025'")
    Y = alt.Y("mae:Q", scale=alt.Scale(domain=[0, 5.5]), title="Error medio (puestos)")
    barras = val.mark_rule(strokeWidth=2.5).encode(y="lo:Q", y2="hi:Q")
    puntos = val.mark_circle(size=120, opacity=1).encode(y=Y, tooltip=tip)
    valores = val.mark_text(dy=-10, fontSize=11.5, fontWeight=600, baseline="bottom").encode(
        y="hi:Q", text=alt.Text("mae:Q", format=".2f"))
    test = base.transform_filter("datum.muestra == 'test 2026'").mark_point(
        shape="diamond", filled=False, size=120, strokeWidth=2).encode(y="mae:Q", tooltip=tip)
    piso = alt.Chart(pd.DataFrame({"y": [R["comparacion"][0.0]["piso"].mean()]}))
    linea_piso = piso.mark_rule(strokeDash=[6, 4], color=PISO, strokeWidth=1.5).encode(y="y:Q")
    texto_piso = piso.transform_calculate(t="'piso: decir siempre el puesto del medio (' + format(datum.y, '.2f') + ')'")\
        .mark_text(align="left", baseline="bottom", dy=-4, x=4, color="#7a776f", fontSize=11.5).encode(y="y:Q", text="t:N")
    return (linea_piso + texto_piso + barras + puntos + valores + test).properties(
        width=640, height=330,
        title=alt.Title("Error medio de cada predicción, en puestos (más abajo, mejor)",
                        subtitle=["Punto: media de los 5 cortes de la validación; barra: un desvío. Rombo: las 14 carreras de 2026."]))


def g_explora(e, inicial):
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
    carrera = alt.selection_point(name="carrera_e", fields=["c"], value=[{"c": inicial["c"]}], on="click",
                                  clear=False, toggle=False)
    filtro = ("datum.m == momento_e && (temporada_e == 'Todas' || datum.temp == temporada_e) && "
              "(piloto_e == 'Todos' || datum.piloto == piloto_e)")
    errores = dict(em="abs(datum.pred - datum.real)", er="abs(datum.pos - datum.real)")
    pano = (alt.Chart(e).transform_filter(filtro).transform_calculate(**errores)
              .transform_aggregate(modelo="mean(em)", regla="mean(er)", n="count()",
                                   groupby=["c", "gp", "fecha", "temp", "muestra"]))
    fin_test = (pd.to_datetime(e["fecha"].max()) + pd.Timedelta(days=12)).strftime("%Y-%m-%d")
    test = pd.DataFrame({"a": ["2026-01-01"], "b": [fin_test], "t": ["test 2026"]})
    X_F = alt.X("fecha:T", title=None, axis=alt.Axis(format="%Y", tickCount="year", grid=False))
    banda = alt.Chart(test).mark_rect(color="#f2efe6").encode(x=alt.X("a:T"), x2="b:T")
    banda_t = alt.Chart(test).mark_text(align="left", baseline="top", dx=6, y=4, color="#7a776f", fontSize=11.5).encode(
        x="a:T", text="t:N")
    tip_p = [alt.Tooltip("gp:N", title="Carrera"), alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y"),
             alt.Tooltip("muestra:N", title="Predicción"), alt.Tooltip("n:Q", title="Pilotos"),
             alt.Tooltip("modelo:Q", title="Error del modelo", format=".2f"),
             alt.Tooltip("regla:Q", title="Error de «termina donde está»", format=".2f")]
    une = pano.mark_rule(color="#d9d6cd").encode(x=X_F, y="regla:Q", y2="modelo:Q")
    regla = pano.mark_tick(color=NARANJA, thickness=2.5, size=11, opacity=0.9).encode(x=X_F, y="regla:Q", tooltip=tip_p)
    modelo = pano.mark_circle(color=AZUL, opacity=1, cursor="pointer").encode(
        x=X_F, y=alt.Y("modelo:Q", title="Error medio (puestos)", scale=alt.Scale(zero=True)),
        size=alt.condition(carrera, alt.value(190), alt.value(48)),
        stroke=alt.condition(carrera, alt.value(TINTA), alt.value(None)), strokeWidth=alt.value(2),
        tooltip=tip_p).add_params(carrera)
    panorama = (banda + banda_t + une + regla + modelo).properties(
        width=760, height=210,
        title=alt.Title("Todas las carreras con predicción fuera de muestra",
                        subtitle=["Cada punto azul es una carrera: el error medio del modelo. La raya naranja, el de «termina donde está». "
                                  "Hacé clic en una carrera para verla abajo."]))

    sel = alt.Chart(e).transform_filter("datum.m == momento_e").transform_filter(carrera)
    cab = (sel.transform_calculate(**errores)
              .transform_aggregate(modelo="mean(em)", regla="mean(er)", groupby=["gp", "temp", "muestra"])
              .transform_calculate(t="datum.gp + ' ' + datum.temp + ' · ' + momento_e + ' · ' + datum.muestra + "
                                     "' — error medio: modelo ' + format(datum.modelo, '.2f') + ', «termina donde está» ' + "
                                     "format(datum.regla, '.2f')")
              .mark_text(align="left", baseline="top", x=0, y=4, fontSize=14, fontWeight=600, color=TINTA)
              .encode(text="t:N").properties(width=760, height=26))
    det = sel.transform_calculate(etq="datum.piloto + ' · ' + datum.equipo",
                                  tuyo="piloto_e != 'Todos' && datum.piloto == piloto_e",
                                  error="abs(datum.pred - datum.real)")
    Y_D = alt.Y("etq:N", sort=alt.EncodingSortField("real", order="ascending"), title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=11.5, ticks=False, domain=False))
    X_D = dict(scale=X_PUESTO, axis=alt.Axis(values=list(range(1, 23)), orient="top", grid=True),
               title="Puesto (1 = adelante)")
    tip_d = [alt.Tooltip("nombre:N", title="Piloto"), alt.Tooltip("equipo:N", title="Equipo"),
             alt.Tooltip("m:N", title="Momento"), alt.Tooltip("pos:Q", title="Dónde iba"),
             alt.Tooltip("pred:Q", title="Lo que dijo el modelo", format=".1f"),
             alt.Tooltip("esp:Q", title="Posición esperada (antes de ordenar)", format=".2f"),
             alt.Tooltip("real:Q", title="Dónde terminó"), alt.Tooltip("sit:N", title="Situación"),
             alt.Tooltip("error:Q", title="Error del modelo (puestos)", format=".1f"),
             alt.Tooltip("muestra:N", title="Predicción")]
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
    return alt.vconcat(panorama, cab, detalle, spacing=14).add_params(momento, temporada, piloto)


ENTRADA = {  # cómo se muestra cada entrada del modelo en los deslizadores de «Probá el modelo»
    "posicion_al_corte": ("Dónde va (puesto)", 1, None),
    "vueltas_detras": ("Vueltas detrás del líder", 1, None),
    "tendencia_10": ("Puestos ganados en las últimas 10 vueltas", 1, None),
    "gap_qualy_pct": ("Qué tan lejos de la pole quedó en la qualy (%)", 0.1, None),
    "log_puntos_equipo": ("Puntos del equipo antes de la carrera", 1, "log"),
    "neumatico_vs_carrera": ("Vueltas de su neumático, contra la mitad del pelotón", 0.5, None),
    "driver_standing_before": ("Puesto en el campeonato antes de la carrera", 1, None),
    "log_puntos": ("Puntos del piloto antes de la carrera", 0.5, "log"),
}


def a_pantalla(col, v):
    """De la unidad del modelo a la del deslizador (los puntos entran al modelo como log(1 + puntos)),
    redondeado al paso del deslizador."""
    x = np.expm1(v) if ENTRADA[col][2] == "log" else v
    paso = ENTRADA[col][1]
    return round(round(float(x) / paso) * paso, 2)


def piloto_inicial(R, c):
    """Un piloto de mitad de pelotón cuyos valores reales caen, en los cuatro momentos, dentro de lo que vio el modelo."""
    t = R["proba"][0.25]["filas"]
    for p in t[t["c"] == c].sort_values("posicion_al_corte")["piloto"]:
        dentro = all(lo <= fila[col] <= hi for P in R["proba"].values()
                     for _, fila in P["filas"][(P["filas"]["c"] == c) & (P["filas"]["piloto"] == p)].iterrows()
                     for col, (lo, hi) in P["rangos"].items())
        pos = t[(t["c"] == c) & (t["piloto"] == p)]["posicion_al_corte"].iloc[0]
        if dentro and 6 <= pos <= 12:
            return p
    return t[t["c"] == c]["piloto"].iloc[0]


def g_proba(m, P, inicial):
    cols, filas = P["columnas"], P["filas"]
    k = CORTO[m]
    ini = filas[(filas["c"] == inicial["c"]) & (filas["piloto"] == inicial["piloto"])].iloc[0]
    params = [alt.param(name=f"carrera_{k}", value=inicial["c"]), alt.param(name=f"piloto_{k}", value=inicial["piloto"])]
    libres = [alt.param(name=f"carrera_{k}", value=inicial["c"]), alt.param(name=f"piloto_{k}", value=inicial["piloto"])]
    calc = {}
    for col in cols:
        paso = ENTRADA[col][1]
        lo, hi = ((np.expm1(v) if ENTRADA[col][2] == "log" else v) for v in P["rangos"][col])
        lo, hi = np.floor(lo / paso + 1e-9) * paso, np.ceil(hi / paso - 1e-9) * paso
        P["rangos_pantalla"] = P.get("rangos_pantalla", {}) | {col: (round(float(lo), 2), round(float(hi), 2), paso)}
        params.append(alt.param(name=f"s{k}_{col}", value=round(float(a_pantalla(col, ini[col])), 2),
                                bind=alt.binding_range(min=float(lo), max=float(hi), step=paso, name=" ",
                                                       element=f"#d{k}-{col}")))
        libres.append(alt.param(name=f"s{k}_{col}", value=round(float(a_pantalla(col, ini[col])), 2)))
        valor = f"log(1 + s{k}_{col})" if ENTRADA[col][2] == "log" else f"s{k}_{col}"
        calc[f"x_{col}"] = f"datum.piloto == piloto_{k} ? {valor} : datum.{col}"
    base = (alt.Chart(filas).transform_filter(f"datum.c == carrera_{k}").transform_calculate(**calc)
              .transform_calculate(esp=P["expr"], tuyo=f"datum.piloto == piloto_{k}")
              .transform_window(orden="row_number()", sort=[alt.SortField("esp"), alt.SortField("x_posicion_al_corte")]))
    if P["tipo"] == "lineal":
        base = base.transform_calculate(dijo="datum.orden", rotulo="'P' + datum.orden")
        frase = (f"datum.piloto + ': va P' + datum.x_posicion_al_corte + ' → el modelo lo pone P' + datum.orden + "
                 f"' (posición esperada ' + format(datum.esp, '.1f') + ')'")
    else:
        base = base.transform_calculate(dijo="datum.esp", rotulo="format(datum.esp, '.1f')")
        frase = (f"datum.piloto + ': larga P' + datum.x_posicion_al_corte + ' → el modelo espera que llegue en el puesto ' + "
                 f"format(datum.esp, '.1f')")
    cambio = ("(datum.dijo < datum.x_posicion_al_corte - 0.5 ? 'gana ' : datum.dijo > datum.x_posicion_al_corte + 0.5 ? "
              "'pierde ' : '') + (abs(datum.dijo - datum.x_posicion_al_corte) < 0.5 ? 'queda donde está' : "
              "format(abs(datum.dijo - datum.x_posicion_al_corte), '.0f') + "
              "(round(abs(datum.dijo - datum.x_posicion_al_corte)) == 1 ? ' puesto' : ' puestos'))")
    cab = (base.transform_filter("datum.tuyo").transform_calculate(t=frase, t2=cambio)
               .mark_text(align="left", baseline="top", x=0, y=2, fontSize=16, fontWeight=700, color=AZUL)
               .encode(text="t:N"))
    cab2 = (base.transform_filter("datum.tuyo").transform_calculate(t2=cambio)
                .mark_text(align="left", baseline="top", x=0, y=26, fontSize=13, color="#5f5d57")
                .encode(text=alt.Text("t2:N")))
    cabecera = (cab + cab2).properties(width=660, height=46)
    filas_ch = base.transform_calculate(etq="datum.piloto + ' · ' + datum.equipo")
    Y = alt.Y("etq:N", sort=alt.EncodingSortField("orden", order="ascending"), title=None,
              axis=alt.Axis(labelLimit=220, labelFontSize=11.5, ticks=False, domain=False))
    XQ = dict(scale=X_PUESTO, axis=alt.Axis(values=list(range(1, 23)), orient="top", grid=True),
              title="Puesto (1 = adelante)")
    tip = [alt.Tooltip("nombre:N", title="Piloto"), alt.Tooltip("equipo:N", title="Equipo"),
           alt.Tooltip("x_posicion_al_corte:Q", title="Dónde va"),
           alt.Tooltip("esp:Q", title="Posición esperada", format=".2f"),
           alt.Tooltip("dijo:Q", title="Lo que dice el modelo", format=".1f")]
    resalte = filas_ch.transform_filter("datum.tuyo").mark_bar(color="#fdf0c4").encode(
        y=Y, x=alt.X(datum=0.5, scale=X_PUESTO), x2=alt.X2(datum=22.5))
    mov = filas_ch.mark_rule(color="#d9d6cd", strokeWidth=2).encode(y=Y, x=alt.X("x_posicion_al_corte:Q", **XQ), x2="esp:Q")
    va = filas_ch.mark_circle(color=NARANJA, size=80, opacity=1).encode(y=Y, x=alt.X("x_posicion_al_corte:Q", **XQ), tooltip=tip)
    esp = filas_ch.mark_circle(color=AZUL, size=95, opacity=1).encode(y=Y, x=alt.X("esp:Q", **XQ), tooltip=tip)
    rot = filas_ch.mark_text(align="left", dx=8, fontSize=11.5, fontWeight=600, color=AZUL).encode(
        y=Y, x=alt.X(datum=22.5, scale=X_PUESTO), text="rotulo:N")
    cuerpo = (resalte + mov + va + esp + rot).properties(width=660, height=alt.Step(19))
    detalle = alt.vconcat(cuerpo).add_params(*libres)

    # la vista principal: la tabla de posiciones que arma el modelo, de arriba hacia abajo
    lineal = P["tipo"] == "lineal"
    t = filas_ch.transform_calculate(
        puesto="'P' + datum.orden" if lineal else "format(datum.esp, '.1f')",
        quien="datum.piloto + '  ' + datum.nombre",
        va=f"'{'va' if m else 'larga'} P' + datum.x_posicion_al_corte",
        flecha=("datum.orden < datum.x_posicion_al_corte ? '▲ ' + (datum.x_posicion_al_corte - datum.orden) : "
                "datum.orden > datum.x_posicion_al_corte ? '▼ ' + (datum.orden - datum.x_posicion_al_corte) : '='"))
    YT = alt.Y("orden:O", axis=None)
    col_x = {"puesto": 8, "quien": 70, "equipo": 300, "va": 470, "flecha": 545}
    capas = [t.transform_filter("datum.tuyo").mark_rect(color="#fdf0c4").encode(y=YT, x=alt.value(0), x2=alt.value(600))]
    for campo, x in col_x.items():
        estilo_txt = dict(fontSize=13.5, fontWeight=700, color=AZUL) if campo == "puesto" else \
                     dict(fontSize=13, color=TINTA if campo in ("quien", "flecha") else "#7a776f")
        capas.append(t.mark_text(align="left", baseline="middle", **estilo_txt).encode(
            y=YT, x=alt.value(x), text=f"{campo}:N", tooltip=tip))
    encab = alt.Chart(pd.DataFrame({"x": list(col_x.values()),
                                    "t": ["Puesto" if lineal else "Esperado", "Piloto", "Equipo", "Ahora", "Cambio"]})) \
        .mark_text(align="left", fontSize=11, fontWeight=700, color="#8a877f").encode(x=alt.X("x:Q", scale=None, axis=None), text="t:N") \
        .properties(width=600, height=16)
    tabla_ch = alt.layer(*capas).properties(width=600, height=alt.Step(23))
    simple = alt.vconcat(cabecera, encab, tabla_ch, spacing=4).add_params(*params)

    aportes = None
    if lineal:   # cómo se arma el número de tu piloto: un aporte por entrada, en puestos
        T = P["terminos"]
        calc_a = {f"a_{c}": f"{w} * (datum.x_{c} - {mu}) / {sd}" for c, w, mu, sd in T["partes"]}
        etiqueta = " : ".join(f"datum.k == 'a_{c}' ? '{ENTRADA[c][0]}'" for c, *_ in T["partes"]) + " : 'punto de partida (constante)'"
        a = (base.transform_filter("datum.tuyo").transform_calculate(a_base=str(T["intercepto"]), **calc_a)
                 .transform_fold([f"a_{c}" for c, *_ in T["partes"]] + ["a_base"], as_=["k", "v"])
                 .transform_calculate(etq=etiqueta, lado="datum.v < 0 ? 'lo adelanta' : 'lo atrasa'"))
        Ya = alt.Y("etq:N", sort=alt.EncodingSortField("v", op="sum", order="ascending"), title=None,
                   axis=alt.Axis(labelLimit=330, ticks=False, domain=False))
        barras = a.mark_bar(height=16, color=AZUL).encode(
            y=Ya, x=alt.X("v:Q", title="Puestos (negativo: lo adelanta · positivo: lo atrasa)"),
            opacity=alt.condition("datum.v < 0", alt.value(1), alt.value(0.45)),
            tooltip=[alt.Tooltip("etq:N", title="Entrada"), alt.Tooltip("v:Q", title="Aporte (puestos)", format="+.2f")])
        nums = a.mark_text(align="left", dx=4, fontSize=11.5, color=TINTA).encode(
            y=Ya, x=alt.value(0), text=alt.Text("v:Q", format="+.2f"))
        cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a877f").encode(x="x:Q")
        aportes = (cero + barras + nums).properties(
            width=420, height=alt.Step(26),
            title=alt.Title("Cómo arma el modelo el número de tu piloto",
                            subtitle=["Cada barra: cuánto lo mueve esa entrada, comparado con un piloto promedio de 2022–2025.",
                                      "Posición esperada = dónde va + la suma de todas las barras."])).add_params(*libres)
    return {"simple": simple, "detalle": detalle, "aportes": aportes}


PLANTILLA = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>¿Dónde termina un piloto?</title>
<meta name="description" content="Proyecto integrador de Ciencia de Datos (UTN FRM, 2026): predecir el puesto de llegada de cada piloto de Fórmula 1 en cuatro momentos de la carrera.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/vega@5.30.0"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5.20.1"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6.26.0"></script>
<style>
:root{--fondo:#f6f5f1;--papel:#ffffff;--tinta:#1d1c1a;--suave:#5f5d57;--tenue:#8a877f;--borde:#e3e0d7;
--rojo:#c8102e;--azul:#2563c9;--naranja:#e2711d;--vino:#a3326f;--amarillo:#fdf0c4}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;background:var(--fondo);color:var(--tinta);font:16px/1.6 Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
a{color:var(--azul)}
code{font-size:.86em;background:#efede6;padding:1px 5px;border-radius:4px}
header.barra{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);border-bottom:1px solid var(--borde)}
.barra .dentro{max-width:1040px;margin:0 auto;padding:10px 20px;display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.marca{font-weight:800;letter-spacing:-.01em;display:flex;align-items:center;gap:8px;white-space:nowrap}
.marca .bandera{width:18px;height:18px;border-radius:3px;background:conic-gradient(#1d1c1a 25%,#fff 0 50%,#1d1c1a 0 75%,#fff 0) 0 0/9px 9px;border:1px solid #1d1c1a}
nav{display:flex;align-items:center;gap:4px;flex-wrap:wrap;font-size:14px}
nav .grupo{color:var(--tenue);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin:0 4px 0 12px}
nav a{color:var(--suave);text-decoration:none;padding:4px 9px;border-radius:999px}
nav a:hover{background:#f0eee7;color:var(--tinta)}
nav a.activo{background:var(--tinta);color:#fff}
main{max-width:1040px;margin:0 auto;padding:0 20px 40px}
section{padding:54px 0 18px;border-bottom:1px solid var(--borde)}
section:last-of-type{border-bottom:0}
.parte{display:inline-flex;align-items:center;gap:8px;font-size:12px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;color:var(--suave)}
.parte::before{content:"";width:22px;height:4px;border-radius:2px;background:var(--rojo)}
h1{font-size:clamp(32px,5vw,50px);line-height:1.08;letter-spacing:-.025em;margin:14px 0 18px;max-width:20ch}
h2{font-size:clamp(22px,3vw,29px);line-height:1.22;letter-spacing:-.015em;margin:10px 0 12px;max-width:34ch}
h4{font-size:15px;margin:22px 0 6px}
.idea{font-size:clamp(18px,2.2vw,22px);line-height:1.5;max-width:46ch;color:#2b2a26;margin:0 0 26px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:0 0 22px}
.kpi{background:var(--papel);border:1px solid var(--borde);border-radius:14px;padding:18px 18px 16px;border-top:4px solid var(--azul)}
.kpi:nth-child(3){border-top-color:var(--naranja)}
.kpi .num{font-size:34px;font-weight:800;letter-spacing:-.02em;line-height:1.1;font-variant-numeric:tabular-nums}
.kpi .que{color:var(--suave);font-size:14px;line-height:1.45;margin-top:6px}
.guia{color:var(--suave);font-size:15px;max-width:70ch}
.lectura{max-width:72ch;color:#2b2a26;margin:0 0 16px}
.grafico{background:var(--papel);border:1px solid var(--borde);border-radius:14px;padding:14px;overflow-x:auto;margin:6px 0 14px}
.grafico .vega-embed{display:block}
.leyenda{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--suave);margin:4px 0 8px}
.leyenda span{display:inline-flex;align-items:center;gap:6px}
.pt{width:11px;height:11px;border-radius:50%;display:inline-block}
.rombo{width:10px;height:10px;transform:rotate(45deg);display:inline-block;background:var(--tinta)}
.cruz{color:var(--vino);font-weight:800;font-size:15px;line-height:1}
.franja{width:22px;height:11px;background:var(--amarillo);display:inline-block;border-radius:2px}
details.como{background:var(--papel);border:1px solid var(--borde);border-radius:14px;margin:6px 0 10px}
details.como>summary{cursor:pointer;list-style:none;padding:13px 18px;font-weight:700;display:flex;align-items:center;gap:10px}
details.como>summary::-webkit-details-marker{display:none}
details.como>summary::before{content:"+";width:22px;height:22px;border-radius:50%;background:var(--tinta);color:#fff;display:inline-grid;place-items:center;font-size:16px;line-height:1}
details.como[open]>summary::before{content:"–"}
details.como>summary small{font-weight:400;color:var(--tenue)}
details.como .cuerpo{padding:0 20px 16px;font-size:15px;color:#2b2a26}
details.como .cuerpo p{max-width:80ch}
.tabla{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;font-size:13.5px;min-width:100%}
th,td{padding:7px 10px;border-bottom:1px solid #ecebe5;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--suave);background:#faf9f6}
table.num td:not(:first-child),table.num th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td .n{color:var(--tenue);font-size:12px}
.nota{font-size:14px;color:var(--suave)}
.controles{display:flex;flex-wrap:wrap;gap:12px 26px;align-items:flex-end;background:var(--papel);border:1px solid var(--borde);border-radius:14px;padding:14px 16px;margin:8px 0}
.control{display:flex;flex-direction:column;gap:6px;font-size:13px}
.control>.titulo{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--tenue);font-weight:700}
.vega-bind-name{display:none}
.vega-bind label,.vega-bind{font-size:14px}
.vega-bind select,.control select{font:inherit;font-size:14px;padding:6px 8px;border:1px solid var(--borde);border-radius:8px;background:#fff;max-width:280px}
.vega-bind-radio{display:inline-flex;gap:4px;flex-wrap:wrap}
.vega-bind-radio input{display:none}
.vega-bind-radio label{padding:5px 11px;border:1px solid var(--borde);border-radius:999px;cursor:pointer;background:#fff}
.vega-bind-radio label:has(input:checked){background:var(--tinta);color:#fff;border-color:var(--tinta)}
.pastillas{display:inline-flex;gap:4px;flex-wrap:wrap}
.pastillas button{font:inherit;font-size:14px;padding:5px 11px;border:1px solid var(--borde);border-radius:999px;background:#fff;cursor:pointer}
.pastillas button.on{background:var(--tinta);color:#fff;border-color:var(--tinta)}
button.volver{font:inherit;font-size:14px;padding:7px 12px;border-radius:8px;border:1px solid var(--tinta);background:#fff;cursor:pointer}
button.volver:hover{background:var(--tinta);color:#fff}
.deslizadores{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:10px;margin:10px 0}
.desl{background:var(--papel);border:1px solid var(--borde);border-radius:12px;padding:10px 12px}
.desl .etq{display:block;font-size:13.5px;font-weight:600;margin-bottom:2px}
.desl .rango{display:block;font-size:12px;color:var(--tenue)}
.desl input[type=range]{width:72%;accent-color:var(--tinta);vertical-align:middle}
.desl .vega-bind-value{font-weight:700;font-variant-numeric:tabular-nums;margin-left:8px}
.modelo-usado{font-size:13.5px;color:var(--suave);margin:8px 0 0}
.aviso{background:#fff6e0;border:1px solid #f0d48a;color:#6b4e00;border-radius:10px;padding:10px 14px;font-size:14px;margin:8px 0}
.queToca{font-size:14px;color:var(--suave);max-width:80ch}
footer{max-width:1040px;margin:0 auto;padding:26px 20px 50px;color:var(--tenue);font-size:13px;border-top:1px solid var(--borde)}
@media (max-width:760px){.kpis{grid-template-columns:1fr}nav .grupo{display:none}section{padding-top:40px}}
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
  <h1>¿Dónde va a terminar cada piloto de Fórmula 1?</h1>
  <p class="idea">__IDEA__</p>
  <div class="kpis">__KPIS__</div>
  <p class="guia">Predecimos el puesto oficial de llegada en cuatro momentos: antes de largar y cuando el líder completa
  el 25, el 50 y el 75 % de las vueltas, con lo que se sabe hasta ese momento. Las tres secciones que siguen cuentan qué
  encontramos; cada una tiene un «Cómo lo sabemos» con la evidencia. Al final, los datos y el modelo quedan en tus manos.</p>
</section>

<section id="datos">
  <span class="parte">1 · Los datos</span>
  <h2>__T_DATOS__</h2>
  <p class="lectura">__L_DATOS__</p>
  <div class="grafico" id="g-datos"></div>
  <details class="como"><summary>Cómo lo sabemos <small>pipeline, filas, filtros y fuga</small></summary><div class="cuerpo">__C_DATOS__</div></details>
</section>

<section id="hallazgo">
  <span class="parte">2 · Lo que encontramos</span>
  <h2>__T_HALLAZGO__</h2>
  <p class="lectura">__L_HALLAZGO__</p>
  <div class="grafico" id="g-hallazgo"></div>
  <details class="como"><summary>Cómo lo sabemos <small>una hipótesis que se sostuvo y una que no</small></summary><div class="cuerpo">__C_HALLAZGO__</div></details>
</section>

<section id="modelo">
  <span class="parte">3 · El modelo</span>
  <h2>__T_MODELO__</h2>
  <p class="lectura">__L_MODELO__</p>
  <div class="grafico" id="g-modelo"></div>
  <details class="como"><summary>Cómo lo sabemos <small>métrica, validación, comparación, test, errores y en qué se apoya</small></summary><div class="cuerpo">__C_MODELO__</div></details>
</section>

<section id="explora">
  <span class="parte">En tus manos · Explorá</span>
  <h2>__CARRERAS_EXPLORA__ carreras, cada piloto con lo que dijo el modelo y lo que pasó</h2>
  <p class="lectura">Arriba, el panorama: cada carrera con su error medio. Elegí el momento, achicá a una temporada o seguí a
  un piloto, y hacé clic en una carrera para abrirla abajo, piloto por piloto. Pasá el mouse por cualquier punto para ver el
  detalle, incluida la predicción. Son __N_EXPLORA__ predicciones hechas por un modelo que no había visto esa carrera.</p>
  <div class="controles">
    <div class="control"><span class="titulo">Momento</span><div id="c-momento"></div></div>
    <div class="control"><span class="titulo">Temporada</span><div id="c-temporada"></div></div>
    <div class="control"><span class="titulo">Seguir a un piloto</span><div id="c-piloto"></div></div>
  </div>
  <div class="leyenda">
    <span><i class="pt" style="background:var(--naranja)"></i>dónde iba en ese momento (lo que dice «termina donde está»)</span>
    <span><i class="pt" style="background:var(--azul)"></i>lo que dijo el modelo</span>
    <span><i class="rombo"></i>dónde terminó</span>
    <span><b class="cruz">✕</b>abandonó</span>
    <span><i class="franja"></i>el piloto que seguís</span>
  </div>
  <div class="grafico" id="g-explora"></div>
  <p class="queToca"><strong>Qué toca cada control.</strong> El momento filtra las filas de los dos gráficos. La temporada y el
  piloto filtran las filas antes de promediar, así que cambian los puntos del panorama (con un piloto, cada punto es su error en
  esa carrera). El clic es una selección: filtra el detalle a esa carrera y agranda su punto. El piloto que seguís, además,
  se marca con la franja amarilla.</p>
</section>

<section id="proba">
  <span class="parte">En tus manos · Probá el modelo</span>
  <h2>Armá la situación de un piloto y mirá en qué puesto lo hace terminar el modelo</h2>
  <p class="lectura">Partí de un piloto real de una carrera de 2026 (el modelo no las vio al entrenar) y mové sus valores. El
  modelo recalcula a toda la carrera con su fórmula: los demás pilotos quedan como estaban y, del 25 % en adelante, se
  reordena a todos 1, 2, 3… Los deslizadores sólo llegan hasta donde llegaron los datos de entrenamiento; si el valor real de
  un piloto queda afuera, la página avisa.</p>
  <div class="controles">
    <div class="control"><span class="titulo">Momento</span><div class="pastillas" id="p-momento">
      <button data-k="0">Antes de largar</button><button data-k="25" class="on">25 %</button><button data-k="50">50 %</button><button data-k="75">75 %</button></div></div>
    <div class="control"><span class="titulo">Carrera de 2026</span><select id="p-carrera"></select></div>
    <div class="control"><span class="titulo">Piloto</span><select id="p-piloto"></select></div>
    <div class="control"><button type="button" class="volver" id="p-volver">Volver a sus valores reales</button></div>
  </div>
  __PANELES__
</section>
</main>
<footer>
  Grupo 8 · 5K09 · Luconi Martina, Olea Facundo, Miranda Mauricio, Mezzabota Tomás, De Coninck Ramiro, XXXX Victoria.<br>
  Datos: corrida del pipeline del 23/09/2026 (Jolpica-F1, FastF1, OpenF1). Página generada con <code>app/construir_app.py</code>
  el __FECHA__ · <a href="https://github.com/Maaurimiranda/f1-predictpositions">repositorio</a>.
</footer>
<script>
const SPECS = __SPECS__;
const PRESETS = __PRESETS__;
const INICIAL = __INICIAL_P__;
const OPC = {actions: false, renderer: "svg",
  formatLocale: {decimal: ",", thousands: ".", grouping: [3], currency: ["$", ""]}};
const vistas = {};
const listo = Promise.all(Object.entries(SPECS).map(([id, spec]) =>
  vegaEmbed("#" + id, spec, OPC).then(r => { vistas[id] = r.view; })));

// Probá el modelo: el momento muestra un panel; carrera y piloto cargan los valores reales en los deslizadores
const selC = document.getElementById("p-carrera"), selP = document.getElementById("p-piloto");
const num = x => (Math.round(x * 10) / 10).toString().replace(".", ",");
for (const [c, d] of Object.entries(PRESETS.carreras)) selC.add(new Option(d.gp, c));
function pilotos(c, elegido) {
  selP.innerHTML = "";
  for (const [p, txt] of PRESETS.carreras[c].pilotos) selP.add(new Option(txt, p));
  if (elegido) selP.value = elegido;
}
async function cargar() {
  const c = selC.value, p = selP.value;
  for (const k of Object.keys(PRESETS.valores)) {
    const v = vistas["g-proba-" + k], real = PRESETS.valores[k][c][p], avisos = [];
    v.signal("carrera_" + k, c).signal("piloto_" + k, p);
    for (const [col, x] of Object.entries(real)) {
      const [lo, hi] = PRESETS.rangos[k][col], y = Math.min(hi, Math.max(lo, x));
      if (y !== x) avisos.push(`<strong>${PRESETS.etiquetas[k][col]}</strong>: su valor real es ${num(x)}, pero al entrenar ` +
        `el modelo sólo vio valores de ${num(lo)} a ${num(hi)}. El deslizador queda en ${num(y)}.`);
      v.signal("s" + k + "_" + col, y);
    }
    await v.runAsync();
    const caja = document.getElementById("aviso-" + k);
    caja.innerHTML = avisos.length ? "⚠ Fuera de lo que vio el modelo. " + avisos.join(" ") : "";
    caja.hidden = !avisos.length;
  }
}
document.querySelectorAll("#p-momento button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#p-momento button").forEach(x => x.classList.toggle("on", x === b));
  document.querySelectorAll(".panel-proba").forEach(pn => { pn.hidden = pn.dataset.k !== b.dataset.k; });
}));
selC.addEventListener("change", () => { pilotos(selC.value); cargar(); });
selP.addEventListener("change", cargar);
document.getElementById("p-volver").addEventListener("click", cargar);
selC.value = INICIAL.c; pilotos(INICIAL.c, INICIAL.piloto);
listo.then(() => {
  for (const k of Object.keys(PRESETS.valores)) {
    const v = vistas["g-proba-" + k];
    const otras = ["-det", "-apo"].map(s => vistas["g-proba-" + k + s]).filter(Boolean);
    const nombres = ["carrera_" + k, "piloto_" + k, ...Object.keys(PRESETS.rangos[k]).map(c => "s" + k + "_" + c)];
    for (const nm of nombres) v.addSignalListener(nm, (_, x) => otras.forEach(o => o.signal(nm, x).runAsync()));
  }
  return cargar();
});

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
    C, T, el = R["comparacion"], R["test"], DECISIONES["elegido"]
    M = (0.0, 0.25, 0.5, 0.75)
    mod = {m: C[m][el[m]].mean() for m in M}
    regla = {m: C[m]["regla"].mean() for m in M}
    ventaja = {m: regla[m] - mod[m] for m in M}
    gana_test = [m for m in M if T[m]["modelo"] < T[m]["regla"]]
    rho = corr.pivot(index="momento", columns="serie", values="rho")
    rv = lambda mm: rho.loc[ETIQUETA[mm], "dónde va en ese momento"]
    z = {}
    z["idea"] = (f"Dónde va un piloto dice casi todo sobre dónde termina; el modelo afina esa posición con lo que pasa en "
                 f"pista y erra entre {f(mod[0.0], 1)} puestos antes de largar y {f(mod[0.75], 1)} al 75 % de la carrera, "
                 f"siempre algo menos que suponer que el piloto termina donde está.")
    z["kpis"] = [(f"{f(mod[0.0], 1)} → {f(mod[0.75], 1)}", "puestos de error del modelo, de antes de largar al 75 % de la carrera (validación 2022–2025)"),
                 (f(ventaja[0.0]), f"puestos menos que «termina donde está» antes de largar, en los 5 cortes; al 75 % la ventaja baja a {f(ventaja[0.75])}"),
                 (f"{len(gana_test)} de 4", "momentos en que esa ventaja se sostuvo en las 14 carreras de 2026: "
                  + " y ".join(ETIQUETA[m].lower() if m == 0 else f"al {ETIQUETA[m]}" for m in gana_test))]
    # tallo
    z["t_datos"] = ("La mitad de los pilotos termina a dos puestos o menos de donde larga; los que se alejan mucho son, "
                    "sobre todo, los que abandonan")
    z["l_datos"] = (f"La franja del medio junta al {pct(N['cerca'])} de los {n(N['por_momento'])} pilotos-carrera: llegaron a "
                    f"dos puestos o menos de su lugar de largada. A la derecha está lo que ningún dato anterior anticipa: los "
                    f"{n(N['abandonos'])} abandonos ({pct(N['abandonos_pct'])}) caen {f(N['caida_abandono'], 1)} puestos de "
                    f"media y son {pct(N['grandes_abandono'])} de los que pierden cinco o más. Por eso medimos el error en "
                    f"puestos, con el error absoluto: esos saltos existen, pero no deberían mandar.")
    z["t_hallazgo"] = ("A partir del 25 % de la carrera, el lugar en el que va un piloto dice más de su llegada que el "
                       "campeonato y el equipo")
    z["l_hallazgo"] = (f"Antes de largar, la grilla ({f(rv(0.0))}) y el campeonato "
                       f"({f(rho.loc[ETIQUETA[0.0], 'puesto en el campeonato'])}) dicen casi lo mismo. Después, la línea "
                       f"naranja sube en cada momento hasta {f(rv(0.75))} al 75 %, mientras que lo que se sabía antes de "
                       f"largar no cambia. Por eso hay un modelo por momento, y por eso «termina donde está» es el rival "
                       f"a vencer: cada vez es más difícil de mejorar.")
    momentos_test = " y ".join(ETIQUETA[m].lower() if m == 0 else f"al {ETIQUETA[m]}" for m in gana_test)
    z["t_modelo"] = (f"El modelo erra entre {f(mod[0.0], 1)} y {f(mod[0.75], 1)} puestos y le gana a «termina donde está» "
                     f"en los cuatro momentos; en 2026, la ventaja se sostuvo {momentos_test}")
    z["l_modelo"] = (f"Mirá la distancia entre cada punto azul y su naranja: es lo que agrega el modelo. Antes de largar le "
                     f"saca {f(ventaja[0.0])} puestos a la regla; al 75 % apenas {f(ventaja[0.75])}, menos que el desvío de "
                     f"cualquiera de los dos. Los rombos son el test: en 2026 el modelo erró {f(T[0.0]['modelo'])} antes de "
                     f"largar contra {f(T[0.0]['regla'])} de la regla, y al 75 % perdió ({f(T[0.75]['modelo'])} contra "
                     f"{f(T[0.75]['regla'])}). Los dos quedan muy lejos del piso.")
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
<p>Datos: las {n(N['por_momento'])} filas de antes de largar. Transformación: llegada − largada, contada por valor y por
situación. Marca: barras apiladas. Color: abandonó (estado distinto de «Finished» o «+N Lap») contra terminó; el mismo
color marca a los que abandonaron en toda la página.</p>"""


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
<h4>Cómo se construyó el gráfico</h4>
<p>Datos: 2022–2025 (el test no se mira), un valor por momento. Transformación: la correlación de Spearman entre el puesto de
llegada y cada dato, que compara órdenes y no distancias. «Dónde va» es la posición rehecha en cada corte (antes de largar,
la grilla); el puesto del equipo sale de ordenar a los equipos de cada carrera por los puntos que sumaban sus pilotos antes
de esa carrera. Las dos líneas grises son planas porque son datos de antes de largar: valen lo mismo en los cuatro
momentos. El eje empieza en 0,4 porque las tres series viven entre 0,5 y 0,9; es un gráfico de líneas, no de barras.</p>"""


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
    for s_ in ["terminó", "abandonó después", "ya había abandonado"]:
        fila = [{"terminó": "terminó la carrera", "abandonó después": "abandonó después del corte",
                 "ya había abandonado": "ya había abandonado al corte"}[s_]]
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
    sit_d = S[S["sit"] == "abandonó después"]
    ya25 = S[(S["m"] == 0.25) & (S["sit"] == "ya había abandonado")]["modelo"].iloc[0]
    return f"""
<h4>La métrica, y por qué</h4>
<p>El MAE en puestos: si un piloto llega 5.º y el modelo dijo 3.º, el error es 2. Se lee directo y cada puesto pesa lo mismo.
La elegimos antes de probar modelos: los abandonos producen errores de cinco o diez puestos que nadie puede prever, y con el
error cuadrático esos pocos casos dominarían el número.</p>
<h4>El piso, y un rival más exigente</h4>
<p>El piso es decir siempre el puesto del medio: erra {f(C[0.0]['piso'].mean())} puestos (en 2026, con 22 autos,
{f(T[0.0]['piso'])}). El rival de verdad es «termina donde está»: cada piloto llega en el puesto que tiene en ese momento
(antes de largar, el de la grilla). Erra {", ".join(f"{f(regla[m])} {('antes de largar' if m == 0 else 'al ' + ETIQUETA[m])}" for m in M)}.</p>
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
<p>En el navegador no corre Python. Del 25 % en adelante el modelo elegido es una regresión lineal, y su fórmula está escrita
en la página tal cual (coeficientes, medias y desvíos del escalado). Antes de largar el elegido es un boosting de 100 árboles,
que no se puede escribir así. Para «Probá el modelo» usamos el <strong>árbol ajustado</strong> (max_depth=6,
min_samples_leaf=30): erra {f(arbol.mean(), 3)} ± {f(arbol.std(), 3)} contra {f(bo.mean(), 3)} ± {f(bo.std(), 3)} del boosting.
La diferencia, {f(arbol.mean() - bo.mean(), 3)} puestos, es menor que un desvío del boosting ({f(bo.std(), 3)}), y el árbol
se escribe como preguntas encadenadas. En «Explorá» y en todo el tallo, los números son los del boosting.</p>"""


R_EXPLORA_TRAIN = [0]


def pagina(R, N, hist, corr, salida):
    z, mod, regla, ventaja = textos(R, N, corr)
    e = R["explora"].copy()
    e["gp"] = e["gp"].map(lambda g_: "GP de " + GP.get(g_, g_))
    R_EXPLORA_TRAIN[0] = int((e["m"] == "Antes de largar").sum()) - N["filas_test"]
    inicial_e = {"m": "Antes de largar", "c": "2026-06"}
    inicial_p = {"c": "2026-14", "piloto": piloto_inicial(R, "2026-14")}
    specs = {"g-datos": estilo(g_datos(hist, N)), "g-hallazgo": estilo(g_hallazgo(corr)),
             "g-modelo": estilo(g_modelo(R)), "g-explora": estilo(g_explora(e, inicial_e))}
    presets = {"carreras": {}, "valores": {}, "rangos": {}, "etiquetas": {}}
    paneles = []
    for m in (0.0, 0.25, 0.5, 0.75):
        P, k = R["proba"][m], CORTO[m]
        P["filas"]["gp"] = P["filas"]["gp"].map(lambda g_: "GP de " + GP.get(g_, g_))
        vistas_p = g_proba(m, P, inicial_p)
        specs[f"g-proba-{k}"] = estilo(vistas_p["simple"])
        specs[f"g-proba-{k}-det"] = estilo(vistas_p["detalle"])
        if vistas_p["aportes"] is not None:
            specs[f"g-proba-{k}-apo"] = estilo(vistas_p["aportes"])
        mas = (f'<div class="grafico" id="g-proba-{k}-apo"></div>' if vistas_p["aportes"] is not None else
               '<p class="nota">Antes de largar el modelo de esta página es un árbol: no suma aportes, hace preguntas '
               'encadenadas (¿larga antes o después del puesto tal? ¿cuántos puntos tiene?) y cada camino termina en '
               'cuántos puestos gana o pierde.</p>')
        etq = {c: ("Lugar de largada" if (m == 0 and c == "posicion_al_corte") else ENTRADA[c][0]) for c in P["columnas"]}
        presets["rangos"][k], presets["etiquetas"][k] = P["rangos_pantalla"], etq
        presets["valores"][k] = {c: {r["piloto"]: {col: round(float(a_pantalla(col, r[col])), 2) for col in P["columnas"]}
                                     for _, r in t.iterrows()} for c, t in P["filas"].groupby("c")}
        deslizadores = "".join(f'<div class="desl"><span class="etq">{etq[c]}</span><div id="d{k}-{c}"></div>'
                               f'<span class="rango">de {f(P["rangos_pantalla"][c][0], 1 if P["rangos_pantalla"][c][2] < 1 else 0)} '
                               f'a {f(P["rangos_pantalla"][c][1], 1 if P["rangos_pantalla"][c][2] < 1 else 0)}</span></div>'
                               for c in P["columnas"])
        modelo_txt = ("Árbol ajustado (el boosting elegido no se puede escribir en la página; ver «Cómo lo sabemos» del modelo)"
                      if P["tipo"] == "árbol" else "Regresión lineal elegida en la Entrega 3, con su fórmula escrita en la página")
        paneles.append(f'<div class="panel-proba" data-k="{k}"{" hidden" if m != 0.25 else ""}>'
                       f'<p class="modelo-usado">{modelo_txt}.</p><div class="deslizadores">{deslizadores}</div>'
                       f'<p class="aviso" id="aviso-{k}" hidden></p><div class="grafico" id="g-proba-{k}"></div>'
                       f'<details class="como"><summary>Ver más <small>la carrera en el gráfico y por qué lo pone ahí</small></summary>'
                       f'<div class="cuerpo"><div class="leyenda"><span><i class="pt" style="background:var(--naranja)"></i>dónde va</span>'
                       f'<span><i class="pt" style="background:var(--azul)"></i>posición esperada por el modelo</span></div>'
                       f'<div class="grafico" id="g-proba-{k}-det"></div>{mas}</div></details></div>')
    t = R["proba"][0.25]["filas"]
    for c, g_ in t.groupby("c"):
        presets["carreras"][c] = {"gp": g_["gp"].iloc[0] + " " + c[:4],
                                  "pilotos": [[p, f"{nm} · {eq}"] for p, nm, eq in
                                              g_.sort_values("real")[["piloto", "nombre", "equipo"]].itertuples(index=False)]}
    specs_json = json.dumps({k: v.to_dict() for k, v in specs.items()}, ensure_ascii=False, separators=(",", ":"))
    kpis = "".join(f'<div class="kpi"><div class="num">{a}</div><div class="que">{b}</div></div>' for a, b in z["kpis"])
    html = (PLANTILLA.replace("__IDEA__", z["idea"]).replace("__KPIS__", kpis)
            .replace("__T_DATOS__", z["t_datos"]).replace("__L_DATOS__", z["l_datos"]).replace("__C_DATOS__", como_datos(N))
            .replace("__T_HALLAZGO__", z["t_hallazgo"]).replace("__L_HALLAZGO__", z["l_hallazgo"])
            .replace("__C_HALLAZGO__", como_hallazgo(corr))
            .replace("__T_MODELO__", z["t_modelo"]).replace("__L_MODELO__", z["l_modelo"])
            .replace("__C_MODELO__", como_modelo(R, mod, regla))
            .replace("__N_EXPLORA__", n(len(e))).replace("__CARRERAS_EXPLORA__", str(e["c"].nunique()))
            .replace("__PANELES__", "".join(paneles))
            .replace("__FECHA__", time.strftime("%d/%m/%Y"))
            .replace("__SPECS__", specs_json).replace("__PRESETS__", json.dumps(presets, ensure_ascii=False))
            .replace("__INICIAL_P__", json.dumps(inicial_p)))
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
    N, hist, corr = datos_y_hallazgo(g)
    print(f"4. Gráficos y página…  ({time.time() - t0:.0f} s)")
    html = pagina(R, N, hist, corr, a.salida)
    print(f"Listo: {a.salida} ({len(html) / 1e6:.1f} MB) en {time.time() - t0:.0f} s.")
    if AVISOS:
        print(f"\n{len(AVISOS)} aviso(s): la aplicación ya no coincide del todo con el notebook. Revisá los textos.")
