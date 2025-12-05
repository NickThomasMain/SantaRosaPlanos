"""
Descripción general
-------------------
Este script carga una nube de puntos (CSV), filtra el clúster principal,
detecta planos por RANSAC de forma iterativa, calcula métricas y residuos
por punto, genera resúmenes por plano, guarda resultados en CSV, crea
gráficos de diagnóstico y visualiza los planos en 3D.

Notas:
- El flujo principal está implementado en `ejecutar_analisis_planos`.
- Funciones auxiliares:
    * detectar_planos_global: detecta planos iterativamente con RANSAC.
    * analizar_planos_detallado: calcula residuos por punto y agrega contexto.
    * analizar_planos_compacto: resume métricas por plano.
    * guardar_tablas: exporta CSV.
    * plot_residuos: genera gráficos para inspección.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import RANSACRegressor, LinearRegression
from vizualication import visualizar_planos_3d
from noise_removal import filter_main_cluster


def calcular_inclinacion_y_direccion(coef):
    """
    Calcula la inclinación (en grados) y la dirección cardinal del plano.

    Parámetros
    ----------
    coef : array-like de longitud 2
        Coeficientes [a, b] de la ecuación del plano z = a*x + b*y + c.

    Retorna
    -------
    inclinacion_deg : float
        Ángulo de inclinación en grados (0 = plano horizontal).
    direccion : str
        Dirección cardinal aproximada hacia la cual el plano desciende más rápidamente.
        (N, NE, E, SE, S, SW, W, NW)
    """
    a, b = coef

    # Magnitud del gradiente -> pendiente (tangente del ángulo)
    pendiente = np.sqrt(a**2 + b**2)

    # Convertir la pendiente a grados: arctan(pendiente)
    inclinacion_deg = np.degrees(np.arctan(pendiente))

    # Ángulo del vector gradiente en grados, con referencia Este y rotación CCW
    ang = np.degrees(np.arctan2(b, a))
    ang = (ang + 360) % 360  # normalizar a [0, 360)

    # Mapeo a 8 direcciones cardinales
    direcciones = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((ang + 22.5) % 360) / 45)  # dividir el círculo en 8 sectores
    direccion = direcciones[idx]

    return inclinacion_deg, direccion


def detectar_planos_global(df, tolerancia=None, n_min=30,
                           max_iter=10, cobertura_objetivo=0.8):
    """
    Detecta planos iterativamente sobre una nube de puntos usando RANSAC.

    Algoritmo (resumen)
    -------------------
    1. Toma la nube completa (east, north, altitud).
    2. Ejecuta RANSAC para ajustar un plano z = a*x + b*y + c.
    3. Si el número de inliers >= n_min, registra el plano.
    4. Remueve los inliers y repite hasta alcanzar max_iter o cobertura objetivo.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con columnas obligatorias: 'east', 'north', 'altitud', 'orig_id'.
    tolerancia : float or None
        Umbral residual para considerar un punto como inlier en RANSAC.
        Si None, RANSAC decide internamente (no recomendado si buscas control).
    n_min : int
        Número mínimo de inliers para aceptar un plano.
    max_iter : int
        Máximo número de iteraciones/planos a intentar.
    cobertura_objetivo : float (0..1)
        Fracción de la nube total que se desea cubrir con planos antes de parar.

    Retorna
    -------
    planos : list of dict
        Lista de diccionarios, uno por plano detectado. Cada diccionario contiene:
            - id: identificador secuencial del plano
            - coef: array [a, b]
            - intercept: término independiente c
            - puntos_idx: array de orig_id de los inliers
            - east_extent, north_extent: dimensiones espaciales del conjunto de inliers
            - inclinacion, direccion: métricas geométricas del plano
    """

    # Extraer matriz de puntos (N x 3) y array de índices globales
    puntos = df[["east", "north", "altitud"]].to_numpy()
    total_puntos = len(puntos)
    idx_global = df.index.to_numpy()  # indices relativos al dataframe original

    planos = []

    for i in range(max_iter):
        # Si quedan pocos puntos, interrumpir
        if len(puntos) < n_min:
            break

        X = puntos[:, :2]  # columnas east, north
        y = puntos[:, 2]   # columna altitud

        # Configuración de RANSAC (linear regression como estimador)
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=tolerancia,
            min_samples=3,
            max_trials=2000,
            random_state=42
        )
        ransac.fit(X, y)

        inliers = ransac.inlier_mask_
        n_inliers = inliers.sum()

        # Si no hay suficientes inliers, terminar las iteraciones
        if n_inliers < n_min:
            break

        # Extraer coeficientes del estimador lineal ajustado: coef = [a, b]
        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_

        # Recuperar los IDs globales (orig_id) de los inliers
        puntos_ids = idx_global[inliers]

        # Subconjunto de dataframe con los puntos inliers para métricas espaciales
        sub = df.loc[df["orig_id"].isin(puntos_ids)]
        east_extent = sub["east"].max() - sub["east"].min()
        north_extent = sub["north"].max() - sub["north"].min()

        # Calcular inclinación y dirección para describir el plano
        inclinacion, direccion = calcular_inclinacion_y_direccion(coef)

        # Registrar el plano detectado
        planos.append({
            "id": len(planos) + 1,
            "coef": coef,
            "intercept": intercept,
            "puntos_idx": puntos_ids,
            "east_extent": east_extent,
            "north_extent": north_extent,
            "inclinacion": inclinacion,
            "direccion": direccion
        })

        # Eliminar los inliers del conjunto de puntos para la siguiente iteración
        puntos = puntos[~inliers]
        idx_global = idx_global[~inliers]

        # Cálculo de cobertura acumulada
        cobertura_actual = (total_puntos - len(puntos)) / total_puntos
        print(f"Iter {i+1}: {n_inliers} puntos, cobertura = {cobertura_actual:.2%}")

        # Si alcanzamos la cobertura objetivo, salir
        if cobertura_actual >= cobertura_objetivo:
            break

    return planos


def analizar_planos_detallado(df, planos):
    """
    Genera una tabla detallada por punto para todos los planos detectados.

    Para cada punto inlier de cada plano se calcula:
      - residual = altura_observada - altura_predicha_por_el_plano
      - se guarda la desviación estándar de los residuos del plano

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con las columnas originales y 'orig_id'.
    planos : list of dict
        Lista de planos devuelta por detectar_planos_global().

    Retorna
    -------
    pandas.DataFrame
        DataFrame con columnas:
        ['plan_id','point_id','height','residual','residual_std',
         'east_extent','north_extent','inclinacion','direccion']
    """

    registros = []

    # Iterar cada plano y calcular residuos para sus puntos inliers
    for plano in planos:
        coef = plano["coef"]
        intercept = plano["intercept"]
        point_ids = plano["puntos_idx"]

        # Subconjunto de puntos pertenecientes al plano
        sub = df.loc[df["orig_id"].isin(point_ids)]

        # Predicción Z a partir del plano y cálculo de residuo
        z_pred = coef[0] * sub["east"] + coef[1] * sub["north"] + intercept
        resid = sub["altitud"] - z_pred

        # Desviación estándar de los residuos del plano (para diagnósticos)
        std_res = resid.std()

        # Registrar una fila por punto
        for pid, h, r in zip(sub["orig_id"], sub["altitud"], resid):
            registros.append({
                "plan_id": plano["id"],
                "point_id": pid,
                "height": h,
                "residual": r,
                "residual_std": std_res,
                "east_extent": plano["east_extent"],
                "north_extent": plano["north_extent"],
                "inclinacion": plano["inclinacion"],
                "direccion": plano["direccion"]
            })

    return pd.DataFrame(registros)


def analizar_planos_compacto(df_detallado):
    """
    Resume métricas de interés por plano en una tabla compacta.

    Métricas incluidas:
    - numero de puntos
    - estadísticos de residuo (media, std, min, max)
    - rango de alturas (min, max)
    - dimensiones espaciales east/north extent
    - inclinación y dirección

    Parámetros
    ----------
    df_detallado : pandas.DataFrame
        Salida de analizar_planos_detallado().

    Retorna
    -------
    pandas.DataFrame
        Tabla resumida con una línea por plano (plan_id).
    """

    resumen = df_detallado.groupby("plan_id").agg({
        "point_id": "count",
        "residual": ["mean", "std", "min", "max"],
        "height": ["min", "max"],
        "east_extent": "first",
        "north_extent": "first",
        "inclinacion": "first",
        "direccion": "first"
    })

    # Aplanar multi-índice de columnas y renombrar para claridad
    resumen.columns = [
        "num_puntos",
        "residual_mean",
        "residual_std",
        "residual_min",
        "residual_max",
        "height_min",
        "height_max",
        "east_extent",
        "north_extent",
        "inclinacion",
        "direccion"
    ]

    return resumen.reset_index()


def guardar_tablas(df_detallado, df_compacto, out_dir="resultados_planos"):
    """
    Guarda los DataFrames detallado y compacto en CSV en la carpeta out_dir.

    Parámetros
    ----------
    df_detallado : pandas.DataFrame
        Tabla con un registro por punto-inlier.
    df_compacto : pandas.DataFrame
        Tabla resumen por plano.
    out_dir : str
        Carpeta de salida; se crea si no existe.
    """

    # Crear carpeta de salida si no existe
    os.makedirs(out_dir, exist_ok=True)

    path1 = os.path.join(out_dir, "planos_detallado.csv")
    path2 = os.path.join(out_dir, "planos_compacto.csv")

    # Escribir CSV sin índices (más portable)
    df_detallado.to_csv(path1, index=False)
    df_compacto.to_csv(path2, index=False)

    print(f"Archivos CSV guardados:\n - {path1}\n - {path2}")


def plot_residuos(df_detallado, df_clean, out_dir="resultados_planos"):
    """
    Genera y guarda gráficos de diagnóstico para los residuos por plano.

    Gráficos generados:
      1) Boxplot de residuos por plano.
      2) Histograma de residuos para cada plano.
      3) Scatter espacial (east,north) coloreado por residuo para cada plano.

    Parámetros
    ----------
    df_detallado : pandas.DataFrame
        Salida de analizar_planos_detallado().
    df_clean : pandas.DataFrame
        DataFrame con la nube filtrada (usado para coordenadas).
    out_dir : str
        Carpeta donde se guardan las imágenes.
    """

    os.makedirs(out_dir, exist_ok=True)

    # --- 1) Boxplot de residuos por plano ---
    plt.figure(figsize=(10, 6))
    df_detallado.boxplot(column="residual", by="plan_id")
    plt.title("Boxplot de residuos por plano")
    plt.suptitle("")  # eliminar título automático
    plt.xlabel("ID del plano")
    plt.ylabel("Residuo [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplot_residuos_por_plano.png"))
    plt.close()

    # --- 2) Histogramas por plano ---
    for pid, grupo in df_detallado.groupby("plan_id"):
        plt.figure(figsize=(8, 5))
        plt.hist(grupo["residual"], bins=25)
        plt.title(f"Histograma de residuos – Plano {pid}")
        plt.xlabel("Residuo [m]")
        plt.ylabel("Frecuencia")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_residuos_plano_{pid}.png"))
        plt.close()

    # --- 3) Scatter espacial por plano coloreado por residuo ---
    for pid, grupo in df_detallado.groupby("plan_id"):
        # Seleccionar puntos originales correspondiente a los point_id
        df_sub = df_clean[df_clean["orig_id"].isin(grupo["point_id"])]

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(
            df_sub["east"],
            df_sub["north"],
            c=grupo["residual"],
            s=10
        )
        plt.colorbar(sc, label="Residuo [m]")
        plt.title(f"Distribución espacial de residuos – Plano {pid}")
        plt.xlabel("East")
        plt.ylabel("North")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"scatter_residuos_plano_{pid}.png"))
        plt.close()

    print(f"Gráficos guardados en: {out_dir}")


def ejecutar_analisis_planos(
        ruta_csv="asro_centroides_peaks_mayor_2450.csv",
        tolerancia=6,
        n_min=150,
        max_iter=30,
        cobertura_objetivo=0.7,
        min_cluster_size=400,
        min_samples=20,
        out_dir="resultados_planos"
    ):
    """
    Orquesta el flujo completo de análisis de planos y generación de salidas.

    Pasos
    -----
    1) Carga el CSV y prepara columnas de visualización.
    2) Filtra el clúster principal (usando filter_main_cluster).
    3) Detecta planos por RANSAC iterativo (detectar_planos_global).
    4) Calcula residuos por punto (analizar_planos_detallado).
    5) Resume métricas por plano (analizar_planos_compacto).
    6) Guarda resultados en CSV (guardar_tablas).
    7) Crea gráficos de diagnóstico (plot_residuos).
    8) Visualiza los planos detectados en 3D (visualizar_planos_3d).

    Parámetros
    ----------
    ruta_csv : str
        Ruta al fichero CSV con la nube de puntos.
    tolerancia : float
        Umbral residual para RANSAC.
    n_min : int
        Número mínimo de inliers por plano.
    max_iter : int
        Máximo de iteraciones/planos a detectar.
    cobertura_objetivo : float
        Fracción objetivo de cobertura de la nube total.
    min_cluster_size, min_samples : int
        Parámetros para el filtrado de clúster principal.
    out_dir : str
        Carpeta de salida para CSV e imágenes.
    """

    print("Cargando datos...")
    df = pd.read_csv(ruta_csv)
    df["orig_id"] = df.index  # id persistente para referencia de puntos

    # Preparar columnas de visualización consistentes con resto de pipeline
    df["x"] = df["east"]
    df["y"] = df["north"]
    df["z"] = df["altitud"]

    print("Filtrando clúster principal...")
    df_clean, labels_clean = filter_main_cluster(
        df,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        visualize=False
    )

    print("Detectando planos globales con RANSAC...")
    planos = detectar_planos_global(
        df=df_clean,
        tolerancia=tolerancia,
        n_min=n_min,
        max_iter=max_iter,
        cobertura_objetivo=cobertura_objetivo
    )

    print(f"{len(planos)} planos detectados.")

    print("Generando tabla detallada de residuos por punto...")
    df_detallado = analizar_planos_detallado(df_clean, planos)

    print("Generando tabla compacta de métricas por plano...")
    df_compacto = analizar_planos_compacto(df_detallado)

    print("Guardando tablas en CSV...")
    guardar_tablas(df_detallado, df_compacto, out_dir=out_dir)

    print("Generando gráficos de análisis...")
    plot_residuos(df_detallado, df_clean, out_dir=out_dir)

    print("Visualizando planos en 3D...")
    visualizar_planos_3d(df_clean, planos)

    print("Análisis completado.")
    return


# Llamada de ejemplo (parámetros por defecto ajustados en la invocación original)
ejecutar_analisis_planos(
    ruta_csv="asro_centroides_peaks_mayor_2450.csv",
    tolerancia=6,
    n_min=200,
    max_iter=30,
    cobertura_objetivo=0.7,
    min_cluster_size=300,
    min_samples=10,
    out_dir="resultados_planos"
)
