import pandas as pd
import numpy as np

from sklearn.linear_model import RANSACRegressor, LinearRegression
from vizualication import visualizar_planos_3d
from noise_removal import filter_main_cluster


def apply_tiling(df, tile_size=5000):
    """
    Aplica un proceso de 'tiling' (división espacial) a la nube de puntos.
    """
    df = df.copy()
    df["tile_x"] = (df["east"]  // tile_size).astype(int)
    df["tile_y"] = (df["north"] // tile_size).astype(int)
    return df


def detectar_planos_global(df, tolerancia=None, n_min=30,
                           max_iter=10, cobertura_objetivo=0.8):
    """
    Detecta planos globales mediante RANSAC con eliminación iterativa de inliers.
    """
    puntos = df[["east", "north", "altitud"]].to_numpy()
    total_puntos = len(puntos)
    idx_global = df.index.to_numpy()

    planos = []

    for i in range(max_iter):
        if len(puntos) < n_min:
            break

        X = puntos[:, :2]
        y = puntos[:, 2]

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

        if n_inliers < n_min:
            break

        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_
        puntos_ids = idx_global[inliers]

        planos.append({
            "id": len(planos) + 1,
            "coef": coef,
            "intercept": intercept,
            "puntos_idx": puntos_ids,
        })

        puntos = puntos[~inliers]
        idx_global = idx_global[~inliers]

        cobertura_actual = (total_puntos - len(puntos)) / total_puntos
        print(f"Iter {i+1}: {n_inliers} puntos, cobertura = {cobertura_actual:.2%}")

        if cobertura_actual >= cobertura_objetivo:
            break

    return planos


def analizar_planos(planos):
    """
    Calcula pendiente e inclinación cardinal para cada plano detectado.
    """

    def calcular_orientacion(a, b):
        """Calcula la orientación cardinal principal."""
        angulo_rad = np.arctan2(b, a)
        angulo_deg = (np.degrees(angulo_rad) + 360) % 360
        direcciones = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int(((angulo_deg + 22.5) % 360) / 45)
        return direcciones[idx]

    salida = []

    for plano in planos:
        a, b = plano["coef"]
        pendiente_rad = np.arctan(np.sqrt(a**2 + b**2))
        pendiente_grados = np.degrees(pendiente_rad)
        direccion = calcular_orientacion(a, b)

        salida.append({
            "id": plano["id"],
            "pendiente_grados": round(pendiente_grados, 2),
            "direccion": direccion,
            "puntos_idx": (
                plano["puntos_idx"].tolist()
                if hasattr(plano["puntos_idx"], 'tolist')
                else plano["puntos_idx"]
            )
        })

    return salida


# ============================================================
# EJECUTA TODA LA PIPELINE
# ============================================================

def ejecutar_pipeline_planos(
        ruta_csv="data/asro_centroides_peaks_mayor_2450.csv",
        tile_size=20000,
        tolerancia=6,
        n_min=200,
        max_iter=30,
        cobertura_objetivo=0.8,
        min_cluster_size=400,
        min_samples=20):
    """
    Ejecuta toda la pipeline de detección y análisis de planos:
    - Carga de datos
    - Filtrado de ruido
    - Tiling espacial
    - Detección de planos por RANSAC
    - Análisis geométrico
    - Visualización 3D final

    Parámetros
    ----------
    ruta_csv : str
        Ruta al archivo CSV con columnas 'east', 'north', 'altitud'.
    tile_size : int
        Tamaño del mosaico espacial.
    tolerancia : float
        Umbral de RANSAC (residual_threshold).
    n_min : int
        Mínimo de puntos para aceptar un plano.
    max_iter : int
        Máximo de iteraciones RANSAC por tile.
    cobertura_objetivo : float
        Proporción mínima cubierta por los planos.
    min_cluster_size : int
        Tamaño mínimo de clúster para filtrado de ruido.
    min_samples : int
        Min_samples para la etapa de filtrado.

    Devuelve
    --------
    dict
        Resultados completos de la ejecución:
        - df_clean
        - planos
        - analisis
    """

    print("Cargando datos...")
    df = pd.read_csv(ruta_csv)
    df["orig_id"] = df.index
    df["x"] = df["east"]
    df["y"] = df["north"]
    df["z"] = df["altitud"]

    print("Filtrando ruido...")
    df_clean, labels_clean = filter_main_cluster(
        df,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    print("Aplicando tiling espacial...")
    df_tiled = apply_tiling(df_clean, tile_size=tile_size)

    print("Detectando planos por tile...")
    planos = []
    for (tx, ty), chunk in df_tiled.groupby(["tile_x", "tile_y"]):

        if len(chunk) < n_min:
            continue

        planos_tile = detectar_planos_global(
            chunk,
            tolerancia=tolerancia,
            n_min=n_min,
            max_iter=max_iter,
            cobertura_objetivo=cobertura_objetivo
        )

        for p in planos_tile:
            p["tile"] = (tx, ty)

        planos.extend(planos_tile)

    print("Analizando geometría de los planos...")
    analisis = analizar_planos(planos)

    for p in analisis:
        print(f"Plano {p['id']}: "
              f"inclinación = {p['pendiente_grados']:.2f}°, "
              f"dirección = {p['direccion']}, "
              f"puntos = {len(p['puntos_idx'])}")

    print("Generando visualización 3D...")
    visualizar_planos_3d(df_clean, planos)

    return {
        "df_clean": df_clean,
        "planos": planos,
        "analisis": analisis
    }


ejecutar_pipeline_planos()
