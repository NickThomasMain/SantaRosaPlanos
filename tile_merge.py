"""
Módulo compacto para detección y postprocesado de planos en una nube de puntos.
Versión revisada: código simplificado, con docstrings y comentarios en español.
Mantiene la funcionalidad esencial: tiling, detección por RANSAC, fusión (merge),
resumen de planos y visualización final (depende de funciones externas de visualización
y filtrado de ruido).
"""

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import RANSACRegressor, LinearRegression

# funciones externas esperadas por la pipeline (deben existir en tu proyecto)
from vizualication import visualizar_planos_3d
from noise_removal import filter_main_cluster


# -------------------------
# Utilidades pequeñas
# -------------------------
def apply_tiling(df, tile_size=5000):
    """
    Añade columnas 'tile_x' y 'tile_y' al DataFrame según tile_size.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'east' y 'north'.
    tile_size : int
        Tamaño del tile en las mismas unidades que 'east'/'north'.

    Retorna
    -------
    pandas.DataFrame
        Copia del DataFrame original con columnas 'tile_x' y 'tile_y'.
    """
    df = df.copy()
    df["tile_x"] = (df["east"] // tile_size).astype(int)
    df["tile_y"] = (df["north"] // tile_size).astype(int)
    return df


def tiles_adjacent(t1, t2):
    """
    Comprueba si dos tiles (tuplas (tx,ty)) son adyacentes (incluye diagonales).

    Retorna True si la distancia Chebyshev <= 1.
    """
    return max(abs(t1[0] - t2[0]), abs(t1[1] - t2[1])) <= 1


# -------------------------
# Cálculo geométrico
# -------------------------
def calcular_inclinacion_y_direccion(coef):
    """
    Calcula la inclinación (grados) y una etiqueta de dirección cardinal
    a partir de los coeficientes [a, b] del plano z = a*x + b*y + c.

    Parámetros
    ----------
    coef : array-like (longitud 2)
        Coeficientes [a, b].

    Retorna
    -------
    (inclinacion_deg, direccion)
    """
    a, b = coef
    pendiente = np.sqrt(a * a + b * b)
    inclinacion_deg = np.degrees(np.arctan(pendiente))

    ang = np.degrees(np.arctan2(b, a))
    ang = (ang + 360) % 360
    direcciones = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((ang + 22.5) % 360) / 45)
    direccion = direcciones[idx]
    return inclinacion_deg, direccion


# -------------------------
# Detección de planos (RANSAC)
# -------------------------
def detectar_planos_global(
    df,
    tolerancia=None,
    n_min=30,
    max_iter=10,
    cobertura_objetivo=0.8,
    max_north_extent=None,
    random_state=42
):
    """
    Detecta planos iterativamente con RANSAC sobre las columnas 'east','north','altitud'.

    Flujo:
      - Ajusta RANSAC sobre (east,north) -> altitud.
      - Si el número de inliers >= n_min y la extensión norte < max_north_extent (si se pasa),
        se acepta el plano y se eliminan sus inliers para la siguiente iteración.
      - Para evitar bucles infinitos, si un plano es rechazado por tamaño, no se eliminan
        puntos; RANSAC intentará otro ajuste en la siguiente iteración.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'east','north','altitud' y 'orig_id' (identificador único).
    tolerancia : float or None
        Umbral de residual para considerar inlier en RANSAC. Si None, RANSAC decidirá.
    n_min : int
        Mínimo de inliers para aceptar un plano.
    max_iter : int
        Máximo número de iteraciones (planos a detectar).
    cobertura_objetivo : float (0..1)
        Fracción de puntos que queremos cubrir antes de detener la búsqueda.
    max_north_extent : float or None
        Si se suministra, rechaza planos cuya extensión norte-sur exceda este valor.
    random_state : int
        Semilla para reproducibilidad.

    Retorna
    -------
    list of dict
        Cada dict contiene: 'id','coef' (array [a,b]), 'intercept', 'puntos_idx' (array de orig_id).
    """
    # Trabajar con copia de los arrays para acelerar operaciones
    puntos = df[["east", "north", "altitud"]].to_numpy()
    ids = df["orig_id"].to_numpy()
    total = len(puntos)

    planos = []
    iter_count = 0

    while iter_count < max_iter and len(puntos) >= n_min:
        iter_count += 1

        X = puntos[:, :2]
        y = puntos[:, 2]

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=tolerancia,
            min_samples=3,
            max_trials=2000,
            random_state=random_state
        )
        ransac.fit(X, y)

        inlier_mask = ransac.inlier_mask_
        n_inliers = int(np.sum(inlier_mask))

        # Si no hay suficientes inliers, terminamos
        if n_inliers < n_min:
            break

        coef = ransac.estimator_.coef_.ravel()[:2]
        intercept = float(ransac.estimator_.intercept_)

        inlier_ids = ids[inlier_mask]

        # Si se solicita máxima extensión norte-sur, calcular y posiblemente rechazar
        if max_north_extent is not None:
            north_vals = df.loc[df["orig_id"].isin(inlier_ids), "north"].to_numpy()
            north_extent = float(north_vals.max() - north_vals.min()) if north_vals.size > 0 else 0.0
            if north_extent > max_north_extent:
                # Rechazar este plano pero no remover puntos para intentar otro ajuste
                print(
                    f"Iter {iter_count}: plano rechazado por north_extent={north_extent:.1f} > {max_north_extent}"
                )
                continue

        # Aceptar plano
        planos.append({
            "id": len(planos) + 1,
            "coef": np.asarray(coef, dtype=float),
            "intercept": intercept,
            "puntos_idx": inlier_ids
        })

        # Eliminar inliers del conjunto de búsqueda
        mask_keep = ~inlier_mask
        puntos = puntos[mask_keep]
        ids = ids[mask_keep]

        covered = (total - len(puntos)) / total
        print(f"Iter {iter_count}: aceptado {n_inliers} puntos, cobertura={covered:.2%}")

        if covered >= cobertura_objetivo:
            break

    return planos


# -------------------------
# Merge (fusión) de planos candidatos
# -------------------------
def merge_planes(planos, df, angle_thresh=np.deg2rad(5), offset_thresh=3.0, height_tolerance=6.0):
    """
    Fusione planos que sean compatibles geométrica y espacialmente.

    Criterios básicos de fusión:
      - Ángulo entre normales por debajo de angle_thresh.
      - Diferencia de offset (intercept) por debajo de offset_thresh.
      - Al re-ajustar una sola superficie sobre la unión de puntos, el error máximo
        (orthogonal) debe ser <= height_tolerance.

    Parámetros
    ----------
    planos : list of dict
        Lista de planos con 'coef','intercept','puntos_idx'.
    df : pandas.DataFrame
        DataFrame con columnas 'east','north','altitud' y 'orig_id'.
    angle_thresh : float
        Umbral angular (radianes) para considerar planos paralelos.
    offset_thresh : float
        Umbral en unidades de altura para la diferencia de intercept.
    height_tolerance : float
        Error máximo (m) permitido al re-ajustar una unión.

    Retorna
    -------
    list of dict
        Lista de planos resultantes después del merge.
    """
    def plane_normal_from_coef(a, b):
        # Para plano z = a*x + b*y + c -> vector normal (a, b, -1)
        n = np.array([a, b, -1.0], dtype=float)
        return n / (np.linalg.norm(n) + 1e-12)

    def fit_plane_svd(points):
        # points: Nx3 array; devuelve normal unitario y d tal que n·x + d = 0
        C = np.mean(points, axis=0)
        u, s, vh = np.linalg.svd(points - C, full_matrices=False)
        normal = vh[-1, :]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        d = -np.dot(normal, C)
        return normal, d

    def max_orthogonal_error(points, normal, d):
        # Distancias algebraicas / norma -> distancia ortogonal aproximada
        dist = (points @ normal + d)
        return np.max(np.abs(dist))

    # Convertir a formato mutable
    planos_work = [p.copy() for p in planos]

    merged_any = True
    round_idx = 0

    # Precompute lookup for point coordinates by orig_id to avoid repeated .loc cost
    coords_lookup = df.set_index("orig_id")[["east", "north", "altitud"]]

    while merged_any:
        round_idx += 1
        merged_any = False
        new_list = []
        used = set()

        for i, base in enumerate(planos_work):
            if i in used:
                continue

            base_ids = np.asarray(base["puntos_idx"])
            # Obtener puntos de base
            pts_base = coords_lookup.loc[base_ids].to_numpy()

            a1, b1 = base["coef"]
            c1 = base["intercept"]
            n1 = plane_normal_from_coef(a1, b1)

            merged_with_some = False

            for j in range(i + 1, len(planos_work)):
                if j in used:
                    continue

                comp = planos_work[j]
                comp_ids = np.asarray(comp["puntos_idx"])
                pts_comp = coords_lookup.loc[comp_ids].to_numpy()

                a2, b2 = comp["coef"]
                c2 = comp["intercept"]
                n2 = plane_normal_from_coef(a2, b2)

                # Ángulo entre normales
                cos_angle = float(np.dot(n1, n2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                if angle > angle_thresh:
                    continue

                if abs(c1 - c2) > offset_thresh:
                    continue

                # Reajustar a la unión y comprobar error
                union_pts = np.vstack([pts_base, pts_comp])
                normal_u, d_u = fit_plane_svd(union_pts)
                err_max = max_orthogonal_error(union_pts, normal_u, d_u)

                if err_max <= height_tolerance:
                    # Fusionar: crear nuevo plano con unión de puntos
                    merged_any = True
                    used.add(j)
                    merged_with_some = True

                    # Obtener coeficientes en forma z = a*x + b*y + c
                    # Si normal = [nx, ny, nz] con nz != 0, z = -(nx/nz)x - (ny/nz)y - d/nz
                    nx, ny, nz = normal_u
                    a_new = -nx / nz
                    b_new = -ny / nz
                    c_new = -d_u / nz

                    merged_ids = np.concatenate([base_ids, comp_ids])
                    base = {
                        "id": base["id"],  # id puede recalcularse más adelante
                        "coef": np.array([a_new, b_new], dtype=float),
                        "intercept": float(c_new),
                        "puntos_idx": merged_ids
                    }
                    # actualizar pts_base y n1 para permitir fusiones encadenadas en misma iteración
                    pts_base = union_pts
                    n1 = plane_normal_from_coef(a_new, b_new)

            new_list.append(base)
            # marcar i si fue consumido via otro como componente (handled en used)
        # Reindexar ids secuencialmente para claridad
        for k, p in enumerate(new_list, start=1):
            p["id"] = k

        planos_work = new_list

    return planos_work


# -------------------------
# Resumen / análisis simple de planos
# -------------------------
def analizar_planos(planos, df):
    """
    Genera un resumen compacto por plano: número de puntos, inclinación y dirección.

    Parámetros
    ----------
    planos : list of dict
        Lista con 'coef','intercept','puntos_idx'.
    df : pandas.DataFrame
        DataFrame original con 'orig_id','east','north','altitud'.

    Retorna
    -------
    pandas.DataFrame
        Tabla con columnas: ['id','num_puntos','inclinacion_deg','direccion'].
    """
    registros = []
    for p in planos:
        ids = np.asarray(p["puntos_idx"])
        num = len(ids)
        a, b = p["coef"]
        inclin, direccion = calcular_inclinacion_y_direccion([a, b])
        registros.append({
            "id": p.get("id", None),
            "num_puntos": num,
            "inclinacion_deg": round(float(inclin), 2),
            "direccion": direccion
        })
    return pd.DataFrame(registros)


# -------------------------
# Programa principal (flujo)
# -------------------------
def complete_tile_merge(
    ruta_csv="data/asro_centroides_peaks_mayor_2450.csv",
    tolerancia=6,
    n_min_tile=200,
    tile_size=2000,
    n_min_global=200,
    max_iter=30,
    cobertura_obj=0.8,
    max_north_extent=20000,
    merge_angle_deg=0.2,
    merge_offset=1000,
    merge_height_tol=40,
    min_cluster_size=400,
    min_samples=20,
    out_dir="resultados_planos"
):
    """
    Flujo principal: carga, filtrado del clúster principal, tiling, detección por tile,
    fusión de planos y visualización final.

    Parâmetros principales se exponen para facilitar experimentación.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- 1) Cargar datos y preparar columnas mínimas ---
    df = pd.read_csv(ruta_csv)
    df["orig_id"] = df.index
    df["x"] = df["east"]
    df["y"] = df["north"]
    df["z"] = df["altitud"]

    # --- 2) Filtrar clúster principal (función externa) ---
    df_clean, labels = filter_main_cluster(
        df,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )

    # --- 3) Aplicar tiling y detectar por chunk/tile ---
    df_tiled = apply_tiling(df_clean, tile_size=tile_size)

    planos = []
    for (tx, ty), chunk in df_tiled.groupby(["tile_x", "tile_y"]):
        if len(chunk) < n_min_tile:
            continue

        planos_tile = detectar_planos_global(
            chunk,
            tolerancia=tolerancia,
            n_min=n_min_global,
            max_iter=max_iter,
            cobertura_objetivo=cobertura_obj,
            max_north_extent=max_north_extent
        )

        # anotar tile para potencial uso espacial en merging
        for p in planos_tile:
            p["tile"] = (int(tx), int(ty))

        planos.extend(planos_tile)

    if not planos:
        print("No se detectaron planos en ningún tile.")
        return

    # --- 4) Resumen inicial ---
    resumen = analizar_planos(planos, df_clean)
    print("Resumen inicial de planos:")
    print(resumen)

    # --- 5) Merge/fusión de planos compatibles ---
    planos_merged = merge_planes(
        planos,
        df=df_clean,
        angle_thresh=np.deg2rad(merge_angle_deg),
        offset_thresh=merge_offset,
        height_tolerance=merge_height_tol
    )

    resumen_final = analizar_planos(planos_merged, df_clean)
    print("Resumen después de merge:")
    print(resumen_final)

    visualizar_planos_3d(df_clean, planos_merged)


    return planos_merged, resumen_final


complete_tile_merge()