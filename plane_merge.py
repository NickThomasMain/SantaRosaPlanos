import pandas as pd
import numpy as np

from sklearn.linear_model import RANSACRegressor, LinearRegression
from vizualication import visualizar_planos_3d2
from noise_removal import filter_main_cluster


def apply_tiling(df, tile_size=5000):
    df = df.copy()
    df["tile_x"] = (df["east"]  // tile_size).astype(int)
    df["tile_y"] = (df["north"] // tile_size).astype(int)
    return df

def tiles_adjacent(t1, t2):
    return max(abs(t1[0] - t2[0]), abs(t1[1] - t2[1])) <= 1


def detectar_planos_global(df, tolerancia=None, n_min=30,
                           max_iter=10, cobertura_objetivo=0.8,
                           max_north_extent=20000):
    """
    Detecta planos globales usando RANSAC,
    rechazando planos cuya extensión norte-sur exceda max_north_extent.
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

        # ✅ NEW — compute north-extent of this plane
        north_vals = df.loc[puntos_ids, "north"].to_numpy()
        north_extent = north_vals.max() - north_vals.min()

        # ✅ Reject plane if too large
        if north_extent > max_north_extent:
            print(f"⚠️ Plano rechazado por tamaño norte ({north_extent:.1f} m > {max_north_extent})")

            # ❗ DO NOT remove points — keep searching in same data
            # But avoid infinite loop: skip this iteration and continue
            # (RANSAC next round will find different plane)
            continue

        # ✅ Plane accepted
        planos.append({
            "id": len(planos) + 1,
            "coef": coef,
            "intercept": intercept,
            "puntos_idx": puntos_ids,
        })

        # Remove inliers for next iteration
        puntos = puntos[~inliers]
        idx_global = idx_global[~inliers]

        cobertura_actual = (total_puntos - len(puntos)) / total_puntos
        print(f"Iter {i+1}: {n_inliers} puntos, cobertura = {cobertura_actual:.2%}, north_extent = {north_extent:.1f} m ✅")

        if cobertura_actual >= cobertura_objetivo:
            break

    return planos




def merge_planes(planos, df,
                 angle_thresh=np.deg2rad(5),
                 offset_thresh=3.0,
                 height_tolerance=6.0):

    def plane_to_normal(a, b):
        return np.array([a, b, -1.0])

    def fit_plane(points):
        centroid = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - centroid)
        normal = vh[-1]
        d = -np.dot(normal, centroid)
        return normal, d

    def max_height_error(points, normal, d):
        distances = (points @ normal + d) / np.linalg.norm(normal)
        return np.max(np.abs(distances))

    merged = True

    # Koordinaten-Array passend zu df.index
    coords = df[["east", "north", "altitud"]].to_numpy()

    merge_round = 1

    while merged:
        print(f"\n--- Merge Runde {merge_round} ---")
        merged = False
        new_planos = []
        skip = set()

        for i in range(len(planos)):
            if i in skip:
                continue

            base = planos[i]
            a1, b1 = base["coef"]
            c1 = base["intercept"]

            base_normal = plane_to_normal(a1, b1)

            # 🔥 korrekt: puntos_idx -> Positionen in coords
            base_points = coords[ df.index.get_indexer(base["puntos_idx"]) ]

            best_merge = None

            for j in range(i + 1, len(planos)):
                if j in skip:
                    # Tiles müssen benachbart sein
                    if not tiles_adjacent(base.get("tile"), comp.get("tile")):
                        continue

                    

                comp = planos[j]
                a2, b2 = comp["coef"]
                c2 = comp["intercept"]

                comp_normal = plane_to_normal(a2, b2)

                # Winkel zwischen Normalen
                angle = np.arccos(
                    np.clip(
                        np.dot(base_normal, comp_normal) /
                        (np.linalg.norm(base_normal) * np.linalg.norm(comp_normal)),
                        -1, 1
                    )
                )

                if angle > angle_thresh:
                    continue

                if abs(c1 - c2) > offset_thresh:
                    continue

                # 🔥 Punkte des Vergleichsplanes holen
                comp_points = coords[ df.index.get_indexer(comp["puntos_idx"]) ]

                # 🔥 Vereinigung der Punkte
                candidate_points = np.vstack([base_points, comp_points])

                # Neue Ebene fitten
                normal_new, d_new = fit_plane(candidate_points)

                # Prüfung des maximalen Höhenfehlers
                if max_height_error(candidate_points, normal_new, d_new) <= height_tolerance:
                    best_merge = (
                        candidate_points,
                        normal_new, d_new,
                        base["puntos_idx"],
                        comp["puntos_idx"]
                    )

                    skip.add(j)

                    print(f"✅ Merge: Plano {base['id']}  +  Plano {comp['id']}")

            # Falls ein Merge gefunden wurde
            if best_merge is not None:
                pts, normal_new, d_new, ids1, ids2 = best_merge

                merged_ids = np.concatenate([ids1, ids2])  # echte df-Indices

                # Neue Koeffizienten
                a_new = -normal_new[0] / normal_new[2]
                b_new = -normal_new[1] / normal_new[2]
                c_new = -d_new / normal_new[2]

                new_planos.append({
                    "id": len(new_planos) + 1,
                    "coef": np.array([a_new, b_new]),
                    "intercept": c_new,
                    "puntos_idx": merged_ids     # 🔥 nur noch puntos_idx
                })

                merged = True

            else:
                new_planos.append(base)

        planos = new_planos
        merge_round += 1

    print(f"\n✅ Final: {len(planos)} Planos nach Merging\n")
    return planos


def analizar_planos(planos):
    """
    Analiza una lista de planos detectados y calcula su inclinación y dirección principal.

    Parámetros:
    ------------
    planos : list[dict]
        Lista generada por detectar_planos_global(), que contiene coeficientes 'coef' y 'puntos_ids'.

    Devuelve:
    -----------
    lista de diccionarios con:
        - id: número de plano
        - pendiente_grados: inclinación del plano en grados
        - direccion: orientación principal (N, NE, E, SE, S, SW, W, NW)
        - puntos_ids: índices de los puntos pertenecientes al plano
    """

    def calcular_orientacion(a, b):
        """Calcula la orientación cardinal basada en los coeficientes del plano."""
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
            "puntos_idx": plano["puntos_idx"].tolist() if hasattr(plano["puntos_idx"], 'tolist') else plano["puntos_idx"]
        })

    return salida

df = pd.read_csv("asro_centroides_peaks_mayor_2450.csv")
df["orig_id"] = df.index

# Visualisierungsspalten anlegen
df["x"] = df["east"]
df["y"] = df["north"]
df["z"] = df["altitud"]

df_clean, labels_clean = filter_main_cluster(
    df,
    min_cluster_size=400,
    min_samples=20,
    visualize=False
)

# 1) Tiles erzeugen
df_tiled = apply_tiling(df_clean, tile_size=20000)

# 2) detectar_planos_global pro Tile ausführen
planos = []

for (tx, ty), chunk in df_tiled.groupby(["tile_x", "tile_y"]):

    if len(chunk) < 200:   # dein n_min
        continue

    planos_tile = detectar_planos_global(
        chunk,
        tolerancia=6,
        n_min=200,
        max_iter=30,
        cobertura_objetivo=0.8
    )

    for p in planos_tile:
        p["tile"] = (tx, ty)

    planos.extend(planos_tile)



analisis = analizar_planos(planos)

for p in analisis:
    print(f"Plano {p['id']}: inclinación = {p['pendiente_grados']:.2f}°, dirección = {p['direccion']}, puntos = {len(p['puntos_idx'])}")



planos2 = merge_planes(
    planos,
    df=df_clean,
    angle_thresh=np.deg2rad(0.2),
    offset_thresh=1000,
    height_tolerance=40
)


analisis = analizar_planos(planos2)

for p in analisis:
    print(f"Plano {p['id']}: inclinación = {p['pendiente_grados']:.2f}°, dirección = {p['direccion']}, puntos = {len(p['puntos_idx'])}")

visualizar_planos_3d2(df_clean, planos2)