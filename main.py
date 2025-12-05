import pandas as pd
import numpy as np

from sklearn.linear_model import RANSACRegressor, LinearRegression
from vizualication import visualizar_planos_3d, visualizar_nube_3d

from noise_removal import filter_main_cluster
from hdb_clustering import run_hdbscan, grid_search_weights, feature_vector


def detectar_planos_pipeline(
        df,
        labels=None,
        per_cluster=True,
        tolerancia=6,
        n_min=30,
        max_iter=10,
        cobertura_objetivo=0.8,
        verbose=True
    ):
    """
    Detecta planos con RANSAC por cluster (o globalmente).
    Requisitos:
      - df debe contener una columna 'orig_id' con la ID global de cada punto.
      - labels debe ser un array con la misma longitud que df (orden por fila).
    Retorna dict {cluster_id: [planos,...]} con 'puntos_idx' = global orig_id array.
    """

    # ---- Validar columnas ----
    if set(["x", "y", "z"]).issubset(df.columns):
        xcol, ycol, zcol = "x", "y", "z"
    elif set(["east", "north", "altitud"]).issubset(df.columns):
        xcol, ycol, zcol = "east", "north", "altitud"
    else:
        raise ValueError("DataFrame must contain either ('x','y','z') or ('east','north','altitud') columns.")

    # ---- Validar orig_id presente ----
    if "orig_id" not in df.columns:
        raise ValueError("df debe contener la columna 'orig_id' con IDs globales. Añádela antes de llamar.")

    results = {}

    # Helper: buscar la mejor plano (varios intentos) pero dentro de un SUBSET dado por posiciones (pos)
    def _detect_on_positions(positions, label_key):
        """
        positions: array of integer positions (row positions in df, i.e. para iloc)
        Devuelve lista con 0 o 1 plano (el mejor encontrado), donde 'puntos_idx' son orig_id globales.
        """
        if len(positions) < n_min:
            if verbose:
                print(f"[{label_key}] Not enough points ({len(positions)}) < n_min ({n_min}).")
            return []

        # Construir subset por posiciones (ORDEN preserved)
        subset = df.iloc[positions]
        pts = subset[[xcol, ycol, zcol]].to_numpy()
        X = pts[:, :2]
        y = pts[:, 2]

        best_model = None
        best_inliers = 0
        best_mask = None

        n_tries = max_iter
        if verbose:
            print(f"[{label_key}] Buscando mejor plano entre {n_tries} intentos sobre {len(positions)} pts...")

        for it in range(n_tries):
            ransac = RANSACRegressor(
                estimator=LinearRegression(),
                residual_threshold=tolerancia,
                random_state=42 + it
            )
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            n_inliers = int(inlier_mask.sum())

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_model = ransac
                best_mask = inlier_mask.copy()

        if best_model is None or best_inliers < n_min:
            if verbose:
                print(f"[{label_key}] No good plane found (best_inliers={best_inliers}).")
            return []

        # Mapear inliers locales (posición dentro del subset) -> posiciones globales (df.iloc positions)
        local_inlier_pos = np.where(best_mask)[0]          # posiciones dentro del subset: 0..len(subset)-1
        global_positions = np.array(positions)[local_inlier_pos]  # posiciones iloc dentro del df
        # Convertir a orig_id (valores globales que usarás para visualización)
        orig_ids = df.iloc[global_positions]["orig_id"].to_numpy()

        coef = best_model.estimator_.coef_.copy()
        intercept = best_model.estimator_.intercept_.copy()

        if verbose:
            print(f"[{label_key}] Mejor plano: {best_inliers} inliers, cobertura = {best_inliers/len(positions):.2%}")

        plano = {
            "id": 1,
            "coef": coef,
            "intercept": intercept,
            "puntos_idx": orig_ids,       # <-- global IDs
            "n_inliers": int(best_inliers),
            "cobertura": best_inliers / len(positions),
            "model": best_model
        }
        return [plano]


    # ---------- per-cluster ----------
    if per_cluster and (labels is not None):
        # labels -> numpy array (positional)
        if isinstance(labels, pd.Series):
            label_series = labels
            if not label_series.index.equals(df.index):
                if len(label_series) == len(df):
                    if verbose:
                        print("labels: pandas.Series con distinto Index; se usará alineación por posición.")
                    label_array = label_series.to_numpy()
                else:
                    raise ValueError("labels Series index no coincide con df.index y longitudes no coinciden.")
            else:
                label_array = label_series.to_numpy()
        else:
            label_array = np.asarray(labels)
        if label_array.shape[0] != len(df):
            raise ValueError(f"labels length ({label_array.shape[0]}) != number of rows in df ({len(df)}).")

        unique_clusters = sorted(set(label_array))
        unique_clusters = [c for c in unique_clusters if c != -1]
        if verbose:
            print("Clusters encontrados (excluyendo -1):", unique_clusters)

        for c in unique_clusters:
            # pos = posiciones (ilocs) en df donde label == c
            pos = np.where(label_array == c)[0]
            if pos.size == 0:
                if verbose:
                    print(f"[cluster_{c}] no positions found -> skip")
                continue

            planos = _detect_on_positions(pos, label_key=f"cluster_{c}")
            results[int(c)] = planos

    else:
        # Detección global: positions = todas las posiciones 0..len(df)-1
        all_positions = np.arange(len(df))
        planos = _detect_on_positions(all_positions, label_key="global")
        results["global"] = planos

    return results



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


# planos = detectar_planos_global(
#     "asro_centroides_peaks_mayor_2450.csv",
#     tolerancia=6,
#     n_min=300,
#     max_iter=30,
#     cobertura_objetivo=0.8
# )



def main():
    df = pd.read_csv("asro_centroides_peaks_mayor_2450.csv")
    df["orig_id"] = df.index

    # Visualisierungsspalten anlegen
    df["x"] = df["east"]
    df["y"] = df["north"]
    df["z"] = df["altitud"]

    visualizar_nube_3d(df)

    df_clean, labels_clean = filter_main_cluster(
        df,
        min_cluster_size=400,
        min_samples=20
    )

    visualizar_nube_3d(df_clean)

    # weight_grid = {
    #     "w_xy": [1.0, 2.0],      # gemeinsames Gewicht für x und y
    #     "w_z": [1.0, 2.0],
    #     "w_nxy": [0],         # gemeinsames Gewicht für nx und ny
    #     "w_nz": [0],
    #     "w_slope": [2.0, 4.0]
    # }

    # results = grid_search_weights(df_clean, weight_grid)

    # final_outputs = []
    # top3 = results[:3]

    # for entry in top3:
    #     params = entry["params"]

    #     # HDBSCAN erneut mit den selektierten Gewichten
    #     features, df_feat, local_scale = feature_vector(df_clean, **params)

    #     labels_hdb, clusterer = run_hdbscan(
    #         df_clean,
    #         visualize=True,
    #         features_override=features
    #     )

    #     df_tmp = df_clean.copy()
    #     df_tmp["cluster"] = labels_hdb

    #     # Ebenen detektieren
    #     planes = detectar_planos_pipeline(
    #         df_tmp,
    #         labels=labels_hdb,
    #         per_cluster=True,
    #         tolerancia=6,
    #         n_min=30
    #     )

    #     num_planes = sum(len(v) for v in planes.values())

    #     final_outputs.append({
    #         "params": params,
    #         "labels": labels_hdb,
    #         "planes": planes,
    #         "num_planes": num_planes
    #     })


    # # 1) Bestes Ergebnis auswählen
    # # Beispielkriterium: maximale Ebenenzahl
    # best_output = max(final_outputs, key=lambda x: x["num_planes"])

    # best_labels = best_output["labels"]
    # best_planes = best_output["planes"]

    # print("Beste Parameter:", best_output["params"])
    # print("Anzahl gefundener Ebenen:", best_output["num_planes"])

    # # 2) Visualisierung vorbereiten
    # df_clean["cluster"] = best_labels

    # all_planes = []
    # for cluster_id, planes in best_planes.items():
    #     all_planes.extend(planes)

    # # 3) Visualisieren
    # visualizar_planos_3d(df_clean, all_planes)


if __name__ == "__main__":
    main()
