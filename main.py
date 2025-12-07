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
    Detecta planos mediante RANSAC por cluster o de manera global.

    Requisitos:
    -----------
    - El DataFrame `df` debe contener una columna 'orig_id' con la ID global
      de cada punto.
    - `labels` debe ser un array o Serie alineado posicionalmente con `df`.

    Retorna:
    --------
    dict:
        Diccionario del tipo {cluster_id: [planos,...]} donde cada plano es un
        diccionario que contiene:
            - coef: coeficientes del plano (a, b)
            - intercept: término independiente
            - puntos_idx: IDs globales (orig_id) de los puntos inliers
            - n_inliers: número de inliers detectados
            - cobertura: proporción de inliers respecto al total del subset
            - model: modelo RANSAC ajustado
    """

    # ---- Validación de columnas espaciales ----
    if set(["x", "y", "z"]).issubset(df.columns):
        xcol, ycol, zcol = "x", "y", "z"
    elif set(["east", "north", "altitud"]).issubset(df.columns):
        xcol, ycol, zcol = "east", "north", "altitud"
    else:
        raise ValueError(
            "El DataFrame debe contener ('x','y','z') o ('east','north','altitud')."
        )

    # ---- Validar que exista la columna orig_id ----
    if "orig_id" not in df.columns:
        raise ValueError(
            "df debe contener la columna 'orig_id' con IDs globales. Añádela antes de llamar a esta función."
        )

    results = {}

    # ----------------------------------------------------------------------
    # Función auxiliar: detecta el mejor plano dentro de un subconjunto dado
    # ----------------------------------------------------------------------
    def _detect_on_positions(positions, label_key):
        """
        Ejecuta RANSAC sobre un subconjunto de posiciones (iloc) del DataFrame.

        Parámetros:
        -----------
        positions : array[int]
            Índices posicionales dentro de df (para iloc).
        label_key : str
            Etiqueta para trazas de depuración.

        Devuelve:
        ---------
        list:
            Lista que contiene 0 o 1 plano detectado (en forma de dict).
        """

        if len(positions) < n_min:
            if verbose:
                print(f"[{label_key}] Puntos insuficientes ({len(positions)}) < n_min ({n_min}).")
            return []

        # Subconjunto preservando el orden original
        subset = df.iloc[positions]
        pts = subset[[xcol, ycol, zcol]].to_numpy()
        X = pts[:, :2]
        y = pts[:, 2]

        best_model = None
        best_inliers = 0
        best_mask = None

        n_tries = max_iter
        if verbose:
            print(f"[{label_key}] Buscando el mejor plano entre {n_tries} intentos sobre {len(positions)} puntos...")

        # Intentos iterativos de RANSAC
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

        # Si no se encuentra un buen plano
        if best_model is None or best_inliers < n_min:
            if verbose:
                print(f"[{label_key}] No se encontró un plano adecuado (best_inliers={best_inliers}).")
            return []

        # Mapear inliers locales a posiciones globales
        local_inlier_pos = np.where(best_mask)[0]
        global_positions = np.array(positions)[local_inlier_pos]
        orig_ids = df.iloc[global_positions]["orig_id"].to_numpy()

        coef = best_model.estimator_.coef_.copy()
        intercept = best_model.estimator_.intercept_.copy()

        if verbose:
            print(f"[{label_key}] Mejor plano: {best_inliers} inliers (cobertura={best_inliers/len(positions):.2%})")

        plano = {
            "id": 1,
            "coef": coef,
            "intercept": intercept,
            "puntos_idx": orig_ids,
            "n_inliers": int(best_inliers),
            "cobertura": best_inliers / len(positions),
            "model": best_model
        }
        return [plano]

    # ----------------------------------------------------------------------
    # Modo por cluster
    # ----------------------------------------------------------------------
    if per_cluster and (labels is not None):

        # Normalizar labels a array posicional
        if isinstance(labels, pd.Series):
            label_series = labels
            if not label_series.index.equals(df.index):
                if len(label_series) == len(df):
                    if verbose:
                        print("labels: Serie con distinto índice; se usará alineación posicional.")
                    label_array = label_series.to_numpy()
                else:
                    raise ValueError(
                        "El índice de labels no coincide con df.index y las longitudes difieren."
                    )
            else:
                label_array = label_series.to_numpy()
        else:
            label_array = np.asarray(labels)

        if label_array.shape[0] != len(df):
            raise ValueError(
                f"Longitud de labels ({label_array.shape[0]}) distinta al número de filas de df ({len(df)})."
            )

        unique_clusters = sorted(set(label_array))
        unique_clusters = [c for c in unique_clusters if c != -1]  # Excluir ruido
        if verbose:
            print("Clusters detectados (excluyendo -1):", unique_clusters)

        for c in unique_clusters:
            pos = np.where(label_array == c)[0]
            if pos.size == 0:
                if verbose:
                    print(f"[cluster_{c}] No se encontraron posiciones -> Saltando")
                continue

            planos = _detect_on_positions(pos, label_key=f"cluster_{c}")
            results[int(c)] = planos

    # ----------------------------------------------------------------------
    # Modo global (sin clusters)
    # ----------------------------------------------------------------------
    else:
        all_positions = np.arange(len(df))
        planos = _detect_on_positions(all_positions, label_key="global")
        results["global"] = planos

    return results



def analizar_planos(planos):
    """
    Analiza una lista de planos y calcula parámetros geométricos
    como la inclinación (pendiente) y la orientación cardinal.

    Parámetros:
    -----------
    planos : list[dict]
        Lista de planos obtenidos mediante detectar_planos_pipeline().

    Devuelve:
    ---------
    list[dict]:
        Lista con información por plano:
            - id: identificador del plano
            - pendiente_grados: inclinación en grados
            - direccion: orientación cardinal (N, NE, E, SE, S, SW, W, NW)
            - puntos_idx: IDs globales de los puntos inliers
    """

    def calcular_orientacion(a, b):
        """Calcula la dirección cardinal basada en los coeficientes (a, b) del plano."""
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
                if hasattr(plano["puntos_idx"], "tolist")
                else plano["puntos_idx"]
            )
        })

    return salida



def main():
    """
    Pipeline principal:
    -------------------
    1) Carga del dataset.
    2) Visualización inicial.
    3) Eliminación de ruido mediante filter_main_cluster().
    4) Búsqueda en rejilla para optimizar pesos de HDBSCAN.
    5) Ejecución de HDBSCAN con los mejores parámetros.
    6) Detección de planos locales por cluster.
    7) Selección del mejor resultado.
    8) Visualización final de planos detectados.
    """

    df = pd.read_csv("data/asro_centroides_peaks_mayor_2450.csv")
    df["orig_id"] = df.index

    # Crear columnas estándar para visualización y RANSAC
    df["x"] = df["east"]
    df["y"] = df["north"]
    df["z"] = df["altitud"]

    visualizar_nube_3d(df)

    # ---------------------------
    # Eliminación del cluster principal
    # ---------------------------
    df_clean, labels_clean = filter_main_cluster(
        df,
        min_cluster_size=400,
        min_samples=20
    )

    visualizar_nube_3d(df_clean)

    # ---------------------------
    # Grid search de pesos
    # ---------------------------
    steps = np.arange(1, 4.01, 0.5)
    steps_xy = np.arange(1, 2.01, 0.5)

    weight_grid = {
        "w_xy": steps_xy.tolist(),
        "w_z": steps.tolist(),
        "w_nxy": steps_xy.tolist(),
        "w_nz": steps.tolist(),
        "w_slope": steps.tolist()
    }

    results = grid_search_weights(df_clean, weight_grid)

    final_outputs = []
    top3 = results[:3]

    for entry in top3:
        params = entry["params"]

        # HDBSCAN con los pesos seleccionados
        features, df_feat, local_scale = feature_vector(df_clean, **params)

        labels_hdb, clusterer = run_hdbscan(
            df_clean,
            visualize=True,
            features_override=features
        )

        df_tmp = df_clean.copy()
        df_tmp["cluster"] = labels_hdb

        # Detección de planos por cluster
        planes = detectar_planos_pipeline(
            df_tmp,
            labels=labels_hdb,
            per_cluster=True,
            tolerancia=6,
            n_min=30
        )

        num_planes = sum(len(v) for v in planes.values())

        final_outputs.append({
            "params": params,
            "labels": labels_hdb,
            "planes": planes,
            "num_planes": num_planes
        })

    # Seleccionar el mejor resultado (mayor número de planos)
    best_output = max(final_outputs, key=lambda x: x["num_planes"])

    best_labels = best_output["labels"]
    best_planes = best_output["planes"]

    print("Mejores parámetros:", best_output["params"])
    print("Número de planos detectados:", best_output["num_planes"])

    # Preparar df_clean para visualización final
    df_clean["cluster"] = best_labels

    all_planes = []
    for cluster_id, planes in best_planes.items():
        all_planes.extend(planes)

    visualizar_planos_3d(df_clean, all_planes)


if __name__ == "__main__":
    main()
