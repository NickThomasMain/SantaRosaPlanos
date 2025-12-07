import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
import hdbscan
from itertools import product
from sklearn.metrics import silhouette_score
from hdbscan.validity import validity_index as hdbscan_dbcv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------
# estimate_dynamic_k (robusto)
# --------------------------
def estimate_dynamic_k(points, k0=20, k_min=10, k_max=30, debug=True):
    """
    Calcula un valor dinámico de k para cada punto en función de la densidad local.
    Retorna: (k_dynamic (np.ndarray int), local_scale (np.ndarray float))
    """
    points = np.asarray(points)
    N = points.shape[0]
    if N == 0:
        raise ValueError("Arreglo de puntos vacío.")
    if k0 >= N:
        k0 = max(1, N - 1)

    nbrs = NearestNeighbors(n_neighbors=k0+1).fit(points)
    dists, idxs = nbrs.kneighbors(points)

    # Distancia al k0-ésimo vecino: indicador de densidad local
    local_scale = dists[:, k0]

    # Normalización robusta mediante percentiles (reduce influencia de outliers)
    lo = np.percentile(local_scale, 2)
    hi = np.percentile(local_scale, 98)
    s = (local_scale - lo) / (hi - lo + 1e-9)
    s = np.clip(s, 0.0, 1.0)

    k_dynamic = (k_min + s * (k_max - k_min)).astype(int)
    k_dynamic = np.clip(k_dynamic, k_min, k_max)

    if debug:
        print("estimate_dynamic_k: N=%d k_min=%d k_max=%d -> estadísticas de k_dyn: min=%d median=%d max=%d" % (
            N, k_min, k_max, int(k_dynamic.min()), int(np.median(k_dynamic)), int(k_dynamic.max())
        ))

    return k_dynamic, local_scale


def grid_search_weights(df_clean, weight_grid):
    """
    Evalúa todas las combinaciones de pesos especificadas en weight_grid
    y califica cada combinación mediante evaluate_clustering().
    """
    keys = list(weight_grid.keys())
    values = list(weight_grid.values())

    results = []

    for combination in product(*values):
        params = dict(zip(keys, combination))
        print(f"Probando combinación: {params}")

        metrics = evaluate_clustering(df_clean, params)
        print(metrics)
        results.append(metrics)

    # Ordenar por puntuación total
    results_sorted = sorted(results, key=lambda x: x["total_score"], reverse=True)

    print("\nMejores 3 combinaciones:")
    for r in results_sorted[:3]:
        print(r)

    return results_sorted


def evaluate_clustering(df_clean, weight_params):
    """
    Ejecuta HDBSCAN con los pesos proporcionados y calcula:
    - Coeficiente Silhouette
    - Proporción de ruido
    - Estabilidad media de los clústeres
    - Índice DBCV (especial para HDBSCAN)

    Retorna: Diccionario con todas las métricas + puntuación total
    """

    # Construcción de features
    features, df_feat, local_scale = feature_vector(df_clean, **weight_params)

    # Ejecución de HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=10,
        cluster_selection_method="eom",
        metric="euclidean"
    ).fit(features)

    labels = clusterer.labels_

    # 1) Proporción de ruido
    noise_ratio = np.sum(labels == -1) / len(labels)

    # 2) Silhouette Score (si hay más de un clúster válido)
    if len(set(labels)) > 1 and len(set(labels[labels >= 0])) > 1:
        try:
            sil = silhouette_score(features, labels)
        except:
            sil = -1
    else:
        sil = -1

    # 3) Estabilidad del clúster (probabilidades de HDBSCAN)
    stability = float(np.mean(clusterer.probabilities_))

    # 4) DBCV Score
    try:
        dbcv_score = hdbscan_dbcv(features, labels)
    except:
        dbcv_score = -1

    total_score = (sil + stability + dbcv_score - noise_ratio) / 4

    return {
        "params": weight_params.copy(),
        "silhouette": sil,
        "noise_ratio": noise_ratio,
        "stability": stability,
        "dbcv": dbcv_score,
        "total_score": total_score
    }


# --------------------------
# compute_normals (soporta k_dynamic)
# --------------------------
def compute_normals(df, k_neighbors=30, k_dynamic=None, min_k_allowed=3, debug=False):
    """
    Calcula normales por punto; soporta un arreglo k_dynamic opcional.
    - Si k_dynamic es None: utiliza k_neighbors global.
    - Se construye un KNN con n_neighbors = max_k + 1 y se usan indices[i,1:k+1].
    """
    points = df[["x", "y", "z"]].values
    N = points.shape[0]

    # Sanitización de k_dynamic
    if k_dynamic is not None:
        k_dynamic = np.asarray(k_dynamic).flatten()
        if k_dynamic.shape[0] != N:
            raise ValueError(f"Longitud de k_dynamic {k_dynamic.shape[0]} != número de puntos {N}")
        k_dynamic = k_dynamic.astype(int)

    if k_dynamic is None:
        max_k = int(k_neighbors)
    else:
        max_k = int(np.max(k_dynamic))

    # No solicitar más vecinos de los que existen
    if max_k + 1 > N:
        max_k = N - 1
        if max_k < 1:
            raise ValueError("No hay suficientes puntos para calcular normales.")
        if debug:
            print("max_k ajustado a", max_k)

    # Construcción del KNN
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, algorithm="kd_tree")
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    normals = np.zeros((N, 3), dtype=float)

    for i in range(N):

        # Determinar k para este punto
        if k_dynamic is None:
            k = int(k_neighbors)
        else:
            k = int(k_dynamic[i])

        # Limitar a rango válido
        k = int(np.clip(k, min_k_allowed, max_k))

        neigh_idx = indices[i, 1:k+1]
        neigh = points[neigh_idx]

        # Centrado y covarianza
        pts_centered = neigh - np.mean(neigh, axis=0)
        cov = np.dot(pts_centered.T, pts_centered) / max(1, (k - 1))

        # Descomposición en autovalores/autovectores
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, np.argmin(eigvals)]

        # Normalizar y asegurar orientación nz >= 0
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / (norm + 1e-12)
        if normal[2] < 0:
            normal = -normal

        normals[i] = normal

    return normals


# --------------------------
# compute_slope
# --------------------------
def compute_slope(normals):
    """
    Calcula la pendiente como el ángulo entre la normal y el eje vertical.
    slope = arccos(|nz|)  (radianes)
    """
    nz = np.abs(normals[:, 2])
    nz = np.clip(nz, -1.0, 1.0)
    slope = np.arccos(nz)
    return slope


# --------------------------
# feature_vector
# --------------------------
def feature_vector(
        df,
        k_neighbors=20,
        w_xy=1.0, w_z=1.0,
        w_nxy=0, w_nz=0,
        w_slope=2.0,
        normalize=True,
        debug=True
    ):
    """
    Genera el conjunto de features utilizando estimate_dynamic_k + compute_normals.
    Retorna: features_weighted, df_feat, local_scale
    """

    # Asignación de pesos compartidos
    w_x = w_xy
    w_y = w_xy

    w_nx = w_nxy
    w_ny = w_nxy

    df_feat = df.copy()

    # Cálculo de k dinámico por punto
    points = df_feat[["x", "y", "z"]].values
    k_dynamic, local_scale = estimate_dynamic_k(points, k0=k_neighbors, debug=debug)

    # Cálculo de normales usando k por punto
    normals = compute_normals(df_feat, k_neighbors=k_neighbors, k_dynamic=k_dynamic, debug=debug)

    df_feat["nx"] = normals[:, 0]
    df_feat["ny"] = normals[:, 1]
    df_feat["nz"] = normals[:, 2]
    df_feat["k_dynamic"] = k_dynamic

    # Pendiente
    df_feat["slope"] = compute_slope(normals)

    # Matriz de features
    features_raw = np.column_stack([
        df_feat["x"].values,
        df_feat["y"].values,
        df_feat["z"].values,
        df_feat["nx"].values,
        df_feat["ny"].values,
        df_feat["nz"].values,
        df_feat["slope"].values
    ])

    # Normalización (x,y,z y slope); normales se mantienen en esfera unitaria
    if normalize:
        scaler_xyz = StandardScaler()
        scaler_slope = StandardScaler()

        features_raw[:, 0:3] = scaler_xyz.fit_transform(features_raw[:, 0:3])
        features_raw[:, 6:7] = scaler_slope.fit_transform(features_raw[:, 6:7])

    # Aplicación de pesos
    weights = np.array([w_x, w_y, w_z, w_nx, w_ny, w_nz, w_slope], dtype=float)
    features_weighted = features_raw * weights

    return features_weighted, df_feat, local_scale


# --------------------------
# run_hdbscan
# --------------------------
def run_hdbscan(
        df_clean,
        min_cluster_size=100,
        min_samples=10,
        visualize=False,
        features_override=None
    ):
    # -------------------------------------------
    # 1) Selección de features (por override o por defecto)
    # -------------------------------------------
    if features_override is not None:
        features = features_override
        local_scale = np.ones(len(df_clean))
    else:
        features, df_feat, local_scale = feature_vector(df_clean)

    # Ajuste adaptativo de min_samples
    median_scale = np.median(local_scale)
    iqr_scale = np.percentile(local_scale, 75) - np.percentile(local_scale, 25)

    base = 5
    factor = 1 + (iqr_scale / (median_scale + 1e-9))

    min_samples_adaptive = int(np.clip(base * factor, 5, 30))
    print("min_samples adaptativo =", min_samples_adaptive)

    # -------------------------------------------
    # 2) Ejecución de HDBSCAN
    # -------------------------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=6,
        cluster_selection_method="eom",
        metric="euclidean"
    )

    labels = clusterer.fit_predict(features)

    print("Número de clústeres encontrados (sin contar ruido):",
          len(set(labels[labels >= 0])))
    print("Puntos clasificados como ruido:", np.sum(labels == -1))

    # -------------------------------------------
    # 3) Visualización opcional
    # -------------------------------------------
    if visualize:
        try:
            visualize_clusters(
                df_clean,
                labels,
                title_text="HDBSCAN – Clústeres en la nube filtrada",
                point_size=2
            )
        except Exception as e:
            print("Error durante la visualización:", e)

    return labels, clusterer


# --------------------------
# visualize_clusters
# --------------------------
def visualize_clusters(df, labels, title_text="DBSCAN – Clústeres básicos", point_size=2):
    """
    Visualiza los clústeres resultantes en 3D usando Plotly.
    """
    df_plot = df.copy()
    df_plot["cluster"] = labels.astype(str)

    fig = px.scatter_3d(
        df_plot,
        x="x", y="y", z="z",
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.Set1,
        title=title_text,
        opacity=0.8,
        width=900,
        height=750
    )

    fig.update_traces(marker=dict(size=point_size))

    X = df_plot["x"].values
    Y = df_plot["y"].values
    Z = df_plot["z"].values

    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    xy_span = max(x_max - x_min, y_max - y_min)

    z_min, z_max = Z.min(), Z.max()
    z_mid = (z_min + z_max) / 2
    z_span = xy_span * 0.02

    fig.update_scenes(
        xaxis=dict(range=[x_min, x_min + xy_span]),
        yaxis=dict(range=[y_min, y_min + xy_span]),
        zaxis=dict(range=[z_mid - z_span/2, z_mid + z_span/2]),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.5)
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="East",
            yaxis_title="North",
            zaxis_title="Altitud"
        ),
        height=850
    )

    fig.show()
