import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
import hdbscan

# ----------------------------------------------------------
# 1) Cargar datos
# ----------------------------------------------------------
def load_pointcloud(csv_path):
    """
    Carga un archivo CSV con columnas: cat, east, north, altitud.
    Renombra las columnas a x, y, z para estandarizar el proceso.
    """
    df = pd.read_csv(csv_path)

    # Renombrar columnas a convención estándar (x, y, z)
    df = df.rename(columns={
        "east": "x",
        "north": "y",
        "altitud": "z"
    })

    return df[["x", "y", "z"]]   # cat no se usa por ahora


# ----------------------------------------------------------
# 2) Preprocesamiento con ponderación de Z
# ----------------------------------------------------------
def scale_features(df, z_weight=1.5):
    """
    Aplica un factor de ponderación a la columna z (altitud) 
    ANTES de normalizar con StandardScaler.
    
    Parámetros
    ----------
    df : DataFrame con columnas x, y, z
    z_weight : float
        Factor que multiplica la altura para aumentar o reducir
        su impacto en la distancia euclidiana usada por DBSCAN.
        - 1.0  -> sin cambios (baseline)
        - >1.0 -> la altura pesa más
        - <1.0 -> la altura pesa menos

    Retorna
    -------
    pts_scaled : array normalizado (numpy)
    scaler : StandardScaler ajustado (por si se quiere revertir)
    df_mod : df modificado (con z ponderado), útil para debug
    """

    df_mod = df.copy()
    df_mod["z"] = df_mod["z"] * z_weight   # Ponderar altitud

    scaler = StandardScaler()
    pts_scaled = scaler.fit_transform(df_mod[["x", "y", "z"]].values)

    return pts_scaled, scaler, df_mod


def compute_normals(df, k_neighbors=30):
    """
    Berechnet robuste, ausgerichtete Normalen mittels PCA über k-Nachbarn.

    Parameter
    ---------
    df : DataFrame mit Spalten ['x','y','z']
    k_neighbors : int
        Anzahl KNN für die lokale Flächenapproximation.

    Returns
    -------
    normals : ndarray (N,3)
        Ausgerichtete und normalisierte Normalenvektoren.
    """

    points = df[["x", "y", "z"]].values
    N = len(points)

    if N < k_neighbors:
        raise ValueError("Zu wenige Punkte für Normalenberechnung.")

    # --- 1) KNN Suche (KD-Tree) ---
    nn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree')
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    normals = np.zeros((N, 3), dtype=float)

    # --- 2) PCA pro Punkt ---
    for i in range(N):
        neigh = points[indices[i]]
        cov = np.cov(neigh.T)

        # Eigenzerlegung
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Kleinster Eigenwert -> Normalenrichtung
        normal = eigvecs[:, np.argmin(eigvals)]

        # Normalisieren
        normal /= np.linalg.norm(normal) + 1e-12

        normals[i] = normal

    # --- 3) Orientierung erzwingen: Normalen zeigen nach oben ---
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1

    return normals


def compute_slope(normals):
    """
    Pendiente = ángulo entre la normal y el eje vertical.
    slope = acos(|n_z|)
    """
    nz = np.abs(normals[:, 2])
    slope = np.arccos(nz)
    return slope



def build_feature_vector(
        df,
        k_neighbors=20,
        w_x=1.0, w_y=1.0, w_z=1.0,
        w_nx=1.0, w_ny=1.0, w_nz=1.0,
        w_slope=1.0
    ):
    """
    Construye el vector de características y aplica un peso independiente
    a cada característica FINAL (después de calcular normales, slope y roughness).

    Importante:
    - Las normales, slope y roughness se calculan sobre las coordenadas
      métricas originales (sin multiplicar Z antes).
    - Sólo después se aplican los pesos w_* al vector final.
    - Finalmente se devuelve el array de features (sin escalado) y df_feat.
    """
    df_feat = df.copy()

    # --- COMIENZO: cálculo de features geométricas sobre coordenadas originales ---
    normals = compute_normals(df_feat, k_neighbors=k_neighbors)
    df_feat["nx"] = normals[:, 0]
    df_feat["ny"] = normals[:, 1]
    df_feat["nz"] = normals[:, 2]

    df_feat["slope"] = compute_slope(normals)

    # --- FIN cálculo geométrico ---

    # --- Construir vector sin pesos todavía ---
    # Usamos aquí las columnas originales x,y,z (no modificadas)
    features_raw = np.column_stack([
        df_feat["x"].values,
        df_feat["y"].values,
        df_feat["z"].values,
        df_feat["nx"].values,
        df_feat["ny"].values,
        df_feat["nz"].values,
        df_feat["slope"].values
    ])

    # --- Aplicar ponderaciones por columna en el mismo orden ---
    weights = np.array([w_x, w_y, w_z, w_nx, w_ny, w_nz, w_slope], dtype=float)
    features_weighted = features_raw * weights  # broadcasting por columnas

    return features_weighted, df_feat


# ----------------------------------------------------------
# 3) Curva K-distancia para elegir eps
# ----------------------------------------------------------
def plot_k_distance(points_scaled, k=100):
    """
    Traza la curva de distancia al k-ésimo vecino más cercano.
    El "codo" indica un buen valor de eps para DBSCAN.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points_scaled)
    distances, _ = nbrs.kneighbors(points_scaled)
    k_distances = np.sort(distances[:, -1])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=k_distances,
        mode='lines',
        name=f"{k}-NN distance"
    ))
    fig.update_layout(
        title="Curva de distancia k-vecinos (usar el codo para elegir eps)",
        xaxis_title="Punto ordenado",
        yaxis_title=f"Distancia al {k}-ésimo vecino"
    )
    fig.show()

# ----------------------------------------------------------
# 3) Clustering con DBSCAN
# ----------------------------------------------------------
def run_dbscan(points_scaled, eps=0.5, min_samples=100):
    """
    Ejecuta DBSCAN sobre los puntos ya escalados.
    Retorna los labels para cada punto.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points_scaled)
    return labels



# ----------------------------------------------------------
# 4) Visualización con Plotly
# ----------------------------------------------------------
def visualize_clusters(df, labels, title_text="DBSCAN – Clústeres básicos", point_size=2):
    """
    Visualiza los puntos en 3D usando Plotly con colores discretos por clúster,
    ejes proporcionados y un ratio espacial coherente.
    """

    df_plot = df.copy()
    df_plot["cluster"] = labels.astype(str)

    # --- Crear figura ---
    fig = px.scatter_3d(
        df_plot,
        x="x", y="y", z="z",
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.Set1,  # colores NÍTIDOS, no gradientes
        title=title_text,
        opacity=0.8,
        width=900,
        height=750
    )

    fig.update_traces(marker=dict(size=point_size))

    # ============================================================
    #  🔧 Ajuste de rangos y proporciones del espacio 3D
    # ============================================================

    X = df_plot["x"].values
    Y = df_plot["y"].values
    Z = df_plot["z"].values

    # Rango XY combinado
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    xy_span = max(x_max - x_min, y_max - y_min)

    # Para Z: mismo centro, pero rango reducida (p.ej. 40% del XY span)
    z_min, z_max = Z.min(), Z.max()
    z_mid = (z_min + z_max) / 2
    z_span = xy_span * 0.02  # Ajustable: 0.4 = 40%, experimenta si quieres

    fig.update_scenes(
        xaxis=dict(range=[x_min, x_min + xy_span]),
        yaxis=dict(range=[y_min, y_min + xy_span]),
        zaxis=dict(range=[z_mid - z_span/2, z_mid + z_span/2]),

        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.5)  # Z = halb so groß wie XY
    )

    # Ejes + Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="East",
            yaxis_title="North",
            zaxis_title="Altitud"
        ),
        height=850
    )

    fig.show()


# ----------------------------------------------------------
# 5) Pipeline principal
# ----------------------------------------------------------
def weighted_clustering(csv_path, mode="features", k_neighbors=20,
         w_x=1.0, w_y=1.0, w_z=1.0,
         w_nx=1.0, w_ny=1.0, w_nz=1.0,
         w_slope=1.0,
         dbscan_eps=0.5, dbscan_min_samples=25):
    """
    Pipeline:
    - mode="basic": sólo x,y,z escalados (legacy behaviour)
    - mode="features": cálculo geométrico (normales, slope, roughness),
      luego aplicación de pesos uniformes por feature, luego StandardScaler,
      luego clustering.
    """

    df = load_pointcloud(csv_path)

    if mode == "basic":
        pts_scaled, scaler = scale_features(df)   # legacy: z_weight si se usaba ahí
        features = pts_scaled

    elif mode == "features":
        features, df_feat = build_feature_vector(
            df,
            k_neighbors=k_neighbors,
            w_x=w_x, w_y=w_y, w_z=w_z,
            w_nx=w_nx, w_ny=w_ny, w_nz=w_nz,
            w_slope=w_slope
        )

        # 1: standardisieren
        scaler = StandardScaler()
        features_std = scaler.fit_transform(features)

        # 2: Gewichte nach StandardScaler anwenden → wirken wirklich!
        weights = np.array([w_x, w_y, w_z, w_nx, w_ny, w_nz, w_slope])
        features = features_std * weights


    else:
        raise ValueError("mode must be 'basic' or 'features'")

    # K-dist / eps opcional
    # plot_k_distance(features, k=20)

    labels = run_dbscan(features, eps=dbscan_eps, min_samples=dbscan_min_samples)

    visualize_clusters(df, labels)



    # weighted_clustering("asro_centroides_peaks_mayor_2450.csv",
    #  mode="features",
    #  k_neighbors=20,
    #  w_x=1.0, w_y=1.0, w_z=3.0,
    #  w_nx=1.0, w_ny=1.0, w_nz=3.0,
    #  w_slope=2.0,
    #  dbscan_eps=0.8, dbscan_min_samples=30)
