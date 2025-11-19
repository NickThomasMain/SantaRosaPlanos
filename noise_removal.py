import pandas as pd
import numpy as np
import hdbscan
import plotly.express as px

from db_clustering import visualize_clusters


import numpy as np
import hdbscan

def filter_main_cluster(
        df,
        min_cluster_size=400,
        min_samples=20,
        visualize=False
    ):
    """
    Realiza un clustering con HDBSCAN sobre un DataFrame que contiene
    las columnas 'x', 'y', 'z' y devuelve únicamente el clúster más grande.
    Opcionalmente visualiza los clústeres antes y después del filtrado.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada que debe contener las columnas 'x', 'y', 'z'.
    min_cluster_size : int
        Tamaño mínimo de un clúster para HDBSCAN.
    min_samples : int
        Número mínimo de muestras para un clustering más robusto.
    visualize : bool
        Si es True, visualiza los clústeres antes y después del filtrado.

    Retorna
    -------
    df_clean : DataFrame
        DataFrame filtrado que contiene solamente el clúster principal.
    labels_clean : ndarray
        Etiquetas de clúster correspondientes a los puntos filtrados.
    """

    # ----------------------------------------------------------
    # 1) Extraer las columnas espaciales (x, y, z)
    # ----------------------------------------------------------
    X = df[["x", "y", "z"]].values

    # ----------------------------------------------------------
    # 2) Ejecutar HDBSCAN
    # ----------------------------------------------------------
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom"
    )

    labels = clusterer.fit_predict(X)

    # ---- Visualización previa (si se solicita) ----
    if visualize:
        try:
            visualize_clusters(df, labels, title_text="HDBSCAN – Clústeres antes del filtrado", point_size=2)
        except Exception as e:
            print("⚠️ Error durante la visualización previa:", e)

    # ----------------------------------------------------------
    # 3) Identificar el clúster más grande (excluyendo ruido)
    # ----------------------------------------------------------
    valid = labels >= 0
    largest_cluster = np.argmax(np.bincount(labels[valid]))
    mask = labels == largest_cluster

    df_clean = df[mask].copy()
    labels_clean = labels[mask]

    # ----------------------------------------------------------
    # 4) Información útil
    # ----------------------------------------------------------
    print("Puntos originales:", len(df))
    print("Puntos en el clúster principal:", len(df_clean))
    print("Puntos eliminados:", len(df) - len(df_clean))

    # ---- Visualización posterior (si se solicita) ----
    if visualize:
        try:
            visualize_clusters(df_clean, labels_clean, title_text="HDBSCAN – Clúster principal (filtrado)", point_size=2)
        except Exception as e:
            print("⚠️ Error durante la visualización posterior:", e)

    return df_clean, labels_clean