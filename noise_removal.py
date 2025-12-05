import pandas as pd
import numpy as np
import hdbscan
import plotly.express as px

from db_clustering import visualize_clusters


def filter_main_cluster(
        df,
        min_cluster_size=400,
        min_samples=20
    ):
    """
    Realiza un clustering con HDBSCAN sobre un DataFrame que contiene
    las columnas 'x', 'y', 'z' y devuelve únicamente el clúster más grande.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame de entrada que debe contener las columnas 'x', 'y', 'z'.
    min_cluster_size : int
        Tamaño mínimo de un clúster para HDBSCAN.
    min_samples : int
        Número mínimo de muestras para un clustering más robusto.

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


    return df_clean, labels_clean