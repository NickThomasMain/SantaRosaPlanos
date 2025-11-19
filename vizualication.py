import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import alphashape
import trimesh

from scipy.spatial import ConvexHull



def visualizar_poligonos_3d(csv_path, planos, point_size=2):
    """
    Visualiza puntos 3D y los planos detectados en Plotly con polígonos exactos.
    Incluye un menú para activar/desactivar todos los planos a la vez.
    """

    # --- Punkte laden ---
    df = pd.read_csv(csv_path)
    X = df["east"].values
    Y = df["north"].values
    Z = df["altitud"].values

    fig = go.Figure()

    # --- 1) Punktwolke ---
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(size=point_size, color="gray", opacity=0.7),
        name="Puntos",
        showlegend=True
    ))

    # Farben
    colores = px.colors.qualitative.Set2
    plano_traces = []

    # --- 2) Polygone für jede Ebene ---
    for i, p in enumerate(planos):

        # Ebene
        a, b = p["coef"]
        c = p["intercept"]

        # Punkte extrahieren
        pts3d = np.column_stack([
            X[p["puntos_ids"]],
            Y[p["puntos_ids"]],
            Z[p["puntos_ids"]]
        ])

        if pts3d.shape[0] < 3:
            continue

        # Normalenvektor der Ebene
        # Ebene: z = a x + b y + c → ax + by - z + c = 0 → Normal = (a, b, -1)
        normal = np.array([a, b, -1.0])

        # Exaktes Polygon berechnen
        poly = polygon_from_plane(pts3d, normal)

        # Für Mesh3d benötigen wir Dreiecke → triangulieren
        hull2 = ConvexHull(poly[:, :2])  # 2D triangulation reicht

        fig.add_trace(go.Mesh3d(
            x=poly[:, 0],
            y=poly[:, 1],
            z=poly[:, 2],
            i=hull2.simplices[:, 0],
            j=hull2.simplices[:, 1],
            k=hull2.simplices[:, 2],
            color=colores[i % len(colores)],
            opacity=0.45,
            name=f"Plano {p['id']}",
            showlegend=False
        ))

        plano_traces.append(len(fig.data) - 1)

    # --- 3) Buttons für alle Ebenen ---
    total_traces = len(fig.data)

    visible_all = [True] + [trace in plano_traces for trace in range(1, total_traces)]
    visible_none = [True] + [False for _ in range(1, total_traces)]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, y=1.1,
                buttons=[
                    dict(label="Con planos",
                         method="update",
                         args=[{"visible": visible_all}]),
                    dict(label="Sin planos",
                         method="update",
                         args=[{"visible": visible_none}])
                ]
            )
        ]
    )

    # --- 4) Maßstab angleichen ---
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    xy_range = max(x_max - x_min, y_max - y_min)

    fig.update_scenes(
        xaxis=dict(range=[x_min, x_min + xy_range]),
        yaxis=dict(range=[y_min, y_min + xy_range]),
        aspectratio=dict(x=1, y=1, z=0.5)
    )

    fig.update_layout(
        title="Puntos + Polígonos exactos de planos",
        scene=dict(
            xaxis_title="East",
            yaxis_title="North",
            zaxis_title="Altitud"
        ),
        height=800
    )

    fig.show()

def visualizar_planos_3d(csv_path, planos, point_size=2):
    """
    Visualiza puntos 3D y colorea los puntos según el plano al que pertenecen.
    Adicional: muestra las superficies matemáticas de los planos RANSAC.
    """

    df = pd.read_csv(csv_path)
    X = df["east"].values
    Y = df["north"].values
    Z = df["altitud"].values

    fig = go.Figure()

    # --- 1) Todos los puntos en gris ---
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(size=point_size, color="gray", opacity=0.25),
        name="Puntos (todos)",
        showlegend=True
    ))

    # Colores para puntos
    colores = px.colors.qualitative.Set2
    plano_traces = []

    # --- 2) Puntos de cada plano ---
    for i, p in enumerate(planos):
        ids = p["puntos_ids"]

        fig.add_trace(go.Scatter3d(
            x=X[ids],
            y=Y[ids],
            z=Z[ids],
            mode='markers',
            marker=dict(
                size=point_size + 1,
                opacity=0.85,
                color=colores[i % len(colores)]
            ),
            name=f"Plano {p['id']}",
            showlegend=True
        ))
        plano_traces.append(len(fig.data) - 1)

    # --- 2.5) Superficies matemáticas de cada plano (go.Surface) ---
    for i, p in enumerate(planos):
        a, b = p["coef"]
        c = p["intercept"]

        # Bounds del plano basado en los puntos de esta superficie
        ids = p["puntos_ids"]
        x_min, x_max = X[ids].min(), X[ids].max()
        y_min, y_max = Y[ids].min(), Y[ids].max()

        # Crear grid 2D
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 30),
            np.linspace(y_min, y_max, 30)
        )
        zz = a * xx + b * yy + c

        # Añadir superficie
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, colores[i % len(colores)]],
                        [1, colores[i % len(colores)]]],
            opacity=0.35,
            showscale=False,
            name=f"Superficie plano {p['id']}",
        ))

        plano_traces.append(len(fig.data) - 1)

    # --- 3) Dropdown ---
    visible_all  = [True] + [True  for _ in plano_traces]
    visible_none = [True] + [False for _ in plano_traces]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, y=1.1,
                buttons=[
                    dict(label="Con planos", method="update",
                         args=[{"visible": visible_all}]),
                    dict(label="Sin planos", method="update",
                         args=[{"visible": visible_none}])
                ]
            )
        ]
    )

    # --- 4) Achsen ---
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()
    xy_range = max(x_max - x_min, y_max - y_min)

    fig.update_scenes(
        xaxis=dict(range=[x_min, x_min + xy_range]),
        yaxis=dict(range=[y_min, y_min + xy_range]),
        aspectratio=dict(x=1, y=1, z=0.5)
    )

    fig.update_layout(
        title="Puntos y superficies RANSAC por plano",
        scene=dict(
            xaxis_title="East",
            yaxis_title="North",
            zaxis_title="Altitud"
        ),
        height=850
    )

    fig.show()


def plot_3d_points(
    csv_path,
    x_col="east",
    y_col="north",
    z_col="altitud",
    color_col="altitud",
    color_scale="earth",
    point_size=3,
    title="Puntos 3D (interactivo)"
):
    """
    Crea una visualización 3D interactiva de datos puntuales (por ejemplo, puntos DEM).

    Parámetros:
    ------------
    csv_path : str
        Ruta al archivo CSV con los datos de los puntos.
    x_col, y_col, z_col : str
        Nombres de las columnas que contienen las coordenadas.
    color_col : str
        Columna utilizada para la escala de colores.
    color_scale : str
        Nombre de una escala de color válida en Plotly 
        (por ejemplo, 'earth', 'viridis', 'thermal').
    point_size : int o float
        Tamaño de los puntos (por defecto: 3).
    title : str
        Título del gráfico.
    """
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(csv_path)

    # Crear el gráfico 3D
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        color_continuous_scale=color_scale,
        title=title
    )

    # Ajustar el tamaño de los puntos
    fig.update_traces(marker=dict(size=point_size))

    # Mostrar el gráfico de forma interactiva
    fig.show()