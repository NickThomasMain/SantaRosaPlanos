import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px


def visualizar_planos_3d(df_clean, planos, point_size=2):
    """
    Visualiza la nube filtrada (df_clean) y los planos detectados (planos)
    generados por detectar_planos_hdbscan().
    Usa exclusivamente las columnas: east, north, altitud.
    """

    # --- 1) Extraer arrays ---
    X = df_clean["east"].values
    Y = df_clean["north"].values
    Z = df_clean["altitud"].values

    fig = go.Figure()

    # --- 2) Todos los puntos en gris ---
    fig.add_trace(go.Scatter3d(
        x=X, y=Y, z=Z,
        mode='markers',
        marker=dict(size=point_size, color="gray", opacity=0.25),
        name="Puntos (df_clean)",
        showlegend=True
    ))

    colores = px.colors.qualitative.Set2
    n_colors = len(colores)

    trace_indices = []

    # --- 3) Puntos de cada plano ---
    for i, p in enumerate(planos):
        ids = p["puntos_idx"]                     # globale IDs
        pos = df_clean.index.get_indexer(ids)     # lokale Positionen (0..n-1)       
        color = colores[i % n_colors]

        fig.add_trace(go.Scatter3d(
            x=X[pos], y=Y[pos], z=Z[pos],
            mode='markers',
            marker=dict(
                size=point_size + 1,
                opacity=0.9,
                color=color
            ),
            name=f"Plano {p['id']}",
            showlegend=True
        ))

        trace_indices.append(len(fig.data) - 1)

    # --- 4) Superficie del plano ---
    for i, p in enumerate(planos):
        a, b = p["coef"]
        c = p["intercept"]
        ids = p["puntos_idx"]
        color = colores[i % n_colors]

        # límites para la superficie
        pos = df_clean.index.get_indexer(ids)
        x_min, x_max = X[pos].min(), X[pos].max()
        y_min, y_max = Y[pos].min(), Y[pos].max()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 25),
            np.linspace(y_min, y_max, 25)
        )
        zz = a * xx + b * yy + c

        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, color], [1, color]],
            opacity=0.35,
            showscale=False,
            name=f"Superficie plano {p['id']}",
        ))

        trace_indices.append(len(fig.data) - 1)

    # --- 5) Dropdown (Planos ein/ausblenden) ---
    n_traces = len(fig.data)

    visible_all = [True] * n_traces        # alles sichtbar
    visible_none = [False] * n_traces      # nur die Punktwolke sichtbar
    visible_none[0] = True                 # Basispunkte an lassen

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
                         args=[{"visible": visible_none}]),
                ]
            )
        ]
    )


    X = df_clean["x"].values
    Y = df_clean["y"].values
    Z = df_clean["z"].values

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


def visualizar_nube_3d(df_clean, point_size=2):
    """
    Visualiza únicamente la nube de puntos (df_clean)
    usando columnas: east, north, altitud.
    Incluye coloración continua según la altitud (Z).
    """

    # --- 1) Extraer arrays ---
    X = df_clean["east"].values
    Y = df_clean["north"].values
    Z = df_clean["altitud"].values

    # --- 2) Crear figura ---
    fig = go.Figure()

    # --- 3) Scatter 3D con color por altitud ---
    fig.add_trace(go.Scatter3d(
        x=X,
        y=Y,
        z=Z,
        mode="markers",
        marker=dict(
            size=point_size,
            color=Z,                     # Color según altitud
            colorscale="Viridis",        # Paleta continua
            showscale=True,              # Barra de color
            opacity=0.9
        ),
        name="Punto"
    ))

    # --- 4) Escalado uniforme del espacio ---
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    # Größeren XY-Bereich bestimmen
    xy_span = max(x_max - x_min, y_max - y_min)

    # Z-Bereich reduzieren für 2.5D-Effekt (wie in Ihrer Funktion)
    z_min, z_max = Z.min(), Z.max()
    z_mid = (z_min + z_max) / 2
    z_span = xy_span * 0.02  # flache Darstellung

    fig.update_scenes(
        xaxis=dict(range=[x_min, x_min + xy_span]),
        yaxis=dict(range=[y_min, y_min + xy_span]),
        zaxis=dict(range=[z_mid - z_span/2, z_mid + z_span/2]),

        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.5)
    )

    # --- 5) Achsen + Layout ---
    fig.update_layout(
        scene=dict(
            xaxis_title="East",
            yaxis_title="North",
            zaxis_title="Altitud"
        ),
        height=800,
        title="Nube de puntos (solo puntos, color por altitud)"
    )

    fig.show()
