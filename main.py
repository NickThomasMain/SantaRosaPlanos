from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import RANSACRegressor, LinearRegression


def detectar_planos_global(csv_path, tolerancia=6.0, n_min=30, max_iter=10, cobertura_objetivo=0.8):
    """
    Detecta múltiples planos en una nube de puntos completa usando RANSAC de manera iterativa.
    El proceso se detiene cuando se cubre un porcentaje objetivo de los puntos totales.

    Parámetros:
    ------------
    csv_path : str
        Ruta del archivo CSV con las columnas 'east', 'north' y 'altitud'.
    tolerancia : float
        Distancia máxima (en metros) permitida entre un punto y el plano (umbral RANSAC).
    n_min : int
        Número mínimo de puntos necesarios para considerar un plano válido.
    max_iter : int
        Número máximo de planos que se intentarán detectar.
    cobertura_objetivo : float
        Porcentaje (entre 0 y 1) de puntos totales que deben estar cubiertos para detener el proceso.

    Devuelve:
    -----------
    lista de diccionarios con:
        - id: número de plano
        - coef: coeficientes [a, b] del plano z = a·x + b·y + c
        - intercept: valor de intersección c
        - puntos_ids: índices de los puntos que pertenecen al plano
    """
    # Cargar CSV
    df = pd.read_csv(csv_path)
    puntos = df[["east", "north", "altitud"]].to_numpy()
    total_puntos = len(puntos)
    planos = []
    puntos_usados = 0

    for i in range(max_iter):
        if len(puntos) < n_min:
            break

        X = puntos[:, :2]
        y = puntos[:, 2]

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=tolerancia,
            random_state=42
        )
        ransac.fit(X, y)

        inliers = ransac.inlier_mask_
        if inliers.sum() < n_min:
            break

        coef = ransac.estimator_.coef_
        intercept = ransac.estimator_.intercept_

        planos.append({
            "id": i + 1,
            "coef": coef,
            "intercept": intercept,
            "puntos_ids": np.where(inliers)[0]
        })

        # Actualizar porcentaje de cobertura
        puntos_usados += inliers.sum()
        cobertura_actual = puntos_usados / total_puntos

        print(f"🔹 Iteración {i+1}: {inliers.sum()} puntos en el plano, cobertura = {cobertura_actual:.2%}")

        # Eliminar los puntos del plano actual
        puntos = puntos[~inliers]

        # Criterio de parada
        if cobertura_actual >= cobertura_objetivo:
            print(f"✅ Criterio alcanzado: {cobertura_actual:.2%} de los puntos cubiertos.")
            break

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
            "puntos_ids": plano["puntos_ids"].tolist() if hasattr(plano["puntos_ids"], 'tolist') else plano["puntos_ids"]
        })

    return salida


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


# plot_3d_points(
#     "asro_centroides_peaks_mayor_2450.csv",
#     point_size=2,
#     title="Visualización 3D de puntos"
# )

planos = detectar_planos_global(
    "asro_centroides_peaks_mayor_2450.csv",
    tolerancia=6,
    n_min=20,
    max_iter=30,
    cobertura_objetivo=0.8
)


analisis = analizar_planos(planos)

for p in analisis:
    print(f"Plano {p['id']}: inclinación = {p['pendiente_grados']:.2f}°, dirección = {p['direccion']}, puntos = {len(p['puntos_ids'])}")