import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) Cargar datos de la nube de puntos
# ---------------------------------------------------------
df = pd.read_csv("asro_centroides_peaks_mayor_2450.csv")  # columnas: cat,east,north,altitud

x = df["east"].values
y = df["north"].values
z = df["altitud"].values

puntos = np.column_stack((x, y))

# ---------------------------------------------------------
# 2) Crear el grid (raster DEM)
# ---------------------------------------------------------

res = 500  # resolución del grid en unidades CRS (metros, etc.)

# Extensión
xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

# Crear rejilla regular
xi = np.arange(xmin, xmax, res)
yi = np.arange(ymin, ymax, res)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Interpolación (puede ser: nearest, linear, cubic)
dem = griddata(puntos, z, (xi_grid, yi_grid), method="linear")

# ---------------------------------------------------------
# 3) Calcular pendiente (slope) y aspecto (aspect)
# ---------------------------------------------------------

# gradiente en X y Y
dz_dy, dz_dx = np.gradient(dem, res, res)

# pendiente en grados
slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

# aspecto en grados (0-360)
aspect = (np.degrees(np.arctan2(-dz_dx, dz_dy)) + 360) % 360

# ---------------------------------------------------------
# 4) Preparar datos para clustering
# ---------------------------------------------------------

# aplanar las matrices
slope_vec = slope.flatten()
aspect_vec = aspect.flatten()

# quitar celdas NaN (producto de la interpolación)
mask = ~np.isnan(slope_vec)
features = np.column_stack((slope_vec[mask], aspect_vec[mask]))

# ---------------------------------------------------------
# 5) K-Means clustering
# ---------------------------------------------------------

k = 4  # número de clusters
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(features)

# crear raster de etiquetas
cluster_raster = np.full(slope_vec.shape, np.nan)
cluster_raster[mask] = labels
cluster_raster = cluster_raster.reshape(slope.shape)

# ---------------------------------------------------------
# 6) Visualización
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("DEM (altitud)")
plt.imshow(dem, origin="lower")
plt.colorbar()

plt.subplot(1, 4, 2)
plt.title("Slope (pendiente)")
plt.imshow(slope, origin="lower")
plt.colorbar()

plt.subplot(1, 4, 3)
plt.title("Aspect")
plt.imshow(aspect, origin="lower")
plt.colorbar()

plt.subplot(1, 4, 4)
plt.title("Clusters (slope + aspect)")
plt.imshow(cluster_raster, origin="lower")
plt.colorbar()

plt.tight_layout()
plt.show()
