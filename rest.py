def detectar_planos_global(csv_path, tolerancia=4.0, n_min=30, max_iter=10, cobertura_objetivo=0.8):
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