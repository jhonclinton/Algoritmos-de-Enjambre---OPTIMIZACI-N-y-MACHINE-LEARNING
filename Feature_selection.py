# ============================================================
# EJERCICIO 1: FEATURE SELECTION USANDO ABC
# Artificial Bee Colony para selección de características
# Dataset: Breast Cancer
# Modelo evaluador: KNN
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


COLONY_SIZE = 20      # cantidad de abejas empleadas
MAX_ITER = 50         # número máximo de iteraciones
LIMIT = 10            # límite para abandonar una fuente
N_FOLDS = 5           # validación cruzada
ALPHA = 0.99          # peso del accuracy
np.random.seed(42)


data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

N_FEATURES = X.shape[1]

print("Dataset: Breast Cancer")
print("Muestras:", X.shape[0])
print("Características:", N_FEATURES)

# 1. REPRESENTACIÓN DE LA FUENTE DE ALIMENTO
# 1 = característica seleccionada
# 0 = característica descartada

def nueva_fuente():
    fuente = np.random.randint(0, 2, N_FEATURES)

    # Evita que una fuente quede sin ninguna característica
    if fuente.sum() == 0:
        fuente[np.random.randint(N_FEATURES)] = 1

    return fuente

# 2. FUNCIÓN DE APTITUD
# Evalúa accuracy con KNN y penaliza usar muchas características

def fitness(fuente):
    indices = np.where(fuente == 1)[0]

    if len(indices) == 0:
        return 0.0

    X_sub = X[:, indices]

    modelo = KNeighborsClassifier(n_neighbors=3)

    accuracy = cross_val_score(
        modelo,
        X_sub,
        y,
        cv=N_FOLDS,
        scoring="accuracy"
    ).mean()

    ratio_caracteristicas = len(indices) / N_FEATURES

    aptitud = ALPHA * accuracy + (1 - ALPHA) * (1 - ratio_caracteristicas)

    return aptitud

# 3. INICIALIZACIÓN DEL ENJAMBRE

fuentes = [nueva_fuente() for _ in range(COLONY_SIZE)]
aptitudes = [fitness(f) for f in fuentes]
trials = [0] * COLONY_SIZE

mejor_indice = np.argmax(aptitudes)
mejor_fuente = fuentes[mejor_indice].copy()
mejor_fitness = aptitudes[mejor_indice]

historial = []

print("\nINICIO DEL ALGORITMO ABC")
print("Mejor fitness inicial:", round(mejor_fitness, 4))
print("Características iniciales seleccionadas:", mejor_fuente.sum(), "/", N_FEATURES)


# 4. EVOLUCIÓN DEL ALGORITMO ABC

for iteracion in range(1, MAX_ITER + 1):

    # FASE 1: ABEJAS EMPLEADAS
    # Cada abeja modifica una característica de su solución

    for i in range(COLONY_SIZE):
        nueva = fuentes[i].copy()

        bit = np.random.randint(N_FEATURES)
        nueva[bit] = 1 - nueva[bit]

        if nueva.sum() == 0:
            nueva[bit] = 1

        nueva_aptitud = fitness(nueva)

        if nueva_aptitud > aptitudes[i]:
            fuentes[i] = nueva
            aptitudes[i] = nueva_aptitud
            trials[i] = 0
        else:
            trials[i] += 1

    # FASE 2: ABEJAS OBSERVADORAS
    # Escogen fuentes con mejor aptitud

    suma_aptitudes = sum(aptitudes)

    if suma_aptitudes == 0:
        probabilidades = np.ones(COLONY_SIZE) / COLONY_SIZE
    else:
        probabilidades = np.array(aptitudes) / suma_aptitudes

    for _ in range(COLONY_SIZE):
        i = np.random.choice(COLONY_SIZE, p=probabilidades)

        nueva = fuentes[i].copy()

        bit = np.random.randint(N_FEATURES)
        nueva[bit] = 1 - nueva[bit]

        if nueva.sum() == 0:
            nueva[bit] = 1

        nueva_aptitud = fitness(nueva)

        if nueva_aptitud > aptitudes[i]:
            fuentes[i] = nueva
            aptitudes[i] = nueva_aptitud
            trials[i] = 0
        else:
            trials[i] += 1

    # FASE 3: ABEJAS EXPLORADORAS
    # Reemplazan soluciones que no mejoran

    for i in range(COLONY_SIZE):
        if trials[i] >= LIMIT:
            fuentes[i] = nueva_fuente()
            aptitudes[i] = fitness(fuentes[i])
            trials[i] = 0

    # ACTUALIZAR MEJOR SOLUCIÓN GLOBAL

    mejor_indice_actual = np.argmax(aptitudes)

    if aptitudes[mejor_indice_actual] > mejor_fitness:
        mejor_fitness = aptitudes[mejor_indice_actual]
        mejor_fuente = fuentes[mejor_indice_actual].copy()

    historial.append(mejor_fitness)

    if iteracion % 10 == 0:
        print(
            f"Iteración {iteracion} | "
            f"Mejor fitness: {mejor_fitness:.4f} | "
            f"Características: {mejor_fuente.sum()}/{N_FEATURES}"
        )

# 5. RESULTADOS

indices_seleccionados = np.where(mejor_fuente == 1)[0]
nombres_seleccionados = np.array(data.feature_names)[indices_seleccionados]

print("\n" + "=" * 60)
print("RESULTADO FINAL - FEATURE SELECTION CON ABC")
print("=" * 60)

print("Mejor fitness:", round(mejor_fitness, 4))
print("Cantidad de características seleccionadas:", len(indices_seleccionados), "/", N_FEATURES)

print("\nCaracterísticas seleccionadas:")
for nombre in nombres_seleccionados:
    print("-", nombre)

# 6. COMPARACIÓN FINAL
modelo_final = KNeighborsClassifier(n_neighbors=3)

accuracy_todas = cross_val_score(
    modelo_final,
    X,
    y,
    cv=N_FOLDS,
    scoring="accuracy"
).mean()

accuracy_seleccionadas = cross_val_score(
    modelo_final,
    X[:, indices_seleccionados],
    y,
    cv=N_FOLDS,
    scoring="accuracy"
).mean()

print("\nComparación de accuracy:")
print("Accuracy con TODAS las características:", round(accuracy_todas, 4))
print("Accuracy con características seleccionadas:", round(accuracy_seleccionadas, 4))

print("=" * 60)


# 7. GRÁFICA DE CONVERGENCIA


plt.figure(figsize=(8, 5))
plt.plot(historial)
plt.title("Convergencia del algoritmo ABC")
plt.xlabel("Iteración")
plt.ylabel("Mejor fitness")
plt.grid(True)
plt.show()