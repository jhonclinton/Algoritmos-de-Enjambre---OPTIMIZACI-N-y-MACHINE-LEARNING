# ============================================================
# EJERCICIO 4: CLUSTERING USANDO PSO
# Particle Swarm Optimization para agrupamiento de datos
# Dataset generado con make_blobs
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ============================================================
# CONFIGURACIÓN
# ============================================================

N_PARTICLES = 30     # cantidad de partículas
MAX_ITER = 100       # número de iteraciones
N_CLUSTERS = 3       # cantidad de grupos a encontrar

W = 0.7              # inercia
C1 = 1.5             # aprendizaje personal
C2 = 1.5             # aprendizaje social

np.random.seed(42)

# ============================================================
# CARGAR / GENERAR DATASET
# ============================================================

X, y_real = make_blobs(
    n_samples=200,
    centers=N_CLUSTERS,
    cluster_std=1.2,
    random_state=42
)

scaler = StandardScaler()
X = scaler.fit_transform(X)

N_SAMPLES, N_FEATURES = X.shape
DIMENSION = N_CLUSTERS * N_FEATURES

print("Dataset generado con make_blobs")
print("Muestras:", N_SAMPLES)
print("Características:", N_FEATURES)
print("Clusters a encontrar:", N_CLUSTERS)

# ============================================================
# 1. REPRESENTACIÓN DE LA PARTÍCULA
# Cada partícula representa posibles centroides
# ============================================================

def crear_particula():
    return np.random.uniform(
        low=X.min(),
        high=X.max(),
        size=DIMENSION
    )

def obtener_centroides(particula):
    return particula.reshape(N_CLUSTERS, N_FEATURES)

# ============================================================
# 2. ASIGNACIÓN DE CLUSTERS
# ============================================================

def asignar_clusters(centroides):
    distancias = np.linalg.norm(
        X[:, np.newaxis] - centroides,
        axis=2
    )
    etiquetas = np.argmin(distancias, axis=1)
    return etiquetas

# ============================================================
# 3. FUNCIÓN DE APTITUD
# Se minimiza la distancia interna de los clusters
# ============================================================

def fitness(particula):
    centroides = obtener_centroides(particula)
    etiquetas = asignar_clusters(centroides)

    distancia_total = 0

    for i in range(N_CLUSTERS):
        puntos_cluster = X[etiquetas == i]

        if len(puntos_cluster) == 0:
            return float("inf")

        distancia_total += np.sum(
            np.linalg.norm(puntos_cluster - centroides[i], axis=1) ** 2
        )

    return distancia_total

# ============================================================
# 4. INICIALIZACIÓN DEL ENJAMBRE
# ============================================================

particles = np.array([crear_particula() for _ in range(N_PARTICLES)])
velocities = np.random.uniform(-1, 1, (N_PARTICLES, DIMENSION))

pbest = particles.copy()
pbest_scores = np.array([fitness(p) for p in particles])

gbest_index = np.argmin(pbest_scores)
gbest = pbest[gbest_index].copy()
gbest_score = pbest_scores[gbest_index]

historial = []

print("\nINICIO DEL ALGORITMO PSO")
print("Mejor fitness inicial:", round(gbest_score, 4))

# ============================================================
# 5. EVOLUCIÓN DEL ALGORITMO PSO
# ============================================================

for iteration in range(1, MAX_ITER + 1):

    for i in range(N_PARTICLES):

        r1 = np.random.rand(DIMENSION)
        r2 = np.random.rand(DIMENSION)

        velocities[i] = (
            W * velocities[i]
            + C1 * r1 * (pbest[i] - particles[i])
            + C2 * r2 * (gbest - particles[i])
        )

        particles[i] = particles[i] + velocities[i]

        score = fitness(particles[i])

        if score < pbest_scores[i]:
            pbest[i] = particles[i].copy()
            pbest_scores[i] = score

    best_index = np.argmin(pbest_scores)

    if pbest_scores[best_index] < gbest_score:
        gbest = pbest[best_index].copy()
        gbest_score = pbest_scores[best_index]

    historial.append(gbest_score)

    if iteration % 10 == 0:
        print(
            f"Iteración {iteration} | "
            f"Mejor fitness: {gbest_score:.4f}"
        )

# ============================================================
# 6. RESULTADOS FINALES
# ============================================================

centroides_finales = obtener_centroides(gbest)
etiquetas_finales = asignar_clusters(centroides_finales)

silhouette = silhouette_score(X, etiquetas_finales)

print("\n" + "=" * 60)
print("RESULTADO FINAL - CLUSTERING CON PSO")
print("=" * 60)

print("Mejor fitness:", round(gbest_score, 4))
print("Silhouette Score:", round(silhouette, 4))

print("\nCentroides encontrados:")
print(centroides_finales)

print("\nCantidad de datos por cluster:")
for i in range(N_CLUSTERS):
    print(f"Cluster {i}: {np.sum(etiquetas_finales == i)} elementos")

print("=" * 60)

# ============================================================
# 7. GRÁFICA DE CONVERGENCIA
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(historial)
plt.title("Convergencia del algoritmo PSO para Clustering")
plt.xlabel("Iteración")
plt.ylabel("Mejor fitness")
plt.grid(True)
plt.show()

# ============================================================
# 8. GRÁFICA DE CLUSTERS
# ============================================================

plt.figure(figsize=(7, 6))
plt.scatter(X[:, 0], X[:, 1], c=etiquetas_finales)
plt.scatter(
    centroides_finales[:, 0],
    centroides_finales[:, 1],
    marker="X",
    s=200
)

plt.title("Clusters encontrados con PSO")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.grid(True)
plt.show()