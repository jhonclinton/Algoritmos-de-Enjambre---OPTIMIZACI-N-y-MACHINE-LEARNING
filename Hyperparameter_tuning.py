#1. Importamos librerías

import numpy as np      #para manejar arrays
import random           #para números aleatorios
import matplotlib.pyplot as plt  #para la gráfica
import time             #para medir tiempo

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# para que no cambien los resultados cada vez
np.random.seed(42)
random.seed(42)

#2. Cargamos el Dataset

data = load_digits()   #cargo el dataset
X = data.data          #datos (pixeles)
y = data.target        #etiquetas

#3. Parámetros PSO

num_particles = 15     #cantidad de partículas
num_iterations = 25    #máximo de iteraciones

w = 0.7    #inercia
c1 = 1.5   #parte individual
c2 = 1.5   #parte social

# 4.Rangos

#aquí defino hasta dónde puede moverse cada parámetro
bounds =[
    (50, 200),
    (5, 30),
    (2, 15),
    (1, 10),
    (0.3, 1.0)
]

dim =len(bounds)  #cuántos parámetros hay


#5. Inicialización

#creamos partículas con valores aleatorios dentro de los rangos
particles = np.array([
    [random.uniform(b[0], b[1]) for b in bounds]
    for _ in range(num_particles)
])

#al inicio no se mueven
velocities = np.zeros((num_particles, dim))

#guardamos la mejor posición de cada partícula
pbest= particles.copy()

#guardo qué tan buenas son
pbest_scores = np.zeros(num_particles)


#6. FITNESS

def fitness(p):
    #convierto la partícula en un modelo
    model = RandomForestClassifier(
        n_estimators=int(p[0]),
        max_depth=int(p[1]),
        min_samples_split=int(p[2]),
        min_samples_leaf=int(p[3]),
        max_features=p[4],
        random_state=42,
        n_jobs=-1
    )

    #uso validación cruzada para ver qué tan bien funciona
    score = cross_val_score(model, X, y, cv=3).mean()
    return score


#7. PSO

inicio_pso = time.time()

#evalúo al inicio
for i in range(num_particles):
    pbest_scores[i] = fitness(particles[i])

#mejor global
gbest = pbest[np.argmax(pbest_scores)]

historial= []

#early stopping
no_mejora = 0
mejor_global = max(pbest_scores)
paciencia = 5

for t in range(num_iterations):

    for i in range(num_particles):

        #números aleatorios
        r1, r2 = np.random.rand(), np.random.rand()

        #actualización de velocidad (la fórmula clave)
        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest[i] - particles[i])
            + c2 * r2 * (gbest - particles[i])
        )

        #se mueve la partícula
        particles[i] = particles[i] + velocities[i]

        #evitar que se salga de los límites
        for d in range(dim):
            particles[i][d] = np.clip(
                particles[i][d],
                bounds[d][0],
                bounds[d][1]
            )

        #evalúo
        score = fitness(particles[i])

        #si mejora, actualizo
        if score > pbest_scores[i]:
            pbest[i] = particles[i].copy()
            pbest_scores[i] = score

    #mejor global otra vez
    gbest = pbest[np.argmax(pbest_scores)]

    mejor_iter = np.max(pbest_scores)
    historial.append(mejor_iter)

    print(f"Iteración {t}: Mejor accuracy = {mejor_iter:.4f}")

    #veo si mejoró o no
    if mejor_iter > mejor_global:
        mejor_global = mejor_iter
        no_mejora = 0
    else:
        no_mejora += 1

    #si ya no mejora, paro
    if no_mejora >= paciencia:
        print("\nConvergencia alcanzada. Early stopping activado.")
        break

fin_pso = time.time()

#Resultados

print("\n===== PSO =====")
print("n_estimators =", int(gbest[0]))
print("max_depth =", int(gbest[1]))
print("min_samples_split =", int(gbest[2]))
print("min_samples_leaf =", int(gbest[3]))
print("max_features =", round(gbest[4], 2))

print("Accuracy =", max(pbest_scores))
print("Tiempo PSO =", round(fin_pso - inicio_pso, 2))
print("Iteraciones =", len(historial))

#9. Gráfica

plt.plot(historial, marker='o')
plt.title("Evolución del PSO")
plt.xlabel("Iteraciones")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

