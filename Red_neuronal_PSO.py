"""
=============================================================================
EJERCICIO 3: Entrenamiento de Red Neuronal con PSO (sin Backpropagation)
=============================================================================
Dataset  : sklearn load_breast_cancer
Enjambre : Particle Swarm Optimization (PSO)
Red      : 3 capas  →  30 → 16 → 8 → 1
=============================================================================

CICLO DEL ALGORITMO PSO:
  1. Representación de la partícula  → vector de pesos y bias de la red
  2. Inicialización del enjambre     → partículas aleatorias en el espacio
  3. Función de aptitud              → accuracy sobre datos de entrenamiento
  4. Comportamiento de la partícula  → actualización de velocidad y posición
  5. Evolución                       → iteración hasta convergencia
  6. Finalización                    → criterio de parada (iteraciones máximas)
=============================================================================
"""

# ── Librerías (todas disponibles en Google Colab sin instalación extra) ───────
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. ARQUITECTURA DE LA RED NEURONAL
# ─────────────────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Red Neuronal de 3 capas sin backpropagation.
    Los pesos y bias se codifican como un vector 1D (partícula PSO).

    Arquitectura:
        Entrada        : 30 neuronas  (features del dataset)
        Capa oculta 1  : 16 neuronas  (activación: ReLU)
        Capa oculta 2  :  8 neuronas  (activación: ReLU)
        Salida         :  1 neurona   (activación: Sigmoid)
    """

    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.n_weights = self._count_weights()

    def _count_weights(self):
        """Total de parámetros: pesos W + bias b por cada capa."""
        total = 0
        for i in range(len(self.layer_sizes) - 1):
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]  # W
            total += self.layer_sizes[i + 1]                         # b
        return total

    def _decode_weights(self, particle):
        """
        Decodifica el vector 1D → matrices W y vectores b.
        Orden en la partícula: [W1_flat | b1 | W2_flat | b2 | W3_flat | b3]
        """
        weights, biases = [], []
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            rows = self.layer_sizes[i]
            cols = self.layer_sizes[i + 1]
            W = particle[idx: idx + rows * cols].reshape(rows, cols)
            idx += rows * cols
            b = particle[idx: idx + cols]
            idx += cols
            weights.append(W)
            biases.append(b)
        return weights, biases

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X, particle):
        """Propagación hacia adelante con los pesos de la partícula."""
        weights, biases = self._decode_weights(particle)
        A = X
        for i, (W, b) in enumerate(zip(weights, biases)):
            Z = A @ W + b
            A = self.relu(Z) if i < len(weights) - 1 else self.sigmoid(Z)
        return A.flatten()

    def predict(self, X, particle, threshold=0.5):
        """Retorna etiquetas binarias 0 / 1."""
        probs = self.forward(X, particle)
        return (probs >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ALGORITMO PSO
# ─────────────────────────────────────────────────────────────────────────────

class PSO:
    """
    Particle Swarm Optimization — entrena la red neuronal sin gradientes.

    Ecuación de movimiento:
        v(t+1) = w * v(t)
               + c1 * r1 * (pbest - x)   <- componente cognitivo
               + c2 * r2 * (gbest - x)   <- componente social
        x(t+1) = x(t) + v(t+1)
    """

    def __init__(self,
                 n_particles=50,
                 max_iter=200,
                 w=0.7,       # inercia
                 c1=1.5,      # cognitivo
                 c2=1.5,      # social
                 v_max=0.5,
                 bounds=(-3, 3),
                 verbose=True):
        self.n_particles = n_particles
        self.max_iter    = max_iter
        self.w           = w
        self.c1          = c1
        self.c2          = c2
        self.v_max       = v_max
        self.bounds      = bounds
        self.verbose     = verbose

        self.history_gbest_fitness = []
        self.history_gbest_acc     = []

    # PASO 2 ── Inicialización del enjambre ───────────────────────────────────
    def _initialize_swarm(self, dim):
        low, high = self.bounds
        positions  = np.random.uniform(low, high, (self.n_particles, dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.n_particles, dim))
        return positions, velocities

    # PASO 3 ── Función de aptitud ────────────────────────────────────────────
    def _fitness(self, particle, nn, X, y):
        """f(x) = -accuracy  →  minimizar f  ≡  maximizar accuracy."""
        y_pred = nn.predict(X, particle)
        return -accuracy_score(y, y_pred)

    # PASOS 4-5 ── Evolución del enjambre ─────────────────────────────────────
    def optimize(self, nn, X_train, y_train, X_val, y_val):
        dim = nn.n_weights

        # PASO 2: Inicializar posiciones y velocidades
        positions, velocities = self._initialize_swarm(dim)

        pbest_pos = positions.copy()
        pbest_fit = np.array([self._fitness(p, nn, X_train, y_train)
                               for p in positions])

        gbest_idx = np.argmin(pbest_fit)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        if self.verbose:
            print("\n" + "=" * 62)
            print("  PSO — Entrenamiento de Red Neuronal sin Backpropagation")
            print("=" * 62)
            print(f"  Particulas : {self.n_particles}")
            print(f"  Dimension  : {dim}  (pesos + bias de la NN)")
            print(f"  Iteraciones: {self.max_iter}")
            print(f"  w={self.w}, c1={self.c1}, c2={self.c2}")
            print("=" * 62 + "\n")

        # PASO 5: Bucle de evolución
        for iteration in range(self.max_iter):

            # PASO 4: Actualizar cada partícula
            for i in range(self.n_particles):
                r1 = np.random.rand(dim)
                r2 = np.random.rand(dim)

                velocities[i] = (self.w  * velocities[i]
                                + self.c1 * r1 * (pbest_pos[i] - positions[i])
                                + self.c2 * r2 * (gbest_pos    - positions[i]))

                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                positions[i]  = np.clip(positions[i] + velocities[i], *self.bounds)

                fit = self._fitness(positions[i], nn, X_train, y_train)

                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = positions[i].copy()
                    if fit < gbest_fit:
                        gbest_fit = fit
                        gbest_pos = positions[i].copy()

            # Registrar métricas
            acc_train = -gbest_fit
            acc_val   = accuracy_score(y_val, nn.predict(X_val, gbest_pos))
            self.history_gbest_fitness.append(gbest_fit)
            self.history_gbest_acc.append(acc_val)

            if self.verbose and (iteration % 20 == 0 or iteration == self.max_iter - 1):
                print(f"  Iter {iteration + 1:>4d}/{self.max_iter} | "
                      f"Fitness: {gbest_fit:.4f} | "
                      f"Acc Train: {acc_train:.4f} | "
                      f"Acc Val: {acc_val:.4f}")

        # PASO 6: Finalización
        if self.verbose:
            print(f"\n  PSO finalizado.")
            print(f"  Mejor fitness  : {gbest_fit:.4f}")
            print(f"  Acc validacion : {self.history_gbest_acc[-1]:.4f}\n")

        return gbest_pos


# ─────────────────────────────────────────────────────────────────────────────
# 3. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    # Cargar dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    print("\n Dataset: Breast Cancer Wisconsin")
    print(f"  Muestras : {X.shape[0]}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Clases   : {list(data.target_names)}")

    # División train / val / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Normalización
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"\n  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # Crear red neuronal  30 → 16 → 8 → 1
    layer_sizes = [30, 16, 8, 1]
    nn = NeuralNetwork(layer_sizes)
    print(f"\n  Arquitectura NN : {' -> '.join(map(str, layer_sizes))}")
    print(f"  Parametros totales (dim particula): {nn.n_weights}")

    # Ejecutar PSO
    pso = PSO(
        n_particles=50,
        max_iter=200,
        w=0.7,
        c1=1.5,
        c2=1.5,
        v_max=0.5,
        bounds=(-3, 3),
        verbose=True
    )
    best_particle = pso.optimize(nn, X_train, y_train, X_val, y_val)

    # Evaluación final
    y_pred_test = nn.predict(X_test, best_particle)
    acc_test    = accuracy_score(y_test, y_pred_test)

    print("=" * 62)
    print("  RESULTADOS FINALES EN TEST SET")
    print("=" * 62)
    print(f"  Accuracy: {acc_test:.4f}  ({acc_test * 100:.2f}%)\n")
    print(classification_report(y_test, y_pred_test, target_names=data.target_names))

    # ── Visualizaciones ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "PSO - Entrenamiento de Red Neuronal sin Backpropagation\n"
        "Dataset: Breast Cancer Wisconsin",
        fontsize=13, fontweight='bold'
    )

    iters = range(1, len(pso.history_gbest_fitness) + 1)

    # Gráfica 1: Convergencia del fitness
    ax1 = axes[0]
    ax1.plot(iters, [-f for f in pso.history_gbest_fitness], color='steelblue', linewidth=2)
    ax1.axhline(y=acc_test, color='red', linestyle='--', alpha=0.7,
                label=f'Acc Test = {acc_test:.3f}')
    ax1.set_title("Convergencia del PSO\n(Acc Train - gbest)")
    ax1.set_xlabel("Iteracion")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfica 2: Accuracy en validación
    ax2 = axes[1]
    ax2.plot(iters, pso.history_gbest_acc, color='darkorange', linewidth=2)
    ax2.set_title("Accuracy en Validacion\n(por iteracion PSO)")
    ax2.set_xlabel("Iteracion")
    ax2.set_ylabel("Accuracy Validacion")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Gráfica 3: Matriz de confusión
    ax3 = axes[2]
    cm = confusion_matrix(y_test, y_pred_test)
    im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax3.set_title(f"Matriz de Confusion\nAcc Test = {acc_test:.4f}")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(data.target_names, rotation=15)
    ax3.set_yticklabels(data.target_names)
    ax3.set_ylabel('Etiqueta Real')
    ax3.set_xlabel('Etiqueta Predicha')
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.show()   # <-- En Colab muestra la figura inline (sin guardar a ruta local)

    # Resumen del ciclo PSO
    print("\n" + "=" * 62)
    print("  CICLO DEL ALGORITMO PSO - RESUMEN")
    print("=" * 62)
    print(f"""
  1. REPRESENTACION DE LA PARTICULA
     Dimension : {nn.n_weights} valores reales
     Codifica  : W1(30x16)+b1(16) | W2(16x8)+b2(8) | W3(8x1)+b3(1)

  2. INICIALIZACION DEL ENJAMBRE
     Particulas: {pso.n_particles}
     Posicion  : U({pso.bounds[0]}, {pso.bounds[1]})
     Velocidad : U(-{pso.v_max}, {pso.v_max})

  3. FUNCION DE APTITUD
     f(x) = -Accuracy(NN(x), y_train)
     Minimizar f  <=>  Maximizar Accuracy

  4. COMPORTAMIENTO DE LA PARTICULA
     v(t+1) = {pso.w}*v(t) + {pso.c1}*r1*(pbest-x) + {pso.c2}*r2*(gbest-x)
     x(t+1) = x(t) + v(t+1)

  5. EVOLUCION
     {pso.max_iter} iteraciones

  6. FINALIZACION
     Criterio : maximo de iteraciones alcanzado
     Acc TEST : {acc_test:.4f}  ({acc_test * 100:.2f}%)
    """)

    return best_particle, acc_test


# ── Punto de entrada ──────────────────────────────────────────────────────────
best_weights, final_acc = main()