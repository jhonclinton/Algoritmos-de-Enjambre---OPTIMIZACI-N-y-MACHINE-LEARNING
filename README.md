# Algoritmos-de-Enjambre---OPTIMIZACI-N-y-MACHINE-LEARNING

Este repositorio contiene una colección de implementaciones avanzadas de algoritmos de **Inteligencia de Enjambre (Swarm Intelligence)**. El objetivo es demostrar cómo comportamientos colectivos inspirados en la naturaleza pueden resolver problemas complejos de optimización, clasificación y agrupamiento en Machine Learning sin depender de métodos tradicionales basados en gradientes.

---

##  Contenido del Proyecto

El proyecto se divide en cuatro experimentos fundamentales, cada uno enfocado en una etapa distinta.
### 1. Feature Selection con ABC (Artificial Bee Colony) 
**Archivo:** `Feature_selection.py`

Optimización de subconjuntos de características para mejorar la eficiencia y precisión de los modelos.
*   **Concepto:** Se imita el comportamiento de una colmena (abejas empleadas, observadoras y exploradoras) para encontrar la mejor "fuente de alimento", que en este caso es el conjunto óptimo de variables.
*   **Modelo Evaluador:** K-Nearest Neighbors (KNN).
*   **Dataset:** Breast Cancer Wisconsin.
*   **Impacto:** Se redujo el espacio de búsqueda de 30 a **10 características clave**, logrando elevar el Accuracy del **95.78% al 98.07%**.

### 2. Hyperparameter Tuning con PSO 
**Archivo:** `Hyperparameter_tuning.py`

Búsqueda automatizada de la configuración óptima de hiperparámetros para maximizar el rendimiento del modelo.
*   **Concepto:** Las partículas "vuelan" sobre el espacio de hiperparámetros ajustando su velocidad en función de su memoria propia y el éxito del grupo.
*   **Modelo:** Random Forest Classifier.
*   **Parámetros Optimizados:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `max_features`.
*   **Resultado:** El algoritmo convergió rápidamente gracias a un sistema de *Early Stopping*, superando a *Grid Search* con un Accuracy de **0.9343** en menor tiempo.

### 3. Red Neuronal sin Backpropagation (PSO) 
**Archivo:** `Red_neuronal_PSO.py`

Entrenamiento de una Red Neuronal Artificial (ANN) mediante optimización global, eliminando el cálculo de gradientes.
*   **Arquitectura:** MLP de 3 capas: **30 (Entrada) → 16 (H1) → 8 (H2) → 1 (Salida)**.
*   **Desafío:** La partícula debe optimizar un vector de **641 dimensiones** que representa la totalidad de pesos y bias de la red.
*   **Resultado:** Se alcanzó un **92.98% de Accuracy en el set de prueba**, demostrando que PSO es una alternativa viable al entrenamiento tradicional en arquitecturas específicas.

### 4. Clustering No Supervisado con PSO 
**Archivo:** `Clustering_NS_PSO.py`

Agrupamiento automático de datos basado en la minimización de la varianza interna de los grupos.
*   **Concepto:** Cada partícula representa las coordenadas de los centroides de los clusters.
*   **Métrica de Calidad:** Se utiliza el **Silhouette Score** para medir la separación y cohesión de los grupos.
*   **Resultado:** El algoritmo redujo el fitness (distancia interna) de 56.91 a **17.32**, obteniendo un score de silueta de **0.8166**, indicando grupos altamente definidos.

---

## Tecnologías y Librerías
*   **Python:** Lenguaje principal de implementación.
*   **NumPy:** Manejo de vectores multidimensionales para las partículas y pesos.
*   **Scikit-learn:** Utilizado para carga de datasets, métricas de evaluación y modelos base.
*   **Matplotlib:** Generación de gráficas de convergencia y visualización de clusters.

---

##  Resumen de Resultados Comparativos

| Experimento | Algoritmo | Métrica Principal | Resultado Final |
| :--- | :--- | :--- | :--- |
| **Selección de Características** | **ABC** | Accuracy (KNN) | **98.07%** |
| **Ajuste de Hiperparámetros** | **PSO** | Accuracy (RF) | **0.9343** |
| **Entrenamiento de Red Neuronal** | **PSO** | Accuracy (Test) | **92.98%** |
| **Agrupamiento (Clustering)** | **PSO** | Silhouette Score | **0.8166** |

---

##  Instalación y Uso

1.  **Clonar este repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/swarm-intelligence-ml.git](https://github.com/tu-usuario/swarm-intelligence-ml.git)
    ```
2.  **Instalar las dependencias necesarias:**
    ```bash
    pip install numpy scikit-learn matplotlib
    ```
3.  **Ejecución:** Cada script es independiente y genera sus propias visualizaciones de resultados. Por ejemplo:
    ```bash
    python Red_neuronal_PSO.py
    ```

---
*Este proyecto es una muestra de cómo la computación bio-inspirada permite resolver problemas de optimización donde los métodos tradicionales pueden fallar o ser ineficientes.*
