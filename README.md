# Algoritmos-de-Enjambre---OPTIMIZACI-N-y-MACHINE-LEARNING


Este repositorio contiene una colección de implementaciones avanzadas de algoritmos de **Inteligencia de Enjambre (Swarm Intelligence)**. El objetivo es demostrar cómo comportamientos colectivos inspirados en la naturaleza pueden resolver problemas complejos de optimización, clasificación y agrupamiento en Machine Learning sin depender de métodos tradicionales basados en gradientes.

---

## 📑 Contenido del Proyecto

El proyecto se divide en cuatro experimentos fundamentales, cada uno enfocado en una etapa distinta del pipeline de Ciencia de Datos:

### 1. Feature Selection con ABC (Artificial Bee Colony) 🐝
**Archivo:** `Feature_selection.py`

Optimización de subconjuntos de características para mejorar la eficiencia y precisión de los modelos.
*   **Concepto:** Se imita el comportamiento de una colmena (abejas empleadas, observadoras y exploradoras) para encontrar la mejor "fuente de alimento", que en este caso es el conjunto óptimo de variables[cite: 2].
*   **Modelo Evaluador:** K-Nearest Neighbors (KNN)[cite: 2].
*   **Dataset:** Breast Cancer Wisconsin[cite: 2].
*   **Impacto:** Se redujo el espacio de búsqueda de 30 a **10 características clave**, logrando elevar el Accuracy del **95.78% al 98.07%**[cite: 2].

### 2. Hyperparameter Tuning con PSO 🏎️
**Archivo:** `Hyperparameter_tuning.py`

Búsqueda automatizada de la configuración óptima de hiperparámetros para maximizar el rendimiento del modelo.
*   **Concepto:** Las partículas "vuelan" sobre el espacio de hiperparámetros ajustando su velocidad en función de su memoria propia y el éxito del grupo[cite: 3].
*   **Modelo:** Random Forest Classifier[cite: 3].
*   **Parámetros Optimizados:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `max_features`[cite: 3].
*   **Resultado:** El algoritmo convergió rápidamente gracias a un sistema de *Early Stopping*, superando a *Grid Search* con un Accuracy de **0.9343** en menor tiempo[cite: 3].

### 3. Red Neuronal sin Backpropagation (PSO) 🧠
**Archivo:** `Red_neuronal_PSO.py`

Entrenamiento de una Red Neuronal Artificial (ANN) mediante optimización global, eliminando el cálculo de gradientes.
*   **Arquitectura:** MLP de 3 capas: **30 (Entrada) → 16 (H1) → 8 (H2) → 1 (Salida)**[cite: 4].
*   **Desafío:** La partícula debe optimizar un vector de **641 dimensiones** que representa la totalidad de pesos y bias de la red[cite: 4].
*   **Resultado:** Se alcanzó un **92.98% de Accuracy en el set de prueba**, demostrando que PSO es una alternativa viable al entrenamiento tradicional en arquitecturas específicas[cite: 4].

### 4. Clustering No Supervisado con PSO 📍
**Archivo:** `Clustering_NS_PSO.py`

Agrupamiento automático de datos basado en la minimización de la varianza interna de los grupos.
*   **Concepto:** Cada partícula representa las coordenadas de los centroides de los clusters[cite: 1].
*   **Métrica de Calidad:** Se utiliza el **Silhouette Score** para medir la separación y cohesión de los grupos[cite: 1].
*   **Resultado:** El algoritmo redujo el fitness (distancia interna) de 56.91 a **17.32**, obteniendo un score de silueta de **0.8166**, indicando grupos altamente definidos[cite: 1].

---

## 🛠️ Tecnologías y Librerías
*   **Python:** Lenguaje principal de implementación[cite: 1, 2, 3, 4].
*   **NumPy:** Manejo de vectores multidimensionales para las partículas y pesos[cite: 1, 4].
*   **Scikit-learn:** Utilizado para carga de datasets, métricas de evaluación y modelos base[cite: 1, 2, 3, 4].
*   **Matplotlib:** Generación de gráficas de convergencia y visualización de clusters[cite: 1, 2, 3, 4].

---

## 📈 Resumen de Resultados Comparativos

| Experimento | Algoritmo | Métrica Principal | Resultado Final |
| :--- | :--- | :--- | :--- |
| **Selección de Características** | **ABC** | Accuracy (KNN) | **98.07%** |
| **Ajuste de Hiperparámetros** | **PSO** | Accuracy (RF) | **0.9343** |
| **Entrenamiento de Red Neuronal** | **PSO** | Accuracy (Test) | **92.98%** |
| **Agrupamiento (Clustering)** | **PSO** | Silhouette Score | **0.8166** |

---

---
*Este proyecto es una muestra de cómo la computación bio-inspirada permite resolver problemas de optimización donde los métodos tradicionales pueden fallar o ser ineficientes.*
