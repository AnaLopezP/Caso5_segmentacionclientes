# Para hacer el modelo kmeans, primero hacemos la regla del codo para determinar el número de clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#leemos los datos limpios
df = pd.read_csv('Client_segment_limpio.csv', sep=';', encoding='latin1')
print(df.head())

# 2. Escala los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. Calcula WCSS para diferentes valores de k
wcss = []
k_values = range(1, 11)  # Probar con k de 1 a 10
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# 4. Graficar la regla del codo
plt.figure(figsize=(8, 5))
plt.plot(k_values, wcss, marker='o', linestyle='--')
plt.title('Regla del Codo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(k_values)
plt.grid()
plt.show()

from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# Calcular la puntuación de silueta para diferentes valores de k
for k in range(2, 7):  # Evalúa de 2 a 6 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    silhouette_avg = silhouette_score(df_scaled, labels)
    print(f"Para k = {k}, la puntuación de silueta promedio es: {silhouette_avg}")

    # Graficar la silueta para cada cluster
    sample_silhouette_values = silhouette_samples(df_scaled, labels)
    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 10  # Separación entre clusters

    plt.title(f"Gráfico de Silueta para k={k}")
    plt.xlabel("Puntuación de la Silueta")
    plt.ylabel("Clusters")
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.show()

from sklearn.decomposition import PCA

# Aplicar PCA
pca = PCA(n_components=2)  # Cambia el número de componentes si es necesario
df_pca = pca.fit_transform(df_scaled)

# Varianza explicada
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
print("Varianza total explicada:", sum(pca.explained_variance_ratio_))

# Graficar los datos en las dos primeras componentes principales
plt.scatter(df_pca[:, 0], df_pca[:, 1], c='blue', s=5)
plt.title("Datos proyectados en PCA1 y PCA2")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid()
plt.show()

# Entrenar el modelo con el número óptimo de clusters
optimal_k = None # hay que cambiarlo por el valor óptimo
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(df_scaled)

# Agregar etiquetas al df
#Frame original
df['Cluster'] = kmeans.labels_
print(df.head())

cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)


