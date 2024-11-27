# Para hacer el modelo kmeans, primero hacemos la regla del codo para determinar el número de clusters
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
#ignorar warnings
import warnings
from analisis_datos import df_scaled
warnings.filterwarnings('ignore')

# 3. Calcula WCSS para diferentes valores de k
wcss = []
k_values = range(1, 20)  # Probar con k de 1 a 20 (ponemos que son 8 o 9 los optimos)
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
plt.savefig('imagenes/regla_del_codo.png')
plt.show()

from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

# Calcular la puntuación de silueta para diferentes valores de k
for k in range(5, 10):  # Evalúa de 5 a 9 clusters
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
    plt.savefig(f'imagenes/silueta_k{k}.png')
    plt.show()

from sklearn.decomposition import PCA
koptimo = 6
# Kmeans con el número óptimo de clusters
kmeans = KMeans(n_clusters=koptimo, random_state=42)
cluster = kmeans.fit(df_scaled)

# Añadir los labels de cluster 
df_scaled['Cluster'] = cluster
print(df_scaled.head())

#Obtener los índices de las columas Generacion y Campanna_anno
gen_index = df_scaled.index('Generacion') #ERROR AL COGER EL ÍNDICE
campanna_index = df_scaled.index('Campanna_anno')

centroide = kmeans.cluster_centers_

# Visualizar el scatter plot con la edad
plt.figure(figsize=(10, 6))
sns.scatterplot(x = df_scaled['Generacion'], y = df_scaled['Campanna_anno'], data = df_scaled, hue = 'Cluster', s=100, alpha=0.5, palette='viridis')
plt.scatter(centroide[:, gen_index], centroide[:, campanna_index], s=300, c='red', label='Centroides', marker='x', edgecolors='black')
plt.title('Segmentación de Clientes')
plt.xlabel('Edad')
plt.ylabel('Ingresos')
plt.legend()
plt.grid(True)
plt.savefig('imagenes/segmentacion_clientes.png')
plt.show()



'''# Aplicar PCA
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
optimal_k = 2 # hay que cambiarlo por el valor óptimo
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(df_scaled)

# Agregar etiquetas al df
#Frame original
df['Cluster'] = kmeans.labels_
print(df.head())

cluster_summary = df.groupby('Cluster').mean()
print(cluster_summary)

# representación gráfica de los clusters
# Graficar los datos en las dos primeras componentes principales
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, s=5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroides')
plt.title("Datos proyectados en PCA1 y PCA2")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid()
plt.legend()
plt.show()
'''

