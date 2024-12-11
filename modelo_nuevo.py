import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

# Leer el dataset limpio
df = pd.read_csv('dataset_con_clusters.csv', sep=',', encoding='latin1', decimal=',')

# Verificar el dataset
print(df.head())

# Identificar variables binarias y continuas
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nColumnas numéricas:", numerical_cols)

binary_cols = []
continuous_cols = []

for col in numerical_cols:
    unique_values = df[col].dropna().unique()
    if sorted(unique_values) == [0, 1]:
        binary_cols.append(col)
    else:
        continuous_cols.append(col)

print("\nVariables binarias (0 y 1):", binary_cols)
print("\nVariables numéricas continuas:", continuous_cols)

# Seleccionar solo variables numéricas continuas para clustering
df_continuous = df[continuous_cols]
print("\nVariables numéricas continuas seleccionadas para clustering:", df_continuous.columns.tolist())

# Manejo de valores faltantes: eliminar filas con valores faltantes
print("\nValores faltantes por columna:")
print(df_continuous.isnull().sum())
df_cleaned = df_continuous.dropna()
print(f"\nNúmero de filas después de eliminar valores faltantes: {df_cleaned.shape[0]}")

# Método del Codo
inercia = []
rangos_k = range(1, 20)

for k in rangos_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_cleaned)
    inercia.append(kmeans.inertia_)

# Graficar el Método del Codo
plt.figure(figsize=(10, 6))
plt.plot(rangos_k, inercia, marker='o', linestyle='--')
plt.title('Método del Codo para Determinar K Óptimo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inercia')
plt.xticks(rangos_k)
plt.grid(True)
plt.show()

# Análisis de Silueta
silhueta_media = []
rangos_k_sil = range(2, 20)

for k in rangos_k_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_cleaned)
    sil_score = silhouette_score(df_cleaned, labels)
    silhueta_media.append(sil_score)

# Graficar el Análisis de Silueta
plt.figure(figsize=(10, 6))
plt.plot(rangos_k_sil, silhueta_media, marker='o', linestyle='--', color='green')
plt.title('Análisis de Silueta para Determinar K Óptimo')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Media')
plt.xticks(rangos_k_sil)
plt.grid(True)
plt.show()

# Seleccionar K óptimo basado en análisis previos (supongamos que K=4)
k_optimo = 4

# Aplicar K-Means con K óptimo
kmeans = KMeans(n_clusters=k_optimo, random_state=42)
clusters = kmeans.fit_predict(df_cleaned)

# Añadir los labels de los clusters al DataFrame limpio
df_cleaned['Cluster'] = clusters

print("\nAsignación de Clusters:")
print(df_cleaned.head())

# Visualización: Scatter Plot con Clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Edad', y='Dias_cliente', hue='Cluster', data=df_cleaned, palette='Set1', s=100, alpha=0.7)
plt.title(f'K-Means Clustering (K={k_optimo})')
plt.xlabel('Edad')
plt.ylabel('Dias_cliente')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Visualización: Centroides en el Scatter Plot
centroides = kmeans.cluster_centers_

plt.figure(figsize=(10, 8))
sns.scatterplot(x='Edad', y='Dias_cliente', hue='Cluster', data=df_cleaned, palette='Set1', s=100, alpha=0.7)
plt.scatter(centroides[:, 0], centroides[:, 1], s=300, c='yellow', label='Centroides', edgecolor='black', marker='X')
plt.title(f'K-Means Clustering con Centroides (K={k_optimo})')
plt.xlabel('Edad')
plt.ylabel('Dias_cliente')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# Índice de Davies-Bouldin
from sklearn.preprocessing import StandardScaler

# Escalar datos antes del clustering
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

# Cálculo del índice Davies-Bouldin
dbi_scores = []
rangos_k = range(2, 20)

for k in rangos_k:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df_scaled)
    dbi = davies_bouldin_score(df_scaled, labels)
    dbi_scores.append(dbi)
    print(f"K={k}: Davies-Bouldin Index={dbi:.4f}")

# Graficar los resultados de Davies-Bouldin
plt.figure(figsize=(10, 6))
sns.lineplot(x=list(rangos_k), y=dbi_scores, marker='o', color='blue')
plt.title('Índice de Davies-Bouldin para Diferentes Valores de K')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Davies-Bouldin Index')
plt.xticks(rangos_k)
plt.grid(True)
plt.show()

'''# Añadir los clusters al dataset original
df['Cluster'] = df_cleaned['Cluster']
'''