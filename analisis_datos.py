import pandas as pd
import numpy as np

df = pd.read_csv('Client_segment_limpio.csv', sep=';', encoding='latin1')

# Calculamos la media de cada variable y centralizamos
df_mean = df.mean()
df_centered = df - df_mean
print(df_centered.head())

# Calculamos la matriz de covarianza
cov_matrix = np.cov(df_centered, rowvar=False)
print(cov_matrix)

# Calculamos los autovalores y autovectores
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(eigenvalues)
print(eigenvectors)

# Ordenamos los autovalores de mayor a menor
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Calculamos la matriz de vectores propios
eigenvectors_matrix = eigenvectors
print(eigenvectors_matrix)

# Proyectamos los datos en el nuevo espacio
df_pca = np.dot(df_centered, eigenvectors_matrix)
print(df_pca)

# Graficamos los datos centrados
import matplotlib.pyplot as plt
plt.scatter(df_pca[:, 0], df_pca[:, 1], c='blue', s=5)
plt.title("Datos proyectados en PCA1 y PCA2")
plt.xlabel("Componente Principal 1 (PCA1)")
plt.ylabel("Componente Principal 2 (PCA2)")
plt.grid()
plt.savefig('imagenes/PCA.png')
plt.show()
