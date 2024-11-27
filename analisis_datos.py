import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# 1. Leer los datos
df = pd.read_csv('Client_segment_limpio.csv', sep=';', encoding='latin1')
print(df.head())


# 2. Escala los datos
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# 3. Los convertimos a dataframe
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# 4. Aplicamos PCA
pca = PCA(n_components=2)
df_pca_componentes_principales = pca.fit_transform(df_scaled)
df_pca = pd.DataFrame(df_pca_componentes_principales, columns=['pca1', 'pca2'])
print(df_pca.head())

# 5. Obtener el porcentaje de varianza explicada
varianza_explicada = pca.explained_variance_ratio_
print("-------- VARIANZA EXPLICADA --------")
print(pca.explained_variance_ratio_)
for i, var in enumerate(varianza_explicada):
    print(f'PCA {i+1}: {var*100:.2f}%')
    
# 6. Graficar los datos en el espacio PCA
def graficar_pca(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['pca1'], df['pca2'], alpha=.5)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Gráfico de dispersión PCA')
    plt.savefig('imagenes/grafico_pca.png')
    plt.show()
    

'''plt.figure(figsize=(8, 6))
plt.scatter(df_pca['pca1'], df_pca['pca2'], alpha=.5)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Gráfico de dispersión PCA')
plt.savefig('imagenes/grafico_pca.png')
plt.show()
'''
# Dibujar los vectores de las variables originales
cargas = pca.components_.T * np.sqrt(pca.explained_variance_)
cargas_df = pd.DataFrame(cargas, columns=['pca1', 'pca2'], index=df.columns)
print("-------- CARGAS --------")
print(cargas_df)

def graficar_cargar_general(df, cargas):
    plt.figure(figsize=(8, 6))
    plt.scatter(df['pca1'], df['pca2'], alpha=.5)
    for i in cargas.index:
        plt.arrow(0, 0, cargas.loc[i, 'pca1'], cargas.loc[i, 'pca2'], color='r', alpha=.5, head_width=0.1)
        plt.text(cargas.loc[i, 'pca1']*1.15, cargas.loc[i, 'pca2']*1.15, i, color='g')
    
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Gráfico de dispersión PCA con vectores de carga')
    plt.savefig('imagenes/grafico_pca_vectores_carga.png')
    plt.show()


'''plt.figure(figsize=(8, 6))
plt.scatter(df_pca['pca1'], df_pca['pca2'], alpha=.5)
for i in cargas_df.index:
    plt.arrow(0, 0, cargas_df.loc[i, 'pca1'], cargas_df.loc[i, 'pca2'], color='r', alpha=.5, head_width=0.1)
    plt.text(cargas_df.loc[i, 'pca1']*1.15, cargas_df.loc[i, 'pca2']*1.15, i, color='g')
    
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Gráfico de dispersión PCA con vectores de carga')
plt.savefig('imagenes/grafico_pca_vectores_carga.png')
plt.show()
'''
# gráfico horizontal de cargas
def graficar_cargas (df, cargas, pca_num):
    plt.figure(figsize=(8, 6))
    plt.barh(df.columns, np.abs(cargas[f'pca{pca_num}']), alpha=.5)
    plt.xticks(rotation=90)
    plt.ylabel('Carga')
    plt.title(f'Cargas de las variables en PCA {pca_num}')
    plt.savefig(f'imagenes/grafico_cargas_pca{pca_num}.png')
    plt.show()
    

'''plt.figure(figsize=(8, 6))
plt.barh(df.columns, np.abs(cargas_df['pca1']), alpha=.5)
plt.xticks(rotation=90)
plt.ylabel('Carga')
plt.title('Cargas de las variables en PCA 1')
plt.savefig('imagenes/grafico_cargas_pca1.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.barh(df.columns, np.abs(cargas_df['pca2']), alpha=.5)
plt.xticks(rotation=90)
plt.ylabel('Carga')
plt.title('Cargas de las variables en PCA 2')
plt.savefig('imagenes/grafico_cargas_pca2.png')
plt.show()'''

if __name__ == '__main__':
    graficar_pca(df_pca)
    graficar_cargar_general(df_pca, cargas_df)
    graficar_cargas(df, cargas_df, 1)
    graficar_cargas(df, cargas_df, 2)

