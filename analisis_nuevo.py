import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configurar estilos para las visualizaciones
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Cargar el dataset
df = pd.read_csv('Client_segment_limpio.csv', sep=';', encoding='latin1')
# 2. Verificar los nombres de las columnas
print("\nNombres de las columnas en el dataset:")
print(df.columns.tolist())



# 4. Inspeccionar las primeras filas
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# 5. Preprocesamiento de los Datos

# Seleccionar columnas numéricas excluyendo identificadores
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'patient_id' in numerical_cols:
    numerical_cols.remove('patient_id')

print("\nColumnas numéricas seleccionadas para PCA:", numerical_cols)

# Verificar valores faltantes en columnas numéricas
print("\nValores faltantes por columna:")
print(df[numerical_cols].isnull().sum())

# Eliminar filas con valores faltantes en columnas numéricas
df_cleaned = df.dropna(subset=numerical_cols)
print(f"\nNúmero de filas después de eliminar valores faltantes: {df_cleaned.shape[0]}")

# Estandarizar las variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned[numerical_cols])

# Convertir a DataFrame para mayor claridad
df_scaled = pd.DataFrame(X_scaled, columns=numerical_cols)
print("\nDatos estandarizados:")
print(df_scaled.head())

# 6. Aplicar PCA

# Inicializar PCA con 2 componentes
pca = PCA(n_components=2)

# Ajustar PCA a los datos estandarizados y transformar los datos
principal_components = pca.fit_transform(X_scaled)

# Crear un DataFrame con las componentes principales
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# No hay una columna 'Class', así que no la concatenamos
final_df = pca_df

print("\nComponentes principales:")
print(final_df.head())

# Obtener el porcentaje de varianza explicada
varianza_explicada = pca.explained_variance_ratio_
print("\nVarianza explicada por cada componente:")
for i, var in enumerate(varianza_explicada, start=1):
    print(f"PC{i}: {var*100:.2f}%")

# 7. Visualizar los Resultados

# a. Visualización de las Componentes Principales
plt.figure(figsize=(10,8))
sns.scatterplot(x='PC1', y='PC2', data=final_df, s=100, alpha=0.7)
plt.title('PCA de Thyroid Dataset (Sin Clase)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()

# b. Dibujar los Vectores (Cargas) de las Variables Originales

# Obtener las cargas (vectores propios)
cargas = pca.components_.T * np.sqrt(pca.explained_variance_)

# Crear un DataFrame para las cargas
cargas_df = pd.DataFrame(cargas, index=numerical_cols, columns=['PC1', 'PC2'])

print("\nCargas de las variables en las componentes principales:")
print(cargas_df)

# Dibujar los vectores
plt.figure(figsize=(10,8))
sns.scatterplot(x='PC1', y='PC2', data=final_df, s=100, alpha=0.7)

for var in numerical_cols:
    plt.arrow(0, 0, cargas_df.loc[var, 'PC1'], cargas_df.loc[var, 'PC2'], 
              color='black', alpha=0.5, head_width=0.05)
    plt.text(cargas_df.loc[var, 'PC1']*1.1, cargas_df.loc[var, 'PC2']*1.1, 
             var, color='black', ha='center', va='center')

plt.title('PCA con Vectores de Variables Originales')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.show()



'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configurar estilos para las visualizaciones
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Cargar el dataset
df = pd.read_csv('Client_segment_limpio.csv', sep=';', encoding='latin1')
print("Dataset cargado exitosamente.")

# 2. Seleccionar columnas numéricas
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nColumnas numéricas:", numerical_cols)

# 3. Identificar variables binarias
binary_cols = []
continuous_cols = []

for col in numerical_cols:
    unique_values = df[col].dropna().unique()
    if sorted(unique_values) == [0, 1]:
        binary_cols.append(col)
    else:
        continuous_cols.append(col)

print("\nVariables binarias (0 y 1):", binary_cols)
print("\nVariables numéricas continuas antes de excluir 'patient_id':", continuous_cols)

# 4. Excluir 'patient_id' de las variables continuas si está presente
if 'patient_id' in continuous_cols:
    continuous_cols.remove('patient_id')
    print("\n'patient_id' excluida de las variables continuas.")

print("\nVariables numéricas continuas seleccionadas para PCA:", continuous_cols)

# 5. Seleccionar solo variables numéricas continuas
df_continuous = df[continuous_cols]
print("\nVariables numéricas continuas seleccionadas para PCA:", df_continuous.columns.tolist())

# 6. Manejo de valores faltantes
print("\nValores faltantes por columna:")
print(df_continuous.isnull().sum())

# Eliminar filas con valores faltantes
df_cleaned = df_continuous.dropna()
print(f"\nNúmero de filas después de eliminar valores faltantes: {df_cleaned.shape[0]}")

# 7. Estandarizar las variables numéricas continuas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned)

df_scaled = pd.DataFrame(X_scaled, columns=continuous_cols)
print("\nDatos estandarizados:")
print(df_scaled.head())

# 8. Aplicar PCA
n_components = 2  # Puedes ajustar este número según tus necesidades
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, n_components+1)])
print("\nComponentes principales:")
print(pca_df.head())

# 9. Varianza explicada
varianza_explicada = pca.explained_variance_ratio_
for i, var in enumerate(varianza_explicada, start=1):
    print(f"PC{i}: {var*100:.2f}%")

# 10. Visualización: Scatter Plot
plt.figure(figsize=(10,8))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=100, alpha=0.7)
plt.title('PCA de Thyroid Dataset (Variables Numéricas Continuas)')
plt.xlabel(f'PC1 ({varianza_explicada[0]*100:.2f}% Varianza)')
plt.ylabel(f'PC2 ({varianza_explicada[1]*100:.2f}% Varianza)')
plt.grid(True)
plt.show()

# 11. Visualización: Biplot Mejorado
cargas = pca.components_.T
escalamiento = 3  # Ajusta este valor según tus datos
cargas_scaled = cargas * escalamiento

plt.figure(figsize=(12,10))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=50, alpha=0.7)

# Dibujar los vectores
for i, var in enumerate(continuous_cols):
    plt.arrow(0, 0, cargas_scaled[i,0], cargas_scaled[i,1], 
              color='r', alpha=0.5, head_width=0.05)
    plt.text(cargas_scaled[i,0]*1.1, cargas_scaled[i,1]*1.1, 
             var, color='r', ha='center', va='center')

plt.title('PCA de Thyroid Dataset con Vectores de Variables (Variables Continuas)')
plt.xlabel(f'PC1 ({varianza_explicada[0]*100:.2f}% Varianza)')
plt.ylabel(f'PC2 ({varianza_explicada[1]*100:.2f}% Varianza)')
plt.grid(True)
plt.axhline(0, color='grey', linewidth=0.5)
plt.axvline(0, color='grey', linewidth=0.5)
plt.show()

# 12. Visualización: Gráficos de Cargas Separados
cargas_df = pd.DataFrame(cargas, index=continuous_cols, columns=['PC1', 'PC2'])
print(cargas_df)

# Gráfico de Cargas para PC1
plt.figure(figsize=(10,8))
plt.barh(cargas_df.index, cargas_df['PC1'], color='skyblue')
plt.xlabel('Cargas en PC1')
plt.title('Cargas de las Variables en PC1')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()

# Gráfico de Cargas para PC2
plt.figure(figsize=(10,8))
plt.barh(cargas_df.index, cargas_df['PC2'], color='lightgreen')
plt.xlabel('Cargas en PC2')
plt.title('Cargas de las Variables en PC2')
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.show()
print(cargas_df)


#Edad en PC1, Dias_cliente en pc2'''