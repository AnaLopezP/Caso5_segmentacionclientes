import pandas as pd
import numpy as np

#leemos los datos
df = pd.read_csv('Client_segment_MODIFICADO.csv', sep=';', encoding='latin1')
print(df.head())

#miramos los tipos de datos
print(df.dtypes)
print(df.describe())
print(df.info())


#Variables inecesarias: anno_nacimiento, provincia, digital_encuesta
#df.drop(['anno_nacimiento', 'Provincia', 'Digital_encuesta'], axis=1, inplace=True)

#miramos los valores unicos de cada variable
for i in df.columns:
    print(i)
    print(df[i].unique())
    
#Sexo: M, F
#Casado: Si, No, nan (1, 0, 0)
#graduado: Si, No, nan (1, 0, 0)
#Profesion: Funcionario , Ingeniero, Servicios,  Artista, Ejecutivo, Medico, Construccion, Negocios/empresa, Otros
#Num profesion: 1, 2, 3, 4, 5, 6, 7, 8, 0
#Experiencia laboral: '<10annos' '+20annos' '10-20annos'
#experiencia laboral num: 1, 3, 2
#gastoscore: 1, 2, 3 (Bajo, medio, alto)
#generación: 'Generacion Z' 'Millennials' 'Generacion X' 'Generacion Silenciosa'
#generacion num: 4, 3, 2, 1
#Campanna_anno: nan '3' '1' '0,5' '4' --> num de veces que se le ha mandado cositas por ciertos años de renovación. 
#Campanna_anno num: 0, 3, 1, 0.5, 4

#Cambiamos las variables categoricas a numericas
#quitamos los 3 primeros caracteres de la columna ID
df['ID'] = df['ID'].str[3:]
df['ID'] = df['ID'].astype(int)
df['Genero'] = df['Genero'].map({'M': 0, 'F': 1})
df['Casado'] = df['Casado'].map({'Si': 1, 'No': 0, np.nan: 0})
df['Graduado'] = df['Graduado'].map({'Si': 1, 'No': 0, np.nan: 0})
df['Profesion'] = df['Profesion'].map({'Funcionario': 1, 'Ingeniero': 2, 'Servicios': 3, 'Artista': 4, 'Ejecutivo': 5, 'Medico': 6, 'Construccion': 7, 'Negocios/empresa': 8, 'Otros': 0})
df['Experiencia laboral'] = df['Experiencia laboral'].map({'<10annos': 1, '+20annos': 3, '10-20annos': 2})
df['Gastoscore'] = df['Gastoscore'].map({'Bajo': 1, 'Medio': 2, 'Alto': 3})
df['Generacion'] = df['Generacion'].map({'Generacion Z': 4, 'Millennials': 3, 'Generacion X': 2, 'Generacion Silenciosa': 1})
df['Campanna_anno'] = df['Campanna_anno'].map({np.nan: 0, '3': 3, '1': 1, '0,5': 0.5, '4': 4})

#Cambiamos la , por . de Digital_encuesta
df['Digital_encuesta'] = df['Digital_encuesta'].str.replace(',', '.')

#Creamos un mapa para Provincia: 'Malaga' 'Salamanca' 'Burgos' 'Gerona' 'i\x81vila' 'Orense' 'Almeri\xada'
#  'LLeida' 'Segovia' 'Castellon' 'Ciudad Real' 'Zamora' 'Madrid' 'Soria'
#  'Zaragoza' 'Jaen' 'Leon' 'Ceuta' 'Toledo' 'Las Palmas' 'Albacete'
#  'Vizcaya' 'Huelva' 'Guadalajara' 'Granada' 'La Rioja' 'Alicante'
#  'i\x81lava' 'Badajoz' 'Sevilla' 'Teruel' 'Caceres' 'La Corui±a'
#  'Pontevedra' 'Cadiz' 'Barcelona' 'Cuenca' 'Palencia' 'Baleares'
#  'Santa Cruz de Tenerife' 'Valencia' 'Huesca' 'Melilla' 'Guipiºzcoa'
#  'Tarragona' 'Navarra' 'Valladolid' 'Lugo' 'Cordoba' 'Murcia'

df['Provincia'] = df['Provincia'].map({
    'Malaga': 0, 'Salamanca': 1, 'Burgos': 2, 'Gerona': 3, 'i\x81vila': 4, 
    'Orense': 5, 'Almeri\xada': 6, 'LLeida': 7, 'Segovia': 8, 'Castellon': 9, 
    'Ciudad Real': 10, 'Zamora': 11, 'Madrid': 12, 'Soria': 13, 'Zaragoza': 14, 
    'Jaen': 15, 'Leon': 16, 'Ceuta': 17, 'Toledo': 18, 'Las Palmas': 19, 
    'Albacete': 20, 'Vizcaya': 21, 'Huelva': 22, 'Guadalajara': 23, 
    'Granada': 24, 'La Rioja': 25, 'Alicante': 26, 'i\x81lava': 27, 
    'Badajoz': 28, 'Sevilla': 29, 'Teruel': 30, 'Caceres': 31, 
    'La Corui±a': 32, 'Pontevedra': 33, 'Cadiz': 34, 'Barcelona': 35, 
    'Cuenca': 36, 'Palencia': 37, 'Baleares': 38, 'Santa Cruz de Tenerife': 39, 
    'Valencia': 40, 'Huesca': 41, 'Melilla': 42, 'Guipiºzcoa': 43, 
    'Tarragona': 44, 'Navarra': 45, 'Valladolid': 46, 'Lugo': 47, 
    'Cordoba': 48, 'Murcia': 49
})


 



#miramos cuantos valores nulos hay 
print(df.isnull().sum())
#No hay NaN



#guardamos los datos limpios
df.to_csv('Client_segment_limpio.csv', sep=';', index=False)


