import pandas as pd
import numpy as np

#leemos los datos
df = pd.read_csv('Client_segment_MODIFICADO.csv', sep=';', encoding='latin1')
print(df.head())

#Quitamos nan
df.fillna(0, inplace=True)

#miramos cuantos valores nulos hay 
print(df.isnull().sum())
#No hay NaN

#miramos los tipos de datos
print(df.dtypes)