import pandas as pd
import numpy as np

#leemos los datos
df = pd.read_csv('client_seg_telco/client_seg_telco.csv', sep=';')
print(df.head())

#Quitamos nan
df.fillna(0, inplace=True)

#miramos cuantos valores nulos hay 
print(df.isnull().sum())
#No hay NaN

#miramos los tipos de datos
print(df.dtypes)