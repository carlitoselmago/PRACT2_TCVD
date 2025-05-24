import pandas as pd
import os
from helpers import *

df = cargar_dataset()

# Creamos una muestra de la población de total para poder hacer los siguientes cálculos de una manera mas ágil

muestra = df.sample(n=100000,random_state=123)

print(muestra.head())


# Selección de columnas de interés
columnas_interes = [
    'Vict Sex', 'Vict Age', 'Vict Descent',
    'LAT', 'LON', 'AREA NAME', 'Premis Desc',
    'DATE OCC', 'TIME OCC',
    'Crm Cd Desc', 'Part 1-2',
    'Weapon Desc'
]

# Crear un nuevo DataFrame con solo esas columnas
df_resumen = muestra[columnas_interes].copy()

# Resumen estadístico para todas las columnas (numéricas y categóricas)
resumen = df_resumen.describe(include='all').transpose()

# Añadir manualmente tipo de dato y mostrar valores únicos (más legibles)
resumen['Tipo de dato'] = df_resumen.dtypes.astype(str)
resumen['Valores únicos'] = df_resumen.nunique()

# Mostrar filas para revisar
print("Resumen de variables seleccionadas:\n")
print(resumen[['Tipo de dato', 'count', 'unique', 'top', 'freq', 'mean', 'min', 'max', 'Valores únicos']])

guardar_dataset(resumen, '_resumen_estadistico')
guardar_dataset(muestra,'_02')