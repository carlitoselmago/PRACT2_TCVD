import pandas as pd
import os
from helpers import *


df = cargar_dataset('_02')

# Creamos una máscara para filtrar las columnas que contengan ceros, NaN o cadenas de texto vacias
mask = (df == 0) | (df.isna()) | (df == '')
columnas_a_tratar = mask.any()[mask.any()].index.tolist()
print("columnas que contengan ceros, NaN o cadenas de texto vacias:", columnas_a_tratar)

# De las columnas a tratar, analizamos cual es el porcentaje de datos vacios
# Total de filas
total_filas = len(df)

# Diccionario para guardar los porcentajes por columna
porcentajes_por_columna = {}

for col in columnas_a_tratar:
    cantidad_con_problemas = ((df[col] == 0) | (df[col].isna()) | (df[col] == '')).sum()
    porcentaje = cantidad_con_problemas / total_filas * 100
    porcentajes_por_columna[col] = round(porcentaje, 2)

# Mostrar resultados
for col, pct in porcentajes_por_columna.items():
    print(f"Columna '{col}': {pct}% de valores vacios")

# Eliminación de filas vacias cuando el porcentaje es muy bajo
##############################################################

# Latitud y Longitud
df = df.dropna(subset=['LAT', 'LON'])

# Premis Desc
df = df.dropna(subset=['Premis Desc'])


# Proceso de imputación
###############################################################

# Para edad de la victima

# Consideramos que 0 en 'Vict Age' representa dato faltante
df['Vict Age'] = df['Vict Age'].replace(0, pd.NA)

# Ahora imputamos
imputados = imputar_con_regresion(df, 'Vict Age', ['Vict Sex', 'Vict Descent', 'Premis Desc'])

# Asignamos si hay predicciones
if imputados is not None and len(imputados) > 0:
    df.loc[df['Vict Age'].isna(), 'Vict Age'] = imputados

# Para la etnicidad de la víctima, utilizamos símplemente el modo
df['Vict Descent'] = df['Vict Descent'].fillna(df['Vict Descent'].mode()[0])

# Creación de nuevas columnas con valores combinados
################################################################

# Creamos una columna que suma el número total de crímenes adicionales, con 1 por defecto (siendo el principal)
df['num_offenses'] = df[['Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']].notna().sum(axis=1)

# Luego eliminamos estas columnas para mejor visualización del dataframe
df = df.drop(['Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4'], axis=1)
print(df)

# Conversión de tipos de variable
################################################################

# Primero analizamos los tipos asignados
print(df.info())

# Variables tipo categórica / factor
df['AREA'] = df['AREA'].astype('category')
df['AREA NAME'] = df['AREA NAME'].astype('category')
df['Part 1-2'] = df['Part 1-2'].astype('category')
df['Vict Sex'] = df['Vict Sex'].astype('category')
df['Vict Descent'] = df['Vict Descent'].astype('category')
df['Status'] = df['Status'].astype('category')


# Guardamos el dataset como Pickle para que se mantengan los tipos asignados
guardar_dataset_procesado(df,'_03')