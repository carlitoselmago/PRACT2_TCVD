from helpers import *
import matplotlib.pyplot as plt
import numpy as np

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
df["DATE OCC"] = pd.to_datetime(df["DATE OCC"])

# Creación de la columna HOUR para poder procesar la hora del crimen (sin minutos)
df["HOUR"] = df["TIME OCC"].astype(str).str.zfill(4).str[:2].astype(int)

# Mostramos la distribución temporal de los crímenes
df["HOUR"].value_counts().sort_index().plot(kind="bar")

plt.xlabel("Hora del día (0–23)")
plt.ylabel("Número de casos")
plt.title("Occurencia de crímenes por horario")
plt.savefig('assets/C_limpieza_01.png')
plt.show()

# Creamos una nueva columna con el nombre del día de la semana
df["WEEKDAY"] = df["DATE OCC"].dt.day_name()

# Orden de los días 
dias_semana = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Mostrarmos la distribución por día de la semana
df["WEEKDAY"].value_counts().reindex(dias_semana).plot(kind="bar")
plt.xlabel("Día de la semana")
plt.ylabel("Número de casos")
plt.title("Ocurrencia de crímenes por día de la semana")
plt.savefig('assets/C_limpieza_02.png')
plt.show()


# Histograma de criemenes serios/no serios por área

# Agrupamos por área y tipo de crimen
counts = df.groupby(["AREA NAME", "Part 1-2"]).size().unstack(fill_value=0)

# Reordenar columnas por claridad
# 1 = serios, 2 = menos serios
counts = counts[[1, 2]]  

# Crear posiciones en X
areas = counts.index.tolist()
x = np.arange(len(areas))
width = 0.35 

# Creamos el gráfico
plt.figure(figsize=(14,6))
plt.bar(x - width/2, counts[1], width, label="Crímenes serios (Part 1)", color="firebrick")
plt.bar(x + width/2, counts[2], width, label="Crímenes menos serios (Part 2)", color="steelblue")

plt.xlabel("Área")
plt.ylabel("Número de casos")
plt.title("Comparativa de crímenes por área")
plt.xticks(x, areas, rotation=45, ha="right")
plt.savefig('assets/C_limpieza_03.png')
plt.show()




# Tratamiento de las variables de tiempo
# Convertir la columna 'DATE OCC' a datetime + tratamiento de variables derivadas
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df['DIA_SEMANA'] = df['DATE OCC'].dt.day_name()
df['MES'] = df['DATE OCC'].dt.month

# Crear franjas horarias basadas en 'TIME OCC' (de tipo HHMM, como 1430)
df['FRANJA_HORARIA'] = df['TIME OCC'].apply(categorizar_hora)


# Guardamos el dataset como Pickle para que se mantengan los tipos asignados
guardar_dataset_procesado(df,'_03')