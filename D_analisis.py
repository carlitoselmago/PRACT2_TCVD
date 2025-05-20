# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:55:09 2025

@author: mahp4
"""

import pandas as pd
import os
from helpers import *
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

df = cargar_dataset_procesado('_03')


# Modelo supervisado

# Seleccionar variables relevantes
df_modelo = df[['Part 1-2', 'Crm Cd Desc', 'Weapon Desc', 'Vict Age', 'Vict Sex', 'TIME OCC']].copy()

# Filtrar filas con datos faltantes
df_modelo = df_modelo.dropna()

# Convertir Part 1-2: Parte 1 = 1, Parte 2 = 0
df_modelo['Part 1-2'] = df_modelo['Part 1-2'].apply(lambda x: 1 if x == 1 else 0)

# Asegurar que 'Vict Age' es numérico
df_modelo['Vict Age'] = pd.to_numeric(df_modelo['Vict Age'], errors='coerce')
df_modelo = df_modelo.dropna(subset=['Vict Age'])  # eliminar filas si hay errores de conversión

# Variables predictoras y objetivo
X = df_modelo.drop('Part 1-2', axis=1)
y = df_modelo['Part 1-2']

# Variables categóricas y numéricas
cat_features = ['Crm Cd Desc', 'Weapon Desc', 'Vict Sex']
num_features = ['Vict Age', 'TIME OCC']

# Preprocesamiento
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', 'passthrough', num_features)
])

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('logreg', LogisticRegression(max_iter=1000))
])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Extraer el modelo entrenado
model = pipeline.named_steps['logreg']

# Obtener nombres de variables tras preprocesamiento
feature_names = pipeline.named_steps['preproc'].get_feature_names_out()

# Combinar con coeficientes
coef_df = pd.DataFrame({
    'Variable': feature_names,
    'Coeficiente': model.coef_[0]
}).sort_values(by='Coeficiente', ascending=False)

# Mostrar las variables más influyentes
print("Variables que más aumentan la probabilidad de crimen grave (Parte 1):")
print(coef_df.head(10).to_string(index=False))

print("\n Variables que más disminuyen la probabilidad de crimen grave (Parte 1):")
print(coef_df.tail(10).to_string(index=False))

# Modelo no supervisado


# Filtrar columnas necesarias
df_cluster = df[['LAT', 'LON', 'TIME OCC', 'Weapon Desc']].copy()
# Eliminar ubicaciones inválidas (ej. coordenadas nulas)
df_cluster = df_cluster[(df_cluster['LAT'] > 33) & (df_cluster['LAT'] < 35) &
                        (df_cluster['LON'] < -117) & (df_cluster['LON'] > -119)]

# Eliminar filas con valores faltantes
df_cluster = df_cluster.dropna()

# Preprocesamiento: normalización + one-hot para Weapon Desc
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['LAT', 'LON', 'TIME OCC']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Weapon Desc'])
])

# Aplicar preprocesamiento
X = preprocessor.fit_transform(df_cluster)

# Elegir número de clusters usando el método del codo
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Graficar el codo
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del codo para seleccionar k')
plt.grid(True)
plt.show()

# Entrenar modelo final con k=3
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init='auto')
cluster_labels = kmeans_final.fit_predict(X)

# Agregar los clústeres al DataFrame original
df_cluster['cluster'] = cluster_labels

# Visualización por ubicación geográfica
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
scatter = plt.scatter(df_cluster['LON'], df_cluster['LAT'], c=df_cluster['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Clusters de crímenes por ubicación (k=3)")
plt.grid(True)
plt.colorbar(scatter, label='Cluster')
plt.show()

# Número de crímenes por clúster
print("Número de crímenes por clúster:")
print(df_cluster['cluster'].value_counts(), '\n')

# Armas más comunes por clúster
print("Armas más comunes por clúster:")
for c in sorted(df_cluster['cluster'].unique()):
    print(f"\nCluster {c}:")
    print(df_cluster[df_cluster['cluster'] == c]['Weapon Desc'].value_counts().head(5))

# Hora promedio del crimen por clúster
print("\n Hora promedio del crimen por clúster:")
hora_media = df_cluster.groupby('cluster')['TIME OCC'].mean().round(0).astype(int)
print(hora_media)


# Contraste de hipótesis

# Filtrar columnas necesarias y eliminar valores faltantes
df_chi = df[['Vict Sex', 'Crm Cd Desc']].dropna()

# Crear tabla de contingencia
tabla = pd.crosstab(df_chi['Vict Sex'], df_chi['Crm Cd Desc'])

# Aplicar test chi-cuadrado
chi2, p, dof, expected = chi2_contingency(tabla)

# Mostrar resultados
print("Chi-cuadrado:", round(chi2, 2))
print("Grados de libertad:", dof)
print("Valor p:", round(p, 4))

# Interpretación
alpha = 0.05
if p < alpha:
    print("Se rechaza H₀: Existe relación significativa entre el sexo de la víctima y el tipo de crimen.")
else:
    print("No se rechaza H₀: No hay evidencia suficiente de relación entre el sexo de la víctima y el tipo de crimen.")