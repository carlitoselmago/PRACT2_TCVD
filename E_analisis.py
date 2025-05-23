# -*- coding: utf-8 -*-
"""
Created on Sat May 17 15:55:09 2025

@author: mahp4
"""

from helpers import *

# Preparar los datos

# carga y copia del dataset
df = cargar_dataset_procesado('_03')

# Modelo supervisado

# Seleccionar variables relevantes
df_modelo = df[['Part 1-2', 'Vict Descent', 'Vict Age','Vict Sex', 'TIME OCC', 'DIA_SEMANA', 'MES', 'FRANJA_HORARIA', 'AREA NAME']].copy()
df_modelo = df_modelo.dropna()


# Convertir Part 1-2: Parte 1 = 1, Parte 2 = 0
df_modelo['Part 1-2'] = df_modelo['Part 1-2'].apply(lambda x: 1 if x == 1 else 0)

# Variables predictoras y objetivo
X = df_modelo.drop('Part 1-2', axis=1)
y = df_modelo['Part 1-2']

# Variables categóricas y numéricas
cat_features = ['Vict Sex','Vict Descent','DIA_SEMANA', 'FRANJA_HORARIA', 'AREA NAME']
num_features = ['Vict Age', 'TIME OCC', 'MES']

# Preprocesamiento
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

# Pipeline completo
pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('logreg', LogisticRegression(solver='saga', max_iter=10000))
])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print(coef_df.head(20).to_string(index=False))

print("\n Variables que más disminuyen la probabilidad de crimen grave (Parte 1):")
print(coef_df.tail(20).to_string(index=False))

# Evaluación del modelo
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

# Mostrar las variables con más impacto
coef_df_abs = coef_df.copy()
coef_df_abs['Impacto'] = coef_df_abs['Coeficiente'].abs()
coef_df_abs = coef_df_abs.sort_values(by='Impacto', ascending=False)

# Gráfico de barras con las variables más influyentes
# Elegimos las 20 más influyentes
top_coef = coef_df_abs.head(20)

# Diccionario de reemplazo de etiquetas
map_labels = {
    'cat__Vict Descent_L': 'Ascendencia tailandesa',
    'cat__Vict Descent_D': 'Ascendencia camboyana',
    'cat__Vict Descent_P': 'Ascendencia isleña del pacífico',
    'cat__Vict Descent_G': 'Ascendencia guanameña',
    'cat__Vict Descent_V': 'Ascendencia vietnamita',
    'cat__Vict Descent_H': 'Ascendencia hispana',
    'cat__Vict Descent_C': 'Ascendencia china',
    'cat__Vict Descent_B': 'Ascendencia afroamericana',
    'cat__Vict Descent_S': 'Ascendencia samoana',
    'cat__Vict Descent_J': 'Ascendencia japonesa',
    'cat__Vict Descent_Z': 'Ascendencia del sur de Asia',
    'cat__Vict Sex_M': 'Sexo masculino',
    'cat__Vict Sex_H': 'Sexo otro',
    'cat__Vict Descent_U': 'Ascendencia hawaiana',
    'cat__AREA NAME_Pacific': 'Área: Oeste de LA',
    'cat__Vict Descent_O': 'Otras etnias',
    'cat__Vict Descent_F': 'Ascendencia filipina',
    'cat__FRANJA_HORARIA_Noche': 'Franja horaria: noche',
    'cat__FRANJA_HORARIA_Madrugada': 'Franja horaria: madrugada',
    'cat__Vict Descent_K': 'Ascendencia coreana',
    'cat__Vict Descent_W': 'Ascendencia blanca',
    'cat__AREA NAME_Foothill': 'Área: Foothill',
    'cat__AREA NAME_Wilshire': 'Área: Wilshire',
    'cat__Vict Sex_F': 'Sexo femenino',
    'cat__AREA NAME_Central': 'Área: Centro',
    'cat__AREA NAME_Harbor': 'Área: Puerto',
}

# Crear columna de etiquetas más amigables
top_coef['Etiqueta'] = top_coef['Variable'].map(map_labels).fillna(top_coef['Variable'])

# Separar positivos y negativos
top_coef_pos = top_coef[top_coef['Coeficiente'] > 0]
top_coef_neg = top_coef[top_coef['Coeficiente'] < 0]

# Crear gráfico
fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Gráfico de coeficientes positivos
sns.barplot(x='Impacto', y='Etiqueta', data=top_coef_pos,
            palette='Reds', ax=axes[0])
axes[0].set_title('Variables que aumentan la probabilidad de ser víctima de un crimen grave')
axes[0].set_xlabel('Importancia (valor absoluto del coeficiente)')
axes[0].set_ylabel('')

# Gráfico de coeficientes negativos
sns.barplot(x='Impacto', y='Etiqueta', data=top_coef_neg,
            palette='Blues', ax=axes[1])
axes[1].set_title('Variables que disminuyen la probabilidad de ser víctima de un crimen grave')
axes[1].set_xlabel('Importancia (valor absoluto del coeficiente)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# AUC-ROC
# Calcular probabilidades predichas para la clase positiva
y_probs = pipeline.predict_proba(X_test)[:, 1]

# Calcular el AUC-ROC
auc = roc_auc_score(y_test, y_probs)

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Graficar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

auc






# Modelo no supervisado

# Filtrar columnas necesarias
df_cluster = df[['LAT', 'LON', 'FRANJA_HORARIA', 'Weapon Desc']].copy()

# Eliminar ubicaciones inválidas (ej. coordenadas nulas o fuera de LA)
df_cluster = df_cluster[(df_cluster['LAT'] > 33) & (df_cluster['LAT'] < 35) &
                        (df_cluster['LON'] < -117) & (df_cluster['LON'] > -119)]

# Eliminar filas con valores faltantes
df_cluster = df_cluster.dropna()

# Preprocesamiento: normalización para coordenadas y one-hot para categorías
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['LAT', 'LON']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['FRANJA_HORARIA', 'Weapon Desc'])
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
plt.title('Método del codo')
plt.grid(True)
plt.show()

# Entrenar modelo final con k=3 (si se mantiene como óptimo)
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init='auto')
cluster_labels = kmeans_final.fit_predict(X)

# Agregar etiquetas de cluster
df_cluster['cluster'] = cluster_labels

# Visualización geográfica
plt.figure(figsize=(8,6))
scatter = plt.scatter(df_cluster['LON'], df_cluster['LAT'], c=df_cluster['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel("Longitud")
plt.ylabel("Latitud")
plt.title("Clusters de crímenes por ubicación")
plt.grid(True)
plt.colorbar(scatter, label='Cluster')
plt.show()

# Número de crímenes por cluster
print("Número de crímenes por cluster:")
print(df_cluster['cluster'].value_counts(), '\n')

# Armas más comunes por cluster
print("Armas más comunes por cluster:")
for c in sorted(df_cluster['cluster'].unique()):
    print(f"\nCluster {c}:")
    print(df_cluster[df_cluster['cluster'] == c]['Weapon Desc'].value_counts().head(5))

# Franja horaria más común por cluster
print("\nFranja horaria más frecuente por cluster:")
print(df_cluster.groupby('cluster')['FRANJA_HORARIA'].agg(lambda x: x.mode().iloc[0]))


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
    
# Apmliación del análisis a partir de los resultados
# Filtrar columnas necesarias y eliminar nulos
df_chi = df[['Vict Sex', 'Crm Cd Desc']].dropna() 

# Diccionario de reemplazo
sexo_map = {
    'F': 'Femenino',
    'M': 'Masculino',
    'X': 'Otro',
    'H': 'Desconocido'  # Asumo que H no es masculino sino algún valor extra
}
df_chi['Vict Sex'] = df_chi['Vict Sex'].replace(sexo_map)

# Crímenes más frecuentes por sexo
top_crimes = df_chi['Crm Cd Desc'].value_counts().head(10).index
df_top = df_chi[df_chi['Crm Cd Desc'].isin(top_crimes)]


plt.figure(figsize=(10, 6))
sns.countplot(data=df_top, y='Crm Cd Desc', hue='Vict Sex', order=top_crimes)
plt.title('Top 10 tipos de crimen por sexo de la víctima')
plt.xlabel('Número de casos')
plt.ylabel('Tipo de crimen')
plt.legend(title='Sexo')
plt.tight_layout()
plt.show()   

# Porcentajes relativos
# Calcular porcentajes
tabla_pct = pd.crosstab(df_top['Vict Sex'], df_top['Crm Cd Desc'], normalize='index') * 100

# Transponer para graficar
tabla_pct.T.plot(kind='barh', figsize=(10, 6), stacked=False)
plt.title('Distribución porcentual de los 10 crímenes más frecuentes por sexo')
plt.xlabel('Porcentaje (%)')
plt.ylabel('Tipo de crimen')
plt.legend(title='Sexo')
plt.tight_layout()
plt.show()    

# Mapa de calor
# Tabla completa
tabla = pd.crosstab(df_chi['Vict Sex'], df_chi['Crm Cd Desc'])

# Heatmap (puedes limitar a top 20 crímenes si hay muchos)
top_20 = tabla.sum().sort_values(ascending=False).head(20).index
tabla_top20 = tabla[top_20]

plt.figure(figsize=(12, 5))
sns.heatmap(tabla_top20, annot=True, fmt='d', cmap='Blues')
plt.title('Mapa de calor: crímenes por sexo (top 20)')
plt.xlabel('Tipo de crimen')
plt.ylabel('Sexo de la víctima')
plt.tight_layout()
plt.show()    

# Residuos
chi2, p, dof, expected = chi2_contingency(tabla)

# Calcular residuos estandarizados
residuos = (tabla - expected) / np.sqrt(expected)

# Limitar visualización a los 20 crímenes más frecuentes
residuos_top20 = residuos[top_20]

plt.figure(figsize=(12, 5))
sns.heatmap(residuos_top20, annot=True, center=0, cmap='RdBu_r')
plt.title('Residuos estandarizados (chi-cuadrado) - diferencias respecto a lo esperado')
plt.xlabel('Tipo de crimen')
plt.ylabel('Sexo de la víctima')
plt.tight_layout()
plt.show()