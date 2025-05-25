import pandas as pd
import os
import shutil
import kagglehub
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
import numpy as np

#Variables globales
archivo = 'Crime_Data_from_2020_to_Present'
dest_folder = os.path.join(os.getcwd(), 'dataset')

def cargar_dataset(sufijo=''):
    df = pd.read_csv(os.path.join(dest_folder, archivo+sufijo+'.csv'))
    return df

def cargar_dataset_procesado(sufijo=''):
    df= pd.read_pickle(os.path.join(dest_folder, archivo+sufijo+'.pkl'))
    return df

def guardar_dataset(df,sufijo=''):
    df.to_csv(os.path.join(dest_folder, archivo+sufijo+'.csv'))

def guardar_dataset_procesado(df,sufijo=''):
    df.to_pickle(os.path.join(dest_folder, archivo+sufijo+'.pkl'))

def imputar_con_regresion(df, columna_objetivo, columnas_predictoras):
    df_entrenamiento = df[df[columna_objetivo].notna()]
    df_imputar = df[df[columna_objetivo].isna()]

    # Si no hay nada que imputar, devolver None o una lista vacía
    if df_imputar.empty:
        print("No hay valores faltantes que imputar.")
        return []

    # Convertir variables categóricas a dummies
    X_train = pd.get_dummies(df_entrenamiento[columnas_predictoras])
    X_pred = pd.get_dummies(df_imputar[columnas_predictoras])

    # Alinear columnas
    X_pred = X_pred.reindex(columns=X_train.columns, fill_value=0)

    y_train = df_entrenamiento[columna_objetivo]

    # Entrenar modelo
    modelo = RandomForestRegressor()
    modelo.fit(X_train, y_train)

    # Verificar que X_pred tenga filas antes de predecir
    if X_pred.empty:
        raise ValueError("X_pred está vacío. Verifica que hay datos válidos para imputar.")
    
    predicciones = modelo.predict(X_pred)

    return predicciones

def categorizar_hora(hora):
    if pd.isna(hora): return 'Desconocida'
    hora = int(hora)
    if 500 <= hora < 1200:
        return 'Mañana'
    elif 1200 <= hora < 1700:
        return 'Tarde'
    elif 1700 <= hora < 2100:
        return 'Noche'
    else:
        return 'Madrugada'