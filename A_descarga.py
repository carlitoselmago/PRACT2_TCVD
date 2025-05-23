import os
import shutil
import pandas as pd
import kagglehub
from helpers import *


dest_path = os.path.join(dest_folder, archivo+'.csv')

# Creamos una carpeta destino si no existe
os.makedirs(dest_folder, exist_ok=True)

# Descargamos el csv solo si no existe
if not os.path.exists(dest_path):
    print("Archivo no encontrado, descargando con kagglehub...")
    temp_path = kagglehub.dataset_download("ishajangir/crime-data")
    temp_file_path = os.path.join(temp_path, archivo+'.csv')

    if not os.path.exists(temp_file_path):
        raise FileNotFoundError(f"No se encontró el archivo esperado en el dataset: {temp_file_path}")

    shutil.copy2(temp_file_path, dest_path)
    print("Archivo copiado a:", dest_path)
else:
    print("El archivo ya existe en:", dest_path)

# Leemos CSV como Pandas dataset
df = cargar_dataset()
print(df.describe())

