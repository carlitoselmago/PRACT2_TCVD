from helpers import *

df = cargar_dataset_procesado('_03')

guardar_dataset(df,'_final')

print("Dataset exportado como %_final.csv")