import pandas as pd
import os
from helpers import *

df = cargar_dataset()

# Creamos una muestra de la población de total para poder hacer los siguientes cálculos de una manera mas ágil

muestra = df.sample(n=100000,random_state=123)

print(muestra.head())

guardar_dataset(muestra,'_02')