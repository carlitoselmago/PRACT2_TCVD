### Tipología y ciclo de vida de los datos - PRACT2

## Dataset utilizado
https://www.kaggle.com/datasets/ishajangir/crime-data


### Instalación

Ejecuta
```
pip install -r requirements.txt
```

### Archivos del repositorio

- carpeta 'dataset' : contiene los archivos del dataset utilizado
- carpeta 'assets' : contiene gráficos relativos a la práctica

- A_decarga : Descarga el dataset utilizando la librería kagglehub en la carpeta 'dataset'
- B_integracion_y_seleccion : Muestreo de la población total
- C_limpieza: proceso de limpieza, imputación y combinación de variables
- D_analisis: Análisis estadístico
- E_exportar: Símplemente exporta el dataset a csv

Los archivos se deben ejecutar en orden alfabético, cada archivo procesará el dataset y creará un archivo nuevo si pertoca para el siguiente paso del proceso.

Para ejecutarlos todos con una sola llamada ejecutar:
```
python main.py
```