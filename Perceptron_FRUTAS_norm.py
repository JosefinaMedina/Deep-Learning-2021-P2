import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import RN_Perceptron as rn

# Leer FrutasTrain.csv
os.chdir('../Datos/')

datos = pd.read_csv("FrutasTrain.csv")
nColum = list(datos.columns.values)

#--- DATOS DE ENTRENAMIENTO ---
entradas = np.array(datos.iloc[:,0:2])

normalizarEntrada = 0  # 1 si normaliza; 0 si no
if normalizarEntrada:
    # Escala los valores entre 0 y 1
    # normalizador = preprocessing.MinMaxScaler()
    
    # Normaliza utilizando la media y el desvio
    normalizador= preprocessing.StandardScaler()
    entradas = normalizador.fit_transform(entradas)


#--- SALIDA BINARIA ---
opciones = datos['Clase'].unique()
salida = datos['Clase'] == opciones[1]  #es boolean
salida = np.array(salida * 1)  #lo convierte en binario

ORDEN = "RAND"
if ORDEN == "ASC":
    #-- orden ASCENDENTE ---
    orden = np.argsort(salida)
if ORDEN == "DESC":
    #-- orden DESCENDENTE ---
    orden = np.argsort(-1*salida)
else:
    ##--- orden ALEATORIO ---
    orden=np.random.permutation(len(salida))
        
salida = salida[orden]
entradas = entradas[orden, :]



alfa = 0.01
MAX_ITE = 350
dibuja=1
titulos = nColum[0:2]
[W, b, ite] = rn.entrena_Perceptron(entradas, salida,alfa, MAX_ITE, dibuja, titulos)    
yTrain = rn.aplica_Perceptron(entradas, W, b)


print("%% de aciertos en el entrenamiento:", 100*np.sum(yTrain==salida)/len(salida))
      
# -- El perceptron ya está entrenado ---
# W y b determinan la recta que separa los ejemplos 

# Leer FrutasTest.csv y ver que responde el perceptrón
datosTest = pd.read_csv("FrutasTest.csv")
salidaTest = datosTest['Clase'] == opciones[1]  #es boolean
salidaTest = np.array(salidaTest * 1)  #lo convierte en binario

xTest = np.array(datosTest.iloc[:,0:2])
if normalizarEntrada:
    xTest = normalizador.transform(xTest)

# Calcular las respuestas del perceptron
yTest = rn.aplica_Perceptron(xTest,W,b)
print("%% de aciertos en el testeo:", 100*np.sum(yTest==salidaTest)/len(salidaTest))

    