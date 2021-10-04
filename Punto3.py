import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import RN_Perceptron as rn

# Leer FrutasTrain.csv
os.chdir('../02_Perceptron/')

datos = pd.read_csv("Semillas1.csv")
nColum = list(datos.columns.values)

#--- DATOS DE ENTRENAMIENTO ---
entradas = np.array(datos.iloc[:,0:-1])

normalizarEntrada = 1  # 1 si normaliza; 0 si no
if normalizarEntrada:
    # Escala los valores entre 0 y 1
    # normalizador = preprocessing.MinMaxScaler()
    
    # Normaliza utilizando la media y el desvio
    normalizador= preprocessing.StandardScaler()
    entradas = normalizador.fit_transform(entradas)

salidas = np.array(datos['Clase']=="Tipo2") * 1
nomClase = ['Otra', 'Tipo2']

ORDEN = "RAND"
if ORDEN == "ASC":
    #-- orden ASCENDENTE ---
    orden = np.argsort(salidas)
if ORDEN == "DESC":
    #-- orden DESCENDENTE ---
    orden = np.argsort(-1*salidas)
else:
    ##--- orden ALEATORIO ---
    orden=np.random.permutation(len(salidas))
        
salida = salidas[orden]
entradas = entradas[orden]

poraciertos=[]
i=0
for i in range(50):

    alfa = 0.05
    MAX_ITE = 200
    dibuja=1
    titulos = nColum[0:-1]
    [W, b, ite] = rn.entrena_Perceptron(entradas, salida,alfa, MAX_ITE, dibuja, titulos)    
    yTrain = rn.aplica_Perceptron(entradas, W, b)
    #print("%% de aciertos en el entrenamiento:", 100*np.sum(yTrain==salida)/len(salida))
    poraciertos.append(100*np.sum(yTrain==salida)/len(salida))
    i=+1

print("El porcentaje de aciertos es ", np.mean(poraciertos))
      
# -- El perceptron ya está entrenado ---
# W y b determinan la recta que separa los ejemplos 


    