import pandas as pd
import numpy as np
from sklearn import preprocessing
import RN_Perceptron as rn

# Leer FrutasTrain.csv
datos = pd.read_csv("../Datos/FrutasTrain.csv")
nColum = list(datos.columns.values)

#--- DATOS DE ENTRENAMIENTO ---
entradas = np.array(datos.iloc[:,0:2])

#--- SALIDA BINARIA ---
salida = datos['Clase'] == 'Melon'  #es boolean
salida = np.array(salida * 1)  #lo convierte en binario

alfa = 0.01
MAX_ITE = 350
dibuja=1
titulos = nColum[0:2]
[W, b, ite] = rn.entrena_Perceptron(entradas, salida,alfa, MAX_ITE, dibuja, titulos)    

yTrain = rn.aplica_Perceptron(entradas, W, b)
print("%% de aciertos en el entrenamiento:", 100*np.sum(yTrain==salida)/len(salida))

# Leer FrutasTest.csv y ver que responde el perceptrón
datosTest = pd.read_csv("../Datos/FrutasTest.csv")
xTest = np.array(datosTest.iloc[:,0:2])

salidaTest = datosTest['Clase'] == 'Melon'  #es boolean
salidaTest = np.array(salidaTest * 1)  #lo convierte en binario

# Calcular las respuestas del perceptron
yTest = rn.aplica_Perceptron(xTest,W,b)
print("%% de aciertos en el testeo:", 100*np.sum(yTest==salidaTest)/len(salidaTest))
