import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import RN_feedforward as rn
    
# Leer FrutasTrain.csv
os.chdir('../03_CombinadorLineal/')
datos = pd.read_csv("Semillas.csv")

#--- EJEMPLO DE ENTRENAMIENTO ---
entradas = np.array(datos.iloc[:,0:6])
normalizarEntrada = 1

if normalizarEntrada:
    # Escala los valores entre 0 y 1
    normalizador = preprocessing.MinMaxScaler()
    entradas = normalizador.fit_transform(entradas)

#--SALIDA BINARIA--
salida=datos['Clase'] == 'Tipo2' #es boolean
salida=np.array(salida*1) #lo convierte en binario

alfa = 0.2
MAX_ITE = 500
FUN = 'logsig'
if (FUN=='tansig'):
    salida = 2* np.array(salida * 1)-1  #lo convierte en [-1,1]
    
# Entrenamos la neurona no lineal para aproximar los valores de clase
[W, b, ite] = rn.entrena_NeuronaGradiente(entradas, salida, alfa, MAX_ITE, FUN, CotaError=10e-05, verIte=1, dibuja=1, titulos=['Diametro', 'Color'])

netas = np.sum(entradas*W,axis=1)+b
salidas = rn.evaluar(FUN,netas)
if (FUN=='tansig'):
    y_pred1 = 2*((salidas>0)*1)-1 
if (FUN=='logsig'):
    y_pred1 = (salidas>0.5)*1

y_pred = rn.neurona_predice(entradas, W, b, FUN)
# yTrain = rn.evaluar(FUN, W @ entradas.T + b)
# # usamos el plano Z=0 como punto de corte
# yTrain[yTrain>0] = 1
# yTrain[yTrain<0] = -1 

print("cantidad aciertos Train: %d" % np.sum(y_pred==salida))
# -- La neurona no lineal ya está entrenada ---
# Leer FrutasTest.csv y ver que responde el perceptrón


