import pandas as pd
import numpy as np
from sklearn import preprocessing

datos = pd.read_csv("../Datos/FrutasTrain.csv")
xTrain = np.array(datos.iloc[:,0:2])

# Escala los valores entre 0 y 1
normalizador = preprocessing.MinMaxScaler()
xTrain = normalizador.fit_transform(xTrain)

# Leer FrutasTest.csv 
datosTest = pd.read_csv("../Datos/FrutasTest.csv")
xTest = np.array(datosTest.iloc[:,0:2])

#--- noramlizando los datos de testep ---
xTest = normalizador.transform(xTest)
