import pandas as pd
import numpy as np
from sklearn import model_selection,preprocessing

import RN_Perceptron as rn
    
datos = pd.read_csv('../02_Perceptron/Iris.csv')
entradas = np.array(datos.iloc[:,:-1])   #-- todas las columnas menos la última

# # convirtiendo los atributos nominales en numericos
# entradas = np.array(pd.get_dummies(datos.iloc[:,:-1]))

## nombres de los atributos
#titulos = list(entradas.columns.values)

#--- SALIDA BINARIA : 1 si es "drugY" ; 0 si no ---
salidas = np.array(datos['class']=="Iris-setosa") * 1
nomClase = ['Otra', 'Iris-setosa']
#--- CONJUNTOS DE ENTRENAMIENTO Y TESTEO ---
X_train, X_test, T_train, T_test = model_selection.train_test_split(
        entradas, salidas, test_size=0.30, random_state=42)

normalizarEntrada = 1  # 1 si normaliza; 0 si no
if normalizarEntrada:
    # Escala los valores entre 0 y 1
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

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
entradas = entradas[orden, :]

i=0
for i in range(10):
   
    alfa = 0.01
    MAX_ITE = 650
    [W, b, ite] = rn.entrena_Perceptron(X_train, T_train,alfa, MAX_ITE)
    yTrain = rn.aplica_Perceptron(X_train, W, b)

    # Calcular las respuestas del perceptron
    yTest = rn.aplica_Perceptron(X_test,W,b)

    aciertosTrain = 100 * np.sum(yTrain==T_train)/len(T_train)
    aciertosTest  = 100 * np.sum(yTest==T_test)/len(T_test)
    print("iteraciones utilizadas = ",ite)
    print("%% aciertos datos entrenamiento %.2f:" % aciertosTrain)
    print("%% aciertos datos de testeo %.2f:" % aciertosTest)