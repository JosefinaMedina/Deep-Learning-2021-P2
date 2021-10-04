import pandas as pd
import numpy as np
from sklearn import model_selection,preprocessing
import RN_Perceptron as rn
    
datos = pd.read_csv('../02_Perceptron/Sonar.csv')
entradas = np.array(datos.iloc[:,:-1])   #-- todas las columnas menos la última

# # convirtiendo los atributos nominales en numericos
# entradas = np.array(pd.get_dummies(datos.iloc[:,:-1]))

## nombres de los atributos
#titulos = list(entradas.columns.values)

#--- SALIDA BINARIA ---
opciones = datos['class'].unique()
salidas = datos['class'] == opciones[1]  #es boolean
salidas = np.array(salida * 1)  #lo convierte en binario

#--- CONJUNTOS DE ENTRENAMIENTO Y TESTEO ---
X_train, X_test, T_train, T_test = model_selection.train_test_split(
        entradas, salidas, test_size=0.30, random_state=42)

alfa = 0.01
MAX_ITE = 300
[W, b, ite] = rn.entrena_Perceptron(X_train, T_train,alfa, MAX_ITE)
yTrain = rn.aplica_Perceptron(X_train, W, b)

# Calcular las respuestas del perceptron
yTest = rn.aplica_Perceptron(X_test,W,b)

aciertosTrain = 100 * np.sum(yTrain==T_train)/len(T_train)
aciertosTest  = 100 * np.sum(yTest==T_test)/len(T_test)
print("iteraciones utilizadas = ",ite)
print("%% aciertos datos entrenamiento %.2f:" % aciertosTrain)
print("%% aciertos datos de testeo %.2f:" % aciertosTest)