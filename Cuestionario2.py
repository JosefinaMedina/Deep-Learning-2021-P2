import pandas as pd
import numpy as np
from sklearn import model_selection,preprocessing
import RN_Perceptron as rn
    
datos = pd.read_csv(r'C:\Users\Josefina Medina\Documents\FISICA FACULTAD\CUARTO\deep learning\02_Perceptron\Lentes.csv')
entradas = np.array(datos.iloc[:,1:-1])   #-- todas las columnas menos la última



# # convirtiendo los atributos nominales en numericos
# entradas = np.array(pd.get_dummies(datos.iloc[:,:-1]))

## nombres de los atributos
#titulos = list(entradas.columns.values)

salidas = np.array(datos['diagnostico']=="2") * 1
nomClase = ['Otra', '2']

nColum = list(datos.columns.values)

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

poraciertos = []
i=0

for i in range(50):

    alfa = 0.05
    MAX_ITE = 200
    dibuja=1
    titulos = nColum[1:-1]
    [W, b, ite] = rn.entrena_Perceptron(entradas, salida,alfa, MAX_ITE, dibuja, titulos)    
    yTrain = rn.aplica_Perceptron(entradas, W, b)
    #print("%% de aciertos en el entrenamiento:", 100*np.sum(yTrain==salida)/len(salida))
    poraciertos.append(100*np.sum(yTrain==salida)/len(salida))
    i=+1

print("El porcentaje de aciertos es ", np.mean(poraciertos))