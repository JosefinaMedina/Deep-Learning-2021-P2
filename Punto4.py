import os 
import pandas as pd
from sklearn import preprocessing
import RN_Perceptron as rn
import chardet
import numpy as np

os.chdir('../02_Perceptron/')
nomArch='zoo.csv'

#-- detectando la codificacion de caracteres usada----
with open(nomArch, 'rb') as f:
    result = chardet.detect(f.read()) #or readline if the file is large
    
datos= pd.read_csv(nomArch, index_col=0, encoding=result['encoding'])
ncolum= datos.columns.values
nEj=datos.shape[0]

#--SALIDA BINARIA--
salida=datos['Clase'] == 'Reptil' #es boolean
salida=np.array(salida*1) #lo convierte en binario

#dfTipo2 = datos[datos.Clase =='Tipo2']
#---cualitativos a numericos---
#datos=pd.get_dummies(datos.iloc[:,0:-1])

#---DATOS DE ENTRENAMIENTO---
entradas=np.array(datos.iloc[:,0:-1])
orden=np.random.permutation(range(nEj))

salida=salida[orden]
entradas = entradas[orden, :]

normalizarEntrada = 1 #1 si normaliza, 0 si no
if normalizarEntrada:
    min_max_scaler=preprocessing.MinMaxScaler()
    entradas= min_max_scaler.fit_transform(entradas)
    
aciertos=[]
elegidos=[]
poraciertos=[]

itera=[]
masVotados= np.zeros(16)
for intentos in range(30):
    
    alfa=0.05
    MAX_ITE=200
    dibuja=0
    [W, b, ite]=rn.entrena_Perceptron(entradas, salida, alfa, MAX_ITE)
    yTrain=rn.aplica_Perceptron(entradas, W, b)
    
    OK=np.sum(yTrain==salida)
    #print("Intento %d - ite =%2d %%ACIERTOS %.2f" %(intentos, ite, 10))
    
    aciertos.append(OK)
    itera.append(ite)
    ordenElecc = np.argsort(-1*W).tolist()
    masVotados[ordenElecc[0]] = masVotados[ordenElecc[0]]+1
    masVotados[ordenElecc[1]] = masVotados[ordenElecc[1]]+1
    poraciertos.append(100*np.sum(yTrain==salida)/len(salida))
    elegidos.append( ordenElecc )

print("alfa %.2f, MAX_ITE %d, AVG Ite = %2f, AVG aciertos %.2f" %(alfa, MAX_ITE,np.mean(itera),np.mean(aciertos)))

idElecc = np.argsort(-1*masVotados)
print("El porcentaje de aciertos es ", np.mean(poraciertos))
print (masVotados)
print ("las caracteristicas mas elegidas son %s, %s y %s" % 
       (ncolum[idElecc[0]], ncolum[idElecc[1]], ncolum[idElecc[2]]))