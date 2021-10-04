import numpy as np
import RN_Perceptron as rn

X = np.array([[0, 1], [1,0],[0,0],[1,1]])
T = np.array([0, 0, 0, 1])

MAX_ITE = 10
alfa = 0.25
dibuja=1
titulos = ['Atrib1', 'Atrib2']

[W, b, ite] = rn.entrena_Perceptron(X, T, alfa, MAX_ITE, dibuja, titulos)

Y = rn.aplica_Perceptron(X,W,b)    

nAciertos = sum(Y==T)
print("%% de aciertos = %.2f %%" % (100 * nAciertos/X.shape[0]))