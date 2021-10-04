import numpy as np
from grafica import *

# Ejemplos de entrada de la función AND
entradas = np.array([[0,0], [0,1],[1,0],[1,1]])
salida = np.array([0,0,0,1])

# Tamaño de los datos de entrada y títulos
nCantEjemplos = entradas.shape[0]  # nro. de filas
nAtrib = entradas.shape[1]         #nro. de columnas
titulos = ['X1', 'X2']

# Inicializar la recta
W = np.array(np.random.uniform(-0.5, 0.5, size=2))
b = np.random.uniform(-0.5, 0.5)

# graficar
dibuPtos(entradas, salida, titulos)
ph = dibuRecta(entradas, W, b)

MAX_ITE = 10
alfa = 0.01
ite=0
while (ite<MAX_ITE):
    for e in range(nCantEjemplos):
        # Calcular y  (la salida del perceptron)


        # Si no es correcta, corregir  W y b


        # graficar la recta
        ph = dibuRecta(entradas, W, b, ph)
        
    ite = ite + 1
    print("ite %d" % ite)
    