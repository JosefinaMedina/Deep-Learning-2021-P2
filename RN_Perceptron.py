import numpy as np
from matplotlib import pyplot as plt
from grafica import *

def entrena_Perceptron(X, T, alfa, MAX_ITE, dibuja=1, titulos=[]):
    # Tamaño de los datos de entrada y títulos
    nCantEjemplos = X.shape[0]  # nro. de filas
    nAtrib = X.shape[1]         #nro. de columnas

    # Inicializar la recta
    W = np.random.uniform(-0.5, 0.5, size=nAtrib)
    b = np.random.uniform(-0.5, 0.5)
    
    if dibuja: # graficar
        dibuPtos(X, T, titulos)
        ph = dibuRecta(X, W, b)
        
    hubo_cambio=True
    ite=0
    while (hubo_cambio and (ite<MAX_ITE)):
        hubo_cambio=False
        ite = ite + 1
        
        # para cada ejemplo
        for i in range(nCantEjemplos):
            # Calcular la T
            # neta=b
            # for j in range(nAtrib):
            #      neta = neta + W[j] * X[i,j]
            neta = b + sum(W * X[i,:])
            y = 1 * (neta>0)
    
            # Si no es correcta, corregir W y b  
            if not(y==T[i]):
                hubo_cambio=True
                #    actualizamos los pesos W y b
                # for j in range(nAtrib):
                #     W[j] = W[j] + alfa *(T[i]-y)*X[i,j]
                W = W + alfa *(T[i]-y)*X[i,:]    
                b = b + alfa *(T[i]-y)
                        
        if dibuja: # graficar
            print(ite)
            ph = dibuRecta(X, W, b, ph)
            
    return([W,b,ite])
    
def aplica_Perceptron(X, W, b):
    cantEjemplos = X.shape[0]
    nAtrib = X.shape[1]
    Y = []
    for e in range(cantEjemplos):
        neta = b
        for j in range(nAtrib):
            neta = neta + W[j]*X[e,j]
        Y.append((neta>0)*1)    
      
    # ----  Otra forma de hacer lo mismo  ----
    #netas = W @ entradas.T + b
    #Y = 1* (netas>0)
    return(Y)
