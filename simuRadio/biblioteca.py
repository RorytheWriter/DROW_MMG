import numpy as np 
from sklearn.model_selection import train_test_split
from numpy import mean
import statistics
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pylab import *  
from rsome import dro
import rsome as rso
from rsome import E
import pandas as pd
import numpy as hsplit
#---------------------------------------------------
def superior(train, epsilon):
    n = len(train)
    media = mean(train) 
    model = dro.Model(n)  
    x = model.dvar(1)  
    z = model.rvar(1)  
    u = model.rvar(1)  
    fset = model.ambiguity() 
    for s in range(n):         
         #fset[s].suppset(rso.norm(z - train[s], 1) <= u, z >= 0, z <= media) 
         fset[s].suppset(rso.norm(z - train[s], 1) <= u, z >= 0) 
    fset.exptset(E(u) <= epsilon) 
    pr = model.p                     
    fset.probset(pr == 1/n)  # probabilidad de los escenarios
    model.minsup(E(x*z), fset) # optimizando
    model.st(x == 1)  #restricciones           
    model.solve()   # solucionando                   
    model.get() 
    return model.get()
#---------------------------------------------------
def inferior(train,epsilon):
    n = len(train)
    media = mean(train) 
    model = dro.Model(n)  
    x = model.dvar(1)  
    z = model.rvar(1)  
    u = model.rvar(1)  
    fset = model.ambiguity() 
    for s in range(n):         
         #fset[s].suppset(rso.norm(z - train[s], 1) <= u, z >= 0, z <= media) 
         fset[s].suppset(rso.norm(z - train[s], 1) <= u, z >= 0) 
    fset.exptset(E(u) <= epsilon) 
    pr = model.p                     
    fset.probset(pr == 1/n)  # probabilidad de los escenarios
    model.maxinf(E(x*z), fset) # optimizando
    model.st(x == 1)  #restricciones           
    model.solve()   # solucionando                   
    model.get() 
    return model.get()
#--------------------------------------------------
def validacion (test):
    media = mean(test)
    return media
#-----------------------------------------------------
def dataSet(dirExcel,numCol): # "numCol" = número de columnas a agrupar (debe ser divisor exacto de las col de la matriz)
   df = pd.read_excel(dirExcel, header=None) # leyendo hoja de Excel
   matriz = df.to_numpy() # obteniendo matriz afín a numpy
   F,C = matriz.shape # obteniendo número de filas y columnas de la matriz
   delta = int(C/numCol) # determinando número de submatrices
   vProm = np.ones((numCol,1))*(1/numCol) # vector promediador de dimensión "numCol"
   data = np.empty((F,0), int) # vector columna vacio de "F" filas
   d = np.hsplit(matriz,delta) # arreglo de submatrices de "delta" columnas a partir de "matriz"
   for k in range(len(d)):
       w = np.dot(d[k],vProm) # producto punto entre la primera submatriz y "unos"
       data = np.append(data, w, axis=1) # adicionando resultado anterior (columna) a data
   return (data)
#---------------------------
#data = dataSet("data\prueba.xlsx",3)
#data = dataSet("data\kaggleData.xlsx",8)
#print(data)
#filas,columnas = data.shape
#print(filas,columnas)
#---------------------------------------------------------------
