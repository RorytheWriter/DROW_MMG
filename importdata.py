import pandas as pd
import numpy as np
import numpy as hsplit

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
#------------------------------------------------------------------


#data = dataSet("data\prueba.xlsx",3)
data = dataSet("kaggleData.xlsx",8)
print(data)
filas,columnas = data.shape
print(filas,columnas)


