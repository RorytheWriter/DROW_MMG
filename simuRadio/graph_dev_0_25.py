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

import traceback

from biblioteca import *
#---------- The "empirical dataset"----------------------------------------------------
n = 100 # number of records in the historical data
np.random.seed(1) # semilla
mu_00, sigma_00 = 9, 0.25 # media y desvio estandar
zhat_00 = np.random.normal(mu_00, sigma_00, n) # The empirical dataset
#-------------------------------------------------------------------------------------
K = 200   # iteraciones para el cálculo de la confianza de epsilon
T = 200 # número de valores del radio epsilon a evaluar
stp = 0.0075 # incrementos del radio epsilon.... T * stp = 1.5 aprox !!!!
#-----------------------------------------------
epsilon = 0; c_00=[]; ep_00=[]; ug=[]; lg=[]; vg=[]
for e in range(T):
   cont, u, l, v, um, vm, lm = 0, 0, 0, 0, 0, 0, 0  
   for i in range(K): 
        print(f"\n Now with radius t: {e} and iteration k: {i}")
        train, test = train_test_split(zhat_00 , test_size = 0.50, shuffle = True) # datos de entrenamiento y validación
        try:
            u = (superior(train, epsilon)); 
            l = (inferior(train, epsilon)); 
            v = (validacion(test));
        except Exception:
            print("---------------- \n Uh oh Error:")
            print(traceback.format_exc())
            print("---------------- \n")
        else:
            if (v > l and v < u):
               cont = cont + 1  
   c_00.append(cont/K)
   ep_00.append(epsilon)
   epsilon = epsilon + stp # <<<<<<<-------- incremento del radio
#----------------------------------------------------------------------
# graph
fig, axes = plt.subplots()
axes.set_xlabel("Wasserstein radio")
axes.set_ylabel("reliability")
axes.plot(ep_00,c_00, color="black")
plt.show()  
#----- Salvando datos en archivos txt -----------
np.savetxt('ep_00.txt', ep_00)
np.savetxt('c_00.txt', c_00)
# ----------------------------------------------
