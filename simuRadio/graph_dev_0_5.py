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

from biblioteca import *
#---------- The "empirical dataset"----------------------------------------------------
n = 100 # number of records in the historical data
np.random.seed(1) # semilla
mu_01, sigma_01 = 9, 0.5 # media y desvio estandar
zhat_01 = np.random.normal(mu_01, sigma_01, n) # The empirical dataset
#-------------------------------------------------------------------------------------
K = 200   # iteraciones para el cálculo de la confianza de epsilon
T = 200 #320 # número de valores del radio epsilon a evaluar
stp = 0.0075 #0.0046875 # incrementos del radio epsilon.... T * stp = 1.5 aprox !!!!
#-----------------------------------------------
epsilon = 0; c_01=[]; ep_01=[]; ug=[]; lg=[]; vg=[]
for e in range(T):
   cont, u, l, v, um, vm, lm = 0, 0, 0, 0, 0, 0, 0  
   for i in range(K): 
        train, test = train_test_split(zhat_01 , test_size = 0.50, shuffle = True) # datos de entrenamiento y validación
        u = (superior(train, epsilon)); 
        l = (inferior(train, epsilon)); 
        v = (validacion(test));
        print(f"t: {e} and k: {i}")
        if (v > l and v < u):
           cont = cont + 1  
   c_01.append(cont/K)
   ep_01.append(epsilon)
   epsilon = epsilon + stp # <<<<<<<-------- incremento del radio
#----------------------------------------------------------------------
# graph
fig, axes = plt.subplots()
axes.set_xlabel("Wasserstein radio")
axes.set_ylabel("reliability")
axes.plot(ep_01,c_01, color="black")
plt.show()  
#----- Salvando datos en archivos txt -----------
np.savetxt('ep_01.txt', ep_01)
np.savetxt('c_01.txt', c_01)
# ----------------------------------------------
