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
mu_04, sigma_04 = 9, 2.0 # media y desvio estandar
zhat_04 = np.random.normal(mu_04, sigma_04, n) # The empirical dataset
#-------------------------------------------------------------------------------------
K = 200   # iteraciones para el cálculo de la confianza de epsilon
T = 200 # número de valores del radio epsilon a evaluar
stp = 0.0075 # incrementos del radio epsilon.... T * stp = 1.5 aprox !!!!
#-----------------------------------------------
epsilon = 0; c_04=[]; ep_04=[]; ug=[]; lg=[]; vg=[]
for e in range(T):
   cont, u, l, v, um, vm, lm = 0, 0, 0, 0, 0, 0, 0  
   for i in range(K): 
        train, test = train_test_split(zhat_04 , test_size = 0.50, shuffle = True) # datos de entrenamiento y validación
        u = (superior(train, epsilon)); 
        l = (inferior(train, epsilon)); 
        v = (validacion(test));
        print(f"t: {e} and k: {i}")
        if (v > l and v < u):
           cont = cont + 1  
   c_04.append(cont/K)
   ep_04.append(epsilon)
   epsilon = epsilon + stp # <<<<<<<-------- incremento del radio
#------------------------------------------------------
# graph
fig, axes = plt.subplots()
axes.set_xlabel("Wasserstein radio")
axes.set_ylabel("reliability")
axes.plot(ep_04,c_04, color="black")
plt.show()
#----- Salvando datos en archivos txt -----------
np.savetxt('ep_04.txt', ep_04)
np.savetxt('c_04.txt', c_04)
# ----------------------------------------------


