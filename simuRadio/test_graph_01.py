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

ep_00 = np.loadtxt('ep_00.txt')
ep_01 = np.loadtxt('ep_01.txt')
ep_02 = np.loadtxt('ep_02.txt')
ep_03 = np.loadtxt('ep_03.txt')
ep_04 = np.loadtxt('ep_04.txt')

c_00 = np.loadtxt('c_00.txt')
c_01 = np.loadtxt('c_01.txt')
c_02 = np.loadtxt('c_02.txt')
c_03 = np.loadtxt('c_03.txt')
c_04 = np.loadtxt('c_04.txt')

fig, axes = plt.subplots(figsize=(12,7))
axes.set_xlabel(r'Wasserstein radius',fontsize=24)
axes.set_ylabel(r'Reliability',fontsize=24)
axes.plot(ep_00,c_00, color="black",label=r'$\sigma=0.25$',linewidth=2)
axes.plot(ep_01,c_01, color="black",linestyle="dashed",label=r'$\sigma=0.5$',linewidth=2)
axes.plot(ep_02,c_02, color="dimgray", label=r'$\sigma=1.0$',linewidth=2)
axes.plot(ep_03,c_03, color="dimgray",linestyle="dashed",label=r'$\sigma=1.5$',linewidth=2)
axes.plot(ep_04,c_04, color="silver",label=r'$\sigma=2.0$',linewidth=2)
axes.set_xlim([-0.05,1])
plt.rcParams.update({'font.size': 24})
#axes.set_xscale('log')
plt.grid(visible=True, which='major', axis='both')
plt.legend(fontsize=24)
plt.savefig('wasserradii.pdf', format='pdf')
plt.show()



