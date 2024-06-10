import matplotlib.pyplot as plt
import numpy as np 

filename = 'pBs_2body_decay.txt'
q0s = np.loadtxt(filename)
print("q0s = ", q0s)
print("type(q0s) = ", type(q0s))
print("shape(q0s) = ", np.shape(q0s))
Nbins=50
plt.hist(q0s, bins=Nbins)  
plt.show()