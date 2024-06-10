import matplotlib.pyplot as plt
import numpy as np 

"""

In this module, we will load the data generated in example_Gauss.py
and plot it to make sure we get the same result

"""

"here we save the generated points into a text file"
filename='data.txt'
m0s = np.loadtxt(filename)

"""
the rest is pretty similar to  example_Gauss.py"""
Nbins = 25
hist, bin_edges = np.histogram(m0s,bins=Nbins)

plt.hist(m0s, bins=Nbins)  



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel('Number of points',size=20)
plt.xlabel('m0',size=20, position=(1,1.2))

"here save the figure"
plt.savefig('./example_Gauss_check.pdf',
				bbox_inches='tight', 
				transparent=True)