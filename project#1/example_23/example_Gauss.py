import matplotlib.pyplot as plt
import numpy as np 

"""

In this module, we will generate distribution of points
that are centered about a mean value of m0,
with some deviation described by dm0.
The points will distributed according to a Gaussian distribution, 
also known as normal distribution 

https://en.wikipedia.org/wiki/Normal_distribution

"""

"first, let's define some basic functions"
pi= np.pi
def sqrt(x): return np.sqrt(x)
def exp(x): return np.exp(x)

"first, let's define some basic functions"
def f_gaus(x,mu,sig):
	amp = 1.0/sqrt(2.0 * pi * pow(sig,2))
	arg = pow(x-mu,2)/( 2.0 * pow(sig,2) )
	return amp * exp(-arg)
	
m0, dm0 = 1, .01

"""
Python has many built in functions, 
including a normal distributions
https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
"""
Npoints = 1000
m0s = np.random.normal(m0, dm0, Npoints)

"""here we save the generated points into a text file using 
https://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html"""
filename='data.txt'
np.savetxt(filename,m0s)

"""
we will seperate the number of elements having a given 'x'. 
We will do this by separating them into Nbins groups
 """
Nbins = 25
hist, bin_edges = np.histogram(m0s,bins=Nbins)

"""
to plot it, we use the hist function: 
	https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html?highlight=histogram#numpy.histogram
	"""
plt.hist(m0s, bins=Nbins)  

x=np.arange(m0-4*dm0,m0+4*dm0,dm0/100.)

"""
here we will normalize Gaussian distribution 
so that its peak matches the histogram peak
"""
tmp = f_gaus(x,m0,dm0)
tmp = tmp * max(hist) / max(tmp) 

plt.plot(x,tmp ,color='r')

print("hist",hist)



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.ylabel('Number of points',size=20)
plt.xlabel('m0',size=20, position=(1,1.2))


"here save the figure"
plt.savefig('./example_Gauss.pdf',
				bbox_inches='tight', 
				transparent=True)
