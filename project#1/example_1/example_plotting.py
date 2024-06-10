import matplotlib.pyplot as plt
import numpy as np 

"""
here we define two arrays of real valued numbers, 
the first, ranges from 0 to 1.99 in steps of .01
the second,  ranges from 0 to 1.9 in steps of .1
"""
x=np.arange(0,2,.01)
x1=np.arange(0,2,.1)

"""
let's print these out to visualize them
'"""
print("************************")
print("x = ",x)
print("")
print("************************")
print("x1 = ",x1)

"""
here we define two functions, 
the first, returns x to the power of 1.5
the second, returns x squared
"""
def f1(x):
	return pow(x,1.5)
	
def f2(x):
	return pow(x,2)
	

"""
here we break the figure into three panels. 
In the first panel, we plot both f1 and f2 in red and blue, respectively
"""
plt.subplot(311)
plt.plot(x,f1(x),color='red')
plt.plot(x,f2(x),color='blue')
"""
this limits the x-range of the plot
"""
plt.xlim([min(x), max(x)])
"""
this makes vertical and horizontal black lines of width=1
"""
plt.axvline(x=0,color='k',linewidth=1)
plt.axhline(y=0,color='k',linewidth=1)

"""
In the second panel, we plot both f1 as a continuous red line 
and compare it to a discrete red circles...
the latter are plotted using the errorbar function
"""
plt.subplot(312)
plt.plot(x,f1(x),color='red')
plt.errorbar(x1,f1(x1),markersize=8,fmt='o',color='r',mfc='white',mec='r', elinewidth=2, capsize=4, mew=1.4)	
plt.xlim([min(x), max(x)])
plt.axvline(x=0,color='k',linewidth=1)
plt.axhline(y=0,color='k',linewidth=1)

"""
In the third panel, we plot both f2 as a continuous blue line 
and compare it to a discrete blue squares
"""
plt.subplot(313)
plt.plot(x,f2(x),color='b')
plt.errorbar(x1,f2(x1),markersize=8,fmt='s',color='b',mfc='white',mec='b', elinewidth=2, capsize=4, mew=1.4)	
plt.xlim([min(x), max(x)])
plt.axvline(x=0,color='k',linewidth=1)
plt.axhline(y=0,color='k',linewidth=1)


"here save the figure"
plt.savefig('./example_plot.pdf',
				bbox_inches='tight', 
				transparent=True)