import matplotlib.pyplot as plt
import numpy as np 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

"""
Here we consider particle distributions described by 
a simplified version of a Breit-Wigner distribution 
https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution

P(E) = pow(M*Gamma,2) / (pow(pow(E,2)- pow(M,2) , 2) + pow(M*Gamma , 2))
"""

"first, let's define some basic functions"
pi= np.pi
def sqrt(x): return np.sqrt(x)
def exp(x): return np.exp(x)


def P_BW(M,Gamma,E): 
	tmp1 = pow(pow(E,2)- pow(M,2) , 2)
	tmp2 =  pow(M*Gamma,2)
	tmp3 =  pow(M,3)*Gamma
	return tmp3 / ( tmp1 + tmp2 )

BW_plot1='y'
if BW_plot1 == 'y':
	Es = np.arange(2,5,.01)
	M = 2.5

	Gamma = 1
	plt.plot(Es, P_BW(M,Gamma,Es),color='r')
	Gamma = 2.
	plt.plot(Es, P_BW(M,Gamma,Es),color='b')
	Gamma = 3.
	plt.plot(Es, P_BW(M,Gamma,Es),color='g')

	plt.xlim([min(Es), max(Es)])
	plt.savefig('./BW_example.pdf',
					bbox_inches='tight', 
					transparent=True)

def chi2_BW(M_Gamma):
	M,Gamma = M_Gamma
	Pis = P_BW(M,Gamma,Xs)
	
	chi20 = np.sum(pow((Pis-Ys)/sigs,2))
	dof = len(sigs)-2.0
	return chi20
	
def find_min_chi2_BW():
	M_Gamma_guess = [2.,1.0]
	"""
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
	"""
	from scipy import optimize
	M_Gamma = optimize.minimize(chi2_BW, M_Gamma_guess, method='nelder-mead')
	print("M_Gamma",M_Gamma)
	return M_Gamma.x
	
BW_plot2='n'
if BW_plot2 == 'y':
	plt.figure(figsize=(9,10))
	sig = .2

	M = 3.0
	Gamma = .5

	Xs  = np.arange(2,5,.2)
	Ys = P_BW(M,Gamma,Xs)
	sigs  = np.ones(len(Xs))*sig
	plt.errorbar(Xs, Ys, yerr= sigs, markersize=8,fmt='o',color='k',mfc='white',mec='k', elinewidth=2, capsize=6, mew=1.4,zorder=10)	

			
	Es = np.arange(2,5,.01)
	M = 2.0
	Gamma = .3
	
	print("chi2_BW(M,Gamma)",chi2_BW([M,Gamma]))
	plt.plot(Es, P_BW(M,Gamma,Es),color='r')

	M = 2.4
	Gamma = .4	
	print("chi2_BW(M,Gamma)",chi2_BW([M,Gamma]))
	plt.plot(Es, P_BW(M,Gamma,Es),color='b')

	M = 2.8
	Gamma = .45	
	print("chi2_BW(M,Gamma)",chi2_BW([M,Gamma]))
	plt.plot(Es, P_BW(M,Gamma,Es),color='g')

	plt.xlim([min(Es), max(Es)])
	plt.axhline(y=0,color='k',linewidth=1)
	plt.savefig('./BW_example_2.pdf',
					bbox_inches='tight', 
					transparent=True)


def rgb_to_hex(rgb):
	return '#%02x%02x%02x' % rgb

	
BW_plot3='n'
if BW_plot3 == 'y':
	plt.figure(figsize=(9,10))
	sig = .2

	M = 3.0
	Gamma = .5

	Xs  = np.arange(2,5,.2)
	Ys = P_BW(M,Gamma,Xs)
	sigs  = np.ones(len(Xs))*sig

			
	Ns=4
	for n0 in np.arange(Ns):
		Gamma = .1*n0+.2
		j0=n0/float(Ns)
		x=(1-j0)*47+j0*192
		y=(1-j0)*122+j0*39
		z=(1-j0)*121+j0*45
		colorf=rgb_to_hex((int(x),int(y),int(z)))
		
		tmp =[]
		Ms= np.arange(2,5,.01)
		for M in Ms:
			chi20=chi2_BW([M,Gamma])
			tmp.append(chi20)
		#plt.errorbar(M, chi20, markersize=8,fmt='o',color=colorf,mfc='white',mec=colorf, elinewidth=2, capsize=6, mew=1.4,zorder=10)	
		plt.plot(Ms,tmp,color=colorf)
		
		
		y_label = r'$\Gamma = '+str(round(Gamma,2))+'$'

		plt.text(4.0,2000-450*n0, y_label,size=20,color=colorf ,ha="center", va="center")

		plt.axhline(y=0,color='k',linewidth=1)

		plt.xlim([min(Ms), max(Ms)])

		plt.ylim([0,7000])

		plt.ylabel(r'$\chi^2$ ',size=25)
		
		plt.xlabel(r'$M$',size=25, position=(1,1.2))	

		plt.savefig('./BW_example_3.pdf',
						bbox_inches='tight', 
						transparent=True)

BW_plot4='y'
if BW_plot4 == 'y':
	plt.figure(figsize=(9,10))
	sig = .2

	M = 3.0
	Gamma = .5

	Xs  = np.arange(2,5,.2)
	Ys = P_BW(M,Gamma,Xs)
	sigs  = np.ones(len(Xs))*sig
	plt.errorbar(Xs, Ys, yerr= sigs, markersize=8,fmt='o',color='k',mfc='white',mec='k', elinewidth=2, capsize=6, mew=1.4,zorder=10)	

	M,Gamma = find_min_chi2_BW()
	Es = np.arange(2,5,.01)
	plt.plot(Es, P_BW(M,Gamma,Es),color='r')


	plt.xlim([min(Es), max(Es)])
	plt.axhline(y=0,color='k',linewidth=1)
	plt.savefig('./BW_example_4.pdf',
					bbox_inches='tight', 
					transparent=True)


