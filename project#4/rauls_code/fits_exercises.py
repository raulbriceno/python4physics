import matplotlib.pyplot as plt
import numpy as np 

"""

"""

"first, let's define some basic functions"
pi= np.pi
def sqrt(x): return np.sqrt(x)
def exp(x): return np.exp(x)

def qq(Ei, mB, mC ):
	Dms = pow(mB,2) - pow(mC,2)
	Sms = pow(mB,2) + pow(mC,2)
	
	num = pow(Ei,4) + pow(Dms,2) - 2 * pow(Ei,2) * Sms
	
	return num / (4.0 * pow(Ei,2))

def missing_mass(mA, mB, pB ):
	tmp = pow(mA,2) + pow(mB,2)
	tmp = tmp - 2* mA * sqrt(pow(mB,2) + pow(pB,2))
	return sqrt(tmp)

"""
###############################################
###############################################
				constant example
###############################################
###############################################
"""

def constant(a):
	return a
	
def chi2_MN(a):
	Pis = constant(a)
	chi20 = np.sum(pow((Pis-Ys)/sigs,2))
	return chi20
	
def find_min_chi2_cont():
	M_Gamma_guess = Ys[0]
	"""
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
	"""
	from scipy import optimize
	a = optimize.minimize(chi2_MN, M_Gamma_guess, method='nelder-mead')
	print("a",a)
	return a.x
	

	
constant_plot = 'n'
if constant_plot == 'y':
	plt.figure(figsize=(13,10))

	Ns= 100
	
	
	Xs  = np.arange(Ns)
	Ys = 3*(np.random.random(Ns)-.5) + 1000
	sigs  = (np.random.random(Ns)+.5)*3
	

	plt.errorbar(Xs, Ys, yerr= sigs, markersize=8,fmt='o',color='k',mfc='white',mec='k', elinewidth=2, capsize=6, mew=1.4,zorder=10)	

	data = np.ones((3,len(Xs)),dtype=float)
	data[0]= Xs
	data[1]= Ys
	data[2]= sigs
	
	filename = 'const_data.txt'
	np.savetxt(filename,data.T)

	
	plt.xlim([min(Xs),max(Xs)])
	plt.ylim([np.mean(Ys)-5*np.mean(sigs),np.mean(Ys)+5*np.mean(sigs)])
		
	plt.savefig('./const.pdf',
					bbox_inches='tight', 
					transparent=True)

	a = find_min_chi2_cont()
	plt.plot(Xs,constant(a)*np.ones(len(Xs)),linewidth=4)
	plt.xlim([min(Xs),max(Xs)])
	plt.ylim([np.mean(Ys)-5*np.mean(sigs),np.mean(Ys)+5*np.mean(sigs)])

	plt.savefig('./const_fit.pdf',
					bbox_inches='tight', 
					transparent=True)

	
load_const_data = 'y'
if load_const_data == 'y':
	filename = 'const_data.txt'
	const_data = np.loadtxt(filename)
	print("shape(const_data) = ",np.shape(const_data))
	Xs,Ys,sigs = const_data.T
	print("shape(Xs) = ",np.shape(Xs))
	print("shape(Ys) = ",np.shape(Ys))
	print("shape(sigs) = ",np.shape(sigs))

"""
###############################################
###############################################
				MN example
###############################################
###############################################
"""

def line(a,b,mpi):
	return a+mpi*b
	
def chi2_MN(ab):
	a,b = ab
	Pis = line(a,b,mpis)
	chi20 = np.sum(pow((Pis-Ys)/sigs,2))
	return chi20
	
def find_min_chi2_mN():
	M_Gamma_guess = [800,1.0]
	"""
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
	"""
	from scipy import optimize
	ab = optimize.minimize(chi2_MN, M_Gamma_guess, method='nelder-mead')
	return ab.x
	

	
MN_plot = 'n'
if MN_plot == 'y':
	plt.figure(figsize=(13,10))

	filename = 'MN_data.txt'
	MN_data = np.loadtxt(filename)
	mpis,Ys,sigs = MN_data.T

	
		
	plt.errorbar(mpis, Ys, yerr= sigs, markersize=8,fmt='o',color='k',mfc='white',mec='k', elinewidth=2, capsize=6, mew=1.4,zorder=10)	
	a,b = find_min_chi2_mN()
	
	mpis = np.arange(0,800,1)
	plt.xlim([min(mpis),max(mpis)])
	plt.ylim([750,1700])
	plt.ylabel(r'$M_N/{\rm MeV}$ ',size=25)
		
	plt.xlabel(r'$m_\pi/{\rm MeV}\sim \sqrt{m_q}$',size=25, position=(1,1.2))
	plt.savefig('./MN.pdf',
					bbox_inches='tight', 
					transparent=True)
	plt.plot(mpis,line(a,b,mpis))
	plt.xlim([min(mpis),max(mpis)])
	plt.ylim([750,1700])
	plt.savefig('./MN_fit.pdf',
					bbox_inches='tight', 
					transparent=True)

"""
###############################################
###############################################
				BW example
###############################################
###############################################
"""


def P_BW(M,Gamma,E): 
	tmp1 = pow(pow(E,2)- pow(M,2) , 2)
	tmp2 =  pow(M*Gamma,2)
	tmp3 =  pow(M,3)*Gamma
	return tmp3 / ( tmp1 + tmp2 )

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

	
BW_plot='n'
if BW_plot == 'y':
	plt.figure(figsize=(13,10))
	sig = .1

	M = 4.0
	Gamma = 1.5

	Xs  = np.arange(2,6,.3)
	
	dXs =  (np.random.random(len(Xs))-1)/4.
	
	Xs = Xs + dXs
	Ys = P_BW(M,Gamma,Xs)+(np.random.random(len(Xs))/100.)
	sigs  = (1.0-np.random.random(len(Xs))/2.)*sig
	

	
	plt.errorbar(Xs, Ys, yerr= sigs, markersize=8,fmt='o',color='k',mfc='white',mec='k', elinewidth=2, capsize=6, mew=1.4,zorder=10)	

	data = np.ones((3,len(Xs)),dtype=float)
	data[0]= Xs
	data[1]= Ys
	data[2]= sigs
	
	filename = 'BW_data.txt'
	np.savetxt(filename,data.T)

	Es = np.arange(1.,7,.01)
	plt.xlim([min(Es), max(Es)])	
	plt.ylim([0, 3])	
	
	plt.savefig('./BW_data.pdf',
					bbox_inches='tight', 
					transparent=True)
	
	M,Gamma = find_min_chi2_BW()
	plt.plot(Es, P_BW(M,Gamma,Es),color='r')


	plt.xlim([min(Es), max(Es)])	
	plt.ylim([0, 3])	
	
	plt.savefig('./BW_fit.pdf',
					bbox_inches='tight', 
					transparent=True)
