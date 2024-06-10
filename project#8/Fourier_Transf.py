import matplotlib.pyplot as plt
import numpy as np 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

"""
"""

"first, let's define some basic functions"
pi= np.pi
def sqrt(x): return np.sqrt(x)
def exp(x): return np.exp(x)
def sin(x): return np.sin(x)
def cos(x): return np.cos(x)
def log(x): return np.log(x)
def poly0(x): return (pow(x,3)/6.) + (pow(x,4)/8.0)

""""
Simpson integrator for f(x) in [a,b]

f - Function to integrate (supplied by a user)
a - Lower limit of integration
b - Upper limit of integration
s - Result of integration (out)
n - number of intervals

"""

"Trapezoid rule"
def int_trap(f, a, b, n):
	"""
	To assure n is a multiple of two
	we can just use the fact that int(n/2) will floor n/2. 
	
	Therefore, f int(n/2) is n, the n is a multiple of two,
	otherwise we should replace n = int((n+1)/2) 

	"""
	
	if int(n/2)<n/2.0:
		n = int((n+1)/2)
	
	s = 0.0	
	dx = (b-a)/n;
	
	for i0 in np.arange(1,n):
		x = a + i0 * dx
		s = s + f(x)

	"""
	we have to remember to add the pieces not included above, 
	namely the points as x=a, a+dx, b
	"""
	s = s + ((f(a) + f(b))/2.0)
	
	"finally, we need to multiply by dx"
	
	return s*dx
	 


"Simpson rule"
def int_simp(f, a, b, n):
	"""
	To assure n is a multiple of two
	we can just use the fact that int(n/2) will floor n/2. 
	
	Therefore, f int(n/2) is n, the n is a multiple of two,
	otherwise we should replace n = int((n+1)/2) 

	"""
	
	if int(n/2)<n:
		n = int((n+1)/2)
	
	s = 0.0	
	dx = (b-a)/n;
	
	for i0 in np.arange(2,n,2):
		x = a + i0 * dx
		s = s + ( 2 * f(x) + 4.0*f(x+dx))

	"""
	we have to remember to add the pieces not included above, 
	namely the points as x=a, a+dx, b
	"""
	s = s + (f(a) + 4.0*f(a+dx) + f(b))
	
	"finally, we need to multiply by dx/3"
	
	return s*dx/3.0
	 
"Gaus 4pt and 8pt"
def int_Gaus(f, a, b, n):
	"""
	input:
		f - a single argument real function
		a,b - the two end-points (interval of integration)
	
	output:
		r - result of integration 
	"""

	m = (b-a)/2.0
	c = (b+a)/2.0
	"Gauss (4 points using symmetry)"
	if n ==4:
	
		ti = np.array([0.3399810436, 0.8611363116])
		ci = np.array([0.6521451549, 0.3478548451])

	"Gauss (8 points using symmetry)"
	if n ==8:
		ti = np.array([0.1834346424, 0.5255324099, 0.7966664774, 0.9602898564])
		ci = np.array([0.3626837833,0.3137066458, 0.2223810344, 0.1012285362])

	r = np.sum(ci * f(m*ti + c) + ci * f(-m*ti + c))
	return r*m

	
	 


test_func = 'y'
if test_func == 'y':
	print("f = sin")
	print("a,b=0,pi")
	a,b=0,pi
	plt.figure(figsize=(8,4))
	plt.ylabel(r'$\int_0^\pi sin(x)\, dx$ ',size=25)
	x=np.arange(a,b+.001,.001)

	plt.xlabel(r'$n$ ',size=25, position=(1,1.2))
	
	tmp_Gaus4 = int_Gaus(sin, a, b, 4)
	tmp_Gaus8 = int_Gaus(sin, a, b, 8)
	
	print("tmp_Gaus4,", tmp_Gaus4)
	print("tmp_Gaus8,", tmp_Gaus8)
	
	
	for n0 in range(2,12):
		n=pow(2,n0)
		tmp_trap = int_trap(sin, a, b, n)
		if n <40:
			plt.errorbar(n,tmp_trap,markersize=8,fmt='o',color='b',mfc='white',mec='b', elinewidth=2, capsize=6, mew=1.4,zorder=100)	

		tmp_simp = int_simp(sin, a, b, n)
		print("n,trap, simp = ",n,tmp_trap,tmp_simp)

	plt.axhline(y=2,color='k',linewidth=1)
	plt.savefig('sinx_vs_n',
		bbox_inches='tight',
		transparent=True)	

	plt.figure(figsize=(5,5))
	plt.ylabel(r'$\sin(x)$ ',size=25)
	x=np.arange(a,b+.001,.001)
	plt.plot(x,sin(x), color='r')
	plt.xlabel(r'$x$ ',size=25, position=(1,1.2))
	
	plt.axhline(y=0,color='k',linewidth=1)
	plt.savefig('sin_vs_x',
		bbox_inches='tight',
		transparent=True)	


	print("")
	print("")
	print("")
	def f(x): 
		num = x*cos(10*pow(x,2))
		denum = pow(x,2) + 1
		return num / denum 

	tmp_Gaus4 = int_Gaus(f, a, b, 4)
	tmp_Gaus8 = int_Gaus(f, a, b, 8)
	
	print("tmp_Gaus4,", tmp_Gaus4)
	print("tmp_Gaus8,", tmp_Gaus8)

	plt.figure(figsize=(5,5))
	plt.ylabel(r'$x\cos(10\, x)/(x^2+1)$ ',size=25)
	x=np.arange(a,b+.001,.001)
	plt.plot(x,f(x), color='r')
	plt.xlabel(r'$x$ ',size=25, position=(1,1.2))
	
	plt.axhline(y=0,color='k',linewidth=1)
	plt.savefig('fx_vs_x',
		bbox_inches='tight',
		transparent=True)	


	print("f = x*cos(10*pow(x,2)) / (pow(x,2) + 1)")
	print("a,b=0,pi")
	a,b=0,pi
	for n0 in range(2,16):
		n=pow(2,n0)
		tmp_trap = int_trap(f, a, b, n)
		tmp_simp = int_simp(f, a, b, n)
		print("n,trap, simp = ",n,tmp_trap,tmp_simp)

end	 	
def fin_deriv(f, x, a):
	"""
	this defines the rate of change of the function f
	as a function of x
	
	the final point is at xf = x + a
	the initial point is at xi = x - a
	the difference between these two is xf - xi = 2*a
	"""
	
	"the numerator"
	num = f(x+a) - f(x-a)
	"the denominator"
	denum = 2.0 * a
	
	return num / denum



def fin_deriv(f, x, a):
	"""
	this defines the rate of change of the function f
	as a function of x
	
	the final point is at xf = x + a
	the initial point is at xi = x - a
	the difference between these two is xf - xi = 2*a
	"""
	
	"the numerator"
	num = f(x+a) - f(x-a)
	"the denominator"
	denum = 2.0 * a
	
	return num / denum


def second_deriv(f, x, a):
	
	def df(x):
		return fin_deriv(f, x, a)
	
	return fin_deriv(df, x, a)


"""
*****************************
*****************************
  Electric field
*****************************
*****************************
"""	

def E_field(E0, x,t,f):
	"I am setting c =1 "
	c=1
	lamd = c/f
	
	darg0 = 2 * pi * x / lamd
	darg1 = - 2 * pi * f * t
	
	return E0 * cos(darg0 + darg1)

def dE(E0, x,t,f, var):
	
	"the numerator"
	if var =='t':
		num = E_field(E0, x,t+a,f) - E_field(E0, x,t-a ,f)
	elif var =='x':
		num = E_field(E0, x+a,t,f) - E_field(E0, x-a,t ,f)

	"the denominator"
	denum = 2.0 * a
	
	return num / denum

def dE2(E0, x,t,f, var1, var2):
	
	"the numerator"
	if var2 =='t':
		num = dE(E0, x,t+a,f,var1) - dE(E0, x,t-a ,f,var1)
	elif var2 =='x':
		num = dE(E0, x+a,t,f,var1) - dE(E0, x-a,t ,f,var1)

	"the denominator"
	denum = 2.0 * a
	
	return num / denum


def test_wave_Eq(E0, x,t,f):
	
	"derivative with respect to x"
	
	dEtt = dE2(E0, x,t,f, 't', 't')
	dExx = dE2(E0, x,t,f, 'x', 'x')
	
	return dExx - dEtt

"""
*****************************
*****************************
  Electric plots
*****************************
*****************************
"""	


	
Eplots = 'n'
if Eplots =='y':
	
	ts = np.arange(0,2,.01)
	
	x = 0
	a = pow(10,-3)
	
	colors = ['r','orange','green','blue','magenta']
	E0s = [1,1,2,2]
	fs = [1,1.5,1,1.5] 
	
	plt.figure(figsize=(8,5))
	plt.subplot(211)
	plt.axvline(x=0,color='k',linewidth=1)
	plt.ylabel(r'$E(x,t)$ ',size=25)
	plt.subplot(212)
	plt.ylabel(r'$(\partial_x^2 - \partial_t^2 )E(x,t)$ ',size=25)
	plt.xlabel(r'$t$ ',size=25, position=(1,1.2))
	plt.axvline(x=0,color='k',linewidth=1)
	
	for j0 in range(len(E0s)):
		E0 = E0s[j0]
		f = fs[j0]
		colorf = colors[j0]
		
		y1 = E_field(E0, x,ts,f)
		y2 = test_wave_Eq(E0, x,ts,f)

		plt.subplot(211)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.axhline(y=0,color='k',linewidth=1)

		plt.plot(ts, y1, color=colorf)
		plt.ylim([-2.1,2.1])
					
		plt.subplot(212)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)

		plt.axhline(y=0,color='k',linewidth=1)
		plt.axvline(x=0,color='k',linewidth=1)

		plt.plot(ts, y2, color=colorf)
		plt.ylim([-.1,.1])
		
		plt.savefig('E_test_'+str(j0),
			bbox_inches='tight',
			transparent=True)	
	
	
"""
*****************************
*****************************
  sum of cosines
*****************************
*****************************
"""	


	
cos_sum = 'n'
if cos_sum =='y':
	
	ts = np.arange(-2,2,.01)
	
	
	for k0 in range(1,6):
	
		plt.figure(figsize=(8,5))

		colors = ['r','orange','y','green','blue','magenta']
		fs = np.arange(1, k0+1)
	
		plt.subplot(211)
		plt.axvline(x=0,color='k',linewidth=1)
		plt.axhline(y=0,color='k',linewidth=1)
		plt.ylabel(r'$ \cos(2\pi f n t ) $ ',size=25)
		
		plt.subplot(212)
		plt.ylabel(r'$\sum_n \cos(2\pi f n t )$ ',size=25)		
		plt.axvline(x=0,color='k',linewidth=1)
		plt.axhline(y=0,color='k',linewidth=1)

		plt.xlabel(r'$t$',size=25, position=(1,1.2))
	
		ys=np.zeros(len(ts))
	
		for j0 in  range(len(fs)):
		

			f = fs[j0]
			colorf = colors[j0]

			arg0 =  2 * pi * f * ts
			y1 = cos(arg0)
			ys = ys + y1
			
			plt.subplot(211)
			plt.xticks(fontsize=15)
			plt.yticks(fontsize=15)
			plt.axhline(y=0,color='k',linewidth=1)

			plt.plot(ts, y1, color=colorf)
			plt.ylim([-1.1,1.1])
					
		plt.subplot(212)
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.ylim([-2.,6])

		plt.axhline(y=0,color='k',linewidth=1)
		plt.axvline(x=0,color='k',linewidth=1)
			
		plt.plot(ts, ys, color='k')

		plt.savefig('cos_sum_'+str(len(fs)),
			bbox_inches='tight',
			transparent=True)	
	
"""
*****************************
*****************************
  products of cosines
*****************************
*****************************
"""	


	
cos_prod = 'n'
if cos_prod =='y':
	
	Lambda = 100.0
	ts = np.arange(-Lambda,Lambda +.01,.001)
	
	plt.figure(figsize=(8,5))

	for k0 in range(1,5):
	

	
		plt.subplot(4,1,k0)
		plt.axvline(x=0,color='k',linewidth=1)
		plt.axhline(y=0,color='k',linewidth=1)
		plt.ylabel(r'$ n= $ '+str(k0),size=15)
		
		f = k0

		arg0 =  2 * pi * f * ts
		y1 = cos(arg0) * cos(2 * pi * ts)
		
		print("sum(y1)",sum(y1*(ts[1]-ts[0])))
		plt.xticks(fontsize=15)
		plt.yticks(fontsize=15)
		plt.axhline(y=0,color='k',linewidth=1)

		plt.plot(ts, y1, color='k')
		plt.ylim([-1.1,1.1])
		print("k0",k0)
		plt.xlim([-1,1])
		if k0 != 4.0:
			plt.xticks(np.arange(-1,1.5,.5),['','','','',''],size=17)
		else:
			plt.xticks(np.arange(-1,1.5,.5),['-1','-.5','0','.5','1'],size=17)

	plt.savefig('cos_prod',
		bbox_inches='tight',
		transparent=True)	
	
"""
*****************************
*****************************
  Area under the curve
*****************************
*****************************
"""	


	
area_fill = 'n'
if area_fill =='y':
	
	ts = np.arange(-1,1+.01,.001)
	
	plt.figure(figsize=(8,5))

	plt.axvline(x=0,color='k',linewidth=1)
	plt.axhline(y=0,color='k',linewidth=1)
	
	f = 4

	arg0 =  2 * pi * f * ts
	y1 = cos(arg0) * cos(2 * pi * ts)
	print("sum(y1)",sum(y1*(ts[1]-ts[0])))
	
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.axhline(y=0,color='k',linewidth=1)

	plt.fill_between(ts, y1,np.zeros(len(ts)), color='b', alpha=.5)
	plt.ylim([-1.1,1.1])
	plt.xlim([-.5,.5])
	
	plt.xticks(np.arange(-1,1.5,.5),['-1','-.5','0','.5','1'],size=17)

	plt.savefig('aren_under_curve',
		bbox_inches='tight',
		transparent=True)	



"""
*****************************
*****************************
	integral of f(t) = 1 + t
*****************************
*****************************
"""	


	
int_ft = 'n'
if int_ft =='y':
	

	plt.axhline(y=3/2.,color='k',linewidth=1)
	
	plt.ylabel(r'$A$ ',size=25)
	plt.xlabel(r'$\Delta t$ ',size=25, position=(1,1.2))
	for dt in np.arange(0.0001,0.02,.0003):
		
		area = 0
		for t in np.arange(dt/2.,1,dt):
			area = area + ((1+t) * dt)
		
		print(dx,area)
		plt.errorbar(dt,area,markersize=8,fmt='o',color='b',mfc='white',mec='b', elinewidth=2, capsize=6, mew=1.4,zorder=100)	
		

	plt.savefig('int_ft',
		bbox_inches='tight',
		transparent=True)	
	
	
"""
*****************************
*****************************
  calculate pi
*****************************
*****************************
"""	


	
cal_pi = 'n'
if cal_pi =='y':
	
	ts = np.arange(-1,1+.01,.001)

	plt.axhline(y=pi,color='k',linewidth=1)
	
	plt.ylabel(r'$\pi$ ',size=25)
	plt.xlabel(r'$\Delta x$ ',size=25, position=(1,1.2))
	for dx in np.arange(0.001,0.1,.001):
		
		area = 0
		for x in np.arange(-1,1+dx,dx):
			for y in np.arange(-1,1+dx,dx):
			
				r = sqrt(pow(x,2)+pow(y,2))
				if r < 1:
					area = area + pow(dx,2)
		
		print(dx,area)
		plt.errorbar(dx,area,markersize=8,fmt='o',color='b',mfc='white',mec='b', elinewidth=2, capsize=6, mew=1.4,zorder=100)	
		

	plt.savefig('calc_pi2',
		bbox_inches='tight',
		transparent=True)	




"""
*****************************
*****************************
  Fourier series...
*****************************
*****************************
"""	


	
FT_cos = 'n'
if FT_cos =='y':
	
	Lambda = 9.5
	dt = .01
	ts = np.arange(-Lambda+(dt/2),Lambda ,dt)
	
	plt.figure(figsize=(8,5))
	yt = np.zeros(len(ts))
	
	for f in range(1,5):

		yt = yt + cos(2 * pi * f * ts)

	plt.subplot(311)
	plt.plot(ts, yt, color='k')
	plt.axhline(y=0,color='k',linewidth=1)
	plt.ylabel(r'$ {f}(t) $ ',size=15)

	fs = np.arange(0,10,.01)
	YfR = []
	YfI = []
	for f0 in fs:
		argR = dt * yt * cos(2 * pi * f0 * ts)  
		YfR.append( sum(argR) )

		argI = dt * yt * sin(2 * pi * f0 * ts)  
		YfI.append( sum(argI) )
		

	plt.subplot(312)
	plt.plot(fs, YfR, color='k')
	plt.axvline(x=0,color='k',linewidth=1)
	plt.axhline(y=0,color='k',linewidth=1)
	plt.ylabel(r'$ \rm{Re}\,\hat{f}(k) $ ',size=15)

	plt.subplot(313)
	plt.plot(fs, YfI, color='k')
	plt.axvline(x=0,color='k',linewidth=1)
	plt.axhline(y=0,color='k',linewidth=1)
	plt.ylabel(r'$ \rm{Im}\,\hat{f}(k) $ ',size=15)

	plt.ylim([-.01,.01])

	plt.savefig('FT_cos',
		bbox_inches='tight',
		transparent=True)	




"""
*****************************
*****************************
  decode the frequencies...
*****************************
*****************************
"""	


	
FT_mistery = 'y'
if FT_mistery =='y':
	
	Lambda = 5
	dt = .01
	ts = np.arange(-Lambda+(dt/2),Lambda ,dt)

	
	plt.figure(figsize=(8,5))
	yt = np.zeros(len(ts))
	
	for f in [1,3,7,8,18]:

		yt = yt + cos(2 * pi * f * ts)

	data = np.ones((2,len(yt)),dtype=float)
	data[0]= ts
	data[1]= yt
	
	filename = 'FT_mystery.txt'
	np.savetxt(filename,data.T)
	
	plt.subplot(211)
	plt.plot(ts, yt, color='k')
	plt.axhline(y=0,color='k',linewidth=1)
	plt.ylabel(r'$ {f}(t) $ ',size=15)

	fs = np.arange(0,20,.01)
	YfR = []
	YfI = []
	for f0 in fs:
		argR = dt * yt * cos(2 * pi * f0 * ts)  
		YfR.append( sum(argR) )

		argI = dt * yt * sin(2 * pi * f0 * ts)  
		print("sum(argI)",sum(argI))
		YfI.append( sum(argI) )



		

	plt.subplot(212)
	plt.plot(fs, YfR, color='k')
	plt.axvline(x=0,color='k',linewidth=1)
	plt.axhline(y=0,color='k',linewidth=1)
	plt.ylabel(r'$ \rm{Re}\,\hat{f}(k) $ ',size=15)

	"""
	plt.subplot(213)
	plt.plot(fs, YfI, color='k')
	plt.axvline(x=0,color='k',linewidth=1)
	plt.axhline(y=0,color='k',linewidth=1)
	plt.ylabel(r'$ \rm{Im}\,\hat{f}(k) $ ',size=15)
	"""
	

	plt.savefig('FT_mystery',
		bbox_inches='tight',
		transparent=True)	
