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


"""
*****************************
*****************************
  plots as a function of a
*****************************
*****************************
"""	

def master_plot_vs_x():


	plt.figure(figsize=(9.0,5))
	plt.subplot(211)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	
	plt.ylabel(ylabel1,size=25)
	plt.plot(x, ydata1, color='r')
	plt.axvline(x=0,color='k',linewidth=1)
					
	plt.subplot(212)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)

	plt.axhline(y=0,color='k',linewidth=1)
	plt.axvline(x=0,color='k',linewidth=1)

	plt.ylabel(ylabel2,size=25)
	plt.plot(x, ydata2, color='b')
	plt.errorbar(x2, ydata3,  markersize=8,fmt='o',color='g',mfc='white',mec='g', elinewidth=2, capsize=4, mew=1.4)			
			
	plt.xlabel(xlabel0,size=25, position=(1,1.2))
	plt.savefig(filename,
		bbox_inches='tight',
		transparent=True)

plot_vs_x ='n'
if plot_vs_x=='y':
	a = pow(10,-3)

	x=np.arange(0,5,.01)
	x2=np.arange(0,5,.5)

	"""
	******************************
	******************************
			exponential
	******************************
	******************************
	"""
	dexp = fin_deriv(exp, x, a)

	ydata1 = exp(x)
	ydata2 = dexp
	ydata3 = exp(x2)
	ylabel1 = r'$\exp(x)$ '
	ylabel2 = r'$\Delta_a \exp(x) /2a$ '
	xlabel0 = r'$x$'
	filename = './exp_der_vs_x.pdf'
	
	master_plot_vs_x()
	
	"""
	******************************
	******************************
			sin
	******************************
	******************************
	"""
	dsin = fin_deriv(sin, x, a)

	ydata1 = sin(x)
	ydata2 = dsin
	ydata3 = cos(x2)
	ylabel1 = r'$\sin(x)$ '
	ylabel2 = r'$\Delta_a \sin(x) /2a$ '
	xlabel0 = r'$x$'
	filename = './sin_der_vs_x.pdf'
	
	master_plot_vs_x()


	"""
	******************************
	******************************
			cos
	******************************
	******************************
	"""
	dcos = fin_deriv(cos, x, a)

	ydata1 = cos(x)
	ydata2 = dcos
	ydata3 = -sin(x2)
	ylabel1 = r'$\cos(x)$ '
	ylabel2 = r'$\Delta_a \cos(x) /2a$ '
	xlabel0 = r'$x$'
	filename = './cosn_der_vs_x.pdf'
	
	master_plot_vs_x()



"""
*****************************
*****************************
  plots as a function of x
*****************************
*****************************
"""
	

def error_fun(Xvec,Yvec):
	
	num = Yvec[0:-1]-Yvec[1:]
	dnum = Xvec[0:-1]-Xvec[1:]
	Ytmp = 100 * abs(num/dnum)
		
	Xtmp = Xvec[0:-1]
		
	return Xtmp, Ytmp

def master_plot_vs_a():


	plt.figure(figsize=(9.0,5))
	plt.subplot(211)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	
	plt.ylabel(ylabel1,size=25)

	plt.axhline(y=ydata2,color='k',linewidth=1)
	plt.plot(a, ydata1, color='r')

	plt.axvline(x=0,color='k',linewidth=1)
					
	plt.subplot(212)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)

	plt.axhline(y=0,color='k',linewidth=1)
	plt.axvline(x=0,color='k',linewidth=1)

	plt.ylabel(ylabel2,size=25)
	plt.plot(xtmp, ydata3, color='g')
			
	plt.xlabel(xlabel0,size=25, position=(1,1.2))
	plt.savefig(filename,
		bbox_inches='tight',
		transparent=True)
			
plot_vs_a ='n'
if plot_vs_a=='y':
	
	x = 1
	a=np.arange(pow(10,-5),.1,pow(10,-3))
	
	print("a",a)
	
		
	"""
	******************************
	******************************
			exponential
	******************************
	******************************
	"""
	dexp = fin_deriv(exp, x, a)
	xtmp, Ytmp = error_fun(a,dexp)
	ydata1 = dexp
	ydata2 = exp(x)
	ydata3 = Ytmp
	ylabel1 = r'$\Delta_a \exp(x) /2a$ '
	ylabel2 = r'$\sigma_a$ '
	xlabel0 = r'$a$'
	filename = './exp_der_vs_a.pdf'
	
	master_plot_vs_a()
		
	"""
	******************************
	******************************
			sin
	******************************
	******************************
	"""
	dsin = fin_deriv(sin, x, a)
	xtmp, Ytmp = error_fun(a,dsin)
	ydata1 = dsin
	ydata2 = cos(x)
	ydata3 = Ytmp
	ylabel1 = r'$\Delta_a \sin(x) /2a$ '
	ylabel2 = r'$\sigma_a$ '
	xlabel0 = r'$a$'
	filename = './sin_der_vs_a.pdf'

	master_plot_vs_a()

	"""
	******************************
	******************************
			cos
	******************************
	******************************
	"""
	dcos = fin_deriv(cos, x, a)
	xtmp, Ytmp = error_fun(a,dcos)
	ydata1 = dcos
	ydata2 = -sin(x)
	ydata3 = Ytmp
	ylabel1 = r'$\Delta_a \cos(x) /2a$ '
	ylabel2 = r'$\sigma_a$ '
	xlabel0 = r'$a$'
	filename = './cos_der_vs_a.pdf'
	
	master_plot_vs_a()


	"""
	******************************
	******************************
			poly
	******************************
	******************************
	"""
	
	print("poly0(x)",poly0(x))
	dx = fin_deriv(poly0, x, a)
	xtmp, Ytmp = error_fun(a,dx)
	ydata1 = dx
	ydata2 = np.ones(x)
	ydata3 = Ytmp
	ylabel1 = r'$\Delta_a (\frac{x^3}{6}+\frac{x^4}{8}) /2a$ '
	ylabel2 = r'$\sigma_a$ '
	xlabel0 = r'$a$'
	filename = './x_der_vs_a.pdf'
	
	master_plot_vs_a()
				
	plt.savefig(filename,
		bbox_inches='tight',
		transparent=True)
	"""
	******************************
	******************************
			log
	******************************
	******************************
	"""
	
	dlog = fin_deriv(log, x, a)
	xtmp, Ytmp = error_fun(a,dlog)
	ydata1 = dlog
	ydata2 = 1.0/np.ones(x)
	ydata3 = Ytmp
	ylabel1 = r'$\Delta_a \log(x) /2a$ '
	ylabel2 = r'$\sigma_a$ '
	xlabel0 = r'$a$'
	filename = './log_der_vs_a.pdf'
	
	master_plot_vs_a()
			


"""
*****************************************
*****************************************
  solving linear differential equations
*****************************************
*****************************************
"""

def lin_diff_eq(g, f0, xs, a):

	"""
	this solves the diff. eq. df/dx = g(x) by discretizing it
	(f(x+a) - f(x-a))/2a = g(x)
	f(x+a) = f(x-a) + 2*a*g(x)
	
	or equivalently 
	
	f(x+2 a) = f(x) + 2*a*g(x+a)
	
	we put the list of points into fs
	
	we initiate this with 
	the initial value of f, which is f0 = f(xs[0])
	
	"""
	
	fs=[f0]
	
	"note in this loop, we skip over the first element"
	for i0 in range(1,len(xs)):

		"we grab the previous term in the list"	
		f0 = fs[-1]
		
		"""
		we need the derivative at x+a, 
		since xs[i0] =  x+2*a 
		we can use the fact that x+a = xs[i0]-a
		"""	
		df = g(xs[i0]-a)*2*a
		
		"we add these together"	
		fs.append(f0+ df)
	
	return fs


def master_diff_eq():


	plt.figure(figsize=(9.0,5))
	plt.subplot(211)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	
	plt.ylabel(ylabel1,size=25)

	plt.plot(xs, ydata1, color='r')
	plt.errorbar(x2, ydata2,  markersize=8,fmt='o',color='g',mfc='white',mec='g', elinewidth=2, capsize=4, mew=1.4)			

	plt.axvline(x=0,color='k',linewidth=1)
					
	plt.subplot(212)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)

	plt.axhline(y=0,color='k',linewidth=1)
	plt.axvline(x=0,color='k',linewidth=1)

	plt.ylabel(ylabel2,size=20)
	plt.plot(xs, ydata3, color='g')
			
	plt.xlabel(xlabel0,size=20, position=(1,1.2))
	plt.savefig(filename,
		bbox_inches='tight',
		transparent=True)


plot_lin_diff_eq ='n'
if plot_lin_diff_eq=='y':
	
	a = pow(10,-3)	
	xs=np.arange(0,5, 2*a)			
	x2=np.arange(0,5, .5)			

	print("a",a)
	
		
	"""
	******************************
	******************************
			g(x) = exp(x)
			f(x) = exp(x)
	******************************
	******************************
	"""
	
	fs = lin_diff_eq(exp, exp(0), xs, a)
	ydata1 = fs
	ydata2 = exp(x2)
	ydata3 = exp(xs)
	ylabel1 = r'$f(x) $ '
	ylabel2 = r'$g(x) = \exp(x)$ '
	xlabel0 = r'$x$'
	filename = './exp_der_vs_a.pdf'
	
	master_diff_eq()
		

	"""
	******************************
	******************************
			g(x) = sin(x)
			f(x) = -cos(x)
	******************************
	******************************
	"""
	
	fs = lin_diff_eq(sin, -cos(0), xs, a)
	ydata1 = fs
	ydata2 = -cos(x2)
	ydata3 = sin(xs)
	ylabel1 = r'$f(x) $ '
	ylabel2 = r'$g(x) = \sin(x)$ '
	xlabel0 = r'$x$'
	filename = './sin_der_vs_a.pdf'
	
	master_diff_eq()


	"""
	******************************
	******************************
			g(x) = cos(x)
			f(x) = sin(x)
	******************************
	******************************
	"""
	
	fs = lin_diff_eq(cos, sin(0), xs, a)
	ydata1 = fs
	ydata2 = sin(x2)
	ydata3 = cos(xs)
	ylabel1 = r'$f(x) $ '
	ylabel2 = r'$g(x)= \cos(x)$ '
	xlabel0 = r'$x$'
	filename = './cos_der_vs_a.pdf'
	
	master_diff_eq()




"""
*****************************************
*****************************************
  solving second differential equations
*****************************************
*****************************************
"""

def second_diff_eq(h, f0, g0, ts, a):

	"""
	this solves the diff. eq. d^2f/dx^2 = h(x) by discretizing 
	re-writting it as a coupled linear equations
	
	dg/dt = h(t)
	df/dt = g(t)
	
	g(t + 2*a) = g(t) + 2 * a * h(t + a)
	f(t + 2*a) = f(t) + 2 * a * g(t + a)
	
	we put the list of points into fs, gs
	
	we initiate this with 
	the initial value of f, which is f0 = f(ts[0])
	the initial value of g, which is g0 = g(ts[0])
	
	note, we need to calculate g(t + a), but we have g(t + 2*a) and g(t)
	we will estimate this with the average of these two, which we will call
	
	gbar = ( g(t + 2*a) + g(t) ) / 2
	"""
	
	fs=[f0]
	gs=[g0]
	
	"note in this loop, we skip over the first element"
	for i0 in range(1,len(ts)):
		"we start with g(t): we grab the previous term in the list"	
		g0 = gs[-1]
		"the shift in g"
		dg = h(ts[i0] + a) * 2*a
		
		gs.append(  g0 + dg )

		"next we calculate f(t): we grab the previous term in the list"	
		f0 = fs[-1]
		"we estimate g(t+a) with the average of the last two point in gs"
		gbar = (gs[-1] + gs[-2]) /2.0
		df = gbar * 2 * a
		
		"we add these together"	
		fs.append(f0 + df)
	
	return fs, gs


def master_sec_diff_eq():


	plt.figure(figsize=(9.0,8))
	
	"x(t) plots"
	plt.subplot(311)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	
	plt.ylabel(ylabel1,size=25)

	plt.plot(ts, ydata1, color='r')
	plt.errorbar(t2, ydata2,  markersize=8,fmt='o',color='g',mfc='white',mec='g', elinewidth=2, capsize=4, mew=1.4)			

	plt.axvline(x=0,color='k',linewidth=1)
					
	"v(t) plots"
	plt.subplot(312)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)

	plt.axvline(x=0,color='k',linewidth=1)

	plt.ylabel(ylabel2,size=20)
	plt.plot(ts, ydata3, color='r')
	plt.errorbar(t2, ydata4,  markersize=8,fmt='o',color='g',mfc='white',mec='g', elinewidth=2, capsize=4, mew=1.4)			
			
	"a plots"
	plt.subplot(313)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)

	plt.axhline(y=0,color='k',linewidth=1)
	plt.axvline(x=0,color='k',linewidth=1)

	plt.ylabel(ylabel3,size=20)
	plt.plot(ts, ydata5, color='b')
			
	plt.xlabel(xlabel0,size=20, position=(1,1.2))
	plt.savefig(filename,
		bbox_inches='tight',
		transparent=True)


plot_sec_diff_eq ='y'
if plot_sec_diff_eq=='y':
	
	a = pow(10,-3)	
	ts=np.arange(0,5, 2*a)			
	t2=np.arange(0,5, .5)			

	print("a",a)
	
		
	def x_func(t, x0, v0, a0):
		x1 = v0 * t
		x2 = a0 * pow(t,2)/2.0
		return x0 + x1 + x2

	def v_func(t, v0, a0):
		return v0 + a0 * t


	"""
	******************************
	******************************
			a(t) = 0
			x(t) = v0*t
	******************************
	******************************
	"""

	x0, v0 = 0,1
	def a_func(t): return 0
	
	xs, vs = second_diff_eq(a_func,x0, v0, ts, a)
	ydata1 = xs
	ydata2 = x_func(t2, x0, v0, a_func(ts))
	ydata3 = vs
	ydata4 = v_func(t2, v0, a_func(ts))
	ydata5 = a_func(ts)*np.ones(len(ts))
	ylabel1 = r'$x(t) $ '
	ylabel2 = r'$v(t)$ '
	ylabel3 = r'$a(t) = 0$ '
	xlabel0 = r'$t$'
	filename = './x_vs_t_a0.pdf'
	
	master_sec_diff_eq()


	"""
	******************************
	******************************
			a(t) = -9.8
			x(t) = v0*t + a t^2/2
	******************************
	******************************
	"""

	x0, v0 = 0,0
	def a_func(t): return -9.8
	xs, vs = second_diff_eq(a_func, x0, v0, ts, a)
	ydata1 = xs
	ydata2 = x_func(t2, x0, v0, a_func(ts))
	ydata3 = vs
	ydata4 = v_func(t2, v0, a_func(ts))
	ydata5 = a_func(ts)*np.ones(len(ts))
	ylabel1 = r'$x(t) $ '
	ylabel2 = r'$v(t)$ '
	ylabel3 = r'$a(t) = -9.8$ '
	xlabel0 = r'$t$'
	filename = './x_vs_t_a9.8.pdf'
	
	master_sec_diff_eq()
	
	"""
	******************************
	******************************
			a(t) = t
			x(t) = v0*t + a t^2/2
	******************************
	******************************
	"""

	x0, v0 = 0,0
	def a_func(t): return t
	xs, vs = second_diff_eq(a_func, x0, v0, ts, a)
	ydata1 = xs
	ydata2 = x_func(t2, x0, v0, a_func(t2))
	ydata3 = vs
	ydata4 = v_func(t2, v0, a_func(t2))
	ydata5 = a_func(ts)*np.ones(len(ts))
	ylabel1 = r'$x(t) $ '
	ylabel2 = r'$v(t)$ '
	ylabel3 = r'$a(t) = t$ '
	xlabel0 = r'$t$'
	filename = './x_vs_t_at.pdf'
	
	master_sec_diff_eq()
		
