import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
# Create steps in time. I will have them cover a period of 1 year.
steps = np.arange(0, 3650, 1)

# A time step will be one tenth of one day.
dt = .1
time = dt * steps



# Define the lists
#  - s = susceptible
#  - i = infected
#  - r = recovered
#  - d = deceased
tlen = len(time)
s = np.empty((tlen))
i = np.empty((tlen))
r = np.empty((tlen))
d = np.empty((tlen))

# Initial conditions
s[0]=998
i[0]=1
r[0]=0
d[0]=0




"this is the change in functions describing the SIRD model, up to the dt scale"
def SIRD(s,i,r,d, beta, gamma, mu):
	"population size"
	N = s+i+r+d
	
	"the shift in the subsceptibles"
	s_dot = -s*i*beta / N
	
	"the shift in the subsceptibles"
	i_dot = (s*i*beta/N) - gamma*i - mu*i
	
	"the shift in the subsceptibles"
	r_dot = gamma*i
	
	"the shift in the subsceptibles"
	d_dot = mu*i
	
	return s_dot, i_dot, r_dot, d_dot


"""
this code returns the solutions for the different equation. 

We added an output option command, this is to just return either the 

number of infected people or the deceased people
"""
def diff_eq_sird(model,s,i,r,d,beta,gamma,mu,steps,dt, output):
    for t in steps:
        if t < len(steps)-1:
        	
        	"note: here we call the model function"
        	s_dot, i_dot, r_dot, d_dot = model(s[t], i[t], r[t], d[t], beta,gamma,mu)
        	s[t+1] = s[t]+dt*s_dot
        	i[t+1] = i[t]+dt*i_dot
        	r[t+1] = r[t]+dt*r_dot
        	d[t+1] = d[t]+dt*d_dot
    
    "only return number of infections"
    if output =='i':
    	return i

    "only return number of deceased"
    if output =='d':
    	return d



# Initial conditions
s0, i0, r0, d0 = 998, 1, 0, 0
s[0]=s0
i[0]=i0
r[0]=r0
d[0]=d0

"""
input parameters, 
beta, gamma, mu = rates of infection, recovery, and mortality
"""
beta,gamma,mu = .09, .01,.005

"""
solution to the differential equations for 
infected and deceased people respectively
"""
i_sol = diff_eq_sird(SIRD,s,i,r,d, beta,gamma,mu,steps,dt,'i')
d_sol = diff_eq_sird(SIRD,s,i,r,d, beta,gamma,mu,steps,dt,'d')

"just a fancy plotting routing"
fig = plt.figure(figsize=(10, 6))
horiz = np.array([250 for i in range(len(time))])
plt.plot(time, horiz, 'm--')
inf_plot, = plt.plot(time,i_sol,markersize=.8,c='r')


inf_plotd, = plt.plot(time,d_sol,markersize=.8,c='k')
plt.xlabel('Time (days)')
plt.ylabel('People in 100,000s')
plt.title('Flattening the Curve', fontsize=21)
plt.legend(['Hospital Capacity','Infected','Deceased'])
plt.xlim([0, 365])
plt.ylim([0,1000])
plt.tight_layout(pad=4.7, w_pad=6.5, h_pad=2.0)

# here we create the slider
slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
a_slider = Slider(slider_ax,
                  'Infection Rate',
                  .05,
                  .2,
                  valinit=.09
                 )


def update(abeta):
    # Initial conditions
	s[0]=s0
	i[0]=i0
	r[0]=r0
	d[0]=d0
	inf_plot.set_ydata(diff_eq_sird(SIRD,s,i,r,d, abeta,gamma,mu,steps,dt,'i'))
	inf_plotd.set_ydata(diff_eq_sird(SIRD,s,i,r,d, abeta,gamma,mu,steps,dt,'d'))
	fig.canvas.draw_idle()
     

a_slider.on_changed(update)

plt.show()
