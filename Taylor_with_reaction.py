### Taylor_with_reaction.py
#Author Sergio Bordel sergio_bordel@hotmail.com

#Permission is hereby granted, free of charge, to any person obtaining a copy 
#of this software and associated documentation files (the "Software"), to deal 
#in the Software without restriction, including without limitation the rights 
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
#copies of the Software, and to permit persons to whom the Software is furnished 
#to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import math
import numpy as np
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

#Physical properties
#Viscosity of Water Reid, Robert C.; Prausnitz, John M.; Poling, Bruce E. (1987), The Properties of Gases and Liquids, McGraw-Hill Book Company
def mu(T):
	mu=1.856e-14*math.exp((4209/float(T))+0.04527*T-3.376e-5*T**2)
	return mu
#Surface tension of water Vargaftik et. al.
def sigma(T):
	sigma=238.8e-3*(((647.15-T)/647.15)**(1.256))*(1-0.625*(647.15-T)/647.15)
	return sigma


#Film thickness according to the correlation of Han and Shikazono

def delta(mu,U,sigma,rho,D):
	Re=rho*U*D/mu
	Ca=mu*U/sigma
	We=(rho*D*U**2)/sigma
	if Re<2000:
		delta_ad=0.67*(Ca**(2/3))/(1+3.13*(Ca**(2/3))+0.504*(Ca**0.672)*(Re**0.589)-0.352*(We**0.629))
	else:
		delta_ad=106*((Ca/Re)**(2/3))/(1+497*((Ca/Re)**(2/3))+7330*((Ca/Re)**0.672)-5000*((Ca/Re)**0.629))
	return delta_ad*D

#Film thickness and bubble rate in function of the superficial rate Us
def delta2(mu,Us,sigma,rho,Rc):
	g=9.8
	D=2*Rc
	con=rho*g/(8*mu)
	S=(con/Us)*(Rc)**2
	u=1
	for i in range(100):
		U=u*Us
		delta_val=delta(mu,U,sigma,rho,D)
		y=1-float(delta_val)/float(Rc)
		u=(1+float(S)*(1+4*(float(y)**4)*(float(3)/float(4)-np.log(float(y))-(1/float(y)**(2)))))/(y**2)
		#print(u)
	return delta_val,u*Us

def fi(t, alpha, beta):
    """Computes function F(t) based on given alpha and beta."""
    return (alpha - beta) * np.cos(t) + ((2/3) * beta - (1/3) * alpha) * (np.cos(t))**3 - (1/5) * beta * (np.cos(t))**5

def ut(t, alpha, beta):
    """Computes ut(t)."""
    return -alpha * np.sin(t) + beta * (np.sin(t))**3
    
    
#Calculate the derivative of f with respect to eta

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp, simps
import math

def ode_system(x, Y, k):
    """ Converts the second-order ODE into a system of first-order ODEs. """
    f, df_dx = Y
    d2f_dx2 = -2 * x * df_dx + k * f
    return np.vstack([df_dx, d2f_dx2])

def bc(Y_a, Y_b):
    """ Boundary conditions: f(0) = 1, f(inf) = 0 (approximated). """
    return np.array([Y_a[0] - 1, Y_b[0]])  # f(0) = 1, f(inf) = 0

def solve_ode(k, x_max=5, num_points=100):
    """ Solves the given differential equation numerically using solve_bvp. """
    x = np.linspace(0, x_max, num_points)  # Finite domain approximation
    Y_guess = np.zeros((2, x.shape[0]))  # Initial guess: f=0, df/dx=0
    Y_guess[0] = np.exp(-x**2)  # Better initial guess for f(x)

    sol = solve_bvp(lambda x, Y: ode_system(x, Y, k), bc, x, Y_guess)

    if sol.success:
        return sol.x, sol.y[0]
    else:
        raise RuntimeError(f"Solver failed to converge for k={k}")

def compute_integral(k, x_max=5, num_points=100):
    """ 
    Solves the ODE for a given k, integrates f(x) over [0, x_max] using Simpson's rule, 
    and returns the result multiplied by (2 + k).
    """
    if k<3000:
    	x_vals, f_vals = solve_ode(k, x_max, num_points)

    	# Compute the integral using Simpson's rule
    	integral_value = simps(f_vals, x_vals)

    	# Multiply by (2 + k)
    	result = (2 + k) * integral_value
    else:
    	result=np.sqrt(k)
    return result

#Compute values for the upper cap
#Divide the interval between tc and zero in 100 intervals
def G(t,alpha,beta):
	tc = np.arcsin(np.sqrt(alpha / beta))
	g=np.sqrt(fi(t,alpha,beta)-fi(tc,alpha,beta))
	return g
def X(t,alpha,beta,k,Rb,Us):
	tc = np.arcsin(np.sqrt(alpha / beta))
	x=(4*k*Rb/Us)*(fi(t,alpha,beta)-fi(tc,alpha,beta))/((np.sin(t)*ut(t,alpha,beta))**2)
	return x

def Ju1(alpha,beta,k,Rb,Us,D):
	tc = np.arcsin(np.sqrt(alpha / beta))
	t_values = np.linspace(tc, 0, 100)[:-1]  # 100 values from tc to 0, decreasing
	G_values=G(t_values, alpha, beta)
	X_values = X(t_values, alpha, beta, k, Rb, Us)
	X_values[0] = 0 
	compute_integral_vectorized = np.vectorize(compute_integral)
	I_results = compute_integral_vectorized(X_values)
	Integ=2*I_results*np.sin(t_values)
	area_trapz = np.trapz(Integ, G_values)

	Ju=2*np.pi*(Rb**2)*np.sqrt(D*Us/(4*Rb))*area_trapz
	residual=2*np.pi*(Rb**2)*np.sqrt(D*k)*((t_values[-1]/float(2))-np.sin(2*t_values[-1])/float(4))
	Ju=Ju+residual
	return Ju
	
def Ju2(alpha,beta,k,Rb,Us,D):
	tc = np.arcsin(np.sqrt(alpha / beta))
	t_values = np.linspace(tc, np.pi/2, 100)
	G_values=G(t_values, alpha, beta)
	X_values = X(t_values, alpha, beta, k, Rb, Us)
	X_values[0] = 0 
	compute_integral_vectorized = np.vectorize(compute_integral)
	I_results = compute_integral_vectorized(X_values)
	Integ=2*I_results*np.sin(t_values)
	area_trapz = np.trapz(Integ, G_values)

	Ju=2*np.pi*(Rb**2)*np.sqrt(D*Us/(4*Rb))*area_trapz
	return Ju

#Compute values for the cylindrical part
#Calculate the superficial velocity
def X_cyl(z,k,Rb,Us,uz,alpha,beta):
	tc = np.arcsin(np.sqrt(alpha / beta))
	x=(4*k*Rb/Us)*(z/uz+(fi(np.pi/2,alpha,beta)-fi(tc,alpha,beta))/(ut(np.pi/2,alpha,beta)**2))
	return x
def Jcyl(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max):
	tc = np.arcsin(np.sqrt(alpha / beta))	
	uz=(Ub/Us)+(9.8*rho/(4*mu_v*Us))*(Rc**2-Rb**2)+((g*rho/(2*mu_v*Us))*(Rb)**2)*np.log(Rb/Rc)
	z_values=np.linspace(0, z_max, 100)
	X_values=X_cyl(z_values,k,Rb,Us,uz,alpha,beta)
	compute_integral_vectorized = np.vectorize(compute_integral)
	I_results = compute_integral_vectorized(X_values)
	Integ=I_results/np.sqrt(z_values/uz+(fi(np.pi/2,alpha,beta)-fi(tc,alpha,beta))/(ut(np.pi/2,alpha,beta)**2))
	area_trapz = np.trapz(Integ, z_values)
	Jcyl=2*np.pi*(Rb**2)*np.sqrt(D*Us/(4*Rb))*area_trapz
	return Jcyl

def G_low(t,alpha,beta,z_max,Rb,Rc,Us,Ub):
	tc = np.arcsin(np.sqrt(alpha / beta))
	uz=(Ub/Us)+(9.8*rho/(4*mu_v*Us))*(Rc**2-Rb**2)+((g*rho/(2*mu_v*Us))*(Rb)**2)*np.log(Rb/Rc)
	f=np.sqrt(fi(t,alpha,beta)-fi(tc,alpha,beta)+(z_max/uz)*(ut(np.pi/2,alpha,beta))**2)
	return f

def X_low(t,alpha,beta,k,Rb,Us,Rc,Ub,z_max):
	tc = np.arcsin(np.sqrt(alpha / beta))
	uz=(Ub/Us)+(9.8*rho/(4*mu_v*Us))*(Rc**2-Rb**2)+((g*rho/(2*mu_v*Us))*(Rb)**2)*np.log(Rb/Rc)
	x=(4*k*Rb/Us)*(fi(t,alpha,beta)-fi(tc,alpha,beta)+(z_max/uz)*(ut(np.pi/2,alpha,beta))**2)/((np.sin(t)*ut(t,alpha,beta))**2)
	return x


def Jd2(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max):
	tc = np.arcsin(np.sqrt(alpha / beta))
	t_values = np.linspace(np.pi/2, np.pi-tc, 100)[:-1]
	G_values=G_low(t_values, alpha, beta,z_max,Rb,Rc,Us,Ub)
	X_values=X_low(t_values,alpha,beta,k,Rb,Us,Rc,Ub,z_max)
	compute_integral_vectorized = np.vectorize(compute_integral)
	I_results = compute_integral_vectorized(X_values)
	Integ=2*I_results*np.sin(t_values)
	area_trapz = np.trapz(Integ, G_values)
	Ju=2*np.pi*(Rb**2)*np.sqrt(D*Us/(4*Rb))*area_trapz
	residual=2*np.pi*(Rb**2)*np.sqrt(D*k)*(((np.pi-t_values[-1])/float(2))-(np.sin(2*np.pi)-np.sin(2*t_values[-1]))/float(4))
	Jd2=Ju+residual
	return Jd2

def G_low1(t,alpha,beta):
	g=np.sqrt(fi(t,alpha,beta)-fi(np.pi,alpha,beta))
	return g
def X_low1(t,alpha,beta,k,Rb,Us):
	x=(4*k*Rb/Us)*(fi(t,alpha,beta)-fi(np.pi,alpha,beta))/((np.sin(t)*ut(t,alpha,beta))**2)
	return x

def Jd1(alpha,beta,k,Rb,Us,D):
	tc = np.arcsin(np.sqrt(alpha / beta))
	t_values = np.linspace(np.pi, np.pi-tc, 100)[:-1]
	G_values=G_low1(t_values, alpha,beta)
	X_values=X_low1(t_values,alpha,beta,k,Rb,Us)
	compute_integral_vectorized = np.vectorize(compute_integral)
	I_results = compute_integral_vectorized(X_values)
	Integ=2*I_results*np.sin(t_values)
	area_trapz = np.trapz(Integ, G_values)
	Jd=2*np.pi*(Rb**2)*np.sqrt(D*Us/(4*Rb))*area_trapz
	residual=-(2*np.pi*(Rb**2)*np.sqrt(D*k)*(((np.pi-tc-t_values[-1])/float(2))-(np.sin(2*(np.pi-tc))-np.sin(2*t_values[-1]))/float(4)))
	Jd1=Jd+residual
	return Jd1

def J_tot(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max):
	Ju1_v=Ju1(alpha,beta,k,Rb,Us,D)
	Ju2_v=Ju2(alpha,beta,k,Rb,Us,D)
	Jcyl_v=Jcyl(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max)
	Jd2_v=Jd2(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max)
	Jd1_v=Jd1(alpha,beta,k,Rb,Us,D)
	J_t=Ju1_v+Ju2_v+Jcyl_v+Jd2_v+Jd1_v
	return J_t

def kla(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max):
	J_t=J_tot(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max)
	Vol=(float(4)/float(3)+z_max)*np.pi*(Rb**3)
	kl=J_t/Vol
	return kl
def kla_op(k,Rc,Us,D,mu_v,sigma_v,rho,z_max):
	delta_val,Ub=delta2(mu_v,Us,sigma_v,rho,Rc)
	u_ratio = Ub / Us
	Rb=Rc-delta_val

	alpha = (3.0 / 4.0) * (2 - u_ratio) - (2.0 / 3.0) * (Rb / Rc) ** 2
	beta = (35.0 / 12.0) * (Rb / Rc) ** 2
	tc = np.arcsin(np.sqrt(alpha / beta))

	kla_val=kla(k,Rb,Rc,Us,Ub,alpha,beta,D,z_max)
	return kla_val




