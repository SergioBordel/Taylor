### pyTaylor.py
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



#constants for water at 25ºC
rho=997.05
mu=0.00089
g=9.8
D=1.88e-9
sigma=0.07199

#Parameters

Rc=0.001
Us=0.1
Ls=0.11
Lb=0.11


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
def delta2(mu,Us,sigma,rho,Rc,g):
	#g=9.8
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



#Calculates the film thickness around the slug (relative to Rc)

def deltas(u):

	
	d=1-(2-u)**0.5
	return d

def c1(n,de,rho,g,mu):
	group=rho*g/mu
	c=-group*(1+math.cos(n*math.pi))*(2*de/(n*math.pi))**2
	return c

def cnb(Us,Rc,rho,g,mu,sigma):
	#Ca=Us*mu/sigma
	group=rho*g/mu
	
	#u=1/(1-1.29*((3*Ca)**(0.6666)))
	#Ub=Us*u
	#con=rho*g/(8*mu)	
	#S=(con/Us)*(Rc)**2
	#de=Rc*deltab(S,u)
	de,Ub=delta2(mu,Us,sigma,rho,Rc,g)
	c0=Ub+(group/3)*(de**2)

	

	cb=[c0]

	for i in range(1,201):
		ci=c1(i,de,rho,g,mu)
		cb=cb+[ci]
	return cb


def c2(n,de,Us,Rc):
	c=-2*Us*((2/(Rc*de))*((2*de/(n*math.pi))**2)*(math.cos(n*math.pi)-1)-(4/(Rc**2))*((2*de/(n*math.pi))**2)*math.cos(n*math.pi))
	return c

def cns(Us,Rc,mu,sigma):
	
	d_val,Ub=delta2(mu,Us,sigma,rho,Rc,g)
	u=Ub/Us	
	de=Rc*deltas(u)

	c0=Ub-4*Us*(de/Rc)+(8/3)*Us*(de/Rc)**2

	

	cs=[c0]

	for i in range(1,201):
		ci=c2(i,de,Us,Rc)
		cs=cs+[ci]
	return cs

#Define coefficients for the initial condition
def a(n,C):
	a=C*(4/(n*math.pi))*math.sin(n*math.pi/2)
	return a

def an(C):
	ab=[0]
	for i in range(1,201):
		ai=a(i,C)
		ab=ab+[ai]
	return ab

#Define exponential coefficients
def alpha(n,de,c0,D):
	alpha=(D/c0)*(n*math.pi/(2*de))**2
	return alpha

def alphan(de,c0,D):
	alphab=[0]
	for i in range(1,201):
		alphai=alpha(i,de,c0,D)
		alphab=alphab+[alphai]
	return alphab


def getKn(a,c,alphab):
	K=[[0],[0,a[1]]]

	for j in range(1,200):

		Kn=[0]

		for i in range(1,len(K[-1])):
			Ai=0
			for m in range(1,len(K[-1])-i+1):
			
				sumando=K[len(K[-1])-m][i]*c[m]/c[0]
			
				Ai=Ai+sumando
		
		
			Kni=Ai/(alphab[len(K[-1])]-alphab[i])
			Kn=Kn+[Kni]

		suma=0
		for ka in Kn:
			suma=suma+ka
		Knn=a[len(K[-1])]-suma
		Kn=Kn+[Knn]
		K=K+[Kn]
	return K

def Saturation(K,z,alphab):
	suma=0
	af=[]
	for i in range(1,len(K)):
		ai=0
		for j in range(1,len(K[i])):
			ai=ai+K[i][j]*math.exp(-alphab[j]*z)
		af=af+[ai]	
		suma=suma+ai*(2/(i*math.pi))*math.sin(i*math.pi/2)
	
	return suma,af

#Change in profile
def Newprofileb(ab,D,rho,g,mu,sigma,Us,Rc,Lb):
	
	de,Ub=delta2(mu,Us,sigma,rho,Rc,g)
	cb=cnb(Us,Rc,rho,g,mu,sigma)
	c0=cb[0]
	alphab=alphan(de,c0,D)
	K=getKn(ab,cb,alphab)
	suma,afi=Saturation(K,Lb,alphab)
	return suma,afi

def Newprofiles(a,D,mu,sigma,Us,Rc,Ls):
	
	d_val,Ub=delta2(mu,Us,sigma,rho,Rc,g)
	u=Ub/Us
	de=Rc*deltas(u)
	cs=cns(Us,Rc,mu,sigma)
	c0=cs[0]
	alphas=alphan(de,c0,D)
	af=[]
	for ai in a:
		af=af+[ai]
	K=getKn(af,cs,alphas)
	suma,afi=Saturation(K,Ls,alphas)
	return suma,afi


def Evolvedprofiles(D,rho,g,mu,sigma,Us,Rc,Lb,Ls):

	ab0=an(1)
	print(len(ab0))
	for i in range(10):
		suma1,afis=Newprofileb(ab0,D,rho,g,mu,sigma,Us,Rc,Lb)
		af0=np.array(an(1))-np.array([0]+afis)
		suma2,ab=Newprofiles(af0,D,mu,sigma,Us,Rc,Ls)
		ab0=np.array(an(1))-np.array([0]+ab)
	Factor=1-suma1-suma2
	
	return Factor,afis,ab

#Fb,Fs,afis,ab=Evolvedprofiles(D,rho,g,mu,sigma,Us,Rc,Lb,Ls)

#Compute values for the upper cap
def Kf(Ub,Us,Rb,Rc,D):
	A=(float(3)/float(4))*(2*Us-Ub)-(float(2)/float(3))*((Rf/Rc)**2)*Us
	B=(float(35)/float(12))*((Rf/Rc)**2)*Us
	x=A/B
	y=(A-B)*(1-(1-x)**0.5)+((float(2)/float(3))*B+A/float(3))*(1-(1-x)**(float(3)/float(2)))-(B/float(5))*(1-(1-x)**(float(5)/float(2)))
	K=float(4)*(D*math.pi*(Rf**3)*y)**0.5
	return K

def Kla(Us,Rc,D,Ls,Lb,rho,g,mu,sigma):
	
	d_val,Ub=delta2(mu,Us,sigma,rho,Rc,g)
	print(d_val)
	Rb=Rc-d_val
	Vs=math.pi*(Ls*(Rc**2)-float(4)*(Rc**3)/float(3))
	
	u=Ub/Us
	Kcaps_v=2*Kf(Ub,Us,Rb,Rc,D)
	fib,afis,ab=Evolvedprofiles(D,rho,g,mu,sigma,Us,Rc,Lb,Ls)
	
	
	F_v=(Ub-Us)*math.pi*((Rc)**2)*fib+Kcaps_v
	
	return F_v/Vs





