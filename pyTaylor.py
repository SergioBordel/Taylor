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


def deltab(S,u):

	def fun(y):
		res=float(u)*(float(y)**2)-1-float(S)*(1+4*(float(y)**4)*(float(3)/float(4)-np.log(float(y))-(1/float(y)**(2))))
		return res

	refer=fun(1)

	for i in range(1000000):
		y=1-float(i)/float(1000000)

		if fun(y)*refer<0:
			
			break
	d=1-y
	return d

#Calculates the film thickness around the slug (relative to Rc)

def deltas(u):

	
	d=1-(2-u)**0.5
	return d

def c1(n,de,rho,g,mu):
	group=rho*g/mu
	c=-group*(1+math.cos(n*math.pi))*(2*de/(n*math.pi))**2
	return c

def cnb(Us,Rc,rho,g,mu,sigma):
	Ca=Us*mu/sigma
	group=rho*g/mu
	
	u=1/(1-1.29*((3*Ca)**(0.6666)))
	Ub=Us*u
	con=rho*g/(8*mu)	
	S=(con/Us)*(Rc)**2
	de=Rc*deltab(S,u)
	c0=Ub+(group/3)*(de**2)

	

	cb=[c0]

	for i in range(1,101):
		ci=c1(i,de,rho,g,mu)
		cb=cb+[ci]
	return cb


def c2(n,de,Us,Rc):
	c=-2*Us*((2/(Rc*de))*((2*de/(n*math.pi))**2)*(math.cos(n*math.pi)-1)-(4/(Rc**2))*((2*de/(n*math.pi))**2)*math.cos(n*math.pi))
	return c

def cns(Us,Rc,mu,sigma):
	Ca=Us*mu/sigma
	
	
	u=1/(1-1.29*((3*Ca)**(0.6666)))
	Ub=Us*u
	con=rho*g/(8*mu)	
	de=Rc*deltas(u)

	c0=Ub-4*Us*(de/Rc)+(8/3)*Us*(de/Rc)**2

	

	cs=[c0]

	for i in range(1,101):
		ci=c2(i,de,Us,Rc)
		cs=cs+[ci]
	return cs

#Define coefficients for the initial condition
def a(n,C):
	a=C*(4/(n*math.pi))*math.sin(n*math.pi/2)
	return a

def an(C):
	ab=[0]
	for i in range(1,101):
		ai=a(i,C)
		ab=ab+[ai]
	return ab

#Define exponential coefficients
def alpha(n,de,c0,D):
	alpha=(D/c0)*(n*math.pi/(2*de))**2
	return alpha

def alphan(de,c0,D):
	alphab=[0]
	for i in range(1,101):
		alphai=alpha(i,de,c0,D)
		alphab=alphab+[alphai]
	return alphab


def getKn(a,c,alphab):
	K=[[0],[0,a[1]]]

	for j in range(1,100):

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
	afi=[]
	for a in af:
		afi=afi+[a/suma]
	afi=[0]+afi
	return suma,afi

#Change in profile
def Newprofileb(ab,D,rho,g,mu,sigma,Us,Rc,Lb):
	con=rho*g/(8*mu)	
	S=(con/Us)*(Rc)**2
	Ca=Us*mu/sigma
	u=1/(1-1.29*((3*Ca)**(0.6666)))
	Ub=Us*u
	de=Rc*deltab(S,u)
	cb=cnb(Us,Rc,rho,g,mu,sigma)
	c0=cb[0]
	alphab=alphan(de,c0,D)
	K=getKn(ab,cb,alphab)
	suma,afi=Saturation(K,Lb,alphab)
	return suma,afi

def Newprofiles(a,D,mu,sigma,Us,Rc,Ls):
	Ca=Us*mu/sigma
	u=1/(1-1.29*((3*Ca)**(0.6666)))
	Ub=Us*u
	de=Rc*deltas(u)
	cs=cns(Us,Rc,mu,sigma)
	c0=cs[0]
	alphas=alphan(de,c0,D)
	af=[]
	for ai in a:
		af=af+[-ai]
	K=getKn(af,cs,alphas)
	suma,afi=Saturation(K,Ls,alphas)
	return suma,afi

#Saturation functions
def Fb(D,rho,g,mu,sigma,Us,Rc,z):
	con=rho*g/(8*mu)	
	S=(con/Us)*(Rc)**2
	Ca=Us*mu/sigma
	u=1/(1-1.29*((3*Ca)**(0.6666)))
	Ub=Us*u
	de=Rc*deltab(S,u)
	cb=cnb(Us,Rc,rho,g,mu,sigma)
	c0=cb[0]
	alphab=alphan(de,c0,D)
	ab=an(1)
	K=getKn(ab,cb,alphab)
	F,afi=Saturation(K,z,alphab,1)
	return F


def Fs(D,mu,sigma,Us,Rc,z):
	Ca=Us*mu/sigma
	u=1/(1-1.29*((3*Ca)**(0.6666)))
	Ub=Us*u
	de=Rc*deltas(u)
	cs=cns(Us,Rc,mu,sigma)
	c0=cs[0]
	alphas=alphan(de,c0,D)
	af=an(-1)
	K=getKn(af,cs,alphas)
	F,afi=Saturation(K,z,alphas,-1)
	F=-F
	return F
def Evolvedprofiles(D,rho,g,mu,sigma,Us,Rc,Lb,Ls):

	ab=an(1)
	for i in range(10):
		suma1,afis=Newprofileb(ab,D,rho,g,mu,sigma,Us,Rc,Lb)
		suma2,ab=Newprofiles(afis,D,mu,sigma,Us,Rc,Ls)
	Fb=1-suma1
	Fs=1+suma2
	return Fb,Fs,afis,ab

Fb,Fs,afis,ab=Evolvedprofiles(D,rho,g,mu,sigma,Us,Rc,Lb,Ls)

def Kf(Ub,Us,Rf,Rc,D):
	A=(float(3)/float(4))*(2*Us-Ub)-(float(2)/float(3))*((Rf/Rc)**2)*Us
	B=(float(35)/float(12))*((Rf/Rc)**2)*Us
	x=A/B
	y=(A-B)*(1-(1-x)**0.5)+((float(2)/float(3))*B+A/float(3))*(1-(1-x)**(float(3)/float(2)))-(B/float(5))*(1-(1-x)**(float(5)/float(2)))
	K=float(4)*(D*math.pi*(Rf**3)*y)**0.5
	return K

def Kla(Us,Rc,D,Ls,Lb,rho,g,mu,sigma):
	#Calculate capilarity number
	Ca=Us*mu/sigma
	#Calculate the ratio between velocities
	u=1/(1-1.29*((3*Ca)**(0.6666)))
	#Get adimensional number S
	con=rho*g/(8*mu)	
	S=(con/Us)*(Rc)**2
	#Get delta
	d=deltab(S,u)
	
	Rb=Rc*(1-d)
	Vs=math.pi*(Ls*(Rc**2)-float(4)*(Rc**3)/float(3))
	Ub=u*Us
	Kcaps=Kf(Ub,Us,Rb,Rc,D)
	fib,fis,afis,ab=Evolvedprofiles(D,rho,g,mu,sigma,Us,Rc,Lb,Ls)
	
	
	F=Us*math.pi*((Rc)**2)*(1-2*u*(1-d)**2+u)/((1/fis)+(1/fib)-1)
	caps=2*Kcaps/Vs
	film=F/Vs
	return caps,film



