from pylab import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def stab_squeez(x,theta,dela,kapa,ra,kta,phia,La,thetapa,kapb,rb,ktb,phib,Lb,thetapb,nui,nuf,N,fig):
	### PARAMETERS ####
	#Most of the parameters are given without the constant pi or 2pi.
	kapa = kapa*2.*pi       #\kappa_a (overall coupling for mode a)
	kapb = kapb*2.*pi       #\kappa_b (overall coupling for mode b)
	theta = theta*pi        #\theta (coherent pump phase)
	thetapa = thetapa*pi    #\theta^\prime_a (local oscillator phase for mode a)
	thetapb = thetapb*pi    #\theta^\prime_b (local oscillator phase for mode b)
	phia = phia*pi          #\phi_a (overall phase shift for mode a)
	phib = phib*pi          #\phi_b (overall phase shift for mode b)
	N = N*1000 +1           #resolution of frequencies

	#Frequency range initialization (nui:min, nuf:max, N:number of points)
	nu = linspace(nui*2.*pi,nuf*2.*pi,N)

	#Parameters calculated from the input parameters
	eps = x*kapa*exp(1j*theta)                #\epsilon (the coherent pump amplitude)
	Del = kapa*dela                           #\Delta (detuning of modes a, same as for b but in the opposite direction)
	ta   = kta/kapa                           #\tau_a (time delay in mode a)
	tb   = ktb/kapb                           #\tau_b (time delay in mode b)
	kap1a = kapa * ra                         #\kappa_{1,a} (coupling on the right hand side in mode a)
	kap1b = kapb * rb                         #\kappa_{1,b} (coupling on the right hand side in mode b)
	kap2a = kapa * (1-ra)                     #\kappa_{2,a} (coupling on the left hand side in mode a)
	kap2b = kapb * (1-rb)                     #\kappa_{2,b} (coupling on the left hand side in mode b)
	ka = 2*sqrt(ra*(1-ra)*(1-La/100.))*kapa    #k_a (feedback strength in mode a)
	kb = 2*sqrt(rb*(1-rb)*(1-Lb/100.))*kapb    #k_b (feedback strength in mode b)

	### EXPRESSIONS ###
	Ema = np.exp(1j*(-nu*ta+phia))
	Emb = np.exp(1j*(-nu*tb+phib))
	Epa = np.exp(1j*(nu*ta+phia))
	Epb = np.exp(1j*(nu*tb+phib))
	dma = kapa-1j*(nu-Del) + ka*Epa         #d_{-,a}(\nu)
	dmb = kapb-1j*(nu+Del) + kb*Epb         #d_{-,b}(\nu)
	dpa = kapa-1j*(nu+Del) + ka*conj(Ema)   #d_{+,a}(\nu)
	dpb = kapb-1j*(nu-Del) + kb*conj(Emb)   #d_{+,b}(\nu)
	Lab = abs(eps)**2 - dpa*dmb          #\Lambda_{ab}(\nu)
	Lba = abs(eps)**2 - dpb*dma          #\Lambda_{ba}(\nu)

	alpha1p = sqrt(2*kap1a)+sqrt(2*kap2a*(1-La/100.))*Epa   #\alpha_1(\nu)
	alpha2p = sqrt(2*kap2a)+sqrt(2*kap1a*(1-La/100.))*Epa   #\alpha_2(\nu)
	beta1p  = sqrt(2*kap1b)+sqrt(2*kap2b*(1-Lb/100.))*Epb   #\beta_1(\nu)
	beta2p  = sqrt(2*kap2b)+sqrt(2*kap1b*(1-Lb/100.))*Epb   #\beta_2(\nu)
	alpha1m = sqrt(2*kap1a)+sqrt(2*kap2a*(1-La/100.))*Ema   #\alpha_1(-\nu)
	alpha2m = sqrt(2*kap2a)+sqrt(2*kap1a*(1-La/100.))*Ema   #\alpha_2(-\nu)
	beta1m  = sqrt(2*kap1b)+sqrt(2*kap2b*(1-Lb/100.))*Emb   #\beta_1(-\nu)
	beta2m  = sqrt(2*kap2b)+sqrt(2*kap1b*(1-Lb/100.))*Emb   #\beta_2(-\nu)

	Da  = sqrt(1-La/100.)*Lba*Epa+dpb*alpha1p*alpha2p
	Db  = sqrt(1-Lb/100.)*Lab*Epb+dpa*beta1p*beta2p
	Ea  = sqrt(La/100.)*(sqrt(2*kap2a)*dpb*alpha2p+Lba)
	Eb  = sqrt(Lb/100.)*(sqrt(2*kap2b)*dpa*beta2p+Lab)
	Mba = exp(1j*(theta-(thetapa+thetapb)/2))*alpha2m*(Db*np.conj(beta1p)+sqrt(2*Lb/100.*kap2b)*Eb)
	Mab = exp(1j*(theta-(thetapa+thetapb)/2))*beta2m*(Da*np.conj(alpha1p)+sqrt(2*La/100.*kap2a)*Ea)
	Nba = abs(eps)/2.*(np.abs(alpha2m)**2*(np.abs(beta1p)**2+2*Lb/100.*kap2b)+\
	np.abs(beta2p)**2*(np.abs(alpha1m)**2+2*La/100.*kap2a))
	Nab = abs(eps)/2.*(np.abs(beta2m)**2*(np.abs(alpha1p)**2+2*La/100.*kap2a)+\
	np.abs(alpha2p)**2*(np.abs(beta1m)**2+2*Lb/100.*kap2b))

	### CORRELATIONS ###
	ncorr = abs(eps)*(1/np.abs(Lab)**2*(np.real(Mba)+Nba)+1/np.abs(Lba)**2*(np.real(Mab)+Nab))   #(normalized)
	Sout  = 10*np.log10(1+ncorr)

	## Functions ##
	def Rar(lambr,lambi):
		return lambr+kapa+ka*cos(phia)*cos(2*np.pi*lambi*ta)*exp(-lambr*ta)
	def Rai(lambr,lambi):
		return 2*np.pi*lambi-ka*cos(phia)*sin(2*np.pi*lambi*ta)*exp(-lambr*ta)
	def Iar(lambr,lambi):
		return Del+ka*sin(phia)*cos(2*np.pi*lambi*ta)*exp(-lambr*ta)
	def Iai(lambr,lambi):
		return -ka*sin(phia)*sin(2*np.pi*lambi*ta)*exp(-lambr*ta)
	def Rbr(lambr,lambi):
		return lambr+kapb+kb*cos(phib)*cos(2*np.pi*lambi*tb)*exp(-lambr*tb)
	def Rbi(lambr,lambi):
		return 2*np.pi*lambi-kb*cos(phib)*sin(2*np.pi*lambi*tb)*exp(-lambr*tb)
	def Ibr(lambr,lambi):
		return Del-kb*sin(phib)*cos(2*np.pi*lambi*tb)*exp(-lambr*tb)
	def Ibi(lambr,lambi):
		return kb*sin(phib)*sin(2*np.pi*lambi*tb)*exp(-lambr*tb)

	## Equations ##
	def eq1r(lambr, lambi):
		return (Rar(lambr,lambi)-Iai(lambr,lambi))*(Rbr(lambr,lambi)-Ibi(lambr,lambi)) - \
		(Rai(lambr,lambi)+Iar(lambr,lambi))*(Rbi(lambr,lambi)+Ibr(lambr,lambi))-eps**2
	def eq1i(lambr,lambi):
		return (Rar(lambr,lambi)-Iai(lambr,lambi))*(Rbi(lambr,lambi)+Ibr(lambr,lambi)) + \
		(Rai(lambr,lambi)+Iar(lambr,lambi))*(Rbr(lambr,lambi)-Ibi(lambr,lambi))
	def equation1(l):
		equations = [eq1r(l[0],l[1])]
		equations.append(eq1i(l[0],l[1]))
		return equations
	def eq2r(lambr, lambi):
		return (Rar(lambr,lambi)+Iai(lambr,lambi))*(Rbr(lambr,lambi)+Ibi(lambr,lambi)) - \
		(Rai(lambr,lambi)-Iar(lambr,lambi))*(Rbi(lambr,lambi)-Ibr(lambr,lambi))-eps**2
	def eq2i(lambr,lambi):
		return (Rar(lambr,lambi)+Iai(lambr,lambi))*(Rbi(lambr,lambi)-Ibr(lambr,lambi)) + \
		(Rai(lambr,lambi)-Iar(lambr,lambi))*(Rbr(lambr,lambi)+Ibi(lambr,lambi))
	def equation2(l):
		equations = [eq2r(l[0],l[1])]
		equations.append(eq2i(l[0],l[1]))
		return equations
	
	## Grid ##
	lr = np.linspace(-20,20,200)
	li = np.linspace(-30,30,200)
	lr,li = np.meshgrid(lr,li)

	## Solving the stability equations ##
	lamb1 = np.zeros(2)
	lamb2 = np.zeros(2)
	lamb1 = fsolve(equation1,[10,5])
	lamb2 = fsolve(equation2,[10,5])
	lambr = np.array([lamb1[0],lamb1[0],lamb2[0],lamb2[0]])
	lambi = np.array([lamb1[1],-lamb1[1],lamb2[1],-lamb2[1]])    
    
	## Figures ##
	#fig = plt.figure()
	### Linestyles ###
	colors={'brown':(200/255.,154/255.,125/255.),\
			'green':(0,151/255.,78/255.),\
			'blue':(0,120/255.,180/255.),\
			'yellow':(1., 215/255., 0),\
			'black':(0.,0.,0.)}
	collab = ['brown','green','blue','yellow','black']
	linewidth = [2,2,3,3,4]
	linestyle = ['solid','dashed','dashdot','dotted','solid']

	### Squeezing spectrum ###
	axsq = fig.add_axes([.15,.15,.83,.83]) # main axes
	axsq.hold(True)
	axsq.set_xlabel('$\\nu/(2\pi)$ (MHz)',fontsize = 25)
	axsq.set_ylabel('$S_{out,\\theta^\\prime} (\\nu)$ (dB)',fontsize = 25)
	axsq.set_ylim(-40,40)
	axsq.xaxis.set_label_coords(0.5, -0.08)
	axsq.yaxis.set_label_coords(-0.08,0.5)
	plt.xticks(fontsize = 17)
	plt.yticks(fontsize = 17)
	axsq.plot(nu/2/pi,Sout,color = colors["brown"],linewidth = 2,linestyle="solid")
	axsq.grid(True)
	#for i in range(4):
	#    axsq.plot(nu,Squeez[:,i],color = colors[collab[i]],linewidth = linewidth[i],linestyle=linestyle[i])
	#    axsq.grid(True)

	### Stability inset ###
	axst=fig.add_axes([0.2,0.2,0.25,0.35])
	axst.contour(lr,li,eq1r(lr,li),[0],colors='b')
	axst.hold(True)
	axst.contour(lr,li,eq2r(lr,li),[0],colors='b')
	axst.contour(lr,li,eq1i(lr,li),[0],colors='g')
	axst.contour(lr,li,eq2i(lr,li),[0],colors='g')
	axst.contour(lr,li,lr,[0],colors='grey',linestyles='dashed')
	axst.plot(lambr,lambi,'o',color = 'r')
	axst.grid(True)
	axst.set_xlabel('$\lambda_r (MHz)$')
	axst.set_ylabel('$\lambda_i (MHz)$')
	axst.xaxis.set_label_coords(1.25, -0.0015)
	axst.yaxis.set_label_coords(-0.0015,1.25)
	
#stab_squeez(x,theta,dela,kapa,ra,kta,phia,La,thetapa,kapb,rb,ktb,phib,Lb,thetapb,nui,nuf,N,fig):
fig = plt.figure()
stab_squeez(1.5,1,0.,10,0.5,0.4,0,0.,0,10,0.5,2,0,0.0,0,-30,30,20,fig)
fig2 = plt.figure()
stab_squeez(0.6875,1,0.0,10,0.5,2,0,0.,0,10,0.5,2,0.,0.0,0,-30,30,20,fig2)
fig3 = plt.figure()
stab_squeez(0.5,1,0.0,10,0.5,4.2,0,0.,0,10,0.5,2,0,0.0,0,-30,30,20,fig3)
plt.show()
