from pylab import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.interpolate import griddata
import mpl_toolkits.mplot3d as mp3d
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import time

start=time.clock()

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rc('font',family='FreeSerif')
mpl.rc('xtick',labelsize=30)
mpl.rc('ytick',labelsize=30)

colors={'red':(241/255.,88/255.,84/255.),\
        'orange':(250/255,164/255.,58/255.),\
        'pink':(241/255,124/255.,176/255.),\
        'brown':(178/255,145/255.,47/255.),\
        'purple':(178/255,118/255.,178/255.),\
        'green':(96/255,189/255.,104/255.),\
        'blue':(93/255,165/255.,218/255.),\
        'yellow':(222/255., 207/255., 63/255),\
        'black':(0.,0.,0.)}
collab = ['blue','orange','green','pink','brown']
fsfont = {'fontname':'FreeSerif'}

#Function to determine the two-mode squeezing with coherent unidirectional feedback connecting the two sides
def Squeez(eps,theta,Del,kapa,ra,kta,phia,La,thetapa,kapb,rb,ktb,phib,Lb,thetapb,nu):
	kap1a = kapa * ra                         #\kappa_{1,a} (coupling on the right hand side in mode a)
	kap1b = kapb * rb                         #\kappa_{1,b} (coupling on the right hand side in mode b)
	kap2a = kapa * (1-ra)                     #\kappa_{2,a} (coupling on the left hand side in mode a)
	kap2b = kapb * (1-rb)                     #\kappa_{2,b} (coupling on the left hand side in mode b)
	ka = 2*sqrt(ra*(1-ra)*(1-La/100.))*kapa    #k_a (feedback strength in mode a)
	kb = 2*sqrt(rb*(1-rb)*(1-Lb/100.))*kapb    #k_b (feedback strength in mode b)
	ta    = kta/kapa                           #\tau_a (time delay in mode a)
	tb    = ktb/kapb                           #\tau_b (time delay in mode b)
	N    = int(len(nu))                           #\tau_b (time delay in mode b)

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
	Nba = abs(eps)*(np.abs(alpha2m)**2*(np.abs(beta1p)**2+2*Lb/100.*kap2b)+\
	np.abs(beta2p)**2*(np.abs(alpha1m)**2+2*La/100.*kap2a))
	Nab = abs(eps)*(np.abs(beta2m)**2*(np.abs(alpha1p)**2+2*La/100.*kap2a)+\
	np.abs(alpha2p)**2*(np.abs(beta1m)**2+2*Lb/100.*kap2b))
	### CORRELATIONS ###
	ncorrm = abs(eps)*(1/np.abs(Lab)**2*(2*np.real(Mba)+Nba))  #(normalized)
	ncorrp = abs(eps)*(1/np.abs(Lba)**2*(2*np.real(Mab)+Nab))   #(normalized)
#	Soutm  = 10*np.log10(1.+ncorrm)
#	Soutp  = 10*np.log10(1.+ncorrp)
	Sout   = np.zeros(N)
	Sout[0:int(N/2)+2]  = 10*np.log10(1.+ncorrm[0:int(N/2)+2])
	Sout[int(N/2)+2:N]  = 10*np.log10(1.+ncorrp[int(N/2)+2:N])
#	Sout[int(N/2)+1:N]  = 10*np.log10(1.+ncorrm[0:int(N/2)])
#	Sout[0:int(N/2)+1]  = 10*np.log10(1.+ncorrp[int(N/2):N])
	return Sout#,Soutm,Soutp


ktaua=1.8832785157443013#0.#3.08647680594#
ktaub=1.8832785157443013#0.#3.08647680594#
phia=0.#1.8832785157443013#0.#3.08647680594#
phib=0.#1.8832785157443013#0.#3.08647680594#
ra =.5#0.933012701892219#
nu = np.linspace(0,100,10000)*2*np.pi
eps = np.linspace(0,1,601)
Del = np.linspace(5,10.2,601)
Smin = np.zeros((len(eps),len(Del)))
nus = np.zeros((len(eps),len(Del)))
SminL = np.zeros((len(eps),len(Del)))
nusL = np.zeros((len(eps),len(Del)))
epsm,Delm = np.meshgrid(eps,Del)

for ia in range(0, len(eps)):
    for ib in range(0,len(Del)):
        Sout_fb = Squeez(eps[ia]*2*np.pi*10,1*np.pi,Del[ib]*2*np.pi*10,10*2.*pi,ra,ktaua,phia,0,0,10*2.*pi,ra,ktaub,phib,0,0,nu)
        Sout_fbL = Squeez(eps[ia]*2*np.pi*10,1*np.pi,Del[ib]*2*np.pi*10,10*2.*pi,ra,ktaua,phia,5,0,10*2.*pi,ra,ktaub,phib,5,0,nu)
        ind = np.argmin(Sout_fb)
        indL = np.argmin(Sout_fbL)
        nus[ia,ib] = nu[ind]/2/np.pi
        nusL[ia,ib] = nu[indL]/2/np.pi
#        if np.min(Sout_fb)>=-40 and np.min(Sout_fb)<0:
        Smin[ia,ib]=np.min(Sout_fb)
        SminL[ia,ib]=np.min(Sout_fbL)
#        else:
#            Smin[ia,ib]=-40

np.savetxt("./eps-Del_map_18833ktaua.txt",np.concatenate((epsm,Delm,nus,Smin,nusL,SminL),axis=0))

end=time.clock()
durd=int((end-start)/60./60./24.)
durh=int((end-start)/60./60-24*durd)
durm=int((end-start)/60.-(24*durd+durh)*60.)
durs=int((end-start)-((24*durd+durh)*60+durm)*60)
print("Elapsed time: %d:%d:%d:%d" % (durd,durh,durm,durs))

