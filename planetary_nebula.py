#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:30:46 2020

@author: zachary_hockenbery

Calculate emission spectrum from a planetary nebula. Models a central star emitting a 
black body spectrum. B(T,nu) is propagated through a cloud of partially ionized hydrogen
gas. Interactions which are included in this are:
  - bound-free ionization
  - free-free collisions
  - bound-bound excitations and de-excitations (only up to 3rd excited state of H)

What is not included:
  - free-bound recombination of electrons and ions

"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import scipy.stats as st
import scipy.special as sp
import sys
import copy
from math import *
from decimal import Decimal

__Msol__ = 1.989e33     # solar mass, grams
__e__ = 1.602e-19
__hJ__ = 6.626e-34      # Planck const, J*s
__h__ = 4.135e-15       # Planck const, eV*s
__c__ = 3e8             # units m/s
__k__ = 8.617e-5        # Boltzmann const, eV/K
__alpha__ = 1/137
__eVtoJ__ = 1.602e-19   # Conversion of eV to J
__a0__ = 5.291772e-11   # bohr radius, meters
__massH__ = 1.6727e-27  # mass of hydrogen atom, kg
__hbar__ = 6.582119e-16 # reduced planck constant, eV*sec
__me__ = 9.11e-31       # electron mass, kg
__mp__ = 1835*__me__    # proton mass, kg
__mec2__ = 510998.95    # electron rest mass, eV
__mpc2__ = 938000       # proton rest mass, keV
__Avogadro__ = 6.022e23 # atoms/mol
__LY__ = 9.4607e15      # lightyear, meters


__sigmaFFe__ = 6.63e-29 # Thomson scattering cross section, m^2
__sigmaFFi__ = (__me__/__mp__)**2 *__sigmaFFe__ # much smaller than the electron cross-section

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

#%% Some data sets 

# Ionization energies of various elements in eV. Sourced from NIST ASD
ionizationEnergy = {
    1:[13.59843449], # hydrogen
    2:[24.58738880, 54.4177650], # helium
    8:[13.618055,   35.12112, 54.93554, 77.41350, 113.8990, 138.1189, 739.32682, 871.40988], # oxygen
    10:[21.564540,   40.96297, 63.4233,  97.1900,  126.247,  157.934,  207.271,   239.0970, 1195.80783, 1362.19915], # nitrogen
    }

#%% Oscillator strength. Sourced from Rybicki and Lightmann textbook. (Chapter 9?)
"""
Each key is the Z value for the atomic species. The first element
is for ni,nf = 12, 13, 14
          then 23, 24,
          then 34

Dimensionless
"""
oscillatorStrength = {
        1:[np.array([4.162e-1, 7.910e-2, 2.899e-2]),
           np.array([6.408e-1, 1.193e-1]),
           np.array([8.420e-1])],
        2:[1],
        8:[1],
        10:[1],
        }


""" Einstein Coefficients for de-exitation of the atomic species. Sourced from NIST ASD
Ordered as:
    21
    31, 32
    41, 42, 43

Units of 1/s. Invert to get the decay time of the state
"""
einsteinCoeff = {
        1:[np.array([4.6986e8]),
           np.array([5.575e7, 4.4101e7]),
           np.array([1.278e7, 8.4193e6, 8.986e6]),
           ],
        2:[0],
        }


#%% definitions

def f(beta, theta):
    ''' Probability density function '''
    mu = np.cos(theta)
    return -(1-beta*mu)*np.sin(theta)

def F(beta, theta):
    ''' Cumulative distribution function'''
    mu = np.cos(theta)
    return -1*(1-beta*mu)**2/(2*beta) + (1-beta)**2/(2*beta)

def mapUniform(beta, y):
    """ Inverting the function F above.
    Takes points y from a uniform distribution.
    """
    gamma = 1/np.sqrt(1-beta**2)
    # Takes a uniform distribution and
    num = 1 - np.sqrt((1-beta)**2-(y*2*beta/gamma))
    den = beta
    
    return num/den

def fThomson(theta):
    ''' the probability density function for Thomson scattering. In
    the rest frame of the electron '''
    return 1 + (np.cos(theta))**2

def boostedIncidenceEnergy(Ein, beta, mu):
    """ Given an incident energy in lab frame and an incidence angle mu,
    calculate the incidence energy in the rest frame of the electron
    """
    gamma = 1/np.sqrt(1-beta**2)
    return Ein*gamma * (1 - beta * mu)

def boostedFinalEnergy(Eout, beta, mu):
    """ Boost from rest frame energy back to the lab frame energy
    """
    gamma = 1/np.sqrt(1-beta**2)
    return Eout*gamma* (1 + beta*mu) 

def generateAlpha2(N):
    
    xy_min = [0, -1]
    xy_max = [2, +1]
    
    output = np.zeros(N)
    i=0
    var=0
    while i < N:
        var = np.random.uniform(low=xy_min, high=xy_max, size=(1,2))
        if var[0][0] < 1 + var[0][1]**2:
            output[i] = var[0][1]
            i+=1
        else:
            var=0
            continue
    return output

def theta(zi,za):
    # Giving phi_f' from alpha', phi_i', and theta', all in the electron rest frame
    za_sin = np.sqrt(1-za**2)
    zi_sin = np.sqrt(1-zi**2)
    phi = np.random.uniform(0,2*np.pi,1)
    return za*zi-zi_sin*za_sin*np.cos(phi)

def generatePlanck(N, kT):
    # For N photons, generate x = hv 
    beta = 1/(kT)
    B = lambda x: (15*x**3 / np.pi**4)/(np.exp(x)-1) # x = hv/kT
    
    x_peak = 2.821
    height = B(x_peak)
    
    xymin = [0, 0]
    xymax = [20, height] # B is scaled below so we only go up to 1
    output = [[] for n in range(N)]
    
    i = 0
    vals = 0
    while i < N:
        vals = np.random.uniform(low=xymin, high=xymax, size=(1,2))
        if vals[0][1] < B(vals[0][0]):
            output[i] = [ vals[0][0]*kT]
            i+=1
        else:
            vals=0
            continue
    return output


def generateMaxwellian(N, kT, mass):
    # needs mass in kg
    
    kT = kT * __eVtoJ__ # eV converted to Joules
    
    out_x = np.zeros(N)
    out_y = np.zeros(N)
    out_z = np.zeros(N)
    i=0
    while i < N:
        vx = np.random.normal(loc=0, scale=np.sqrt(kT/mass), size=1)
        vy = np.random.normal(loc=0, scale=np.sqrt(kT/mass), size=1)
        vz = np.random.normal(loc=0, scale=np.sqrt(kT/mass), size=1)
        if np.sqrt(vx**2+vy**2+vz**2)/__c__ > 1:
            continue
        else:
            out_x[i] = vx
            out_y[i] = vy
            out_z[i] = vz
            i+=1
            
    return out_x, out_y, out_z
        
def Planck(x):
    
    kT = 100 # eV
    beta = 1/(kT)
    return (2*__h__*1.602e-19*x**3 / __c__**2)/(np.exp(__h__*x*beta)-1)

def generateDopplerAngle(beta):
    """ For a given speed, we draw from a distribution of photon flux at
    a given angle. 
    """
    gamma = 1/np.sqrt(1-beta**2)
    
    lowBound = gamma*((1-beta)**2 - (1+beta)**2)/(2*beta)
    highBound = 0
    
    val = np.random.uniform(low=lowBound, high=highBound, size=1)[0]
    
    return mapUniform(beta, val)


def sahaEq(kT):
    """ Purely for hydrogen. Returns (ne*n+)/n, the product of electron and ion number densities divided
    by the total number density.
    """    
    return 2 * ((2*np.pi*__mec2__*kT)/(__h__*__c__)**2)**(1.5) * np.exp(-13.6/kT)

    
def lineShape(nu, initial, final):
    """ Based on the frequency of the photon in question, returns a value from 0 to 1.
    Basically a scaling factor for how close the photon is to the transition frequency.
    This is for the BB absorption cross section.
    
    Gamma is the decay width = 2 DeltaE = hbar / tau = hbar * lambda
    """
    deltaE = hydrogenSpacing(initial, final)
    
    nu0 = deltaE/__h__ 
    l = __h__*einsteinCoeff[1][final-2][initial-1]
    
    maxVal = st.cauchy.pdf(nu0, loc=nu0, scale=l)
    
    return st.cauchy.pdf(nu, loc=nu0, scale=l)/maxVal

def lineShapeDist(final, initial, nu_c):
    """ For BB decays. Pulls a random energy from the line shape distribution when
    given the initial and final states. Uses Einstein coefficients from NIST ASD.
    
    Returns the energy
    """
    nu_c = 4.926413194201546e-08
    # center frequency
    nu0 = hydrogenSpacing(final, initial)/__h__
#    print("nu0 = %s"%str(nu0))
    l = __h__*einsteinCoeff[1][initial-2][final-1] + 2*nu_c
    
#    print("l = %s"%str(l))
    # a randomly chosen output frequency
    nuOut = st.cauchy.rvs(loc=nu0, scale=l)
    
#    print("nuOut = %s"%str(nuOut))
    
    e_fp = nuOut*__h__
#    print("nu0 = %s"%str(nu0))
    
    beta_a = np.random.choice(atom_speeds)/__c__
    gamma_a = 1/np.sqrt(1-beta_a**2)
    
    
    # grab a random angle
    mu = np.cos(np.random.uniform(low=0,high=2*np.pi, size=1)[0])
    
    
    # photon final energy
    e_f = e_fp*gamma_a*(1+beta_a*mu)

    
    return e_f
    


def bbCrossSection(nu, Z, n):
    """ Takes in the energy of the photon and the atomic state
    and returns cross sections for the available BB transitions. I.e. if the state
    is 1, then the size of the returned array is 3, for the transitions 12, 13, 14.

    New feature: doppler shift. Now requires an input for the velocity of the atom.
    Using this, we generate random angles for the photon 
    """
    # currently I only have data for up to 4. size is 4-n
    final_states = range(n+1, 4+1) # ground, first excited, second excited, etc
    
    nu = dopplerShift(nu)
    
    # returns an array of values between zero and 1. If a photon is not nearly in the energy range for
    # absorption then this will be very close to zero.
    lineShapes = np.array([*map(lambda x: lineShape(nu, n, x), final_states)])

    lamb = __c__/nu # lambda, units of m
    unitFactor = lamb**2 / (8*np.pi)
    
    sigmas = oscillatorStrength[Z][n-1]*unitFactor*lineShapes
    return sigmas

def dopplerShift(nu):
    """Used for Doppler shifting the photon energy in the rest-frame of the ion.
    Will be used for both BF and BB cross sections.
    
    We also need to doppler shift the emitted radiation from de-excitation, which
    will broaden the line.
    
    Use pieces of the collision defs above.
    """
    e_i = nu*__h__ # convert to energy
    beta_i = np.random.choice(atom_speeds)/__c__ # obtain ion speed
    gamma_i = 1/np.sqrt(1-beta_i**2) # convery to ion gamma factor
    
    # flux is Doppler shifted by (1-beta*mu)
    mu_i_i = generateDopplerAngle(beta_i) # generate the angle of the photon seen by the atom

    # incident energy in atom rest frame
    e_ip = e_i*gamma_i*(1-beta_i*mu_i_i)

    nu_out = e_ip / __h__
    
    return nu_out


def energySTDev(initial, final):
    """It's difficult to get the absorption lines to show, so what I am going to do is generate
    photons that are within the absorption band and then simulate that. Therefore a function for
    approximating the absorption band is necessary.
    
    We could even compare to the Doppler shift giving the Voigt function."""
    
    spacing = __h__*einsteinCoeff[1][final-2][initial-1]
    
    return spacing

#%%
nuSpace = np.linspace(2,3, 10000)/__h__
out = [lineShape(nu, 2, 4) for nu in nuSpace]

#maxVal = st.cauchy.pdf(2.5/__h__,loc=2.5/__h__,scale=1/__h__)

#norm = np.array([st.cauchy.pdf(nu, loc=2.5/__h__, scale=1.0/__h__) for nu in nuSpace])

plt.figure(figsize=(12,8))
plt.plot(nuSpace*__h__, out)
    


#%%

def bfCrossSection(nu, Z, n):
    """Returns the cross section for ionizing an atom in state n with a photon
    of energy nu. Units are METERS^2
    
    for h*nu below the ionization energy, the cross section is simply zero.
    
    So, if we have a helium atom, both of the electrons can be ionized, therefore we
    need some extra consideration.
    """
    nu = dopplerShift(nu)
    
    # calculate cutoff ionization energy based on excitation state n
    if __h__*nu < ionizationEnergy[Z][0] - hydrogenSpacing(1, n):
        return 0.0
    # above that cutoff, we have the standard curve shape
    else:
        g_bf = 1
        omega_n = (__alpha__**2 * __mec2__* Z**2)/(2*n**2 * __hbar__)
        omega = nu/(2*np.pi) # convert orbital frequency to angular
        pref = 64*np.pi*n*g_bf*__alpha__/(2*np.sqrt(3))
        f1 = (__a0__/Z)**2
        f2 = (omega_n/omega)**3
        return pref*f1*f2
    

def hydrogenSpacing(n1, n2):
    """ self-explanatory
    c*h/output gives the wavelength we expect
    """
    return ionizationEnergy[1][0] * ((1/n1)**2 - (1/n2)**2)

def electronCollision(e_i):
    """ takes in the energy of the photon and performs
    a collision with a random electron taken from the Maxwell
    distribution.
    
    Energy given as photon energy hv in eV
    
    Outputs the final energy of the photon in lab frame
    """
    beta_e = np.random.choice(electron_speeds)/__c__
    gamma_e = 1/np.sqrt(1-beta_e**2)
    mu_e_i = generateDopplerAngle(beta_e)
    
    # incident energy in electron rest frame
    e_ip = e_i*gamma_e*(1-beta_e*mu_e_i)
    
    # boost doppler angle to electron rest frame
    mu_e_ip = (mu_e_i - beta_e)/(1 - mu_e_i*beta_e)
    mu_e_ip_sin = np.sqrt(1 - mu_e_ip**2)
    
    # Alpha, angle between incident and outgoing photon
    mu_e_a = generateAlpha2(1)[0]
    mu_e_a_sin = np.sqrt(1-mu_e_a**2)
    
    # scattered energy of photon in electron rest frame
    e_fp = e_ip/(1 + e_ip*(1-mu_e_a)/(__mec2__))
    
    # grab a random azimuthal angle
    mu_th = np.cos(np.random.uniform(low=0,high=2*np.pi, size=1)[0])
    
    # outgoing angle in electron rest frame
    mu_fp = (mu_e_a*mu_e_ip - mu_e_a_sin*mu_e_ip_sin*mu_th)
    
    # photon final energy
    e_f = e_fp*gamma_e*(1+beta_e*mu_fp)
    
    
    return e_f

def ionCollision(e_i):
    beta_i = np.random.choice(ion_speeds)/__c__
    gamma_i = 1/np.sqrt(1-beta_i**2)
    mu_i_i = generateDopplerAngle(beta_i)

    # incident energy in ion rest frame
    e_ip = e_i*gamma_i*(1-beta_i*mu_i_i)

    # boost doppler angle to ion rest frame
    mu_i_ip = (mu_i_i - beta_i)/(1 - mu_i_i*beta_i)
    mu_i_ip_sin = np.sqrt(1 - mu_i_ip**2)
    
    # Alpha, angle between incident and outgoing photon
    mu_i_a = generateAlpha2(1)[0]
    mu_i_a_sin = np.sqrt(1-mu_i_a**2)
    
    # scattered energy of photon in ion rest frame
    e_fp = e_ip/(1 + e_ip*(1-mu_i_a)/(__mpc2__*Z*2))
    
    # grab a random azimuthal angle
    mu_th = np.cos(np.random.uniform(low=0,high=2*np.pi, size=1)[0])
    
    # outgoing angle in electron rest frame
    mu_fp = (mu_i_a*mu_i_ip - mu_i_a_sin*mu_i_ip_sin*mu_th)
    
    # photon final energy
    e_f = e_fp*gamma_i*(1+beta_i*mu_fp)
    
    return e_f


def boltzmannEq(i, j, kT):
    """Given an temperature and two states to consider, this returns the
    population ratios. Only works for Hydrogen right now
    
    i is the lower state and j is upper state.
    
    degeneracy in Hydrogen is simply 2n^2
    """
    gi = 2*i**2
    gj = 2*j**2
    return (gj/gi)*np.exp(-hydrogenSpacing(i, j)/kT)


#%% Generate the Planck spectrum.    
Nphoton = 10000 # Number of photons
kT_s = 2.58 # Star temperature, eV (about 30,000 Kelvin for an exposed star core)
print("Temperature of planck spectrum is %s Kelvin" % str(kT_s/__k__))
photon_data = generatePlanck(Nphoton, kT_s) # returns x=hv for each photon. A list of length 1 sublists
visual_data = [item for sublist in photon_data for item in sublist] # flatten so it can be histogrammed
visual_wavelength_data = __h__*__c__/np.array(visual_data)

#%% Generate Uniform photons
#Nphoton = 2000
#center = hydrogenSpacing(1,4)
#spacing = energySTDev(1,2)
#photon_data = [[np.random.uniform(low = center-1, high=center+1, size=1)[0]] for i in range(Nphoton)]
#visual_data = [item for sublist in photon_data for item in sublist] # flatten so it can be histogrammed
#visual_wavelength_data = __h__*__c__/np.array(visual_data)

#%%
binspace = np.linspace(0,30,1000)
plt.figure(figsize=(8,5))
plt.hist(visual_data, bins=binspace)
plt.xlabel(r"$h\nu$ [eV]")
plt.ylabel("Counts")
#plt.xscale('log')
plt.axvline(x=13.6, ymin=0, color='r', ls='--', label=r"$\chi_H$")
plt.legend()
#%% Generate the plasma
# Literature gives temperature of the plasma, 2.15 eV (25,000 Kelvin)
kTp = 0.20 # but I find around 0.19 eV is where the balance shifts a lot 0.26
print("Plasma temperature Tp = %s Kelvin \n"%str(int(kTp/__k__) ))

""" Populate excited states using the Boltzmann equation 
this gives the relative number densities for excited states
2, 3 and 4 in comparison to ground state 1."""
n21_rel = boltzmannEq(1,2, kTp)
n31_rel = boltzmannEq(1,3, kTp)
n41_rel = boltzmannEq(1,4, kTp)
""" Prescribe the number density of atoms here. I will guess this
and try to get an overall number density close to what I find in
literature.

The trade-off between the plasma temperature and the number density of atoms is
a bit finicky. I am trying to keep a 'partial plasma' therefore I should try and
keep the ionization fraction below 10% or so.

We are using meters, so 10000/cm^3 is about 0.01/m^3:
UCHII region is about 1,000,000/cm^3 which is 1e12/m^3
"""
#ndensAtoms = 10000*1e4*1e-6# particles / m^3
ndensAtoms = 10e8 # density for UCHII region 
""" Populate the ionized states using the Saha equation.
this gives the ionization fraction n+/n which is the
number density of ions over the total number density
"""
# also in particles/m^3
ndensIons      = np.sqrt( ndensAtoms * sahaEq(kTp) )
ndensElectrons = np.sqrt( ndensAtoms * sahaEq(kTp) )

ndensPlasma = ndensAtoms + ndensIons

ndensTot = ndensAtoms+ndensIons+ndensElectrons

fracElectrons = ndensElectrons/ndensTot
fracIons      = ndensIons/ndensTot
fracHI        = ndensAtoms*(1 - (n21_rel+n31_rel+n41_rel) )/ndensTot # atomic, ground state
fracHII       = n21_rel*ndensAtoms/ndensTot # first excited state
fracHIII      = n31_rel*ndensAtoms/ndensTot # second excited state
fracHIV       = n41_rel*ndensAtoms/ndensTot # third excited state



print("Percentage ions = %s "        %str(100*fracIons))
print("Percentage electrons = %s"   %str(100*fracElectrons))
print("Percentage atoms = %s "%str(100*fracHI))
#print("Number density of the plasma = %s particles/m^3"%str(ndensPlasma))
print("Total number density of plasma = %s particles/m^3"%f"{Decimal(ndensTot):.2E}"+" (%s /cm^3)"%str(ndensTot*1e-6))

# hmm.. seems quite low
print("Relative number density in first metastable excited state = %s"%str(fracHII))
print("Relative number density in second metastable excited state = %s"%str(fracHIII))
print("Relative number density in third metastable excited state = %s"%str(fracHIV))

#%% Cross section plot
#nuSpace = np.linspace(1, 30, 4800)/__h__
#sig_bf = [bfCrossSection(nu, 1, 1) for nu in nuSpace]
#sig_bb1 = [bbCrossSection(nu, 1, 1) for nu in nuSpace]
#sig_bb2 = [bbCrossSection(nu, 1, 2) for nu in nuSpace]
#sig_bb3 = [bbCrossSection(nu, 1, 3) for nu in nuSpace]
#plt.figure(figsize=(10,8))
#plt.plot(nuSpace*__h__, sig_bf, label="BF")
#for bb in [*zip(*sig_bb1)]:
#    plt.plot(nuSpace*__h__, bb, label="2j")
#for bb in [*zip(*sig_bb2)]:
#    plt.plot(nuSpace*__h__, bb, label="3j")
#for bb in [*zip(*sig_bb3)]:
#    plt.plot(nuSpace*__h__, bb, label="4j")
#plt.yscale("log")
#plt.legend()
#%% Velocity distributions of the electrons, ions and atoms. Resultant velocities are in m/s
Nmaxwell = 1000
vel_ions      = generateMaxwellian(int(Nmaxwell), kTp, 2*__mp__)
vel_electrons = generateMaxwellian(int(Nmaxwell), kTp,   __me__)
vel_atoms     = generateMaxwellian(int(Nmaxwell), kTp, 2*__mp__)
ion_speeds      = np.array([*map(lambda x: np.sqrt(x[0]**2+x[1]**2+x[2]**2), zip(vel_ions[0],vel_ions[1],vel_ions[2]))])
electron_speeds = np.array([*map(lambda x: np.sqrt(x[0]**2+x[1]**2+x[2]**2), zip(vel_electrons[0],vel_electrons[1],vel_electrons[2]))])
atom_speeds     = np.array([*map(lambda x: np.sqrt(x[0]**2+x[1]**2+x[2]**2), zip(vel_atoms[0],vel_atoms[1],vel_atoms[2]))])

# collision frequency in units of collisions/sec. Uses atomic cross section and number density of ions and atoms
collision_freq = 1e-20 * ndensPlasma * np.average(atom_speeds)
print("Calculated collision frequency for atoms = %s collisions/second"%str(collision_freq))
#%% NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW
"""
THIS IS THE NEW ALGORITHM!!

DON'T EDIT THE STUFF UP ABOVE!

-----------------------------------------------

NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW NEW

"""
#photon_data = [[12.087497324444444]]

temp_energy = []
newSpectrum = [[] for i in photon_data]

Z = 1
n = 1

cBF = 0
cCollision = 0
cFF = 0
cBB = 0
cNI = 0

# counters for de-excitation
cLyman = 0
cBalmer = 0
cPaschen = 0

#collision_freq = 1e-20

# testing with a synthetic photon_data
#blah = np.linspace(2.50,2.60, 1000)


"""
Lemon Slice nebula is about 0.2 ly in radius. I'm not sure how many chuncks we need to break
this up in to, but lets find out.
 """
# units of m
sSpan = np.linspace(0.01, 1.0, 100)*0.2*__LY__
deltaS = sSpan[1]-sSpan[0]
print("Step size is %s meters!!!"%f"{Decimal(deltaS):.2E}")

for idp, e_i in enumerate(photon_data):
    print("starting primary photon %s"%str(idp)+"/%s"%str(len(photon_data)))
    
    """ We first calculate if an interaction will happen. To do this we calculate the summed
    optical depth:
        tau = s*ndens* sum(sigma_i).
    We consider it has travelled a physical distance of s = delta s... (i'm not sure what do with this yet)
    
    Within this distance we can calculate the rates based on cross section and number density. So we
    then calculate this total tau.
    """
    for ids, s in enumerate(sSpan):
        
        if e_i == []:
            """ I'm doing this for the ionization condition. If a lone photon ionizes, we are supposed to stop
            the interaction loop. But if a photon from a group ionizes, then we continue.
            """
#            print("met ionization condition")
            break
        
#        print("\n e_i is %s"%str(e_i))
        """ Before going in to e_i, we need a temporary array for holding the energies directly produced
        by these photons. This is temp_energy2. the other array, temp_energy is used to hold the photons produced
        by de-excitation"""
        temp_energy2 = [] # this is to hold the sub photons produced by secondary photons
        for idx, energy in enumerate(e_i):
            
            
            # compare cross sections and roll dice for one of the interactions
            # -1 is ff, 0 is bf, >0 is bb
            # these cross sections natively calculate for a random ion speed.
            dsig_bb = bbCrossSection(energy/__h__, Z, n)
            dsig_bf = bfCrossSection(energy/__h__, Z, n)
            
            drate_bb = dsig_bb*ndensAtoms
            drate_bf = dsig_bf*ndensAtoms
            drate_ff = __sigmaFFe__*ndensElectrons
            
            drate_tot = drate_bf + drate_ff + np.sum(drate_bb) # total rate: sum over sigma*ndens
            
            tau_all = deltaS * drate_tot
            
            # 0 for no interaction, 1 for an interaction
            interaction = np.random.choice([0, 1], p = [np.exp(-tau_all), 1-np.exp(-tau_all)])
                
            if interaction:
                
                # compute iType: -1 = FF, 0 = BF, 1+ is BB
                iType = np.random.choice([-1, 0]+[*range(1, len(drate_bb)+1 )], p=np.concatenate([ [drate_ff], [drate_bf], drate_bb])/drate_tot )
                
                temp_energy = [] # temporarily hold the energies produced by de-excitation
                if iType == -1:
                    print("Collision!")
                    """ Electron or ion collision. For now I am treating them both like an electron
                    collision... because I don't know the cross section for scattering from an
                    ion.
                    
                    Here we concatenate to the temp energy list. """
                    temp_energy += [ electronCollision(energy) ]
    #                    print(str(idx)+"index and temp=%s"%str(temp_energy) )
                    cFF += 1
                    
                    
                elif iType == 0:
                    print("Ionization!")
                    """ Ionization, we lose photon 
                    setting temp_energy to += [] helps for when there are more than one photon, but we can't have the break! """
                    temp_energy += [] # empty, so that it isn't looped over! not that it would...
                    cBF += 1
                    
                else:
                    
                    """ Excitation """
                    cBB += 1
                    # state_f is exactly equal to the state, not the index!
                    # it is the state that the atom was excited to
                    state_f = iType + 1
                    print("Excitation to state %s"%str(state_f))

                    
                    # Rates for the state_f to decay or collide
                    rates = np.concatenate([ [collision_freq], 1/einsteinCoeff[1][state_f-2] ])
                    
                    # Dice roll for decay or collide
                    # 0=collision, 1=de-excitation to 1, 2=de-excitation to 2, etc
                    result = np.random.choice([0]+[*range(1, len(rates-1))], p=rates/np.sum(rates))
                    
                    
                    # result exaclty equal to the state, not the index!
                    if result == 0:
#                        print("    excitation lost... :( ")
                        """ If a collision happens, I assume it results in the atom giving
                        its de-excitation energy to the kinetic energy of the collision partner.
                        Therefore we end the compuatation for this photon branch here.
                        """
                        cCollision += 1
                        break
                    elif result == 1: # decay to ground state
#                        print("    Direct decay to %s"%str(result))
                        # Pull random energy from distribution. Requires low state, then
                        cLyman+=1
                        temp_energy += [lineShapeDist(result, state_f, collision_freq)]
#                        if np.nan in temp_energy:
#                            print("NaN found!! decay to ground state")
#                            sys.exit()
    #                        temp_energy += [hydrogenSpacing(result, state_f)]
    #                        print(temp_energy)
                    elif result == 2: # decay to first excited state
#                        print("    Decay to %s"%str(result))
                        cBalmer+=1
                        temp_energy += [lineShapeDist(result, state_f, collision_freq)]
#                        if np.nan in temp_energy:
#                            print("NaN found!! decay to first excited")
#                            sys.exit()
                        
                        # either collide or decay to ground state
                        rates = np.concatenate([[collision_freq], 1/einsteinCoeff[1][0]])
                        result2 = np.random.choice([0, 1], p=rates/np.sum(rates))
#                        print("Rates for 2 decaying: %s"%str(rates))
                        
                        if result2 == 0: # collision
#                            print("        First excited state was lost to collision... :( ")
                            # break the photon loop here
                            cCollision += 1
                            break
                        elif result2== 1: # decay to ground state, add another photon
#                            print("        First excited decayed to ground state!")
                            cLyman+=1
                            temp_energy += [lineShapeDist(result2, result,collision_freq)]
#                            print(temp_energy)
#                            if np.nan in temp_energy:
#                                print("NaN found!! decay to first, then ground")
#                                sys.exit()
                            
                    elif result == 3: # decay to second excited state
#                        print("    Decay to second excited")
                        cPaschen+=1
                        temp_energy += [lineShapeDist(result, state_f,collision_freq)]
#                        if np.nan in temp_energy:
#                            print("NaN found!! decay to second")
#                            sys.exit()
                        
                        # collide, decay to ground, or to first excited
                        rates = np.concatenate([[collision_freq], 1/einsteinCoeff[1][1]])
                        result3 =np.random.choice([0, 1, 2], p=rates/np.sum(rates))
                        
                        if result3 == 0: #collision
#                            print("        Second excited lost to collision... :( ")
                            cCollision += 1
                            break
                        elif result3==1:# decay to ground
#                            print("        Second excited decayed to ground!")
                            cLyman+=1
                            temp_energy += [lineShapeDist(result3, result,collision_freq)]
#                            if np.nan in temp_energy:
#                                print("NaN found!! decay to second, then ground")
#                                sys.exit()
                        elif result3==2: # decay to first...
#                            print("        Second excited decayed to first!")
                            cBalmer+=1
                            temp_energy += [lineShapeDist(result3, result, collision_freq)]
#                            if np.nan in temp_energy:
#                                print("NaN found!! decay to second, then first")
#                                sys.exit()
                            
                            rates = np.concatenate([[collision_freq], 1/einsteinCoeff[1][0]])
                            result4 = np.random.choice([0, 1], p=rates/np.sum(rates))
                            
                            if result4 == 0: #collision
#                                print("            Lost in collision")
                                cCollision += 1
                                break
                            elif result4 == 1:
#                                print("            Decayed to ground!")
                                cLyman+=1
                                temp_energy += [lineShapeDist(result4, result3, collision_freq)]
#                                if np.nan in temp_energy:
#                                    print("NaN found!! decay to second, then first, then ground")
#                                    sys.exit()

                # adds all of the new photons to the energy list for the next interaction
#                print("Copying new enrgies: %s"%str(temp_energy))
                temp_energy2 += temp_energy
#                print("Current state of temp_energy2 = %s"%temp_energy2)
                
            else:
#                sys.exit()
#                print("no interaction")
                cNI += 1
                temp_energy2 += [energy]
#                print("current state of temp_energy2 = %s"%temp_energy2)
                "No interaction occured"
        """ """
        e_i =copy.deepcopy( temp_energy2)
                
    # at the end of sSpan, the single photon is replaced by the array of e_i    
    newSpectrum[idp] = copy.deepcopy(e_i)
        
print("Finished simulation...")
print("Incidence of BF = %s"%str(cBF))
print("Incidence of BB = %s"%str(cBB))
print("Incidence of FF = %s"%str(cFF))
print("Missed interactions = %s"%str(cNI))
print("Lyman transitions: %s"%cLyman)
print("Balmer transitions: %s"%cBalmer)
print("Paschen transitions: %s"%cPaschen)
#%% testing

#plotting_data = [item for sublist in spectrum_data[0] for item in sublist]
plotting_data = [item for sublist in newSpectrum for item in sublist]
plotting_data = np.array(plotting_data)

wavelength_data = __h__*__c__*np.array(plotting_data)

print("New photons ceated is about %s"%str(len(plotting_data)-Nphoton))

#%%
#pyplot.locator_params(axis='y', nbins=6)
#plt.locator_params(axis='x', numticks=4)
binspace = np.linspace(0,30, 100)


plasma = plt.hist(plotting_data, bins=binspace, label="Plasma-ized", alpha=0.5);
visual = plt.hist(visual_data, bins=binspace,   label="Planck ",     alpha=0.5);

fig, ax = plt.subplots(figsize=(12,8))

every_nth = 1
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)
fig.tight_layout()
plt.plot(plasma[1][:-1],plasma[0], ds='steps-mid', label='Plasma-ized')
plt.plot(visual[1][:-1],visual[0], ds='steps-mid', label="Planck")
#plt.axvline(13.6, color='k', ls='--')
plt.axvline(hydrogenSpacing(1,4),color='k', ls='--', label=r"Ly$~\gamma$")
plt.axvline(hydrogenSpacing(1,3),color='k', ls='--')
plt.axvline(hydrogenSpacing(1,2),color='k', ls='--')
#plt.axvline(hydrogenSpacing(2,3),color='k', ls='--')
#plt.axvline(hydrogenSpacing(2,4),color='k', ls='--')
#plt.axvline(hydrogenSpacing(3,4),color='k', ls='--')
plt.xlabel(r"$h\nu$ [eV]")
plt.ylabel("Counts")
plt.legend()

#plt.yscale("log")
#plt.xscale("log")

#%% Plotting wavelengths
waveSpace = np.linspace(0, 2e-5, 1000)
plt.figure(figsize=(12,8))
planckWl = plt.hist(wavelength_data, bins=waveSpace, label='plasma-ized');
plt.plot(planckWl[1][:-1]*1e6, planckWl[0])
#plt.hist(visual_wavelength_data, bins=100, label='Planck')
plt.legend()
#%%
l = einsteinCoeff[1][2][0]*__hbar__
nu0 = hydrogenSpacing(1,4)/__h__

blah = [st.cauchy.rvs(loc=nu0, scale=l) for i in range(1000)]
plt.hist(blah, bins=100)
#a = plt.hist(blah, bins=np.linspace(3083079161879955.5,3083079161881000.5, 100))
#plt.axhline(y = 1613.0, color='r')
#plt.axvline(x = -1, color='r')
#plt.axvline(x = +1, color='r')


