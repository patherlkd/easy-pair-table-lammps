## Written by Dr. Luke Davis: UCL Department of Mathematics 2023 luke.davis@ucl.ac.uk
##
##
#!/usr/bin/env python
# coding: utf-8


import autograd.numpy as np # numerical python library
import math # python math library
from autograd import grad # automatic differentiation (to get the force)


### Generic force function ###
def force(pair,*args):
    f =  grad(pair)
    return -f(*args)


### Pair Potentials ###
def pair_debug(r):
    return r


# Example Lennard Jones potential
def pair_LJ(r,eps,sigma,rc):
    ene = 0.0
    
    # Impose a cutoff
    if r < rc:
        ene += 4.0*eps*((sigma/r)**12 - (sigma/r)**6)+eps
    
    # A pair function should only return the potential energy at r
    return ene

# A more involved example
def pair_1(r,typei,typej,intmatrix,
         eps_excl,sigma, # for the excluded volume
         eps_coh,rcoh, # for the cohesion
         Qi,Qj,dieconst,kappa,rcoul, # for the electrostatics
         ):
    
    # determine what interactions should be included for i and j
    excl_on = intmatrix[typei-1][typej-1][0]
    coh_on = intmatrix[typei-1][typej-1][1]
    coul_on = intmatrix[typei-1][typej-1][2]

    # initialise enes
    excl_ene = 0.0
    coh_ene = 0.0
    coul_ene = 0.0

    # setup somethings
    epsconvfact = (1. + 4.*((sigma/rcoh)**12 - (sigma/rcoh)**6 ) + (2.**(1/6)*sigma - rcoh)*4.*(6.*((sigma**6)/(rcoh**7))-12.*((sigma**12)/(rcoh**13))))
    epsconvfact = 1./epsconvfact
    eps_conv = eps_coh*epsconvfact
    
    rcexcl = 2.0**(1./6.)*sigma
    
    # add contribs to ene
    
    ## Volume exclusion
    if r < rcexcl and excl_on:
        excl_ene += 4.0*eps_excl*((sigma/r)**12 - (sigma/r)**6)+eps_excl
    
    ## Cohesion
    if r < rcoh and coh_on:
        coh_ene += 4.0*eps_conv*((sigma/r)**12 - (sigma/r)**6)
        coh_ene += -4.0*eps_conv*((sigma/rcoh)**12 - (sigma/rcoh)**6)
        coh_ene += -(r-rcoh)*4.0*eps_conv*( ((6.0*sigma**6)/(rcoh**7)) - ((12.0*sigma**12)/(rcoh**13)))

    ## electrostatics
    if r < rcoul and coul_on:
        coul_ene += (56.11 * Qi * Qj)/(dieconst*r)*math.exp(-kappa*r) # note ke * e^2 * (1/(4.11x10^-21)) * 10^9 = 56.11 (see notes)  

    return excl_ene + coh_ene + coul_ene
