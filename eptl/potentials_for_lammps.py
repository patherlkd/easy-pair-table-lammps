""" This module contains pair interaction potential functions. To call one
of these functions with the ``pair_keyword``, use a str with the entire function
name or the function name sans the prefix, "pair_".

Notice that the last argument in all functions is the cutoff radius, ``rc``. Since this
cutoff radius is used exclusively, a consistent placement for this argument is required 
for the ``shift`` feature in :func:`eptl.create_lammps_pairstyle_table.make_table_for_lammps`
"""

## Written by Dr. Luke Davis: UCL Department of Mathematics 2023 luke.davis@ucl.ac.uk
## Added pair_LJ_12_6_4 Jennifer Clark 02/08/2024

import autograd.numpy as np # numerical python library
import math # python math library
from autograd import grad # automatic differentiation (to get the force)
from functools import partial # to pass functions into functions...

### Generic force function ###
def force(pair,*args):
    f =  grad(pair)
    return -f(*args)

### Pair Potentials ###
def pair_debug(r):
    return r

# Example Lennard Jones
def pair_LJ(r,eps,sigma,rc):
    ene = 0.0

    # Impose a cutoff
    if r < rc:
        ene += 4.0*eps*((sigma/r)**12 - (sigma/r)**6)

    # A pair function should only return the potential energy at r
    return ene

def pair_LJ_12_6_4( r, eps, sigma, c4, rc):
    r""" Potential for LJ with additional $r^{-4}$ dependence.
    
    .. math::
        U_{ij}(r_{ij}))=4\epsilon_{ij}\left[ \left( \frac{\sigma_{ij}}{r_{ij}} \right)^{12} -\left( \frac{\sigma_{ij}}{r_{ij}} \right)^{6} \right] - \frac{C_{4}^{ij}}{r_{ij}^4}

    Parameters
    ----------
    r : float
        Distance between two beads
    eps : float
        Interaction energy parameter
    sigma : float
        Characteristic size parameter
        Distance of potential energy minimum
    c4 : float
        Dispersion coefficient
    rc : float
        Interaction cut-off distance

    Returns
    -------
    ene
        Energy of interaction
    """
    ene = 0.0

    # Impose a cutoff
    if r < rc:
        ene += 4*eps*((sigma/r)**12 - (sigma/r)**6) - c4/r**4

    # A pair function should only return the potential energy at r
    return ene


def pair_Mie( r, eps, sigma, lr, la, rc):
    r""" Mie Potential: Generalized LJ with variable exponents
    
    .. math::
        C_{ij} = \frac{l_r}{l_r-l_a}(\frac{l_r}{l_a})^(\frac{l_a}{l_r-l_a})
        U_{ij}(r_{ij}))=C_{ij}\epsilon_{ij}\left[ \left( \frac{\sigma_{ij}}{r_{ij}} \right)^{l_r} -\left( \frac{\sigma_{ij}}{r_{ij}} \right)^{l_a} \right]

    Parameters
    ----------
    r : float
        Distance between two beads
    eps : float
        Interaction energy parameter
    sigma : float
        Characteristic size parameter
        Distance of potential energy minimum
    lr : float
        Repulsive exponent
    la : float
        Attractive exponent
    rc : float
        Interaction cut-off distance

    Returns
    -------
    ene
        Energy of interaction
    """
    ene = 0.0
    prefactor = lr / (lr - la) * ( lr / la )**( la / ( lr - la))

    if r < rc:
        ene += prefactor * eps * ( (sigma/r)**lr - (sigma/r)**la )

    return ene

def pair_CSW( r, lam, m, n, rc):
    r""" Continuous Square-Well (CSW)
    
    DOI: 10.1080/00268976.2018.1481232
    
    .. math::
        U_{ij}(r_{ij}))=\frac{1}}{2} \left[ \left( \frac{1}{r_{ij}} \right)^{n} + \frac{1-\exp{-m(r_{ij}-1)(r_{ij}-lam)}}{1+\exp{-m(r_{ij}-1)(r_{ij}-lam)}} - 1 \right]

    Parameters
    ----------
    r : float
        Distance between two beads
    lam : float
        Potential range, characteristic size
    m : float
        Softness of attractive contribution
    n : float
        Softness of repulsive contribution
    rc : float
        Interaction cut-off distance

    Returns
    -------
    ene
        Energy of interaction
    """
    ene = 0.0

    exp = np.exp(-m*(r-1)*(r-lam))
    if r < rc:
        ene += ( (1/r)**n + (1-exp)/(1+exp) - 1) / 2

    return ene

# Example WCA  potential
def pair_WCA(r,eps,sigma,rc):
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


## Potential to control repulsion and cohesive tail independently whilst preserving smoothness
def ELJcohesion(r, epc,sigma, rc):
    rcrep =2.0**(1./6.)*sigma
    if r < rcrep:
            
        return 0.0
            
    elif r <= rc:
            
        fbegin = force(partial(pair_LJ),rcrep,epc,sigma,rc)
            
        return pair_LJ(r,epc,sigma,rc)-pair_LJ(rc,epc,sigma,rc) + (r-rcrep)*fbegin
            
    else:
            
        return 0.0
            
def ELJrepulsion(r, epr, epc, sigma, rc):
    rcrep =2.0**(1./6.)*sigma
    if r <= rcrep:
            
        fendrep = force(partial(pair_LJ),rcrep,epr,sigma,rcrep)
        return pair_LJ(r,epr,sigma,rcrep) + ELJcohesion(rcrep,epc,sigma,rc) + epr + (r-rcrep)*fendrep

    if r > rcrep:

        return 0.0


def pair_rep_coh( r, epr, epc, sigma, rc):
    if r <= rc:
        return ELJcohesion( r, epc, sigma, rc) + ELJrepulsion( r, epr, epc, sigma, rc)
    else:
        return 0.0


# Version which is shifted and continuous at the tabulated cut-off
def pair_rep_coh_smooth_linear( r, epr, epc, sigma, rc, rctab):
    if r <= rc:
        fend = force(partial(pair_rep_coh), rctab, epr, epc, sigma, rc)
        return pair_rep_coh( r, epr, epc, sigma, rc) - pair_rep_coh( rctab, epr, epc, sigma, rc) + (r-rctab)*fend
    else:
        return 0.0


# Version which is smooth and scaled by a
def pair_rep_coh_smooth_linear_scaled(r, a, epr, epc, sigma, rc, rctab):
    return a*pair_rep_coh_smooth_linear( r, epr, epc, sigma, rc, rctab)
    
