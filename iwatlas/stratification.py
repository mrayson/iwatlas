"""
Functions for handling density stratification data

Main functions:
    `predict_N2`
    `fit_rho_lsq`
"""

import numpy as np
from scipy.optimize import least_squares

from iwatlas import sshdriver

# Global variable
BASETIME = np.datetime64('2000-01-01 00:00:00')

#####
# Parametric profiles
#####
def double_gaussian_N2(z, beta):
    return beta[0,...] + beta[1,...] * (np.exp(- ((z+beta[2,...])/beta[3,...])**2 )  +\
              np.exp(-((z+beta[4,...])/beta[5,...])**2 ) )

def sech(z):
    return 1. / np.cosh(z)

def sech2(z):
    return sech(z)*sech(z)

def double_sech2_N2(z, beta):
    return beta[0] + beta[1] * (sech2((z+beta[2])/beta[3])   +\
              sech2((z+beta[4])/beta[5]) )

#######
# Prediction routines
#######
def harmonic_pred_N2(aa, Aa, Ba, omega, tdays):
    # aa dimension: [nparams, nx]
    # Aa/Ba dimensions: [nomega, nparams, nx]
    # output dimensions: [nparams, nx, nt]
    nomega = len(omega)
    nt = tdays.shape[0]
    amp = aa[...,None] * np.ones_like(tdays)[None,...] 

    tt =  np.ones_like(amp) * tdays 

    for ii in range(nomega):
        amp += Aa[ii,...,None]*np.cos(omega[ii]*tt) + Ba[ii,...,None]*np.sin(omega[ii]*tt)
    
    return amp


def predict_N2(N2file, xpt, ypt, timept, zout):
    """
    Return a buoyancy frequency squared (N^2) vertical profile at a space-time location of choice
    
    Inputs:
    ---
        N2file: filename of the stratification climatology dataset (NWS_2km_GLORYS_hex_2013_2014_Stratification_Atlas.nc)
        xpy,ypt: vectors [nx] of output space points
        timept: vector [nt] of datetime64 time objects
        zout: vector [nz] of output vertical locations (should be positive)
    
    Returns:
        N2_z: array of buoyancy frequency [nz, nx, nt]
    
    """
    
    ds_N2 = sshdriver.load_ssh_clim(N2file)
    
    # Check inputs
    zout = np.abs(zout)
    tsec = (timept - BASETIME).astype('timedelta64[s]').astype(float)

    assert xpt.shape == ypt.shape
    assert zout.ndim == 1
    assert xpt.ndim == 1
    assert timept.ndim == 1

    
    # Step 1: Interpolate the N2 beta parameters in space
    omega = ds_N2._ds.omega.values
    na = ds_N2._ds.dims['Ntide']
    nparams = ds_N2._ds.dims['Nparams']

    nx = xpt.shape
    N2_mu = np.zeros((nparams,)+nx)
    N2_re = np.zeros((na,nparams,)+nx)
    N2_im = np.zeros((na,nparams,)+nx)

    for jj in range(nparams):
        N2_mu[jj,...] = ds_N2.interpolate(ds_N2._ds['N2_mu'][jj,...], xpt, ypt, kind='linear')
        for nn in range(na):
            N2_re[nn,jj,...] = ds_N2.interpolate(ds_N2._ds['N2_re'][nn,jj,...], xpt, ypt, kind='linear')
            N2_im[nn,jj,...] = ds_N2.interpolate(ds_N2._ds['N2_im'][nn,jj,...], xpt, ypt, kind='linear')
        
    # Step 2: reconstruct the time-series
    N2_t = harmonic_pred_N2(N2_mu, N2_re, N2_im, omega, tsec)
    
    # Step 4: reconstruct in the vertical direction
    zpr = -np.log(zout)

    return double_gaussian_N2(zpr[:,None,None], N2_t)

######
# Fitting routines
######
def rho_err(coeffs, rho, z, density_func):
    """
    Returns the difference between the estimated and actual data
    """
    soln = density_func(z, coeffs)

    return rho - soln

def fit_rho_lsq(rho, z, density_func, bounds, initguess):
    """
    Fits an analytical density/N^2 profile to data
    Uses a robust linear regression
    Inputs:
    ---
        rho: vector of density (or N^2) [Nz]
        z : depth [Nz] w/ negative values i.e. 0 at surface, positive: up
        density_func, bounds, initguess: 
    Returns:
    ---
        rhofit: best fit function at z locations
        f0: tuple with analytical parameters
        err: L2-norm of the error vector
    """
    status = 0

    # Use "least_squares" at it allows bounds on fitted parameters to be input
    H = np.abs(z).max()   

    soln =\
        least_squares(rho_err, initguess, args=(rho, z, density_func), \
        bounds=bounds,\
        xtol=1e-10,
        ftol=1e-10,
        loss='cauchy', f_scale=0.1, # Robust
        verbose=0,
        )
    f0 = soln['x']

    rhofit = density_func(z, f0)
    
    err = np.linalg.norm(rhofit - rho)

    return rhofit, f0, err



