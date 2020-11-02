"""
Internal wave mode and parameter calculation routines

These are functions in (github.com/mrayson/iwaves)[iwaves]. Moved here to limit dependencies

"""

import numpy as np
from scipy import linalg, sparse

from sfoda.utils.mynumpy import grad_z

####
# Internal wave parameter calculations
####
def calc_alpha(phi, c, z):
    """Nonlinearity parameter"""
    phi_z = grad_z(phi, -z)
    num = 3*c*np.trapz( np.power(phi_z,3.), z, axis=0)
    den = 2*np.trapz(np.power(phi_z,2), z, axis=0)
    return num/den

def calc_beta(phi, c, z):
    """Dispersion parameter"""
    phi_z = grad_z(phi, -z)
    num = c*np.trapz( np.power(phi, 2.), z, axis=0)
    den = 2*np.trapz( np.power(phi_z, 2.), z, axis=0)
    
    return num/den

def ssh_to_amp_ratio(N2, phi, z):
    """
    SSH to internal tide amplitude ratio
    
    See Zhao 2016 Eq. A8
    
    $$
    \frac{SSH}{a_0} = \frac{1}{g}\int_{-H}^0 phi(z)N^2(z)dz
    $$
    """
    grav=9.81
    
    return 1/grav * np.trapz(phi*N2, z, axis=0)
    
def amp_to_ssh_ratio(N2, phi, z):
    
    return np.power(ssh_to_amp_ratio(N2, phi, z), -1.)


# 1D parameter calculations
def calc_alpha_1d(phi, c, dz):
    phi_z = np.gradient(phi,-dz)
    num = 3*c*np.trapz( phi_z**3., dx=dz)
    den = 2*np.trapz( phi_z**2., dx=dz)

    return num/den

def calc_beta_1d(phi, c, dz):
    phi_z = np.gradient(phi, dz)
    num = c*np.trapz( phi**2., dx=dz)
    den = 2*np.trapz( phi_z**2., dx=dz)

    return num/den


###
# Mode calculations
####
def calc_modes(N2, z, mode=0):
    """
    Wrapper for iwaves uneven mode calculation
    
    Inputs:
        N2: array [nz, ...] buoyancy frequency
        z: array [nz, ...]
        mode (default=0) vertical mode number to return
        
    Outputs:
        phi: array [nz, ...] mode shape
        c: array [...] eigenvalues
    """    
    
    sz = N2.shape
    nx = int(np.prod(sz[1:]))

    # Need to reshape so rows contain time and other dimensions are along the columns
    N2 = np.reshape(N2,(sz[0], nx))
    z = np.reshape(z,(sz[0], nx))
    
    phi_n = np.zeros_like(N2)
    cn = np.zeros((nx,))

    for ii in range(nx):
        phi, c = iwave_modes_uneven(N2[:,ii], z[:,ii])
        phi_n[:, ii] = phi[:, mode]
        cn[ii] = c[mode]
        
    return np.reshape(phi_n, sz), np.reshape(cn, sz[1:])


def iwave_modes_uneven(N2, z):
    """
    Calculates the eigenvalues and eigenfunctions to the internal wave eigenvalue problem:
    
    $$
    \left[ \frac{d^2}{dz^2} - \frac{1}{c_0} \bar{\rho}_z \right] \phi = 0
    $$
    
    with boundary conditions 
    """

    nz = N2.shape[0] 

    dz = np.zeros((nz,))
    zm = np.zeros((nz,))
    dzm = np.zeros((nz,))

    dz[0:-1] = z[0:-1] - z[1:]
    zm[0:-1] = z[0:-1] - 0.5*dz[0:-1]

    dzm[1:-1] = zm[0:-2] - zm[1:-1]
    dzm[0] = dzm[1]
    dzm[-1] = dzm[-2]

    A = np.zeros((nz,nz))
    for i in range(1,nz-1):
        A[i,i] = 1/ (dz[i-1]*dzm[i]) + 1/(dz[i]*dzm[i])
        A[i,i-1] = -1/(dz[i-1]*dzm[i])
        A[i,i+1] = -1/(dz[i]*dzm[i])

    # BC's
    eps = 1e-10
    A[0,0] = -1.
    A[0,1] = 0.
    A[-1,-1] = -1.
    A[-1,-2] = 0.

    # Construct the RHS matrix i.e. put N^2 along diagonals
    B = np.diag(N2,0)

    # Solve... (use scipy not numpy)
    w, phi = linalg.eig(A, b=B)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    phiall = phi[:,idx]

    # Normalize so the max(phi)=1
    for ii in range(nz):
        phi_1 = phiall[:,ii]
        phi_1 = phi_1 / np.abs(phi_1).max()
        phi_1 *= np.sign(phi_1.sum())
        phiall[:,ii] = phi_1

    return phiall, cn


