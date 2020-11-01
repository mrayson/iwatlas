"""
Internal wave mode and parameter calculation routines

These are functions in (github.com/mrayson/iwaves)[iwaves]. Moved here to limit dependencies

"""

import numpy as np
from scipy import linalg, sparse


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


