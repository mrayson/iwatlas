"""
Directional filtering functions
"""
import numpy as np
from scipy.fftpack import fft2, ifft2, fftfreq, fftshift, ifftshift

def dff2d(z, dx, thetalow, thetahigh):
    """
    Two-dimensional directional Fourier transform filter. 
    
    Inputs:
    ------
        z: 2D complex array
        dx: grid spacing
        thetalow: low angle for filter (degrees CCW from E)
        thetahigh: high angle for filter (degrees CCW from E)
        
    Outputs:
    -----
        zf: 2D filtered complex array
    """
    
    # 2D fourier transfrom
    My, Mx = z.shape
    Z = fft2(z)

    # Compute zonal wavenumbers
    k = fftfreq(Mx, dx/(2*np.pi))
    dk = 1/(Mx*dx)

    # Compute meridional wavenumbers
    l = fftfreq(My, dx/(2*np.pi))
    dl = 1/(My*dx)
    
    # Need to re-order the FFT output because positive frequencies are output first from the numpy DFT algorithm
    k_r = fftshift(k)
    l_r = fftshift(l)
    
    # Create a grid for the direction
    Lx,Ly = np.meshgrid(k_r,l_r)
    theta = np.angle(Lx + 1j*Ly)

    thetadeg = np.mod(theta*180/np.pi,360)

    # Create the filter matrix
    H = np.zeros_like(thetadeg)
    filter_idx = (thetadeg > thetalow) & (thetadeg < thetahigh)
    H[filter_idx] = 1

    # Now reorder H into the original FFT ordering
    H_r = ifftshift(H,axes=1)
    H_r = ifftshift(H_r,axes=0)

    # Finally, filter
    zf = ifft2(Z*H_r)
    
    return zf
 

def hilbert_2d(z, dx, dy):
    My, Mx = z.shape
    Z = fft2(z)

    # Compute zonal frequencies
    k = fftfreq(Mx, dx/(2*np.pi))
    dk = 1/(Mx*dx)

    # Compute meridional frequencies
    l = fftfreq(Mx, dx/(2*np.pi))
    dl = 1/(My*dy)
    
    # Create filter matrices for each of the four quadrant
    Z_posk_posl = np.zeros_like(Z)
    Z_posk_posl[:My//2, :Mx//2] = Z[:My//2, :Mx//2] 

    z_posk_posl = ifft2(Z_posk_posl)

    Z_posk_negl = np.zeros_like(Z)
    Z_posk_negl[:My//2, Mx//2::] = Z[:My//2, Mx//2::] 

    z_posk_negl = ifft2(Z_posk_negl)

    Z_negk_negl = np.zeros_like(Z)
    Z_negk_negl[My//2::, Mx//2::] = Z[My//2::, Mx//2::] 

    z_negk_negl = ifft2(Z_negk_negl)

    Z_negk_posl = np.zeros_like(Z)
    Z_negk_posl[My//2::, :Mx//2] = Z[My//2::, :Mx//2] 

    z_negk_posl = ifft2(Z_negk_posl)
    
    return z_posk_posl, z_posk_negl, z_negk_negl, z_negk_posl
