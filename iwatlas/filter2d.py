"""
Directional filtering functions
"""
import numpy as np
from scipy.fftpack import fft2, ifft2, fftfreq

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