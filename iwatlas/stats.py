"""
Statistical metrics used for harmonic analysis
"""

import numpy as np

def harmonic_indices(ntide, na):
    """
    Return the indices for each central (tidal) frequency
    """
    ii=[]
    for ff in range(ntide):
        ii.append((ff+1)*(2*na+1)-na-1) # Index of the fixed harmonics
    
    return ii

def TVFH(amp2, varin):
    """
    Total Variance Fraction explained by the Harmonic model
    """
    return 0.5 * np.sum(amp2, axis=0) / varin * 100

def SVFH(amp2, varin, ntide, na):
    """
    Stationary variance fraction (i.e. the variance fraction of the fixed harmonics)
    """ 
    ii = harmonic_indices(ntide, na)
    return 0.5 * np.sum(amp2[ii,...], axis=0) / varin * 100

def VF_m(amp2, m, ntide, na):
    """
    Variance fraction within a specific frequency band including annual harmonics
    """
    ii = harmonic_indices(ntide, na)
    return  np.sum(amp2[ii[m]-na:ii[m]+na+1,...], axis=0) / np.sum(amp2, axis=0) * 100


def NSVF_m(amp2, m, ntide, na):
    """
    Nonstationary variance fraction of a specific frequency band
    """
    ii = harmonic_indices(ntide, na)

    M2 = np.sum(amp2[ii[m]-na:ii[m]+na+1,...], axis=0)
    return  (M2 - amp2[ii[m],...])/ M2 * 100
