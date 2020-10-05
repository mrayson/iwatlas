"""
Driver routines to output SSH predictions at arbitrary space-time points

 Requires SODA (github.com/mrayson/soda.git)
"""

import numpy as np
from datetime import datetime

from soda.dataio.suntans.sunxray import Sunxray
from soda.utils.myproj import MyProj

from .harmonics import harmonic_pred

def load_ssh_clim(sshfile):
    """
    Load the climatological atlas as a SUNTANS xarray data object
    """
    sun = Sunxray(sshfile,)

    # Reproject into lat/lon
    P = MyProj('merc')
    sun.xp,sun.yp = P.to_ll(sun.xp, sun.yp)
    sun.xv,sun.yv = P.to_ll(sun.xv, sun.yv)
    sun._xy = None
    
    return sun


def extract_hc_ssh(sshfile, x,y,kind='nearest'):
    """
    Extract harmonic consituents from the internal tide SSH atlas
    """
    sun = load_ssh_clim(sshfile)
    
    ntide = sun._ds.Ntide.shape[0]

    if isinstance(x, float):
        aa = np.zeros((1,))
        Aa = np.zeros((ntide,)) 
        Ba = np.zeros((ntide,)) 
    
    elif isinstance(x, np.ndarray):
        sz = x.shape
        aa = np.zeros((1,)+sz)
        Aa = np.zeros((ntide,)+sz) 
        Ba = np.zeros((ntide,)+sz)
        
    aa[0] = sun.interpolate(sun._ds.SSH_BC_aa.values,x,y, kind=kind)
    for ii in range(ntide):
        Aa[ii,...] = sun.interpolate(sun._ds.SSH_BC_Aa.values[ii,:], x,y, kind=kind)
        Ba[ii,...] = sun.interpolate(sun._ds.SSH_BC_Ba.values[ii,:], x,y, kind=kind)
        
    return aa, Aa, Ba, sun._ds.omega.values

def predict_ssh(sshfile, x, y, time, kind='nearest'):
    """
    Perform harmonic predictions at the points in x and y
    """
    
    aa, Aa, Ba, frq = extract_hc_ssh(sshfile, x,y,kind='nearest')
    
    # Get the time in seconds
    tsec = (time - np.datetime64('1990-01-01 00:00:00')).astype('timedelta64[s]').astype(float)
    
    # Need to reshape the time vector for matrix multiplications
    if isinstance(aa,float):
        tsec = tsec
    elif isinstance(aa, np.ndarray):
        ndim = aa.ndim
        if ndim == 1:
            tsec = tsec
        elif ndim == 2:
            tsec = tsec[:,None]
        elif ndim == 3:
            tsec = tsec[:,None, None]
        else:
            raise Exception('unsupported number of dimension in x matrix')
    
    # Do the actual prediction
    return  harmonic_pred(aa, Aa, Ba, frq, tsec)
