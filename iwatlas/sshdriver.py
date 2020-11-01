"""
Driver routines to output SSH predictions at arbitrary space-time points

 Requires SODA (github.com/mrayson/soda.git)
"""

import numpy as np
from datetime import datetime

from sfoda.suntans.sunxray import Sunxray
from sfoda.utils.myproj import MyProj

from . import harmonics
from .filter2d import dff2d

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


def extract_hc_ssh(sshfile, x,y, sun=None, kind='nearest'):
    """
    Extract harmonic consituents from the internal tide SSH atlas
    """
    if sun is None:
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
    return  harmonics.harmonic_pred(aa, Aa, Ba, frq, tsec)

def extract_amp_nonstat(sshobj, xpt, ypt, time, kind='linear'):
    """
    Extract time-varying (nonstationary) amplitude time-series for each tidal frequency
    """

    basetime = np.datetime64(sshobj._ds.attrs['ReferenceDate'])
    tsec = (time - basetime).astype('timedelta64[s]').astype(float)
    
    aa, Aa, Ba, omega = extract_hc_ssh(None, xpt, ypt, kind=kind, sun=sshobj)
    
    na = sshobj._ds.attrs['Number_Annual_Harmonics']
    ntide = sshobj._ds.dims['Ntide']//(2*na+1)

    alpha_hat, beta_hat, alpha_tilde, beta_tilde =\
        harmonics.harmonic_to_seasonal(Aa, Ba, na, ntide)

    A_re, A_im = harmonics.seasonal_amp(alpha_hat, beta_hat, alpha_tilde, beta_tilde, tsec )
    
    return A_re, A_im
def extract_amp_nonstat_dff(ssh, xlims, ylims, dx, time,\
                    thetalow, thetahigh, A_re=None, A_im=None):
    """
    Extract the non-stationary amplitude over a region and perform a
    direction Fourier filter (DFF)
    
    Inputs:
    ------
        ssh: sunxray object
        xlims, ylims: tuples with lower and upper x/y limites
        dx: output grid spacing (interpolate onto this spacing)
        time: output time step
        thetalow: low angle for filter (degrees CCW from E)
        thetahigh: high angle for filter (degrees CCW from E)
        
    Outputs:
    -----
        A_re_f, A_im_f: 2D filtered complex array
    """

    # Interpolate the amplitude onto a grid prior to DFF
    xgrd  = np.arange(xlims[0], xlims[1]+dx, dx)
    ygrd  = np.arange(ylims[0], ylims[1]+dx, dx)
    X,Y = np.meshgrid(xgrd, ygrd)
    My, Mx = X.shape

    # Interpolate (need to flatten the spatial dimension)
    if A_re is None:
        A_re, A_im = extract_amp_nonstat(ssh, X.ravel(), Y.ravel(), time, kind='linear')
        ntide, ntime, ss = A_re.shape
        A_re = A_re.reshape((ntide, ntime, My, Mx))
        A_im = A_im.reshape((ntide, ntime, My, Mx))
    
    ntide, ntime, My, Mx = A_re.shape
    
    # Prepare the output array
    A_re_f = np.zeros_like(A_re)
    A_im_f = np.zeros_like(A_im)
    
    # Loop through and perform the DFF on each 2D slice
    for nn in range(ntide):
        for ii in range(ntime):
            z_f = dff2d(A_re[nn,ii,...] + 1j*A_im[nn,ii,...], dx, thetalow, thetahigh)
            A_re_f[nn,ii,...] = z_f.real
            A_im_f[nn,ii,...] = z_f.imag
            
    
    return A_re_f, A_im_f, A_re, A_im, X, Y


