"""
Driver routines to output SSH predictions at arbitrary space-time points

 Requires SFODA (github.com/mrayson/sfoda.git)
"""

import numpy as np
from datetime import datetime
from scipy.interpolate import RectBivariateSpline


from sfoda.suntans.sunxray import Sunxray
from sfoda.utils.myproj import MyProj

from . import harmonics
from .filter2d import dff2d

def load_ssh_clim(sshfile):
    """
    Load the climatological atlas as a SUNTANS xarray data object
    
    Input:
    ---
        sshfile: atlas netcdf file string OR Sunxray object
    """
    if isinstance(sshfile, Sunxray):
        return sshfile
    
    elif isinstance(sshfile, str):
        sun = Sunxray(sshfile,)

        # Reproject into lat/lon
        P = MyProj('merc')
        sun.xp,sun.yp = P.to_ll(sun.xp, sun.yp)
        sun.xv,sun.yv = P.to_ll(sun.xv, sun.yv)
        sun._xy = None

        return sun
    else:
        raise Exception('Unknown type {}'.format(type(sshfile)))


def extract_hc_ssh(sshfile, x,y, sun=None, kind='linear'):
    """
    Extract harmonic consituents from the internal tide SSH atlas
    """
    #if sun is None:
    # This function can accept a Sunxray object
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

def predict_ssh(sshfile, x, y, time, kind='linear'):
    """
    Perform harmonic predictions at the points in x and y
    """
    
    aa, Aa, Ba, frq = extract_hc_ssh(sshfile, x,y,kind=kind)
    
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

def extract_amp_nonstat(sshfile, xpt, ypt, time, kind='linear'):
    """
    Extract time-varying (nonstationary) amplitude time-series for each tidal frequency
    """
    sshobj = load_ssh_clim(sshfile)


    basetime = np.datetime64(sshobj._ds.attrs['ReferenceDate'])
    tsec = (time - basetime).astype('timedelta64[s]').astype(float)
    
    aa, Aa, Ba, omega = extract_hc_ssh(sshobj, xpt, ypt, kind=kind)
    
    na = sshobj._ds.attrs['Number_Annual_Harmonics']
    ntide = sshobj._ds.dims['Ntide']//(2*na+1)

    alpha_hat, beta_hat, alpha_tilde, beta_tilde =\
        harmonics.harmonic_to_seasonal(Aa, Ba, na, ntide)

    A_re, A_im = harmonics.seasonal_amp(alpha_hat, beta_hat, alpha_tilde, beta_tilde, tsec )
    
    return A_re, A_im

def extract_amp_dff(sshfile, xlims, ylims, dx, 
                    thetalow, thetahigh, A_re=None, A_im=None):
    """
    Extract the non-stationary amplitude harmonic paramters 
    for a region and perform a directional Fourier filter (DFF).
    
    Use this function to extract directional amplitudes of ALL harmonics
    in a dataset.
    
    Inputs:
    ------
        ssh: sunxray object OR file string
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
    
    aa, A_re, A_im, omega = extract_hc_ssh(sshfile, X, Y, kind='linear')
    
    ntide, My, Mx = A_re.shape
    
    # Zero out any nan's
    A_re[np.isnan(A_re)] = 0
    A_im[np.isnan(A_im)] = 0
    
    # Prepare the output array
    A_re_f = np.zeros_like(A_re)
    A_im_f = np.zeros_like(A_im)
    
    # Loop through and perform the DFF on each 2D slice
    for nn in range(ntide):
        z_f = dff2d(A_re[nn,...] + 1j*A_im[nn,...], dx, thetalow, thetahigh)
        A_re_f[nn,...] = z_f.real
        A_im_f[nn,...] = z_f.imag
            
    
    return A_re_f, A_im_f, A_re, A_im, X, Y, omega

def extract_ssh_point_dff(sshfile, x0, y0, timeout, thetalow, thetahigh, 
                    xyrange=2.0, dx=2.0 ):
    """
    Extract the a time-series of SSH at a point that is propagating in a given direction
    
    Inputs:
    ------
        sshfile: sunxray object OR file string
        x0, y0: scalar lon/lat output point
        timeout: output time step
        thetalow: low angle for filter (degrees CCW from E)
        thetahigh: high angle for filter (degrees CCW from E)
        xyrange: (optional) range for box that surrounds the point to perform DFF (default = 2.0 i.e. box is 4x4 degrees)
        dx: (optional, default=0.02 degress) output grid spacing (interpolate onto this spacing)

        
    Outputs:
    -----
        ssh_pt: time-series of SSH at the point
    """
    sshobj = load_ssh_clim(sshfile)
    
    xlims = (x0-xyrange, x0+xyrange)
    ylims = (y0-xyrange, y0+xyrange)


    # Convert the time
    reftime = np.datetime64(sshobj._ds.attrs['ReferenceDate'])
    tsec = (timeout - reftime).astype('timedelta64[s]').astype(float)

    # Extract the amplitude for a region and do the DFF
    A_re_f, A_im_f, A_re, A_im, X, Y, omega = extract_amp_dff(sshobj, xlims, ylims, dx, \
                        thetalow, thetahigh, A_re=None, A_im=None)

    # Interpolate the DFF result back onto the point of interest
    nf, ny, nx = A_re_f.shape

    A_re_pt = np.zeros((nf,))
    A_im_pt = np.zeros((nf,))
    for ff in range(nf):
        F = RectBivariateSpline( Y[:,0], X[0,:], A_re_f[ff,...])
        A_re_pt[ff] = F(y0,x0)
        F = RectBivariateSpline( Y[:,0], X[0,:], A_im_f[ff,...])
        A_im_pt[ff] = F(y0,x0)

    # Generate a time-series
    ssh_pt_f = harmonics.harmonic_pred(0, A_re_pt, A_im_pt, omega, tsec)
    
    return ssh_pt_f

def extract_amp_nonstat_dff(sshfile, xlims, ylims, dx, time,\
                    thetalow, thetahigh, A_re=None, A_im=None):
    """
    Extract the non-stationary amplitude over a region and perform a
    direction Fourier filter (DFF).
    
    Use this function to extract a spatial snapshot of the amplitude
    of a fundamental tidal frequency (e.g. M2) for a given time snapshot.
    
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
    ssh = load_ssh_clim(sshfile)

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
    
    # Zero out any nan's
    A_re[np.isnan(A_re)] = 0
    A_im[np.isnan(A_im)] = 0
    
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

####
# Grid helper functions
def calc_scoord_log(Nz, rfac):
    """
    Return an s-coordinate vector with logarithmic stretching
    """
    s0 = (rfac-1)/(np.power(rfac, Nz-1)-1)
    scoord = np.zeros((Nz,))
    scoord[1] = s0
    for ii in range(2,Nz):
        scoord[ii] =  scoord[ii-1]*rfac

    return np.cumsum(scoord)

def return_zcoord_3d(sshfile, xpt, ypt, nt, nz, scoord=None, rfac=1.04):
    """
    Create a vertical grid array
    
    Inputs:
    ---
        sshfile: Sunxray object or filename of the stratification climatology dataset (NWS_2km_GLORYS_hex_2013_2014_Stratification_Atlas.nc)
        xpt,ypt: vectors [nx] of output space points
        nt: scalar, number of time points
        nz: scalar, number of vertical layers
        scoord: (optional) vector nx, locations of vertical nodes in non-dimensional space i.e. between 0 and 1
        rfac: (optional) scalar 1 to 1.1 logarithmic scaling factor for the vertical coordinate
    
    Returns:
        zout: array of buoyancy frequency [nz, nx, nt]
    """
    ssh = load_ssh_clim(sshfile)

    # Get the depths
    h = ssh.interpolate(ssh._ds.dv, xpt, ypt)

    hgrd = h[:,None] * np.ones((nt,))[None,:]
    
    if scoord is None:
        scoord = calc_scoord_log(nz, rfac)

    return scoord[:,None,None] * hgrd[None, ...]



