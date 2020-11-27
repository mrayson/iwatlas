"""
Driver routines to output velocity predictions at arbitrary space-time points

Requires SFODA (github.com/mrayson/sfoda.git)
"""

import numpy as np

from . import sshdriver
from . import stratification as strat
from . import iwaves

from sfoda.utils.barycentric import BarycentricInterp

def calc_coriolis(latdeg):
    omega = 2*np.pi/86400.
    degrad = np.pi/180.
    return 2*omega*np.sin(latdeg*degrad)

def calc_u_complex(eta_x, eta_y, omega, f, g=9.81, tau=1e6):
    omegaT = omega + 1j/tau
    num = -1j*omegaT*g*eta_x + f*g*eta_y
    den = omegaT**2. - f**2.
    return num/den

def calc_v_complex(eta_x, eta_y, omega, f, g=9.81, tau=1e6):
    omegaT = omega + 1j/tau
    num = -1j*omegaT*g*eta_y - f*g*eta_x
    den = omega**2. - f**2.
    return num/den

def extract_hc_uv_spatial(sshfile):
    """
    Compute the complex u and v amplitude matrices
    """
    ssh = sshdriver.load_ssh_clim(sshfile)
    
    # Step 1: compute the spatial gradients of eta
    omega = ssh._ds['omega'].values
    f_cor = calc_coriolis(ssh.yv)

    ntide = omega.shape[0]

    # Load the full matrix
    eta_re = ssh._ds['SSH_BC_Aa'][...].values
    eta_im = ssh._ds['SSH_BC_Ba'][...].values

    # Calculate the coriolis
    f_cor = calc_coriolis(ssh.yv)

    u = np.zeros((ntide,ssh.Nc), np.complex128)
    v = np.zeros((ntide,ssh.Nc), np.complex128)

    for ii in range(ntide):
        eta_re_dx, eta_re_dy = ssh.calc_grad(eta_re[ii,:])
        eta_im_dx, eta_im_dy = ssh.calc_grad(eta_im[ii,:])

        u[ii,:] = calc_u_complex(eta_re_dx+1j*eta_im_dx, eta_re_dy+1j*eta_im_dy, omega[ii], f_cor)
        v[ii,:] = calc_v_complex(eta_re_dx+1j*eta_im_dx, eta_re_dy+1j*eta_im_dy, omega[ii], f_cor)
        
    return u, v, omega

def predict_uv(sshfile, x, y, time, kind='linear'):
    """
    Perform harmonic predictions of the u/v velocity amplitude at the points in x and y and time
    """
    ssh = sshdriver.load_ssh_clim(sshfile)
    
    # Calculate complex velocity amplitudes from ssh
    u,v, omega = extract_hc_uv_spatial(ssh)
    
    # Mean velocity is zero
    a0 = np.zeros((ssh.Nc,))
    
    # Interpolate the amplitudes in space and reconstruct the time-series
    aa, Aa, Ba, frq = sshdriver.extract_amp_xy(ssh, x, y, a0, np.real(u), np.imag(u), kind=kind )
    ut = sshdriver.predict_scalar( time, aa, Aa, Ba, omega)

    aa, Aa, Ba, frq = sshdriver.extract_amp_xy(ssh, x, y, a0, np.real(v), np.imag(v), kind=kind )
    vt = sshdriver.predict_scalar( time, aa, Aa, Ba, omega)
    
    return ut, vt

def predict_uv_z(sshfile, x, y, time, nz=80, mode=0, kind='linear'):
    """
    Predict the full-depth profile velocity
    """
    
    ssh = sshdriver.load_ssh_clim(sshfile)
    
    ut, vt = predict_uv(ssh, x, y, time, kind=kind)

    # Only compute N^2 at a few time steps
    N2_z, zout = strat.predict_N2(ssh, x, y, time, nz)
    
    # Mode shapes
    phi_n, cn = iwaves.calc_modes(N2_z, zout, mode=mode)
    
    # Calculate the vertical gradient of the modal structure function and normalize
    dphi_dz = iwaves.grad_z(phi_n, zout)
    dphi_dz_norm = dphi_dz/ dphi_dz.max(axis=0)[None,...]
    
    # Compute the velocity profile
    uz = dphi_dz_norm * ut.T 
    vz = dphi_dz_norm * vt.T 
    
    return uz, vz, zout

def extract_uv_dff(sshfile, xlims, ylims, dx, 
                    thetalow, thetahigh):
    """
    Extract the non-stationary velocity amplitude harmonic paramters 
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
    ssh = sshdriver.load_ssh_clim(sshfile)

    # Interpolate the amplitude onto a grid prior to DFF
    xgrd  = np.arange(xlims[0], xlims[1]+dx, dx)
    ygrd  = np.arange(ylims[0], ylims[1]+dx, dx)
    X,Y = np.meshgrid(xgrd, ygrd)
    My, Mx = X.shape
    
    u, v, omega = extract_hc_uv_spatial(ssh)
    ntide, Nc = u.shape
    
    # Interpolate
    aain = np.zeros((Nc,))
    _, u_re, u_im, _ = sshdriver.extract_amp_xy(ssh, X, Y, aain, np.real(u), np.imag(u), kind='linear')
    _, v_re, v_im, _ = sshdriver.extract_amp_xy(ssh, X, Y, aain, np.real(v), np.imag(v), kind='linear')
    
    u_re_f, u_im_f = sshdriver.extract_dff(u_re, u_im, X, Y, dx, thetalow, thetahigh)
    v_re_f, v_im_f = sshdriver.extract_dff(v_re, v_im, X, Y, dx, thetalow, thetahigh)

    return u_re_f, u_im_f, v_re_f, v_im_f, X, Y, omega

def extract_uv_dff_bc(sshfile, xline, yline, thetalow, thetahigh, dx=0.02, xyrange=1.5):
    """
    Extract the velocity amplitude at points along a line. Performs a DFF
    so that velocity is within the arc of propagation.
    
    Use this function to extract boundary conditions for a numerical model, for example.
    
    Inputs:
    ------
        ssh: sunxray object OR file string
        xline, yline: vectors of output points
        thetalow: low angle for filter (degrees CCW from E)
        thetahigh: high angle for filter (degrees CCW from E)
        
    Outputs:
    -----
        U_re, U_im, V_re, U_im: 2D filtered complex array
        omega: frequencies
    """
    
    ssh = sshdriver.load_ssh_clim(sshfile)
    
    xlims = (xline.min()-xyrange, xline.max()+xyrange)
    ylims = (yline.min()-xyrange, yline.max()+xyrange)

    u_re_f, u_im_f, v_re_f, v_im_f, X, Y, omega = extract_uv_dff(ssh, xlims, ylims, dx, thetalow, thetahigh)
    
    # Interpolate the results onto the line
    Fi = BarycentricInterp(np.array([X.ravel(), Y.ravel()]).T, np.array([xline,yline]).T )

    ntide, ny, nx = u_re_f.shape
    nline = xline.shape[0]
    U_re = np.zeros((ntide, nline))
    U_im = np.zeros((ntide, nline))
    V_re = np.zeros((ntide, nline))
    V_im = np.zeros((ntide, nline))

    for ii in range(ntide):
        U_re[ii,...] = Fi(u_re_f[ii,...].ravel())
        U_im[ii,...] = Fi(u_im_f[ii,...].ravel())
        V_re[ii,...] = Fi(v_re_f[ii,...].ravel())
        V_im[ii,...] = Fi(u_im_f[ii,...].ravel())
        
    return U_re, U_im, V_re, V_im, omega