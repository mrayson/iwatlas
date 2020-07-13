"""
Nonstationary harmonic fitting functions

The nonstationary model extends the tidal harmonic approach by allowing the amplitudes 
to vary slowly in time at a much lower frequency. Here we use an annual frequency 
($\omega_A=2\pi / 365.25$ d$^{-1}$) and its first $n$ harmonics 

$$
a_i = \sum_m \alpha_{m,i} \cos(\omega _m t_i) + \beta_{m,i}\sin(\omega_m t_i) + \varepsilon_a \,,
$$
where the real and imaginary amplitudes are now
$$
\alpha_{m,i} =  \sum_{n=0}^3 \hat{\alpha}_n \cos(n\omega _A t_i) + \hat{\beta}_n\sin(n\omega_A t_i)\,,
$$
and 
$$
\beta_{m,i} =  \sum_{n=0}^3 \tilde{\alpha}_n \cos(n\omega _At_i) + \tilde{\beta}_n\sin(n\omega_A t_i) \,,
$$
respectively. The amplitudes vary in time and capture the seasonal modulation of the tidal harmonics. Eqs. (\ref{eq:a0_amp}) - (\ref{eq:a0_amp_im}) are combined (using a trigonometric identity) to give
$$
a_t = \sum_m \sum_{n=-3}^3  A_{m,n}  \cos( [\omega _m +n \omega_A ]t_i) +
B_{m,n}\sin( [\omega _m +n \omega_A ]t_i) + \varepsilon_{a} .
$$
"""

import numpy as np

from soda.utils.harmonic_analysis import harmonic_fit_array


## Functions (to go into iwatlas module)
twopi = 2*np.pi
tdaysec = 86400.

def harmonic_pred(aa, Aa, Ba, omega, tdays):
    nomega = len(omega)
    nt = tdays.shape[0]
    amp = np.ones_like(tdays) * aa
    for ii in range(nomega):
        amp += Aa[ii,...]*np.cos(omega[ii]*tdays) + Ba[ii,...]*np.sin(omega[ii]*tdays)
    
    return amp

def nonstat_harmonic_fit(X, t, omega, na, omega_A=twopi/(365*tdaysec)):
    frq_all =[]
    for ff in omega:
        for n in range(-na,na+1):
            frq_all.append(ff+n*omega_A)

    Y = harmonic_fit_array(X, t, frq_all, axis=0)
    aa = Y[0]
    Aa = Y[1::2]
    Ba = Y[2::2]
    
    return aa, Aa, Ba, frq_all

def harmonic_to_seasonal(Aa, Ba, na, ntide):
    """
    Convert from harmonic to seasonal form
    """

    assert Aa.shape[0] == (ntide)*(2*na+1)
    assert Ba.shape[0] == (ntide)*(2*na+1)

    alpha_hat = np.zeros((ntide,(na+1)))
    alpha_tilde = np.zeros((ntide,(na+1)))
    beta_hat = np.zeros((ntide,(na+1)))
    beta_tilde = np.zeros((ntide,(na+1)))

    for ff in range(ntide):
        ii = (ff+1)*(2*na+1)-na-1 # Location of the fixed harmonic

        alpha_hat[ff,0] = Aa[ii]
        alpha_tilde[ff,0] = Ba[ii]
        for n in range(1,na+1):
            alpha_hat[ff,n] = Aa[ii-n] + Aa[ii+n]
            beta_hat[ff,n] = Ba[ii+n] - Ba[ii-n]
            alpha_tilde[ff,n] = Ba[ii-n] + Ba[ii+n]
            beta_tilde[ff,n] = Aa[ii-n]-Aa[ii+n]  
    
    return alpha_hat, beta_hat, alpha_tilde, beta_tilde

def seasonal_amp(a_hat, b_hat, a_tilde, b_tilde, t, omega_A=twopi/(365*tdaysec)):
    """
    Calculate the real and imaginary modulated (seasonal) amplitudes
    """
    nf, na = a_hat.shape
    nt = t.shape[0]

    A_re = np.zeros((nf,nt))
    A_im = np.zeros((nf,nt))

    for ff in range(nf):
        for n in range(na):
            A_re[ff,:] += a_hat[ff,n]*np.cos(n*omega_A*t) + b_hat[ff,n]*np.sin(n*omega_A*t)
            A_im[ff,:] += a_tilde[ff,n]*np.cos(n*omega_A*t) + b_tilde[ff,n]*np.sin(n*omega_A*t)
    
    return A_re, A_im
