"""
2D Directional Fourier filter test
"""
from iwatlas import sshdriver
import numpy as np
import matplotlib.pyplot as plt

# Interpolate the amplitude onto a grid prior to DFF
###
# xlims = (114,118)
# ylims = (-12,-9.35)

xlims = (122,124.5)
ylims = (-15,-12.43)

dx = 0.02

time = np.array([np.datetime64('2013-06-01 00:00:00'), np.datetime64('2013-09-01 00:00:00')])

# Filtering bands (degrees CCW from East)
thetalow = 0
thetahigh = 45

#######

basedir = '/home/suntans/cloudstor/Data/IWAtlas'
sshfile = '{}/NWS_2km_GLORYS_hex_2013_2014_SSHBC_Harmonics.nc'.format(basedir)
#ampfile = '{}/NWS_2km_GLORYS_hex_2013_2014_Amplitude_Atlas.nc'.format(basedir)
#climfile = '{}/NWS_2km_GLORYS_hex_2013_2014_Climatology.nc'.format(basedir)

ssh = sshdriver.load_ssh_clim(sshfile)

A_re_f, A_im_f, A_re, A_im, X, Y = sshdriver.extract_amp_nonstat_dff(ssh, xlims, ylims, dx, time,\
                    thetalow, thetahigh, A_re=None, A_im=None)

z = A_re[0,0,...] + 1j*A_im[0,0,...]
zf = A_re_f[0,0,...] + 1j*A_im_f[0,0,...]


plt.figure(figsize=(12,6))
plt.subplot(121)
plt.pcolormesh(X,Y,np.abs(z), cmap='Reds')
plt.colorbar()
plt.gca().set_aspect('equal')

plt.subplot(122)
plt.pcolormesh(X,Y,np.abs(zf), cmap='Reds')
plt.colorbar()
plt.tight_layout()
plt.gca().set_aspect('equal')
plt.show()


