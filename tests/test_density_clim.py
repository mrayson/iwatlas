"""
Test the density stratification loading routines
"""

import numpy as np 
import iwatlas.stratification as strat
import matplotlib.pyplot as plt

############
basedir = '/home/suntans/cloudstor/Data/IWAtlas'

# climfile = '{}/NWS_2km_GLORYS_hex_2013_2014_Climatology.nc'.format(basedir)
N2file = '{}/NWS_2km_GLORYS_hex_2013_2014_Stratification_Atlas.nc'.format(basedir)

# Test points
xpt = np.array([120.5,122.1])
ypt = np.array([-12.,-12.])

timept = np.array([np.datetime64('2013-07-01 00:00:00'), 
                   np.datetime64('2013-11-01 00:00:00'),
                   np.datetime64('2014-03-01 00:00:00')])

zout = np.arange(5,1005,10.)

###########

print('Calculating densty stratification data...')

N2_z = strat.predict_N2(N2file, xpt, ypt, timept ,zout)

plt.figure()
plt.plot(N2_z[:,0,0], -zout)
plt.plot(N2_z[:,0,1], -zout)
plt.plot(N2_z[:,0,2], -zout)
plt.legend(timept)
plt.show()
