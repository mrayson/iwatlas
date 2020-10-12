"""
Test a standalone suntans xarray like object
"""

from iwatlas.sshdriver import load_ssh_clim

#@xr.register_dataset_accessor("ugrid")
#class SunDataset:
#    def __init__(self, xarray_obj):
#            self._ds = xarray_obj
#            self._center = None
#
#    @property
#    def center(self):
#        """Return the geographic center point of this dataset."""
#        if self._center is None:
#            # we can use a cache on our accessor objects, because accessors
#            # themselves are cached on instances that access them.
#            lon = self._obj.latitude
#        return self._center
#
#    def plot(self):
#        """
#        Plot data
#        """
#        print('Plotting!')
#
#def open_ugrid_nc(ncfile, **kwargs):
#    """
#    Open a netcdf file as an xarray object that inherits from a Dataset
#    """
#    return xr.open_dataset(ncfile, **kwargs)

##################

basedir = '/home/suntans/cloudstor/Data/IWAtlas'
sshfile = '{}/NWS_2km_GLORYS_hex_2013_2014_SSHBC_Harmonics.nc'.format(basedir)
ampfile = '{}/NWS_2km_GLORYS_hex_2013_2014_Amplitude_Atlas.nc'.format(basedir)
climfile = '{}/NWS_2km_GLORYS_hex_2013_2014_Climatology.nc'.format(basedir)
#ds = open_ugrid_nc(ncfile)
ssh = load_ssh_clim(sshfile)
amp = load_ssh_clim(ampfile)
clim = load_ssh_clim(climfile)

print(ssh, amp, clim)
