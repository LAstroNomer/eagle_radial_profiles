from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.interpolate import interp1d
import astropy.units as u
import numpy as np
import warnings


def get_mag_zp(fname):
    with warnings.catch_warnings(action="ignore"):
        header = fits.getheader(fname)
        zp = header['FPA.ZP']
        exptime = header['EXPTIME']
        return zp + 2.5*np.log10(exptime)


def get_pixscale(fname):
    with warnings.catch_warnings(action="ignore"):
        header = fits.getheader(fname)
        wcs    = WCS(header)
        pixscale = proj_plane_pixel_scales(wcs)   # deg
        return np.mean(pixscale)*3600. # to arcsec


def flux_to_sb(image, pixscale):
    return image/pixscale**2


def sb_to_flux(image, pixscale):
    return image*pixscale**2

def calc_ith_isophote_radius(r, I, zp, mu_):
    mu = zp-2.5*np.log10(I)
    f = interp1d(r, mu, kind='cubic')
    t = np.arange(np.min(r), np.max(r), 0.5)
    r_ = t[np.argmin(abs(f(t) - mu_))]
    return r_

def AB_mag(data):

    # init MJy/sr

    data = data * 10**6 * u.Jy /u.sr

    # convert to Jy/arcsec**2

    data = data.to(u.Jy/u.arcsec**2)

    return data.value 


def get_ell(r, ell, x):
    foo = interp1d(r, ell)
    return foo(x)