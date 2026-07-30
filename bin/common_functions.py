from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.interpolate import interp1d
import astropy.units as u
import numpy as np
import warnings
from matplotlib import pyplot as plt


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

def calc_ith_isophote_radius(sma, inten, zp, target_mag):
    # Переводим интенсивность в звездные величины
    mu = -2.5 * np.log10(inten) + zp
    
    # Удаляем дубликаты, сохраняя порядок
    r_unique = np.array([])
    mu_unique = np.array([])

    for i, r_, mu_ in zip(np.arange(len(sma)), sma, mu):
        if mu_ in mu[i+1:]:
            pass
        else:
            r_unique = np.append(r_unique, r_)
            mu_unique = np.append(mu_unique, mu_)



    
    # Проверяем, достаточно ли точек
    #if len(mu_unique) < 4:
        # Если мало данных, используем линейную интерполяцию
    f = interp1d(mu_unique, r_unique, kind='linear', 
                     bounds_error=False, fill_value='extrapolate')
    #else:
    #    f = interp1d(mu_unique, r_unique, kind='cubic',
    #                 bounds_error=False, fill_value='extrapolate')
    
    r25 = float(f(target_mag))
    if r25<0:
        plt.figure()
        plt.plot(zp-2.5*np.log10(inten), sma)
        plt.plot(np.arange(30, 20, -0.001), f(np.arange(30, 20, -0.001)))
        print('r25 cal', f(25))
        plt.axvline(25)
        plt.show()
    return r25
def AB_mag(data):

    # init MJy/sr

    data = data * 10**6 * u.Jy /u.sr

    # convert to Jy/arcsec**2

    data = data.to(u.Jy/u.arcsec**2)

    return data.value 


def get_ell(r, ell, x):
    foo = interp1d(r, ell, kind=3)
    return foo(x)

