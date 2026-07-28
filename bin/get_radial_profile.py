# %load bin/get_radial_profile.py
from photutils.aperture import EllipticalAnnulus as EA
import warnings
from astropy.stats import sigma_clipped_stats
import numpy as np

def rad_profile(data, mask, xc, yc, ell=0.0, pa=0.0, step=0.03, maxsma=None, ax=None):
    '''
    Построение азимутального профиля галактики с ребра

    '''
    mask = mask > 0
    start = 0.0001
    width = 1.0
    power = 1 + step

    h, w = data.shape
    if maxsma is None:
        maxsma = w//2

    r  = []
    I  = []
    Ie = []

    while True:
        end = start + width

        a_in  = start
        a_out = end
        #print('widTH', start)

        b_in = a_in*(1-ell)
        b_out = a_out*(1-ell)

        ellipse = EA(positions=(xc, yc),  a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out,theta=np.radians(pa))
        apermask = ellipse.to_mask(method='exact', subpixels=5)
        (slc_large, aper_weights, pixel_mask) = apermask._get_overlap_cutouts(data.shape, mask=None)

        #plt.figure()
        #plt.imshow(aper_weights)
        #plt.show()
        #
        if slc_large is None:
            I.append(np.nan)
            Ie.append(np.nan)
        else:
            with warnings.catch_warnings():
                # ignore multiplication with non-finite data values
                warnings.simplefilter('ignore', RuntimeWarning)
                aper_weights[aper_weights < 0.1] = float('nan')
                values = (data[slc_large] * aper_weights)[pixel_mask]
                #print(values)
                ind = np.where( (~np.isnan(values)))
                values = values[ind]
                #plt.figure()
                #plt.hist(values)
                #plt.show()
                mean, med, std = sigma_clipped_stats(values, sigma=1.5, maxiters=10)
                I.append(mean)
                Ie.append(std)



        r.append((start + end) / 2.)

        width = width * power
        start = end
        if end > maxsma:
            break


    return np.array(r), np.array(I), np.array(Ie)
