import sys
import subprocess as sp
sys.path.append('/home/android/iman_new/iraf_fitting')
from iraf_ellipse import main_ell # type: ignore
from plot_iraf_results import read_ell
from matplotlib import pyplot as plt
from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)
from astropy.io import fits
from bin.zoomdown import zoomdown
import numpy as np
import os

if __name__ == '__main__':

    fname = '737885_28_face'
    tmp_image = f'/home/android/eagle/fits_images/{fname}.fits'
    tmp_data      = fits.getdata(tmp_image)
    tmp_band = tmp_data[2]
    tmp_band = zoomdown(tmp_band, 0.5)

    h, w = tmp_band.shape
    xc = w//2
    yc = h//2

    ellip = 0
    pa    = 0
    sma  = 100
    ZP    = 0.0
    pix2sec = 1.0
    step   = 0.03
    minsma = 0.1
    maxsma = 250

    outp_format = 'jpg'

    tmp_mask = np.zeros(tmp_band.shape)
    tmp_mask[tmp_band <=0] = 1

    fflag = 1.0
    olthresh = 0.0
    linear  = 'no'
    hcenter = 'yes'
    hellip  = 'no'
    hpa     = 'no'
    model_file = None
    layers = ['all']

    input_image = 'tmp.fits'
    fits.writeto(input_image, tmp_band, overwrite=True)

    mask = 'tmp_mask.fits'
    fits.writeto(mask, tmp_mask, overwrite=True)

    ell_file = f'{fname}_ellipse_free.txt'
    if not(os.path.exists(ell_file)):
        main_ell(input_image, xc, yc, ellip=ellip, pa=pa, 
             sma0=sma, m0=ZP, pix2sec=pix2sec, step=step, 
             minsma=minsma, maxsma=maxsma, 
             outp_format=outp_format, ell_file=ell_file, fits_mask=mask, 
             fflag=fflag, olthresh=olthresh, linear=linear, 
             layers=layers, hcenter=hcenter, hellip=hellip,
               hpa=hpa, model_file=model_file)
    
    sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4 =read_ell(ell_file)
    
    plt.figure()
    norm = ImageNormalize(tmp_band, interval=MinMaxInterval(), stretch=LogStretch(10_000))

    plt.imshow(tmp_band, norm=norm, origin='lower')
    
    for i, sma_, ell_, PA_ in zip(np.arange(len(sma)), sma, ell, PA):
        x = []
        y = []
        print(sma_, ell_, PA_)
        if i%10 == 0:
            for t in np.arange(0, 2*np.pi, 0.01):
            
                x_ =  sma_*np.cos(t)
                y_ = sma_*(1-ell_)*np.sin(t)

                cosp = np.cos(np.radians(PA_-90))
                sinp = np.sin(np.radians(PA_-90))

                x.append(xc +  x_*cosp - y_*sinp)
                y.append(yc + x_*sinp + y_*cosp)
            plt.plot(x, y, '-', color='r')
    plt.show()