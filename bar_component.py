from model import build_model
from fit import fit_step, save_imfit_to_data, get_fit_state
from visualisation import ShowResults, MakeImage
import numpy as np
import os
from astropy.io import fits
from bin.common_functions import AB_mag
import subprocess as sp

def get_bar_model(hand_fix, gal, path):

    file = f'{gal}_i0'
    if not(os.path.exists(f'fits/{file}')):
        sp.call(f"mkdir fits/{file}", shell=True)
    
    if os.path.exists(f"{path}/{file}_total.fits"):
        image = fits.getdata(f"{path}/{file}_total.fits")
    else:
        return

    epsilon = np.percentile(image[image > 0], 1)
    zp = 8.9 - 2.5*np.log10(AB_mag(1))
    sigma = np.ones_like(image) #np.sqrt(image + epsilon)
    
    h, w = image.shape
    xc = w/2
    yc = h/2

    '''
    result, imfit = fit_step(image, sigma, bulge_model="Sersic", disk_model="ExponentialDisk3D", 
                                bulge_cfg=None, disk_cfg=None, bulge_fix=False, disk_fix=False,
                                xc=xc, yc=yc, is_3D=True, 
                                hand_fix=hand_fix, fast=False)

    
    mi = MakeImage(imfit)
    
    model = mi.get_model_image(mi.labels, image.shape)

   

    state = get_fit_state(imfit)
    '''
    result, imfit = fit_step(image, sigma, bulge_model="Sersic", disk_model="ExponentialDisk3D", 
                                bulge_cfg=None, disk_cfg=None, 
                                bulge_fix=False, disk_fix=False,
                                xc=xc, yc=yc, is_3D=True, 
                                hand_fix=hand_fix, add_bar=True, bar_cfg=None, fast=False,bar_fix=False)

    mi = MakeImage(imfit)
    print('labels', mi.labels)
    model1 = mi.get_model_image(mi.labels, image.shape)
    p10 = np.nanpercentile(image, 10)
    p90 = np.nanpercentile(image, 90)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(221)
    #plt.imshow(image-model, vmin=p10, vmax=p90)
   
    plt.subplot(222)
    plt.imshow(image-model1, vmin=p10, vmax=p90)

    plt.subplot(223)
    plt.imshow(np.log10(mi.get_model_image('bar', image.shape)))
    
    plt.subplot(224)
    plt.imshow(mi.get_model_image('bulge', image.shape), vmin=p10, vmax=p90)
    
    plt.savefig('model.jpg')


    best_fit = ShowResults(image, imfit,zp=zp)
    best_fit.plot_cuts(f"fits/{file}/bar.jpg")
    save_imfit_to_data(result, imfit, f"fits/{file}/bar.dat")
    return get_fit_state(imfit)['functions']['bar']

