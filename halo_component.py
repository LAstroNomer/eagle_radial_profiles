from model import build_model
from fit import fit_step, save_imfit_to_data, get_fit_state
from visualisation import ShowResults, MakeImage
import numpy as np
import os
from astropy.io import fits
from bin.common_functions import AB_mag
import subprocess as sp

def get_halo_model(hand_fix, gal, path):

    file = f'{gal}_i90'
    if not(os.path.exists(f'fits/{file}')):
        sp.call(f"mkdir fits/{file}", shell=True)
    
    if os.path.exists(f"{path}/{file}_total.fits"):
            image = fits.getdata(f"{path}/{file}_total.fits")
    else:
        return

    epsilon = np.percentile(image[image > 0], 1)
    zp = 8.9 - 2.5*np.log10(AB_mag(1))
    sigma = np.sqrt(image + epsilon)
    
    h, w = image.shape
    xc = w/2
    yc = h/2


    result, imfit = fit_step(image, sigma, bulge_model="Sersic", disk_model="ExponentialDisk3D", 
                                bulge_cfg=None, disk_cfg=None, bulge_fix=False, disk_fix=False,
                                xc=xc, yc=yc, is_3D=True, 
                                hand_fix=hand_fix, add_halo=False, halo_cfg=None, fast=True)

    mask = np.zeros_like(image, dtype=bool)
    print(imfit.getModelAsDict().keys())

    z0 = imfit.getModelAsDict()['function_sets'][0]['function_list'][-1]['parameters']['z_0'][0]


    mi = MakeImage(imfit)
    p10 = np.nanpercentile(image, 10)
    p90 = np.nanpercentile(image, 90)
    model = mi.get_model_image(mi.labels, image.shape)

    mask[(image - model) < 0.0] = True
    mask[int(yc-3*z0):int(yc+3*z0), :] = True

    state = get_fit_state(imfit)
    result, imfit = fit_step(image, sigma, mask=mask, bulge_model="Sersic", disk_model="ExponentialDisk3D", 
                                bulge_cfg=state['functions']['bulge'], disk_cfg=state['functions']['disk'], 
                                bulge_fix=True, disk_fix=True,
                                xc=state['xc'], yc=state['yc'], is_3D=True, 
                                hand_fix=hand_fix, add_halo=True, halo_cfg=None, fast=False,halo_fix=False)
    '''
    mi = MakeImage(imfit)
    print('labels', mi.labels)
    model1 = mi.get_model_image(mi.labels, image.shape)

    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(321)
    plt.imshow(image-model, vmin=p10, vmax=p90)
    masked_image = image.copy()
    masked_image[mask] = np.nan
    plt.subplot(322)
    plt.imshow(masked_image-model, vmin=p10, vmax=p90)

    plt.subplot(323)
    plt.imshow(masked_image-model1, vmin=p10, vmax=p90)

    plt.subplot(324)
    plt.imshow(mi.get_model_image('halo', image.shape), vmin=p10, vmax=p90)
    plt.show()
    '''

    best_fit = ShowResults(image, imfit,zp=zp)
    best_fit.plot_cuts(f"fits/{file}/halo.jpg")
    save_imfit_to_data(result, imfit, f"fits/{file}/halo.dat")
    return get_fit_state(imfit)['functions']['halo'], z0

