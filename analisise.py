import os
from astropy.io import fits
from pathlib import Path
import pyimfit
from visualisation import MakeImage
import sys
import numpy as np
import subprocess as sp
sys.path.append('/home/android/iman_new/iraf_fitting')
sys.path.append('/home/android/iman_new/Ellipse_photometry')
from iraf_ellipse import main_ell # type: ignore
from plot_iraf_results import read_ell # type: ignore
from azimProfile import main as azimp # type: ignore
from bin.common_functions import calc_ith_isophote_radius, AB_mag, get_ell
from fit import fit_step, get_fit_state, save_imfit_to_data
from matplotlib import pyplot as plt
from visualisation import ShowResults
from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)


def load_model(file):
    with open(file, 'r') as ff:
        lines = ff.readlines()#[:-8]
        if lines[0][0] == '#':
            lines_new = []
            for i, line in enumerate(lines):
                if 'Sersic' in line:
                    print('line', line)
                    #line = line.split('\n')[0]
                    line += '   # LABEL bulge\n'
                    #print('line', line)
                elif 'FerrersBar2D' in line:
                    line = line.split('\n')[0]
                    line += '   # LABEL bar\n'
                elif 'FUNCTION' in line:
                    line = line.split('\n')[0]
                    line += '   # LABEL disk\n'
                lines_new.append(line)
        else:
            lines_new = lines[:-8]
                
                        
    model_desc = pyimfit.parse_config(lines_new)
    imfit = pyimfit.Imfit(model_desc)

    #mi = MakeImage(imfit)
    return imfit



def get_IRAF_profile(image, state, fname, maxsma=250):

    ellip = state['functions']['disk']['ell'][0]
    pa    = state['functions']['disk']['PA'][0]
    sma0  = 100
    ZP    = 0.0
    pix2sec = 1.0
    step   = 0.03
    minsma = 1
    #maxsma = 250
   
    outp_format = 'jpg'

    tmp_mask = np.zeros(image.shape)
    tmp_mask[image <=0] = 1

    fflag = 1.0
    olthresh = 0.0
    linear  = 'no'
    hcenter = 'yes'
    hellip  = 'no'
    hpa     = 'no'
    model_file = None
    layers = ['all']

    xc = state["xc"]
    yc = state["yc"]

    input_image = 'tmp.fits'
    fits.writeto(input_image, image, overwrite=True)

    mask = 'tmp_mask.fits'
    fits.writeto(mask, tmp_mask, overwrite=True)

    ell_file = f'profiles_new/{fname}_ellipse_flex.txt'
    if not(os.path.exists(ell_file)):

        key = True
        maxsma = 250
        while key:
            main_ell(input_image, xc, yc, ellip=ellip, pa=pa, 
                        sma0=sma0, m0=ZP, pix2sec=pix2sec, step=step, 
                        minsma=minsma, maxsma=maxsma, 
                        outp_format=outp_format, ell_file=ell_file, fits_mask=mask, 
                        fflag=fflag, olthresh=olthresh, linear=linear, 
                        layers=layers, hcenter=hcenter, hellip=hellip,
                        hpa=hpa, model_file=model_file)

            sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4 =read_ell(ell_file)
            if len(sma) > 0:
                key = False
            else:
                os.remove(ell_file)
                maxsma -= 10
        #print('sma', sma)
        zp = 8.9 -2.5*np.log10(AB_mag(1))
        r25 = calc_ith_isophote_radius(sma, inten, zp, 25)
        pa_new = get_ell(sma, PA, r25)
        ellip_new    = get_ell(sma, ell, r25)
        print('pa', pa, 'ellip', ellip)
        main_ell(input_image, xc, yc, ellip=ellip_new, pa=pa_new, 
                        sma0=r25, m0=ZP, pix2sec=pix2sec, step=step, 
                        minsma=minsma, maxsma=maxsma, 
                        outp_format=outp_format, ell_file=ell_file, fits_mask=mask, 
                        fflag=fflag, olthresh=olthresh, linear=linear, 
                        layers=layers, hcenter=hcenter, hellip=hellip,
                        hpa=hpa, model_file=model_file)
    sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4 =read_ell(ell_file)
    return sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4
        

def get_Azim_profile(image, PA, ell, xc, yc, fname):
               
    output_model = None
    azim_tab = f'profiles/{fname}_azim_0.txt'
    tmp_mask = np.zeros(image.shape)
    tmp_mask[image <=0] = 1
    posang = PA
    ellip  = ell
    xcen = xc
    ycen = yc
    sma_min = 1
    sma_max = 250
    sigma_sky = None
    sigma_cal = None
    outside_frame = False
    linear = False
    step   = 0.03


    input_image = 'tmp.fits'
    fits.writeto(input_image, image, overwrite=True)
    
    mask = 'tmp_mask.fits'
    fits.writeto(mask, tmp_mask, overwrite=True)
    

    if not(os.path.exists(azim_tab)):
        azimp(input_image, output_model, 
            azim_tab, mask, xcen, ycen, 
            ellip, posang, sma_min, sma_max, step, 
            sigma_sky, sigma_cal, 
            outside_frame=outside_frame, linear=linear)

    sma, flux, flux_err = np.loadtxt(f'profiles/{fname}_azim_0.txt', skiprows=1, unpack=True)
    return sma, flux, flux_err
            



path = Path('./final_fit')
for dir_ in os.listdir(path):
    for model_file in sorted(os.listdir(Path(path,dir_)), reverse=True):
        file = os.path.splitext(model_file)[0]
        print(file)
        if os.path.exists(f'./final_pics/{file}.jpg'):
            continue

        image_data = fits.getdata(Path('..', 'images_r', file+'_face.fits'))
        imfit = load_model(Path(path, dir_, model_file))

        mi = MakeImage(imfit)

        state = get_fit_state(imfit)
        sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4 = get_IRAF_profile(image_data,state,file)
        #print('sma', sma)
        model_image = mi.get_model_image(label=mi.labels, size=image_data.shape)

        sma_model, inten_model, inten_err_model, ell_model, errell_model, PA_model, errPA_model, x0_model, y0_model, B4_model, errB4_model = get_IRAF_profile(model_image,state,file+"_model")

        sma_azim, int_azim, _ = get_Azim_profile(image_data, state['functions']['disk']['PA'][0],
                          state['functions']['disk']['ell'][0],
                            state['xc'],
                            state['yc'], file)

        
        sma_0, int_0, _ = get_Azim_profile(image_data, 0,0,
                                            state['xc'],
                                            state['yc'], file+'_0')


        

        model_images = []
        for label in mi.labels:
            tmp_image = mi.get_model_image(label=label, size = image_data.shape)
            model_images.append(tmp_image)

                     
        zp = 8.9 - 2.5*np.log10(AB_mag(1))


        plt.figure(figsize=(20,10))
        plt.subplot(241)
        plt.title('Image')
        norm = ImageNormalize(image_data, interval=MinMaxInterval(), stretch=LogStretch(10_000))
        plt.imshow(image_data, norm=norm, origin='lower', cmap='twilight')

        plt.subplot(242)
        plt.title(r"Profile along semi-major axis $a$ (free isophotes)")
        plt.plot(sma*0.2, zp-2.5*np.log10(inten), '-', color='grey', label='data')
        plt.plot(sma_model*0.2, zp-2.5*np.log10(inten_model), color='magenta', label='model')
        plt.legend()

        if 'r_break' in state['functions']['disk']:
            plt.gca().axvline(state['functions']['disk']['r_break'][0]*0.2, ls='--', color='red')
        elif 'r_break1' in state['functions']['disk']:
            plt.gca().axvline(state['functions']['disk']['r_break1'][0]*0.2, ls='--', color='red')
            plt.gca().axvline(state['functions']['disk']['r_break2'][0]*0.2, ls='--', color='red')

        plt.gca().invert_yaxis()
        plt.ylim(30,15)

        plt.subplot(244)
        plt.title(r"Profile along semi-major axis $a$ (circular isophotes)")
        plt.plot(sma_0*0.2, zp-2.5*np.log10(int_0), '-', color='grey', label='data')
        for label, image_ in zip(mi.labels, model_images):
            sma_0_model, int_0_model, _ = get_Azim_profile(image_, 0,0,
                                                        state['xc'],
                                                        state['yc'], file+f'_{label}_0_model')
            plt.plot(sma_0_model*0.2, zp-2.5*np.log10(int_0_model), label=label)

        sma_0_model, int_0_model, _ = get_Azim_profile(model_image, 0,0,
                                            state['xc'],
                                            state['yc'], file+'_0_model')
        plt.plot(sma_0_model*0.2, zp-2.5*np.log10(int_0_model), label='total')
        plt.legend()

        if 'r_break' in state['functions']['disk']:
            plt.gca().axvline(state['functions']['disk']['r_break'][0]*0.2, ls='--', color='red')
        elif 'r_break1' in state['functions']['disk']:
            plt.gca().axvline(state['functions']['disk']['r_break1'][0]*0.2, ls='--', color='red')
            plt.gca().axvline(state['functions']['disk']['r_break2'][0]*0.2, ls='--', color='red')
        plt.gca().invert_yaxis()
        plt.ylim(30,15)



        plt.subplot(243)
        plt.title(r"Profile along semi-major axis $a$ (fixed orientation)") 

        plt.plot(sma_azim*0.2, zp-2.5*np.log10(int_azim), '-', color='grey', label='data')
        for label, image_ in zip(mi.labels, model_images):
            sma_azim_model, int_azim_model, _ = get_Azim_profile(image_, state['functions']['disk']['PA'][0],
                                                      state['functions']['disk']['ell'][0],
                                                        state['xc'],
                                                        state['yc'], file+f'_{label}_model')
            plt.plot(sma_azim_model*0.2, zp-2.5*np.log10(int_azim_model), label=label)
            
        sma_azim_model, int_azim_model, _ = get_Azim_profile(model_image, state['functions']['disk']['PA'][0],
                                                  state['functions']['disk']['ell'][0],
                                                    state['xc'],
                                                    state['yc'], file+'_model')
        


        plt.plot(sma_azim_model*0.2, zp-2.5*np.log10(int_azim_model), label='total')
        plt.legend()
        if 'r_break' in state['functions']['disk']:
            plt.gca().axvline(state['functions']['disk']['r_break'][0]*0.2, ls='--', color='red')
        elif 'r_break1' in state['functions']['disk']:
            plt.gca().axvline(state['functions']['disk']['r_break1'][0]*0.2, ls='--', color='red')
            plt.gca().axvline(state['functions']['disk']['r_break2'][0]*0.2, ls='--', color='red')
        plt.gca().invert_yaxis()
        plt.ylim(30,15)


        plt.subplot(245)
        plt.title('Model')
        plt.imshow(model_image, norm=norm, origin='lower', cmap='twilight')
        plt.subplot(246)
        residual = image_data - model_image
        
        # Берем 2% и 98% процентили вместо min/max
        vmin = np.percentile(residual, 2)
        vmax = np.percentile(residual, 98)

        # Делаем симметричную шкалу относительно нуля
        lim = max(abs(vmin), abs(vmax))

        im = plt.imshow(residual, cmap='bwr', origin='lower', 
                vmin=-lim, vmax=lim)
        cbar = plt.colorbar(im, ax=plt.gca(), fraction=0.046, pad=0.04, 
                    location='right', shrink=0.6)  # shrink makes it shorter
        cbar.set_label('Difference')
        #plt.colorbar(label='Difference')
        plt.title('Residual')
        plt.subplot(247)
        plt.title("Cut along semi-minor axis")
        sr = ShowResults(image_data, imfit, zp = zp, PA=state['functions']['disk']['PA'][0], maj_axis=True, scale=0.2)
        ax = plt.gca()
        ax = sr.plot_x_cut(ax)
        plt.legend()
        plt.subplot(248)
        plt.title("Cut along semi-major axis")
        sr = ShowResults(image_data, imfit, zp = zp, PA=state['functions']['disk']['PA'][0], maj_axis=True, scale=0.2)
        ax = plt.gca()
        ax = sr.plot_y_cut(ax)
        plt.legend()
        plt.savefig(f'final_pics/{file}.jpg', format='jpg')
        plt.cla()
        plt.clf()
        #.show()

        #exit()    



    #exit()