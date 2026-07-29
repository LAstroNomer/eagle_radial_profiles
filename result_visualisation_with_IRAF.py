import sys
import subprocess as sp
sys.path.append('/home/android/iman_new/iraf_fitting')
sys.path.append('/home/android/iman_new/Ellipse_photometry')
from iraf_ellipse import main_ell # type: ignore
from plot_iraf_results import read_ell # type: ignore
from azimProfile import main as azimp # type: ignore
from bin.common_functions import calc_ith_isophote_radius, AB_mag, get_ell
from bin.imfit_fit import plot_slices
from matplotlib import pyplot as plt
from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)
from astropy.io import fits
from bin.zoomdown import zoomdown
import numpy as np
import os

def cheak_not_zero(models):
    dell_keys = []
    for key in models:
        tmp_data = models[key]
        if np.max(tmp_data) <=0:
            dell_keys.append(key)
    for key in dell_keys:
        del models[key]
    return models

if __name__ == '__main__':
    gals = sorted(list(set([a.split('_')[0] for a in os.listdir('../2d_results')])))
    for gal in gals:
        for i in range(28,12,-1):
            list_models = np.array(os.listdir('../2d_results'))
            fname = f'{gal}_{i}'
            if os.path.exists(f'pics/{fname}.jpg'):
                continue
            print(fname)

            for a in list_models:
                if fname in a:
                    tmp_model = a
                    break


            tmp_image = f'/home/android/eagle/fits_images/{fname}_face.fits'
            tmp_data      = fits.getdata(tmp_image)
            tmp_band = tmp_data[2]
            tmp_band = zoomdown(tmp_band, 0.5)/4

            h, w = tmp_band.shape
            xc = w//2
            yc = h//2

            ellip = 0
            pa    = 0
            sma0  = 100
            ZP    = 0.0
            pix2sec = 1.0
            step   = 0.03
            minsma = 1
            maxsma = 230

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

            ell_file = f'profiles/{fname}_ellipse_flex.txt'
            if not(os.path.exists(ell_file)):
                main_ell(input_image, xc, yc, ellip=ellip, pa=pa, 
                        sma0=sma0, m0=ZP, pix2sec=pix2sec, step=step, 
                        minsma=minsma, maxsma=maxsma, 
                        outp_format=outp_format, ell_file=ell_file, fits_mask=mask, 
                        fflag=fflag, olthresh=olthresh, linear=linear, 
                        layers=layers, hcenter=hcenter, hellip=hellip,
                        hpa=hpa, model_file=model_file)
                sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4 =read_ell(ell_file)
                zp = 8.9 -2.5*np.log10(AB_mag(1))
                r25 = calc_ith_isophote_radius(sma, inten, zp, 25)
                pa = get_ell(sma, PA, r25)
                ellip    = get_ell(sma, ell, r25)
                print('pa', pa, 'ellip', ellip)
                main_ell(input_image, xc, yc, ellip=ellip, pa=pa, 
                        sma0=r25, m0=ZP, pix2sec=pix2sec, step=step, 
                        minsma=minsma, maxsma=maxsma, 
                        outp_format=outp_format, ell_file=ell_file, fits_mask=mask, 
                        fflag=fflag, olthresh=olthresh, linear=linear, 
                        layers=layers, hcenter=hcenter, hellip=hellip,
                        hpa=hpa, model_file=model_file)
            sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4 =read_ell(ell_file)

            sp.call(f'cp ../2d_results/{tmp_model} bestfit_parameters_imfit.dat', shell=True)
            _, models = plot_slices(ax=None)
            models = cheak_not_zero(models)
            for key in models:
                fits.writeto(f'model_{key}.fits', models[key], overwrite=True)
            #sp.call(f'makeimage18 ../2d_results/{tmp_model} --refimage={input_image}', shell=True)
            #model_image = 'modelimage.fits'
            #model = fits.getdata(model_image)
            for key in models:
                ell_file = f'profiles/{fname}_ellipse_flex_model_{key}.txt'
                #fits.writeto(f'model_{key}.fits', models[key], overwrite=True)
                model_image = f'model_{key}.fits'

                if not(os.path.exists(ell_file)):
                    main_ell(model_image, xc, yc, ellip=ellip, pa=pa, 
                        sma0=sma0, m0=ZP, pix2sec=pix2sec, step=step, 
                        minsma=minsma, maxsma=maxsma, 
                        outp_format=outp_format, ell_file=ell_file, fits_mask=mask, 
                        fflag=fflag, olthresh=olthresh, linear=linear, 
                        layers=layers, hcenter=hcenter, hellip=hellip,
                        hpa=hpa, model_file=model_file)
                    
            #sma_model, inten_model, inten_err, ell_model, errell, PA_model, errPA, x0, y0, B4, errB4 =read_ell(ell_file)
        
            
            #sma, inten, inten_err, ell, errell, PA, errPA, x0, y0, B4, errB4 =read_ell(ell_file)

            output_model = None
            azim_tab = f'profiles/{fname}_azim_0.txt'
            mask_image = mask
            posang = 0
            ellip    = 0
            xcen = xc
            ycen = yc
            sma_min = minsma
            sma_max = maxsma
            sigma_sky = None
            sigma_cal = None
            outside_frame = False
            linear = False
            if not(os.path.exists(azim_tab)):
                azimp(input_image, output_model, 
                azim_tab, mask_image, xcen, ycen, 
                ellip, posang, sma_min, sma_max, step, 
                sigma_sky, sigma_cal, 
                outside_frame=outside_frame, linear=linear)

            for key in models:
                #ell_file = f'{fname}_ellipse_flex_model_{key}.txt'
                #fits.writeto(f'model_{key}.fits', models[key])
                model_image = f'model_{key}.fits'
                azim_tab = f'profiles/{fname}_azim_0_model_{key}.txt'
                if not(os.path.exists(azim_tab)):
                    azimp(model_image, output_model, 
                    azim_tab, mask_image, xcen, ycen, 
                    ellip, posang, sma_min, sma_max, step, 
                    sigma_sky, sigma_cal, 
                    outside_frame=outside_frame, linear=linear)

            zp = 8.9 -2.5*np.log10(AB_mag(1))
            output_model = None
            azim_tab = f'profiles/{fname}_azim_25.txt'
            mask_image = mask
            
            r25 = calc_ith_isophote_radius(sma, inten, zp, 25)
            posang = get_ell(sma, PA, r25)
            ellip    = get_ell(sma, ell, r25)
            xcen = xc
            ycen = yc
            sma_min = minsma
            sma_max = maxsma
            sigma_sky = None
            sigma_cal = None
            outside_frame = False
            linear = False
            if not(os.path.exists(azim_tab)):
                azimp(input_image, output_model, 
                azim_tab, mask_image, xcen, ycen, 
                ellip, posang, sma_min, sma_max, step, 
                sigma_sky, sigma_cal, 
                outside_frame=outside_frame, linear=linear)

            for key in models:
                #ell_file = f'{fname}_ellipse_flex_model_{key}.txt'
                #fits.writeto(f'model_{key}.fits', models[key])
                model_image = f'model_{key}.fits'
                azim_tab = f'profiles/{fname}_azim_25_model_{key}.txt'
            
                #azim_tab = f'{fname}_azim_25_model.txt'
                if not(os.path.exists(azim_tab)):
                    azimp(model_image, output_model, 
                    azim_tab, mask_image, xcen, ycen, 
                    ellip, posang, sma_min, sma_max, step, 
                    sigma_sky, sigma_cal, 
                    outside_frame=outside_frame, linear=linear)


            sma_0, flux_0, flux_err_0 = np.loadtxt(f'profiles/{fname}_azim_0.txt', skiprows=1, unpack=True)
            #sma_0_model, flux_0_model, flux_err_0_model = np.loadtxt(f'{fname}_azim_0_model.txt', skiprows=1, unpack=True)
            sma_25, flux_25, flux_err_25 = np.loadtxt(f'profiles/{fname}_azim_25.txt', skiprows=1, unpack=True)
            #sma_25_model, flux_25_model, flux_err_25_model = np.loadtxt(f'{fname}_azim_25_model.txt', skiprows=1, unpack=True)

            
            plt.figure(figsize=(20,10))
            plt.subplot(241)
            plt.title('Image')
            norm = ImageNormalize(tmp_band, interval=MinMaxInterval(), stretch=LogStretch(10_000))

            plt.imshow(tmp_band, norm=norm, origin='lower')
            
            for i, sma_, ell_, PA_ in zip(np.arange(len(sma)), sma, ell, PA):
                x = []
                y = []
                #print(sma_, ell_, PA_)
                if i%10 == 0:
                    for t in np.arange(0, 2*np.pi, 0.01):
                    
                        x_ =  sma_*np.cos(t)
                        y_ = sma_*(1-ell_)*np.sin(t)

                        cosp = np.cos(np.radians(PA_-90))
                        sinp = np.sin(np.radians(PA_-90))

                        x.append(xc +  x_*cosp - y_*sinp)
                        y.append(yc + x_*sinp + y_*cosp)
                    plt.plot(x, y, '-', color='r')


            plt.subplot(242)
            plt.title('IRAF Ellipse')
            plt.plot(sma*0.2, zp-2.5*np.log10(inten), '-', color='grey', label='data')
            for key in models:
                ell_file = f'profiles/{fname}_ellipse_flex_model_{key}.txt'
                sma_model, inten_model, inten_err, ell_model, errell, PA_model, errPA, x0, y0, B4, errB4 =read_ell(ell_file)
                plt.plot(sma_model*0.2, zp-2.5*np.log10(inten_model), '-', label=key)

            plt.gca().invert_yaxis()
            plt.ylim(30,)
            plt.legend()
            plt.subplot(243)
            plt.title('Azim round')

            plt.plot(sma_0*0.2, zp-2.5*np.log10(flux_0), '-', color='grey')
            for key in models:
                sma_0_model, flux_0_model, flux_err_0_model = np.loadtxt(f'profiles/{fname}_azim_0_model_{key}.txt', skiprows=1, unpack=True)
                plt.plot(sma_0_model*0.2, zp-2.5*np.log10(flux_0_model), '-', label=key)
            plt.gca().invert_yaxis()
            plt.ylim(30,)
            plt.subplot(244)
            plt.title('Azim Ellipse')
            plt.plot(sma_25*0.2, zp-2.5*np.log10(flux_25), '-', color='grey')
            for key in models:
                sma_25_model, flux_25_model, flux_err_25_model = np.loadtxt(f'profiles/{fname}_azim_25_model_{key}.txt', skiprows=1, unpack=True)

                plt.plot(sma_25_model*0.2, zp-2.5*np.log10(flux_25_model), '-', label=key)
            plt.gca().invert_yaxis()
            plt.ylim(30,)
            plt.subplot(247)
            plt.title('X CUT')
            plt.plot(np.arange(-w//2,w//2)*0.2, zp-2.5*np.log10(tmp_band[yc, :]), '-', color='grey')
            for key in models:
                plt.plot(np.arange(-w//2,w//2)*0.2, zp-2.5*np.log10(models[key][yc, :]), '-',label=key)
            plt.gca().invert_yaxis()
            plt.ylim(30,)
            plt.legend()
            plt.subplot(248)
            plt.title('Y CUT')
            plt.plot(np.arange(-h//2,h//2)*0.2, zp-2.5*np.log10(tmp_band[:, xc]), '-', color='grey')
            for key in models:
                plt.plot(np.arange(-h//2,h//2)*0.2, zp-2.5*np.log10(models[key][:, xc]), '-',label=key)
            plt.gca().invert_yaxis()
            plt.ylim(30,)
            
            plt.subplot(245)
            plt.title('Model')
            #norm = ImageNormalize(tmp_band, interval=MinMaxInterval(), stretch=LogStretch(10_000))
            plt.imshow(models['total'], norm=norm, origin='lower')
            
            import numpy as np

            plt.subplot(246)
            # Находим максимальное абсолютное отклонение, чтобы сделать шкалу симметричной
            residual = tmp_band - models['total']

            # Берем 2% и 98% процентили вместо min/max
            vmin = np.percentile(residual, 2)
            vmax = np.percentile(residual, 98)

            # Делаем симметричную шкалу относительно нуля
            lim = max(abs(vmin), abs(vmax))

            plt.imshow(residual, cmap='bwr', origin='lower', 
                    vmin=-lim, vmax=lim)
            plt.colorbar(label='Difference')
            plt.title('Residual')
            
            
            plt.suptitle(fname)
            #plt.show()
            plt.savefig(f'pics/{fname}.jpg', format='jpg')