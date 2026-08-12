import os
import pyimfit
from astropy.io import fits
import numpy as np


from visualisation import MakeImage

def get_model(lines):
           
    model_desc = pyimfit.parse_config(lines)
    imfit = pyimfit.Imfit(model_desc)
    return imfit

j = 0
gals = sorted(set([a.split('_')[0] for a in os.listdir('../images_r')]))
for gal in gals:
        for i in range(28,11,-1):
            file = f'{gal}_{i}'
            dir_path = f"fits/{file}"
            if not(os.path.exists(dir_path)):
                 continue
            new_model = f"fits/{file}/best_clustered.dat"


            for img in os.listdir(f'../image_results/'):
                  if file in img:
                        old_model = f'../2d_results/{img[:-4]}.dat'

            print('old_model', old_model)
            print('new_model', new_model)

            image = fits.getdata(f"../images_r/{file}_face.fits")

            with open(new_model, 'r') as ff:
                lines = ff.readlines()[:-8] 
                print(lines)
                new_model_imfit = get_model(lines)

            with open(old_model, 'r') as ff:
                lines = ff.readlines()
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
                #print(lines_new)
                old_model_imfit = get_model(lines_new)
                
            old_mi = MakeImage(old_model_imfit)
            new_mi = MakeImage(new_model_imfit)
            epsilon = np.percentile(image[image > 0], 1)
            #zp = 8.9 - 2.5*np.log10(AB_mag(1))


            sigma = np.sqrt(image + epsilon)
            n_params_new = len(new_model_imfit.numberedParameterNames)
            n_params_old = len(old_model_imfit.numberedParameterNames)

            new_model_image = new_mi.get_model_image(label=new_mi.labels, size=image.shape)
            old_model_image = old_mi.get_model_image(label=old_mi.labels, size=image.shape)


            
            aic_new = np.sum(((new_model_image - image)/sigma)**2) + 2*n_params_new
            aic_old = np.sum(((old_model_image - image)/sigma)**2) + 2*n_params_old
            print('AIC', aic_old, aic_new, file)


            log_image = np.log10(image)
            log_image[np.isinf(log_image)] = np.nan

            log_new_image = np.log10(new_model_image)
            log_new_image[np.isinf(log_new_image)] = np.nan

            log_old_image = np.log10(old_model_image)
            log_old_image[np.isinf(log_old_image)] = np.nan


            Lxi_new   = np.nansum((log_new_image - log_image)**2)  + 2*n_params_new # 2*(11**2 + 11)/(500**2-11-1)
            Lxi_old = np.nansum((log_old_image - log_image)**2)  + 2*n_params_old #(13**2 + 13)/(500**2-13-1)
        
            print('Xi new', Lxi_new, aic_new)
            print('Xi old', Lxi_old, aic_old)
            #exit()


            if not os.path.exists('need_bar.csv') or os.path.getsize('need_bar.csv') == 0:
                with open('need_bar.csv', 'w') as ff:
                    print('name','barred',
                          'aic_new','aic_old',
                          'Lxi_new', 'Lxi_old', sep=',',file=ff)           

            with open('need_bar.csv', 'a') as ff:           
                if 'bar' in old_mi.labels:
                    print(file, True, aic_new, aic_old, Lxi_new, Lxi_old, sep=',', file=ff)
                else:
                    print(file, False, aic_new, aic_old, Lxi_new, Lxi_old, sep=',', file=ff)


            j += 1
            #exit()