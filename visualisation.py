import numpy as np
import pandas as pd
import pyimfit  # type: ignore
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)

from fit import get_fit_state
from model import set_parameters_from_dict
from bin.common_functions import calc_ith_isophote_radius

class MakeImage:
    def __init__(self, imfit):
        model = imfit.getModelAsDict()
        self.function_sets = model['function_sets']
        self.xc = self.function_sets[0]['X0'][0]
        self.yc = self.function_sets[0]['Y0'][0]

        self.labels = []
        self.models = []
        for funcs in self.function_sets[0]['function_list']:
            label = funcs['label']
            self.labels.append(label)

            if label == 'bulge':
                self.bulge = {}
                self.bulge['label'] = label
                self.bulge['name'] = funcs['name']
                self.bulge['parameters'] = funcs['parameters']
                self.bulge['model'] = pyimfit.make_imfit_function(self.bulge['name'], label=self.bulge['label'])
                self.init_bulge()
                self.models.append([label, self.bulge])

            if label == 'disk':
                self.disk = {}
                self.disk['label'] = label
                if funcs['name'] == "DoubleBrokenExponential":
                    self.disk['name'] = "doublebroken-exp"
                else:
                    self.disk['name'] = funcs['name']
                self.disk['parameters'] = funcs['parameters']
                #print(self.disk['parameters'])
                self.disk['model'] = pyimfit.make_imfit_function(self.disk['name'], label=self.disk['label'])
                self.init_disk()
                self.models.append([label, self.disk])
            
            if label == 'bar':
                self.bar = {}
                self.bar['label'] = label
                self.bar['name'] = funcs['name']   
                self.bar['parameters'] = funcs['parameters']
                self.bar['model'] = pyimfit.make_imfit_function(self.bar['name'], label=self.bar['label'])
                self.init_bar()
                self.models.append([label, self.bar])
                        
    def init_bulge(self):
        #print(self.bulge)
        set_parameters_from_dict(self.bulge['model'], self.bulge['parameters'])
        
    def init_disk(self):
        set_parameters_from_dict(self.disk['model'], self.disk['parameters'])

    def init_bar(self):
        set_parameters_from_dict(self.bar['model'], self.bar['parameters'])
            

    def get_model_image(self, label, size):
        model = pyimfit.SimpleModelDescription()
        model.x0.setValue(self.xc)
        model.y0.setValue(self.yc)

        # Самый правильный и безопасный способ
        if isinstance(label, list):
            for label_ in label:
                if label_ in self.labels:
                    for tmp_data in self.models:
                        tmp_label = tmp_data[0]
                        if label_ == tmp_label:
                            model.addFunction(tmp_data[1]['model']) 
                else:
                    print(f'No such label:{label_} in model')
                    return

        elif isinstance(label, str):
            if (label in self.labels):
                for tmp_data in self.models:
                    tmp_label = tmp_data[0]
                    if label == tmp_label:
                        #print('hi')
                        #print(tmp_data[1], tmp_label)

                        model.addFunction(tmp_data[1]['model'])   
            else:
                print(f'No such label:{label} in model')
                return
                
        else:
            print(f'Wrong label format: {label}. Need str or list')
            return
        imfit_model = pyimfit.Imfit(model)
        return imfit_model.getModelImage(shape=size)



class ShowResults:
    def __init__(self, image, fit, zp=0.0, PA=0.0, maj_axis=False, scale=1.0):

        self.PA = PA
        self.scale = scale

        if maj_axis:
            self.image = rotate(image, self.PA, reshape=False) 
        else:
            self.image = image 
        self.zp = zp

        makeimage = MakeImage(fit)
        self.makeimage = makeimage
        self.labels = makeimage.labels

        self.models = {}
        for label in self.labels:
            tmp_model = makeimage.get_model_image(label=label, size=image.shape)
            if maj_axis:
                self.models[label] = rotate(tmp_model, self.PA, reshape=False)
            else:
                self.models[label] = tmp_model

        tmp_model = makeimage.get_model_image(label=self.labels, size=image.shape)
        if maj_axis:
                self.models['total'] = rotate(tmp_model, self.PA, reshape=False)
        else:
            self.models['total'] = tmp_model

        self.models['total'] 
        self.labels.append('total')
        self.maj_axis = maj_axis


    def plot_cuts(self, file, cuts=False):
        _fig, axs = plt.subplots(2, 3, figsize=(15,10))
        h, w = self.image.shape
        ax = axs[0,:]
        zp = self.zp
        if self.maj_axis:
            ax[0].set_title('SmajA')
        else:
            ax[0].set_title('X cut')

        ax[0] = self.plot_x_cut(ax[0])

        if self.maj_axis:
            ax[1].set_title('SminA')
        else:
            ax[1].set_title('Y cut')

        ax[1] = self.plot_y_cut(ax[1])        

        if cuts:
            print('rmax1')
            rmax_1 = calc_ith_isophote_radius(np.arange(h/2,0,-1), self.image[h//2:, w//2], zp=zp, target_mag=30.0)
            print('rmax2')
            rmax_2 = calc_ith_isophote_radius(np.arange(0,h/2,1), self.image[:h//2, w//2], zp=zp, target_mag=30.0)
            print('rmax3')
            rmax_3 = calc_ith_isophote_radius(np.arange(w/2,0,-1), self.image[h//2, :w//2], zp=zp, target_mag=30.0)
            print('rmax4')
            rmax_4 = calc_ith_isophote_radius(np.arange(0,w/2,1), self.image[h//2, w//2:], zp=zp, target_mag=30.0)
            rmax = np.nanmax([rmax_1, rmax_2, rmax_3, rmax_4])
            ax[0].set_xlim(-rmax*self.scale, rmax*self.scale)
            ax[1].set_xlim(-rmax*self.scale, rmax*self.scale)
        #plt.show()

        ax = axs[1,:]
        norm = ImageNormalize(self.image, interval=MinMaxInterval(), stretch=LogStretch(10_000))
        ax[0].imshow(self.image, norm=norm, origin='lower', cmap='twilight')
        ax[1].imshow(self.models['total'], norm=norm, origin='lower', cmap='twilight')


        residual = self.image - self.models['total']

        # Берем 2% и 98% процентили
        vmin = np.percentile(residual, 2)
        vmax = np.percentile(residual, 98)

        # Делаем симметричную шкалу относительно нуля
        lim = max(abs(vmin), abs(vmax))

        im = ax[2].imshow(residual, cmap='bwr', origin='lower', 
                        vmin=-lim, vmax=lim)
        ax[2].set_title('Residual')

        # Правильный способ добавить colorbar
        plt.colorbar(im, ax=ax[2], label='Difference')
        plt.savefig(file, format='jpg')

    def convert_int_to_mag(self, img, zp):
        mag = np.full_like(img, np.nan, dtype=float)
        mask = img > 0
        mag[mask] = zp - 2.5 * np.log10(img[mask])
        return mag

    def plot_y_cut(self, ax):

        h, w = self.image.shape

        mag = self.convert_int_to_mag(self.image[:, w//2], self.zp)
        ax.plot(self.scale*(np.arange(h)-h/2), mag, '-', color='grey', label='data')
                
        for label in self.labels:
            mag = self.convert_int_to_mag(self.models[label][:, w//2], self.zp)
            ax.plot(self.scale*(np.arange(h)-h/2), mag, '-', label=label)
            if label == 'disk':
                if 'r_break' in self.makeimage.disk['parameters']:
                    ax.axvline(self.scale*self.makeimage.disk['parameters']['r_break'][0], ls='--', color='r')
                    ax.axvline(-self.scale*self.makeimage.disk['parameters']['r_break'][0], ls='--', color='r')
                elif 'r_break1' in self.makeimage.disk['parameters']:
                    ax.axvline(self.scale*self.makeimage.disk['parameters']['r_break1'][0], ls='--', color='r')
                    ax.axvline(-self.scale*self.makeimage.disk['parameters']['r_break1'][0], ls='--', color='r')
                    ax.axvline(self.scale*self.makeimage.disk['parameters']['r_break2'][0], ls='--', color='r')
                    ax.axvline(-self.scale*self.makeimage.disk['parameters']['r_break2'][0], ls='--', color='r')

        ax.invert_yaxis()
        ax.set_ylim(30,15)
        return ax

    def plot_x_cut(self, ax):
        h, w = self.image.shape

        mag = self.convert_int_to_mag(self.image[h//2,:], self.zp)
        ax.plot(self.scale*(np.arange(w)-w/2), mag, '-', color='grey', label='data')
        
        for label in self.labels:
            mag = self.convert_int_to_mag(self.models[label][h//2, :], self.zp)
            ax.plot(self.scale*(np.arange(w)-w/2), mag, '-', label=label)
            if label == 'disk':
                if 'r_break' in self.makeimage.disk['parameters']:
                    ax.axvline(self.scale*self.makeimage.disk['parameters']['r_break'][0], ls='--', color='r')
                    ax.axvline(-self.scale*self.makeimage.disk['parameters']['r_break'][0], ls='--', color='r')

        ax.invert_yaxis()
        ax.set_ylim(30,15)
        return ax

class FitAnalysis:
    def __init__(self, results):
        self.results = results
        rows = []

        for run in results:
            state = get_fit_state(run["imfit"])

            bulge = state["functions"]["bulge"]
            disk = state["functions"]["disk"]
            if "h1" in disk:
                pass
            else:
                disk["h1"] = disk["h"]
                disk["h2"] = disk["h"]
                disk["r_break"] = [-1, -1, -1]
                disk["alpha"] = [-1, -1, -1]
            rows.append({
                "chi2": run["chi2"],
                "AIC": run["aic"],

                "n": bulge["n"][0],
                "re": bulge["r_e"][0],
    
                "h1": disk["h1"][0],
                "h2": disk["h2"][0],
                "r_break": disk["r_break"][0],
                #"alpha": disk["alpha"][0],
            })

        self.df = pd.DataFrame(rows).sort_values("chi2")

    def table(self):
        print(self.df)

    def hist(self):
        _fig, ax = plt.subplots(2,2, figsize=(10,8))
        ax[0,0].hist(self.df["h1"], bins=20)
        ax[0,1].hist(self.df["h2"], bins=20)
        ax[1,0].hist(self.df["r_break"], bins=20)
        ax[1,1].hist(self.df["alpha"], bins=20)
        plt.show()

    def best_result(self):
        print(self.results[0]['imfit'].getModelDescription())