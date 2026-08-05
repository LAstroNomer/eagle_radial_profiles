import numpy as np
import pandas as pd
import pyimfit  # type: ignore
from matplotlib import pyplot as plt
from scipy.ndimage import rotate

from fit import get_fit_state
from model import set_parameters_from_dict


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

            self.bulge = {}
            if label == 'bulge':
                self.bulge['label'] = label
                self.bulge['name'] = funcs['name']
                self.bulge['parameters'] = funcs['parameters']
                self.bulge['model'] = pyimfit.make_imfit_function(self.bulge['name'], label=self.bulge['label'])
                self.init_bulge()
                self.models.append([label, self.bulge])

            self.disk = {}
            if label == 'disk':
                self.disk['label'] = label
                self.disk['name'] = funcs['name']
                self.disk['parameters'] = funcs['parameters']
                #print(self.disk['parameters'])
                self.disk['model'] = pyimfit.make_imfit_function(self.disk['name'], label=self.disk['label'])
                self.init_disk()
                self.models.append([label, self.disk])
            
            self.bar = {}
            if label == 'bar':
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
    def __init__(self, image, fit, zp=0.0, PA=0.0, maj_axis=False):

        self.PA = PA

        if maj_axis:
            self.image = rotate(image, self.PA, reshape=False) 
        else:
            self.image = image 
        self.zp = zp

        makeimage = MakeImage(fit)
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


    def plot_cuts(self, file):
        _fig, ax = plt.subplots(1, 2, figsize=(10,5))
        h, w = self.image.shape

        zp = self.zp
        if self.maj_axis:
            ax[0].set_title('SmajA')
        else:
            ax[0].set_title('X cut')

        mag = self.convert_int_to_mag(self.image[h//2,:], zp)
        ax[0].plot(np.arange(w), mag, '-', color='grey', label='data')
        
        for label in self.labels:
            mag = self.convert_int_to_mag(self.models[label][h//2, :], zp)
            ax[0].plot(np.arange(w), mag, '-', label=label)
        ax[0].invert_yaxis()
        ax[0].set_ylim(30,)


        if self.maj_axis:
            ax[1].set_title('SminA')
        else:
            ax[1].set_title('Y cut')

        mag = self.convert_int_to_mag(self.image[:, w//2], zp)
        ax[1].plot(np.arange(h), mag, '-', color='grey', label='data')
        
        for label in self.labels:
            mag = self.convert_int_to_mag(self.models[label][:, w//2], zp)
            ax[1].plot(np.arange(h), mag, '-', label=label)
        ax[1].invert_yaxis()
        ax[1].set_ylim(30,15)
        #plt.show()
        plt.savefig(file, format='jpg')

    def convert_int_to_mag(self, img, zp):
        mag = np.full_like(img, np.nan, dtype=float)
        mask = img > 0
        mag[mask] = zp - 2.5 * np.log10(img[mask])
        return mag


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
                "alpha": disk["alpha"][0],
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