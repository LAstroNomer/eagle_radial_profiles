import numpy as np
from astropy.io import fits
import pyimfit

from bin.common_functions import AB_mag
from fit import fit_step, get_fit_state, save_imfit_to_data
from visualisation import MakeImage, ShowResults, FitAnalysis
from model import build_model
from cluster import run_clustering
from finder import multi_start_fit
import subprocess as sp
import os

scatter = {
    "I_0": 0.5,
    "h": 0.3,
    "h1": 0.3,
    "h2": 0.3,
    "h3": 0.3,
    "r_break": 0.4,
    "alpha": 0.5,
    "I_e": 0.4,
    "r_e": 0.2,
    "n": 0.5,
}
import pandas as pd
tab = pd.read_csv('need_double_break.csv')
need_two = tab["gals"].to_numpy()


# -------------------------------
# Read image
# -------------------------------
gals = sorted(set([a.split('_')[0] for a in os.listdir('../images_r')]))
for gal in gals:
    for j in range(28,11,-1):
        file = f"{gal}_{j}"
        print(file)
        if os.path.exists(f'fits/{file}/best_clustered.dat'):
            breaked_model = f'fits/{file}/best_break.dat'
        else:
            continue

        if os.path.exists(f"../images_r/{file}_face.fits"):
            image = fits.getdata(f"../images_r/{file}_face.fits")
        else:
            continue

        if not(file in need_two):
            continue

        epsilon = np.percentile(image[image > 0], 1)
        zp = 8.9 - 2.5*np.log10(AB_mag(1))
        
        sigma = np.sqrt(image + epsilon)
        #sigma = np.ones_like(image)
        if not(os.path.exists(f'fits/{file}')):
            sp.call(f"mkdir fits/{file}", shell=True)

        with open(breaked_model, 'r') as ff:
            lines = ff.readlines()[:-8]
                
                        
        model_desc = pyimfit.parse_config(lines)
        imfit = pyimfit.Imfit(model_desc)
        state = get_fit_state(imfit)


        double_break_model = build_model("Sersic", "doublebroken-exp",
                 bulge_cfg=state["functions"]["bulge"], 
                 disk_cfg=state["functions"]["disk"],
                 bulge_fix=False, disk_fix=False, 
                 xc=state["xc"],
                 yc=state["yc"])
        new_imfit = pyimfit.Imfit(double_break_model)
        result = new_imfit.fit(image, error=sigma)
        #print(result)
        #print(new_imfit.getModelAsDict())
        #exit()

        # ------------------------------------
        state = get_fit_state(new_imfit)
        state["functions"]["bulge"]['PA'] += [-180,180]
        state["functions"]["bulge"]['ell'] += [0.01,0.8]
        state["functions"]["bulge"]['r_e'] += [0.1,25]
        state["functions"]["bulge"]['I_e'] += [0,1e5]
        state["functions"]["bulge"]['n'] += [0.5,5]

        


        #print('state', state)
        results = multi_start_fit(image,
                        sigma,
                        "Sersic",
                        "doublebroken-exp",
                        state["functions"]["bulge"],
                        state["functions"]["disk"],
                        state["xc"],
                        state["yc"],
                        bulge_fix=False,
                        disk_fix=False,
                        n_starts=30,
                        scatter=scatter)

        #fit_an = FitAnalysis(results)
        #fit_an.table()
        #fit_an.hist()
        #fit_an.best_result()

        save_imfit_to_data(results[0]["fit"], results[0]["imfit"], f"fits/{file}/best_break_double.dat")

        for i, res in enumerate(results):
            save_imfit_to_data(res["fit"], res["imfit"], f"fits/{file}/break_double_{i}.dat")
            

        
        #print(results_exp[0]["fit"])
        #simple_exp = ShowResults(image, best_exp, zp=zp)
        #simple_exp.plot_cuts(f"fits/{file}/best_exp.jpg")

        #print(new_result)
        #simple_break = ShowResults(image, new_imfit, zp=zp)
        #simple_break.plot_cuts(f"fits/{file}/break_fix.jpg")

        print(results[0]["fit"])
        best_fit = ShowResults(image, results[0]["imfit"],zp=zp)
        best_fit.plot_cuts(f"fits/{file}/best_double_break.jpg")

        #cluster, summary, best = run_clustering(f'fits/{file}')
        #print(summary)
        #print('best', best['filename'])
        
