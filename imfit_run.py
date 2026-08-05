import numpy as np
from astropy.io import fits

from bin.common_functions import AB_mag
from fit import fit_step, get_fit_state, save_imfit_to_data
from visualisation import MakeImage, ShowResults, FitAnalysis
from finder import multi_start_fit
import subprocess as sp
import os

scatter = {
    "I_0": 0.5,
    "h": 0.3,
    "h1": 0.3,
    "h2": 0.3,
    "r_break": 0.4,
    "alpha": 0.5,
    "I_e": 0.4,
    "r_e": 0.2,
    "n": 0.5,
}
# -------------------------------
# Read image
# -------------------------------
for j in range(27,11,-1):
    file = f"737885_{j}"
    image = fits.getdata(f"../images_r/{file}_face.fits")
    epsilon = np.percentile(image[image > 0], 1)
    sigma = np.sqrt(image + epsilon)
    #sigma = np.ones_like(image)
    if not(os.path.exists(file)):
        sp.call(f"mkdir {file}", shell=True)

    result, imfit = fit_step(image, sigma, bulge_model="Sersic", disk_model="Exponential", 
                bulge_cfg=None, disk_cfg=None, bulge_fix=False, disk_fix=False,
                xc=250, yc=250)

    state = get_fit_state(imfit)
    results_exp = multi_start_fit(image,
                    sigma,
                    "Sersic",
                    "Exponential",
                    state["functions"]["bulge"],
                    state["functions"]["disk"],
                    state["xc"],
                    state["yc"],
                    bulge_fix=False,
                    disk_fix=False,
                    n_starts=10,
                    scatter=scatter)  

    best_exp = results_exp[0]['imfit']
    FitAnalysis(results_exp).table()
    save_imfit_to_data(results_exp[0]["fit"], best_exp, f"{file}/best_exp.dat")


    #---------------------------------
    # Fit ONE Break
    # --------------------------------
    state = get_fit_state(best_exp)

    new_result, new_imfit = fit_step(image, sigma, bulge_model="Sersic", disk_model="BrokenExponential", 
                bulge_cfg=state["functions"]["bulge"],
                disk_cfg=state["functions"]["disk"],
                bulge_fix=True, 
                disk_fix=False,
                xc=state["xc"],
                yc=state["yc"])
        

    # ------------------------------------
    # Flex Bulge
    # ------------------------------------
    state = get_fit_state(new_imfit)

    results = multi_start_fit(image,
                    sigma,
                    "Sersic",
                    "BrokenExponential",
                    state["functions"]["bulge"],
                    state["functions"]["disk"],
                    state["xc"],
                    state["yc"],
                    bulge_fix=False,
                    disk_fix=False,
                    n_starts=10,
                    scatter=scatter)

    fit_an = FitAnalysis(results)
    fit_an.table()
    #fit_an.hist()
    fit_an.best_result()

    save_imfit_to_data(results[0]["fit"], results[0]["imfit"], f"{file}/best_break.dat")

    for i, res in enumerate(results):
        save_imfit_to_data(res["fit"], res["imfit"], f"{file}/break_{i}.dat")
        

    
    zp = 8.9 - 2.5*np.log10(AB_mag(1))
    print(results_exp[0]["fit"])
    simple_exp = ShowResults(image, best_exp, zp=zp)
    simple_exp.plot_cuts(f"{file}/best_exp.jpg")

    print(new_result)
    simple_break = ShowResults(image, new_imfit, zp=zp)
    simple_break.plot_cuts(f"{file}/break_fix.jpg")

    print(results[0]["fit"])
    best_fit = ShowResults(image, results[0]["imfit"],zp=zp)
    best_fit.plot_cuts(f"{file}/best_break.jpg")
    
