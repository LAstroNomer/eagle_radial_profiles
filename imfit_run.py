import os
import subprocess as sp
import time
import numpy as np
from astropy.io import fits
import shutil
import pyimfit

from bin.common_functions import AB_mag
from cluster import get_best_candidate, parse_break_file, choose_model
from finder import multi_start_fit, multy_fit_with_init_guess
from fit import (
    fit_step,
    get_fit_state,
    save_imfit_to_data,
)
from imfit_run_two_breaks import fit_two_break_with_init_guess
from visualisation import FitAnalysis, ShowResults

scatter = {
    "I_0": 0.5,
    "J_0": 0.5,
    "h": 0.3,
    "h1": 0.3,
    "h2": 0.3,
    "h3": 0.3,
    "r_break": 0.4,
    "alpha": 0.5,
    "I_e": 0.4,
    "r_e": 0.2,
    "z_0": 0.2,
    "n": 0.5,
}



# -------------------------------
# Read image
# -------------------------------\
path = '../images_auriga_r'
incs = [0, 25, 36, 45, 53, 60, 66, 72, 78, 84, 90]
gals = sorted(set([a.split('_')[0] for a in os.listdir(path)]))
for gal in gals:
    if gal != 'Au3':
        continue
    for num, j in enumerate(incs):
        start = time.time()
        file = f"{gal}_i{j}"
        #print(file)
        if os.path.exists(f'fits/{file}/best_clustered.dat'):
            continue
        if os.path.exists(f"{path}/{file}_total.fits"):
            image = fits.getdata(f"{path}/{file}_total.fits")
            #print(image.shape)
            #exit()
        else:
            continue
        epsilon = np.percentile(image[image > 0], 1)
        zp = 8.9 - 2.5*np.log10(AB_mag(1))
        
        sigma = np.sqrt(image + epsilon)
        #low_sigma = zp - 2.5*np.log10(epsilon)
        #if low_sigma < 32:
        #    print(file, low_sigma)
        #continue
        h, w = image.shape
        xc = w/2
        yc = h/2
        #sigma = np.ones_like(image)
        if not(os.path.exists(f'fits/{file}')):
            sp.call(f"mkdir fits/{file}", shell=True)

        if j == 0:
            
            hand_fix = {
                "inc":[j,True],
                "n":[1,True],
                "z_0":[1, True],
                "PA": [90, False]
            }
            
            result, imfit = fit_step(image, sigma, bulge_model="Sersic", disk_model="ExponentialDisk3D", 
                        bulge_cfg=None, disk_cfg=None, bulge_fix=False, disk_fix=False,
                        xc=xc, yc=yc, is_3D=True, 
                        hand_fix=hand_fix)

            state = get_fit_state(imfit)
            #print('state', state)
            #exit()
            results_exp = multi_start_fit(image,
                            sigma,
                            "Sersic",
                            "ExponentialDisk3D",
                            state["functions"]["bulge"],
                            state["functions"]["disk"],
                            state["xc"],
                            state["yc"],
                            bulge_fix=False,
                            disk_fix=False,
                            n_starts=10,
                            scatter=scatter,
                            hand_fix=hand_fix,
                            is_3D=True,
                        )  

            best_exp = results_exp[0]['imfit']
            FitAnalysis(results_exp).table()
            save_imfit_to_data(results_exp[0]["fit"], best_exp, f"fits/{file}/best_exp.dat")
            #print(results_exp[0]["fit"])
            simple_exp = ShowResults(image, best_exp, zp=zp)
            simple_exp.plot_cuts(f"fits/{file}/best_exp.jpg")
            for i, res in enumerate(results_exp):
                save_imfit_to_data(res["fit"], res["imfit"], f"fits/{file}/exp_{i}.dat")

            #exit()


            #---------------------------------
            # Fit ONE Break
            # --------------------------------
            state = get_fit_state(best_exp)

            new_result, new_imfit = fit_step(image, sigma, bulge_model="Sersic", 
                        disk_model="BknExp3D", 
                        bulge_cfg=state["functions"]["bulge"],
                        disk_cfg=state["functions"]["disk"],
                        bulge_fix=True, 
                        disk_fix=False,
                        xc=state["xc"],
                        yc=state["yc"],
                        hand_fix=hand_fix,
                        is_3D=True
                        )
            
            # ------------------------------------
            # Flex Bulge
            # ------------------------------------
            state = get_fit_state(new_imfit)

            results = multi_start_fit(image,
                            sigma,
                            "Sersic",
                            "BknExp3D",
                            state["functions"]["bulge"],
                            state["functions"]["disk"],
                            state["xc"],
                            state["yc"],
                            bulge_fix=False,
                            disk_fix=False,
                            n_starts=30,
                            scatter=scatter,
                            hand_fix=hand_fix,
                            is_3D=True
                            )

            fit_an = FitAnalysis(results)
            fit_an.table()
            #fit_an.hist()
            fit_an.best_result()

            save_imfit_to_data(results[0]["fit"], results[0]["imfit"], f"fits/{file}/best_break.dat")

            for i, res in enumerate(results):
                save_imfit_to_data(res["fit"], res["imfit"], f"fits/{file}/break_{i}.dat")

            #print(new_result)
            simple_break = ShowResults(image, new_imfit, zp=zp)
            simple_break.plot_cuts(f"fits/{file}/break_fix.jpg")

            #print(results[0]["fit"])
            best_fit = ShowResults(image, results[0]["imfit"],zp=zp)
            best_fit.plot_cuts(f"fits/{file}/best_break.jpg")

            #exit()
           
            best_file = get_best_candidate(f"fits/{file}", pattern='break_*.dat', image=image)

            results_two = fit_two_break_with_init_guess(f"fits/{file}/{best_file['filename']}", 
                                        image, 
                                        sigma,
                                        scatter,
                                        hand_fix=hand_fix,
                                        is_3D=True
                                        )

            save_imfit_to_data(results_two[0]["fit"], results_two[0]["imfit"], f"fits/{file}/best_break_double.dat")

            for i, res in enumerate(results_two):
                save_imfit_to_data(res["fit"], res["imfit"], f"fits/{file}/double_break_{i}.dat")
            
            best_fit_two = ShowResults(image, results_two[0]["imfit"],zp=zp)
            best_fit_two.plot_cuts(f"fits/{file}/best_double_break.jpg")
            

            dir_path = f'fits/{file}'
            exp_best = parse_break_file(f'{dir_path}/best_exp.dat')
            best_break = get_best_candidate(f"fits/{file}", pattern='break_*.dat', image=image)
            
            best_double = get_best_candidate(f"fits/{file}", pattern='double_break_*.dat', image=image)
            #print('best_break', best_break)
            #exit()
            best_clustered = choose_model(image, exp_best, best_break, dir_path, best_double)

            #print('best_clustered')
            #print(best_clustered)
            shutil.copy2(best_clustered, f'{dir_path}/best_clustered.dat')

            #exit()

            with open(best_clustered, 'r') as ff:
                lines = ff.readlines()[:-8]
            
                    
            model_desc = pyimfit.parse_config(lines)
            #print('lables',model_desc.functionLabelList())
            imfit = pyimfit.Imfit(model_desc)
            best_fit = ShowResults(image, imfit,zp=zp)
            best_fit.plot_cuts(f"fits/{file}/best_clustered.jpg")
            
            print('Time', time.time()-start)
            
        

        else:
            if j > 40:
                hand_fix = {
                    "inc":[j,True],
                    "n":[1,False],
                    "z_0":[1, False],
                    "PA": [90, False]
                }
            else:
                hand_fix = {
                    "inc":[j,True],
                    "n":[1,True],
                    "z_0":[1, True],
                    "PA": [90, False]
                }
            if not(os.path.exists(f'fits/{file}/best_guess.dat')):
                guess_model = f'fits/{gal}_i{incs[num-1]}/best_clustered.dat'

                results = multy_fit_with_init_guess(guess_model, 
                                        image, 
                                        sigma, 
                                        scatter, 
                                        hand_fix, 
                                        is_3D=True)

                save_imfit_to_data(results[0]["fit"], results[0]["imfit"], f"fits/{file}/best_guess.dat")

                for i, res in enumerate(results):
                    save_imfit_to_data(res["fit"], res["imfit"], f"fits/{file}/guess{i}.dat")
            
            best_clustered = get_best_candidate(f"fits/{file}", pattern='guess*.dat', image=image)
            dir_path = f'fits/{file}'
            shutil.copy2(f'{dir_path}/{best_clustered['filename']}', f'{dir_path}/best_clustered.dat')
            with   open(f'{dir_path}/{best_clustered['filename']}', 'r') as ff:
                lines = ff.readlines()[:-8]
            
                    
            model_desc = pyimfit.parse_config(lines)
            #print('lables',model_desc.functionLabelList())
            imfit = pyimfit.Imfit(model_desc)
            best_fit_two = ShowResults(image, imfit,zp=zp)
            best_fit_two.plot_cuts(f"fits/{file}/best_clustered.jpg")
            print('Time', time.time()-start)

    exit()
            
