import numpy as np
from astropy.io import fits

from bin.common_functions import AB_mag
from fit import fit_step, get_fit_state
from visualisation import MakeImage, ShowResults, FitAnalysis
from finder import multi_start_fit

# -------------------------------
# Read image
# -------------------------------
image = fits.getdata("../images_r/746518_28_face.fits")
epsilon = np.percentile(image[image > 0], 1)
sigma = np.sqrt(image + epsilon)

result, imfit = fit_step(image, sigma, bulge_model="Sersic", disk_model="Exponential", 
            bulge_cfg=None, disk_cfg=None, bulge_fix=False, disk_fix=False,
            xc=250, yc=250)


#---------------------------------
# Fit ONE Break
# --------------------------------
state = get_fit_state(imfit)

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
scatter = {
    "I_0": 0.5,
    "h1": 0.3,
    "h2": 0.3,
    "r_break": 0.4,
    "alpha": 0.5,
    "I_e": 0.4,
    "r_e": 0.2,
    "n": 0.5,
}
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
fit_an.hist()
fit_an.best_result()