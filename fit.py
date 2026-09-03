import pyimfit  # type: ignore

from model import build_model
#from finder import multi_start_fit

def fit_step(image, sigma, bulge_model, disk_model, 
             bulge_cfg, disk_cfg, bulge_fix, disk_fix, xc, yc, is_3D=False, hand_fix=None, 
            fast=True, **kwargs):

    model = build_model(bulge_model,
                        disk_model,
                        bulge_cfg,
                        disk_cfg,
                        bulge_fix,
                        disk_fix,
                        xc,
                        yc,
                        is_3D,
                        hand_fix=hand_fix
                        )

    if fast:
        result, imfit = fast_imfit_fit(image, model, sigma,  bulge_model, disk_model, hand_fix, **kwargs)
    else:
        result, imfit = imfit_fit(image, model, sigma,  **kwargs)

    #print(imfit.getModelDescription())
    return result, imfit

def imfit_fit(image, model, sigma, **kwargs):
    imfit = pyimfit.Imfit(model)
    result = imfit.fit(image, error=sigma,ftol=1e-6, verbose=1,**kwargs)

    return result, imfit

def fast_imfit_fit(image, model, sigma, bulge_model, disk_model, hand_fix, **kwargs):
    print('run fast')
    imfit_nm = pyimfit.Imfit(model)
    result_nm = imfit_nm.fit(image, error=sigma, ftol=1e-3, solver='NM', verbose=1)
    state = get_fit_state(imfit_nm)
    new_model = build_model(bulge_model,
                        disk_model,
                        state["functions"]["bulge"],
                        state["functions"]["disk"],
                        False,
                        False,
                        state["xc"],
                        state["yc"],
                        True,
                        hand_fix=hand_fix
                        )
    imfit_lm = pyimfit.Imfit(new_model)
    result_lm = imfit_lm.fit(image, error=sigma, ftol=1e-6, verbose=1, solver='LM')
    return result_lm, imfit_lm

def get_fit_state(imfit):
    model = imfit.getModelAsDict()

    function_dict = {
        f["label"]: f["parameters"]
        for f in model["function_sets"][0]["function_list"]
    }

    return {
        "xc": model["function_sets"][0]["X0"][0],
        "yc": model["function_sets"][0]["Y0"][0],
        "functions": function_dict,
    }

def save_imfit_to_data(result, imfit, output):
    imfit.saveCurrentModelToFile(output, includeImageOptions=True)
    with open(output, "a") as file:
        #print("open")
        for key in result.keys():
            if key == "params":
                continue

            if key == "paramErrs":
                continue

            print(key, result[key], file=file)



