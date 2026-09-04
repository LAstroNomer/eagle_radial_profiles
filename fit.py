import pyimfit  # type: ignore

from model import build_model
#from finder import multi_start_fit

def fit_step(image, sigma, bulge_model, disk_model, 
             bulge_cfg, disk_cfg, bulge_fix, disk_fix, xc, yc, is_3D=False, hand_fix=None, 
            fast=True, add_halo=False, halo_cfg=None, halo_fix=False, mask=None, 
            add_bar=False, bar_cfg=None, bar_fix=False, **kwargs):

    '''
    if not(add_halo):
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
    '''
    #else:
    #print('add halo', halo_fix)
    model = build_model(bulge_model,
                            disk_model,
                            bulge_cfg,
                            disk_cfg,
                            bulge_fix,
                            disk_fix,
                            xc,
                            yc,
                            is_3D,
                            hand_fix=hand_fix, 
                            halo_cfg=halo_cfg, 
                            add_halo=add_halo,
                            halo_fix=halo_fix,
                            add_bar=add_bar,
                            bar_cfg=bar_cfg,
                            bar_fix=bar_fix,
                            )

    if fast:
        result, imfit = fast_imfit_fit(image, model, sigma,  bulge_model, disk_model, hand_fix, bulge_fix, disk_fix, is_3D, mask=mask, **kwargs)
    else:
        print('model', model)
        result, imfit = imfit_fit(image, model, sigma, mask=mask,  **kwargs)

    #print(imfit.getModelDescription())
    return result, imfit

def imfit_fit(image, model, sigma, mask, **kwargs):
    imfit = pyimfit.Imfit(model)
    result = imfit.fit(image, error=sigma,ftol=1e-6, verbose=1, mask=mask, **kwargs)

    return result, imfit

def fast_imfit_fit(image, model, sigma, bulge_model, disk_model, hand_fix, bulge_fix, disk_fix, is_3D, mask, **kwargs):
    print('run fast')
    imfit_nm = pyimfit.Imfit(model)
    result_nm = imfit_nm.fit(image, error=sigma, ftol=1e-3, solver='NM', verbose=1, mask=mask)
    state = get_fit_state(imfit_nm)
    new_model = build_model(bulge_model,
                        disk_model,
                        state["functions"]["bulge"],
                        state["functions"]["disk"],
                        bulge_fix,
                        disk_fix,
                        state["xc"],
                        state["yc"],
                        is_3D,
                        hand_fix=hand_fix
                        )
    imfit_lm = pyimfit.Imfit(new_model)
    result_lm = imfit_lm.fit(image, error=sigma, ftol=1e-6, verbose=1, solver='LM', mask=mask)
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



