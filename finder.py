import copy
import numpy as np
from tqdm import tqdm
import pyimfit

from visualisation import MakeImage
from fit import fit_step, get_fit_state, build_model


def multi_start_fit(
    image,
    sigma,
    bulge_model,
    disk_model,
    bulge_cfg,
    disk_cfg,
    xc,
    yc,
    bulge_fix=False,
    disk_fix=False,
    n_starts=3,
    scatter=None,
    hand_fix=None,
    is_3D=False,
    fast=True,
    halo_cfg=None,
    halo_fix=True,
):
    """
    Запускает fit несколько раз со случайными начальными условиями.

    scatter = 0.2 означает разброс ±20% относительно стартового значения.
    """

    if scatter is None:
        scatter = 0.2
    results = []

    rng = np.random.default_rng()

    for _ in tqdm(range(n_starts)):

        bulge = copy.deepcopy(bulge_cfg)
        disk = copy.deepcopy(disk_cfg)

        # ---- случайно шевелим параметры ----
        if bulge_fix:
            variances = (disk, disk)
        elif disk_fix:
            variances = (bulge, bulge)
        else:
            variances = (bulge, disk)

        for params in variances:

            for name, values in params.items():
                #print(name, values)
                value, low, high = values

                if low == high:
                    continue

                # координаты можно не трогать
                if name in ("PA", "ell", "inc"):
                    continue
                
                #if name == "n":
                #    new_value = rng.uniform(0.6,4)

                #    new_value = np.clip(new_value, low, high)

                #    params[name][0] = new_value
                #    print(name, new_value)
                if name == "r_break":
                    new_value = rng.uniform(low,high)
                    params[name][0] = new_value
                    print(name, new_value)
                    continue

                if name == "r_break1":
                    new_value = rng.uniform(low,high)
                    params[name][0] = new_value
                    print(name, new_value)
                    rb1 = new_value
                    continue

                if name == "r_break2":
                    new_value = rng.uniform(rb1,high)
                    params[name][0] = new_value
                    print(name, new_value)
                    continue
                
                             
                new_value = value * rng.uniform(1 - scatter[name],
                                                1 + scatter[name])

                new_value = np.clip(new_value, low, high)

                params[name][0] = new_value

        result, imfit = fit_step(
            image,
            sigma,
            bulge_model=bulge_model,
            disk_model=disk_model,
            bulge_cfg=bulge,
            disk_cfg=disk,
            bulge_fix=bulge_fix,
            disk_fix=disk_fix,
            xc=xc,
            yc=yc,
            hand_fix=hand_fix,
            is_3D=is_3D,
            fast=fast,
            halo_cfg=halo_cfg,
            halo_fix=halo_fix,
            add_halo=not(halo_cfg is None), 
        )

        state = get_fit_state(imfit)

        if check_correct_fit(imfit, state, image.shape) :


            results.append(
                {
                    "fit": result,
                    "imfit": imfit,
                    "chi2": result.fitStat,
                    "aic": result.aic,
                    "bic": result.bic,
                }
            )

    results.sort(key=lambda x: x["chi2"])

    return results



def check_correct_fit(imfit, state, shape):

    makeimage = MakeImage(imfit)
    if makeimage.disk["name"] == "Exponential" or makeimage.disk["name"] == "ExponentialDisk3D":
        return True

    if 'r_break' in state["functions"]["disk"]:
        if state["functions"]["disk"]["r_break"][0] > 225:
            print("too big rb", state["functions"]["disk"]["r_break"][0])
            return False

    if state["functions"]["disk"]["h1"][0] > 100:
        print("too big h1", state["functions"]["disk"]["h1"][0])
        return False

    if 'r_break1' in state["functions"]["disk"]:
        if state["functions"]["disk"]["r_break1"][0] > state["functions"]["disk"]["r_break2"][0]:
            print("rb1 large rb2", 'rb1', state["functions"]["disk"]["r_break1"][0],
                   'rb2', state["functions"]["disk"]["r_break2"][0])
            return False

        if state["functions"]["disk"]["r_break1"][0] > 250:
            print("too large rb1", state["functions"]["disk"]["r_break1"][0])
            return False
        if state["functions"]["disk"]["r_break2"][0] > 250:
            print("too large rb2", state["functions"]["disk"]["r_break2"][0])
            return False



    #bulge_image = makeimage.get_model_image(label="bulge", size=shape)
    #disk_image = makeimage.get_model_image(label="disk", size=shape)

    #bt = np.sum(bulge_image)/(np.sum(bulge_image) + np.sum(disk_image))

    #if bt > 0.8:
     #   print("too big bulge", bt)
     #   return False

    return True

    

def multy_fit_with_init_guess(guess_model, 
                                image, 
                                sigma, 
                                scatter, 
                                hand_fix, 
                                bulge_fix,
                                is_3D=False, 
                                fast=True,
                                halo_fix=True,
                                halo_cfg=None,
                                add_halo=True):
    with open(guess_model, 'r') as ff:
        lines = ff.readlines()[:-8]
                    
                            
    model_desc = pyimfit.parse_config(lines)
    imfit = pyimfit.Imfit(model_desc)
    state = get_fit_state(imfit)
    
    for line in lines:
        if 'LABEL' in line:
            line_list = line.split()
            if line_list[-1] == 'bulge':
                bulge_model = line_list[1]
            if line_list[-1] == 'disk':
                disk_model  = line_list[1]
    
    double_break_model = build_model(bulge_model, disk_model,
                     bulge_cfg=state["functions"]["bulge"], 
                     disk_cfg=state["functions"]["disk"],
                     bulge_fix=bulge_fix, disk_fix=False, 
                     xc=state["xc"],
                     yc=state["yc"],
                     hand_fix=hand_fix,
                     is_3D=is_3D,
                     add_halo=add_halo,
                     halo_cfg=halo_cfg,
                     halo_fix=halo_fix,
                     )
    new_imfit = pyimfit.Imfit(double_break_model)
    result = new_imfit.fit(image, error=sigma, ftol=1e-4, verbose=1)


    state = get_fit_state(new_imfit)
    state["functions"]["bulge"]['PA'] += [-180,180]
    state["functions"]["bulge"]['ell'] += [0.01,0.8]
    state["functions"]["bulge"]['r_e'] += [0.1,3]
    state["functions"]["bulge"]['I_e'] += [0,1e5]
    state["functions"]["bulge"]['n'] += [0.5,5]

    re = state["functions"]["bulge"]["r_e"][0]
    state["functions"]["disk"]['PA'] += [-180,180]
    if 'ell' in state['functions']['disk']:
        state["functions"]["disk"]['ell'] += [0.01,0.8]
    else:
        state["functions"]["disk"]['inc'] += [0,90]
        state["functions"]["disk"]['J_0'] += [0,1e5]
        if 'h1' in state['functions']['disk']:
            state["functions"]["disk"]['h1'] += [0,64]
            state["functions"]["disk"]['h2'] += [0,64]
            state["functions"]["disk"]['r_break'] += [3*re,64]
        else:
            state["functions"]["disk"]['h'] += [0,64]

        if 'h3' in state['functions']['disk']:
            state["functions"]["disk"]['h3'] += [0,64]
            state["functions"]["disk"]['r_break1'] += [3*re,64]
            state["functions"]["disk"]['r_break2'] += [3*re,64]


        state["functions"]["disk"]['n'] += [1,100]
        state["functions"]["disk"]['z_0'] += [0,64]

    print(state)
    #exit()
        
    results = multi_start_fit(image, sigma,
                            bulge_model,
                            disk_model,
                            state["functions"]["bulge"],
                            state["functions"]["disk"],
                            state["xc"],
                            state["yc"],
                            bulge_fix=bulge_fix,
                            disk_fix=False,
                            n_starts=10,
                            scatter=scatter,
                            hand_fix=hand_fix,
                            is_3D=is_3D, fast=fast,
                            halo_cfg=halo_cfg,
                            halo_fix=halo_fix,
                            )

    return results



