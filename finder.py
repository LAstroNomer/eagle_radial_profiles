import copy
import numpy as np
from tqdm import tqdm

from visualisation import MakeImage
from fit import fit_step, get_fit_state


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
    n_starts=20,
    scatter=None,
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

                value, low, high = values

                if low == high:
                    continue

                # координаты можно не трогать
                if name in ("PA", "ell"):
                    continue

                #if name == "n":
                #    new_value = rng.uniform(0.6,4)

                #    new_value = np.clip(new_value, low, high)

                #    params[name][0] = new_value
                #    print(name, new_value)
                #if name == "r_break":
                #    new_value = rng.uniform(low,high)
                #    params[name][0] = new_value
                #    print(name, new_value)

                #else:                    
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
            yc=yc
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
    if makeimage.disk["name"] == "Exponential":
        return True

    if state["functions"]["disk"]["r_break"][0] > 225:
        print("too big rb")
        return False

    if state["functions"]["disk"]["h1"][0] > 100:
        print("too big h1")
        return False

    bulge_image = makeimage.get_model_image(label="bulge", size=shape)
    disk_image = makeimage.get_model_image(label="disk", size=shape)

    bt = np.sum(bulge_image)/(np.sum(bulge_image) + np.sum(disk_image))

    if bt > 0.8:
        print("too big bulge", bt)
        return False

    return True

    



