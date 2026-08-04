import pyimfit  # type: ignore


def set_parameters_from_dict(component, params_dict, fixed=False):
    """
    Автоматически устанавливает параметры компонента из словаря.
    
    Args:
        component: объект компонента (например, bulge)
        params_dict: словарь с параметрами {имя: [значение, min, max]}
    """
    for param_name, values in params_dict.items():
        #print('value', values)
        value = values[0]
        min_val = values[1]
        max_val = values[2]

        if param_name == "alpha":
            fixed_ = True
            value = 1.0
        else:
            fixed_ = fixed
        
        # Проверяем, есть ли у компонента такой параметр
        if hasattr(component, param_name):
            
            getattr(component, param_name).setValue(value, [min_val, max_val], fixed=fixed_)
        else:
            print(f"Предупреждение: параметр '{param_name}' не найден в компоненте")

def init_default_bulge(bulge_model, bulge_fix=False):
    bulge = pyimfit.make_imfit_function(bulge_model, label='bulge')

    bulge.PA.setValue(90,[0,180], fixed=bulge_fix)
    bulge.ell.setValue(0.2, [0.01,0.7], fixed=bulge_fix)
    bulge.I_e.setValue(100, [0, 10**5], fixed=bulge_fix)
    bulge.r_e.setValue(5, [0.1, 25], fixed=bulge_fix)
    bulge.n.setValue(1,[0.5,4], fixed=bulge_fix)
    return bulge


def init_default_disk(disk_model, disk_fix=False):
    disk = pyimfit.make_imfit_function(disk_model, label='disk')
    
    disk.PA.setValue(90, [0,180], fixed=disk_fix)
    disk.ell.setValue(0.2, [0.01, 0.7], fixed=disk_fix)
    disk.I_0.setValue(20, [0,10**5], fixed=disk_fix)
    disk.h.setValue(25, [0.1,250], fixed=disk_fix)

    return disk



def set_broken_disk_from_exponential(disk_model, disk_cfg, bulge_cfg, disk_fix=False):
    disk = pyimfit.make_imfit_function(disk_model, label='disk')

    disk = pyimfit.make_imfit_function(disk_model, label='disk')
    disk.PA.setValue(disk_cfg["PA"][0], disk_cfg["PA"][1:], fixed=disk_fix)
    disk.ell.setValue(disk_cfg["ell"][0], disk_cfg["ell"][1:], fixed=disk_fix)
    disk.I_0.setValue(disk_cfg["I_0"][0], disk_cfg["I_0"][1:], fixed=disk_fix)

    # заменить h на h1 и h2
    h  = disk_cfg["h"][0]
    re = bulge_cfg["r_e"][0]
    disk.h1.setValue(h, [0.1*h, 10*h], fixed=disk_fix)
    disk.h2.setValue(h, [0.1*h, 10*h], fixed=disk_fix)
    disk.r_break.setValue(3*h, [re*3, 250], fixed=disk_fix)
    disk.alpha.setValue(2, [1, 100], fixed=True)
    return disk

def build_model(bulge_model, disk_model,
                 bulge_cfg=None, disk_cfg=None,
                 bulge_fix=False, disk_fix=False, xc=250, yc=250):

    # -------------------------------
    # Bulge
    # -------------------------------
    if bulge_cfg is None:
       bulge = init_default_bulge(bulge_model, bulge_fix=bulge_fix)
    else:
        bulge = pyimfit.make_imfit_function(bulge_model, label='bulge')
        set_parameters_from_dict(bulge, bulge_cfg, bulge_fix)


    # -------------------------------
    # Disk
    # -------------------------------
    if disk_cfg is None:
        disk = init_default_disk(disk_model, disk_fix)
    elif disk_model == "BrokenExponential":
        if "h1" in disk_cfg:
            disk = pyimfit.make_imfit_function(disk_model, label='disk')
            set_parameters_from_dict(disk, disk_cfg, disk_fix)
        else:
            disk = set_broken_disk_from_exponential(disk_model, disk_cfg, bulge_cfg, disk_fix)
    elif disk_model == "Exponential":
        disk = pyimfit.make_imfit_function(disk_model, label='disk')
        set_parameters_from_dict(disk, disk_cfg, disk_fix)
            

    # -------------------------------
    # Components
    # -------------------------------
    model = pyimfit.SimpleModelDescription()
    model.x0.setValue(xc, [xc - 10, xc + 10])
    model.y0.setValue(yc, [yc - 10, yc + 10])
    model.addFunction(bulge)
    model.addFunction(disk)

    return model