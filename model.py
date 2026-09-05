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
        if (len(values) > 1):
            min_val = values[1]
            max_val = values[2]

        if param_name == "alpha":
            fixed_ = False
            value = 1.0
        else:
            fixed_ = fixed
        
        # Проверяем, есть ли у компонента такой параметр
        if hasattr(component, param_name):
            if len(values) > 1:
                getattr(component, param_name).setValue(value, [min_val, max_val], fixed=fixed_)
            else:
                getattr(component, param_name).setValue(value, fixed=fixed_)

        else:
            print(f"Предупреждение: параметр '{param_name}' не найден в компоненте")

def init_default_bulge(bulge_model, bulge_fix=False):
    bulge = pyimfit.make_imfit_function(bulge_model, label='bulge')

    bulge.PA.setValue(90,[0,180], fixed=bulge_fix)
    bulge.ell.setValue(0.2, [0.0,0.7], fixed=bulge_fix)
    bulge.I_e.setValue(10, [0, 10**5], fixed=bulge_fix)
    bulge.r_e.setValue(5, [0.1, 3], fixed=bulge_fix)
    bulge.n.setValue(1,[0.5,4], fixed=bulge_fix)
    return bulge

def init_default_halo(bulge_model, bulge_fix=False):
    bulge = pyimfit.make_imfit_function(bulge_model, label='halo')

    bulge.PA.setValue(90,[0,180], fixed=bulge_fix)
    bulge.ell.setValue(0.2, [0.01,0.6], fixed=bulge_fix)
    bulge.I_e.setValue(1, [0, 10**5], fixed=bulge_fix)
    bulge.r_e.setValue(25, [10, 60], fixed=bulge_fix)
    bulge.n.setValue(1,[0.5,6], fixed=bulge_fix)
    return bulge


def init_default_disk(disk_model, disk_fix=False, is_3D=False):
    disk = pyimfit.make_imfit_function(disk_model, label='disk')
    
    disk.PA.setValue(90, [0,180], fixed=disk_fix)
    disk.h.setValue(25, [0.1,64], fixed=disk_fix)

    if is_3D:
        disk.n.setValue(1, [1,100], fixed=disk_fix)
        disk.z_0.setValue(1, [0.1, 64], fixed=disk_fix)
        disk.J_0.setValue(10, [0,10**5], fixed=disk_fix)
        disk.inc.setValue(0, [0,90], fixed=disk_fix)
    else:
        disk.ell.setValue(0.2, [0.01, 0.7], fixed=disk_fix)
        disk.I_0.setValue(20, [0,10**5], fixed=disk_fix)

    return disk

def init_default_bar(bar_model, bar_fix, is_3D=True):
    bar = pyimfit.make_imfit_function(bar_model, label='bar')

    bar.PA.setValue(90, [-180, 180], fixed=bar_fix)
    bar.inc.setValue(0, [0, 90], fixed=bar_fix)
    bar.barPA.setValue(45, [0, 180], fixed=bar_fix)

    bar.J_0.setValue(10, [0, 1e5], fixed=bar_fix)
    bar.R_bar.setValue(10, [1, 25], fixed=bar_fix)
    bar.q.setValue(0.3, [0.1, 0.5], fixed=bar_fix)
    bar.q_z.setValue(0.3, [0.1, 0.5], fixed=bar_fix)
    bar.n.setValue(2, fixed=True)
    return bar




def set_broken_disk_from_exponential(disk_model, disk_cfg, bulge_cfg, disk_fix=False, is_3D=False):
    disk = pyimfit.make_imfit_function(disk_model, label='disk')

    disk = pyimfit.make_imfit_function(disk_model, label='disk')
    disk.PA.setValue(disk_cfg["PA"][0], disk_cfg["PA"][1:], fixed=disk_fix)
    
    # заменить h на h1 и h2
    h  = disk_cfg["h"][0]
    re = bulge_cfg["r_e"][0]
    disk.h1.setValue(h, [0.1*h, 10*h], fixed=disk_fix)
    disk.h2.setValue(h, [0.1*h, 10*h], fixed=disk_fix)
    disk.r_break.setValue(3*h, [re*3, 64], fixed=disk_fix)

    if is_3D:
        disk.J_0.setValue(disk_cfg["J_0"][0], disk_cfg["J_0"][1:], fixed=disk_fix)
        disk.n.setValue(disk_cfg["n"][0], disk_cfg["n"][1:], fixed=disk_fix)
        disk.z_0.setValue(disk_cfg["z_0"][0], disk_cfg["z_0"][1:], fixed=disk_fix)
        disk.inc.setValue(disk_cfg["inc"][0], disk_cfg["inc"][1:], fixed=disk_fix)
    else:
        disk.alpha.setValue(1, [1, 100], fixed=False)  
        disk.I_0.setValue(disk_cfg["I_0"][0], disk_cfg["I_0"][1:], fixed=disk_fix)
        disk.ell.setValue(disk_cfg["ell"][0], disk_cfg["ell"][1:], fixed=disk_fix)
   
    return disk


def set_double_broken_disk_from_exponential(
        disk_model, disk_cfg, bulge_cfg, 
        disk_fix=False, is_3D=False):
    
    disk = pyimfit.make_imfit_function(disk_model, label='disk')

    disk = pyimfit.make_imfit_function(disk_model, label='disk')
    disk.PA.setValue(disk_cfg["PA"][0], [-180,180], fixed=disk_fix)
    #disk.ell.setValue(disk_cfg["ell"][0], [0.01, 0.8], fixed=disk_fix)
    #disk.I_0.setValue(disk_cfg["I_0"][0], [0,1e5], fixed=disk_fix)

    # заменить h на h1 и h2
    h1  = disk_cfg["h1"][0]
    h2  = disk_cfg["h2"][0]
    r_break  = disk_cfg["r_break"][0]
    re = bulge_cfg["r_e"][0]
    disk.h1.setValue(h1, [0.1, 100], fixed=disk_fix)
    disk.h2.setValue(h2, [0.1, 100], fixed=disk_fix)
    disk.h3.setValue(h2, [0.1, 100], fixed=disk_fix)
    disk.r_break1.setValue(r_break, [re*3, 128], fixed=disk_fix)
    disk.r_break2.setValue(r_break, [re*3, 128], fixed=disk_fix)
    #disk.alpha.setValue(1, [1, 100], fixed=False)
    if is_3D:
        disk.J_0.setValue(disk_cfg["J_0"][0], [0, 1e5], fixed=disk_fix)
        disk.n.setValue(disk_cfg["n"][0], [1, 100], fixed=disk_fix)
        disk.z_0.setValue(disk_cfg["z_0"][0], [0, 250], fixed=disk_fix)
        disk.inc.setValue(disk_cfg["inc"][0], [0, 90], fixed=disk_fix)
    else:
        disk.alpha.setValue(1, [1, 100], fixed=False)  
        disk.I_0.setValue(disk_cfg["I_0"][0], [0,1e5], fixed=disk_fix)
        disk.ell.setValue(disk_cfg["ell"][0], [0.01, 0.8], fixed=disk_fix)
    



    return disk

def build_model(bulge_model, 
                disk_model,
                bulge_cfg=None,
                disk_cfg=None,
                bulge_fix=False,
                disk_fix=False,
                xc=250,
                yc=250,
                is_3D=False,
                hand_fix=None,
                add_halo=False,
                halo_cfg=None,
                halo_fix=False,
                bar_cfg=None,
                add_bar=False,
                bar_fix=False,
                ):

    #print('halo_fix', halo_fix)
    # -------------------------------
    # Bulge
    # -------------------------------

    if bulge_cfg is None:
       bulge = init_default_bulge(bulge_model, bulge_fix=bulge_fix)
    else:
        bulge = pyimfit.make_imfit_function(bulge_model, label='bulge')
        set_parameters_from_dict(bulge, bulge_cfg, bulge_fix)
    if 'bulge' in hand_fix:
        for key, value in hand_fix['bulge'].items():
            getattr(bulge,key).setValue(value[0], fixed=value[1])


    # -------------------------------
    # Disk
    # -------------------------------
    if disk_cfg is None:
        disk = init_default_disk(disk_model, disk_fix, is_3D=is_3D)
    elif disk_model == "BrokenExponential" or disk_model == "BknExp3D":
        if "h1" in disk_cfg:
            disk = pyimfit.make_imfit_function(disk_model, label='disk')
            set_parameters_from_dict(disk, disk_cfg, disk_fix)
        else:
            disk = set_broken_disk_from_exponential(disk_model, disk_cfg, bulge_cfg, 
                                                    disk_fix, is_3D=is_3D)
    elif disk_model == "Exponential" or disk_model == "ExponentialDisk3D":
        disk = pyimfit.make_imfit_function(disk_model, label='disk')
        set_parameters_from_dict(disk, disk_cfg, disk_fix)

    elif disk_model == "doublebroken-exp" or disk_model == 'DblBknExp3D':
        if "h3" in disk_cfg:
            disk = pyimfit.make_imfit_function(disk_model, label='disk')
            set_parameters_from_dict(disk, disk_cfg, disk_fix)
        else:
            disk = set_double_broken_disk_from_exponential(disk_model, disk_cfg,
                                                            bulge_cfg, disk_fix,is_3D=is_3D)
    if not(hand_fix is None):
        for key, value in hand_fix['disk'].items():
            getattr(disk,key).setValue(value[0], fixed=value[1])
    disk.n.setValue(1, fixed=True)

    #--------------------------------
    # Halo
    #--------------------------------
    if add_halo:
        if halo_cfg is None:
            halo = init_default_halo('Sersic')
        else:
            halo = pyimfit.make_imfit_function('Sersic', label='halo')
            set_parameters_from_dict(halo, halo_cfg, halo_fix)
            if halo_fix:
                #halo.ell.setValue(halo_cfg['ell'][0], [halo_cfg['ell'][1], halo_cfg['ell'][2]], fixed=False )
                halo.I_e.setValue(halo_cfg['I_e'][0], [0.0, halo_cfg['I_e'][0]], fixed=False )
    
    # --------------------------------
    # Bar
    # --------------------------------
    if add_bar:
        print('Add Bar')
        print(bar_cfg)
        if bar_cfg is None:
            bar = init_default_bar('FerrersBar3D', bar_fix)
        else:
            bar = pyimfit.make_imfit_function('FerrersBar3D', label='bar')
            set_parameters_from_dict(bar, bar_cfg, bar_fix)
            if bar_fix:
                #halo.ell.setValue(halo_cfg['ell'][0], [halo_cfg['ell'][1], halo_cfg['ell'][2]], fixed=False )
                bar.q_z.setValue(bar_cfg['q_z'][0], [0.1, 0.8], fixed=False )
        if 'bar' in hand_fix:
            for key, value in hand_fix['bar'].items():
                getattr(bar,key).setValue(value[0], fixed=value[1])
    
    
    # -------------------------------
    # Components
    # -------------------------------
    model = pyimfit.SimpleModelDescription()
    model.x0.setValue(xc, [xc - 10, xc + 10])
    model.y0.setValue(yc, [yc - 10, yc + 10])
    model.addFunction(bulge)
    model.addFunction(disk)
    if add_halo:
        model.addFunction(halo)
    if add_bar:
        model.addFunction(bar)
    return model