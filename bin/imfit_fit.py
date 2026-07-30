from astropy.io import  fits
import subprocess as sp
import numpy as np
from scipy.interpolate import interp1d
from bin.common_functions import flux_to_sb
from scipy.ndimage import rotate
import re

def get_image_params(fname):
    header = fits.getheader(fname)
    gain      = header['CELL.GAIN']
    readnoise = header['CELL.READNOISE']
    n_images  = header['NINPUTS']
    exptime   = header['EXPTIME']
    return gain, readnoise, exptime, n_images


def plot_slices(ax=None, pa_bar=0, w=400, zp=25, pixscale=0.25, flux_convert=False, psf=None):

    header = ''
    models = dict()
    with open('bestfit_parameters_imfit.dat', 'r') as ff:
        for line in filter(None, (line.rstrip() for line in ff)):
            if (line[0] == '#') or (line[0] == 'Y') or (line[0] == 'X'):
                header += line + '\n'
                continue
            if line[0] == 'F':
                tmp_func = line
                models[tmp_func] = []
                continue
            # print(line)
            models[tmp_func].append(line)
    xs = []
    ys = np.zeros(w)
    model_data = dict()
    for key in models:
        with open('tmp.dat', 'w') as ff:
            print(header, file=ff)
            print(key, file=ff)
            for a in models[key]:
                print(a, file=ff)
        if not(psf is None):
            sp.call('makeimage18 tmp.dat --refimage=tmp.fits  -o dd.fits --psf=%s' % psf, shell=True)
        else:
            sp.call('makeimage18 tmp.dat --refimage=tmp.fits  -o dd.fits', shell=True)

        if flux_convert:
            tmp_image = flux_to_sb(fits.getdata('dd.fits'), pixscale)
        else:
            tmp_image = fits.getdata('dd.fits')
        model_data[key.split()[-1]] = tmp_image

        if not(ax is None):
            h, w = tmp_image.shape
            x = np.arange(w) - w // 2
            image_r = rotate(tmp_image, pa_bar, reshape=False)

            y = image_r[:, w // 2]
            ys += y
            ax.plot(x, zp - 2.5 * np.log10((y)), '-', label=key)
    if not(ax is None):
        ax.plot(x, zp - 2.5 * np.log10((ys)), '-', color='magenta', label='total')
    
    total = np.zeros(tmp_image.shape)
    for key  in model_data:
        total += model_data[key]

    model_data['total'] = total

    return ax, model_data


def imfit_edge(image, mask, xc, yc, Ie, re, n, ell_e, pa_e, I0d, hs, rbs, pa_d, z0, sky=0, noise=0, gain=0, readnoise=0,
               exptime=0, n_images=0, psf='', n_z=100):
    fits.writeto('tmp.fits', image, overwrite=True)
    fits.writeto('mask.fits', mask, overwrite=True)

    if len(rbs) == 0:
        cf = '''
X0 %s  
Y0 %s  
FUNCTION Sersic_GenEllipse
PA %s -180,180
ell %s 0,1
c0 0  -1,1
n   %s  0.0,10
I_e %s   0,Infinity             # counts/pixel
r_e %s 0,100
FUNCTION ExponentialDisk3D
PA %s   -180,180
inc 90 fixed
J_0 %s 0,Infinity
h  %s 0.1,1000
n %i fixed
z_0 %s 0,1000
''' % (xc, yc, pa_e, ell_e, n, Ie, re, pa_d, I0d, hs[0], n_z, z0)
    elif len(rbs) == 1:
        cf = '''
X0 %s  
Y0 %s  
FUNCTION Sersic_GenEllipse
PA  %s -180,180
ell %s 0,1
c0 0  -1,1
n   %s  0,10
I_e %s   0,Infinity             # counts/pixel
r_e %s 0,100
FUNCTION BknExp3D
PA  %s -180,180
inc 90 fixed
J_0 %s 0,Infinity
h1  %s  0,1000
h2  %s 0,1000
r_break %s 0,500
n %i fixed
z_0  %s 0,1000

''' % (xc, yc, pa_e, ell_e, n, Ie, re, pa_d, I0d, hs[0], hs[1], rbs[0], n_z, z0)
    else:
        cf = '''
X0 %s  
Y0 %s  
FUNCTION Sersic_GenEllipse
PA  %s -180,180
ell %s 0,1
c0 0 -1,1
n   %s  0,10
I_e %s   0,Infinity             # counts/pixel
r_e %s 0,100
FUNCTION DblBknExp3D
PA %s -180,180
inc 90 fixed
J_0 %s 0,Infinity
h1 %s 0,1000
h2 %s 0,1000
h3 %s 0,1000
r_break1 %s 0,1000
r_break2 %s 0,1000
n %i fixed
z_0 %s 0,1000
''' % (xc, yc, pa_e, ell_e, n, Ie, re, pa_d, I0d, hs[0], hs[1], hs[2], rbs[0], rbs[1], n_z, z0)

    with open('imfit.cfg', 'w') as ff:
        print(cf, file=ff)

    sp.call('makeimage18 imfit.cfg --refimage=tmp.fits --psf=%s' % psf, shell=True)
    sp.call(
        'imfit18 -c=imfit.cfg tmp.fits --mask=mask.fits --save-model=model.fits  --save-residual=res.fits '
        '--psf=%s --gain=%s --readnoise=%s --ftol=1e-4' % (
            psf, gain, readnoise), shell=True)
    model = read_imfit('bestfit_parameters_imfit.dat')
    mod = fits.getdata('model.fits')
    res = fits.getdata('res.fits')

    return mod, res, model, mask


def read_imfit(file):
    res = dict()
    with open(file, 'r') as ff:
        for line in filter(None, (line.rstrip() for line in ff)):
            if (line[0] == "#") or (line[0] == 'F'):
                continue
            # print(line)
            key, val, _, _, err = line.split()
            key1 = key
            for i in range(10):
                if key1 in res:
                    key1 = key + '_' + str(i)
            res[key1] = np.array([float(val), float(err)])
    return res

def parse_imfit_output(filename: str) -> dict[str, any]:
    """
    Парсит выходной файл imfit в структурированный словарь Python.

    Parameters
    ----------
    filename : str
        Путь к файлу с результатами imfit

    Returns
    -------
    Dict[str, Any]
        Словарь с разобранными данными

    Structure:
    --------
    result
├── metadata: {}
│
├── fit_info
│   ├── best_fit_value: float
│   ├── reduced_value: float
│   ├── AIC: float
│   ├── BIC: float
│   ├── algorithm: str
│   ├── statistic: str
│   │
│   └── global_params (опционально)
│       ├── X0
│       │   ├── value: float
│       │   ├── error: float | None
│       │   └── limits: str | None
│       │
│       └── Y0
│           ├── value: float
│           ├── error: float | None
│           └── limits: str | None
│
└── functions: list[]
    │
    └── [0..N]
        ├── name: str
        │
        └── parameters: dict{}
            ├── param1
            │   ├── value: float | str
            │   ├── error: float | str | None
            │   └── limits: str | None
            │
            ├── param2
            │   ├── value: float | str
            │   ├── error: float | str | None
            │   └── limits: str | None
            │
            └── paramN
                ├── value: float | str
                ├── error: float | str | None
                └── limits: str | None
    """
    result = {
        'metadata': {},
        'fit_info': {},
        'functions': []
    }

    current_function = None

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Пропускаем пустые строки
        if not line:
            continue

        # Парсим метаданные из комментариев
        if line.startswith('#'):
            if 'Best-fit value:' in line:
                result['fit_info']['best_fit_value'] = float(line.split(':')[1].strip())
            elif 'Reduced value:' in line:
                result['fit_info']['reduced_value'] = float(line.split(':')[1].strip())
            elif 'AIC:' in line:
                result['fit_info']['AIC'] = float(line.split(':')[1].strip())
            elif 'BIC:' in line:
                result['fit_info']['BIC'] = float(line.split(':')[1].strip())
            elif 'Algorithm:' in line:
                result['fit_info']['algorithm'] = line.split('Algorithm:')[1].split('--')[0].strip()
            elif 'Fit statistic:' in line:
                result['fit_info']['statistic'] = line.split('Fit statistic:')[1].strip()

        # Парсим глобальные параметры (X0, Y0)
        elif line.startswith('X0') or line.startswith('Y0'):
            parts = line.split()
            param_name = parts[0]
            param_value = float(parts[1])

            if 'global_params' not in result['fit_info']:
                result['fit_info']['global_params'] = {}

            # Парсим ошибку, если есть
            error = None
            if '#' in line:
                error_match = re.search(r'\+/-\s*([\d\.]+)', line)
                if error_match:
                    error = float(error_match.group(1))
            limits = None
            if len(line.split()) == 3:
                limits = line.split()[-1]

            result['fit_info']['global_params'][param_name] = {
                'value': param_value,
                'error': error,
                'limits': limits
            }

        # Начало новой функции
        elif line.startswith('FUNCTION'):
            func_name = line.split()[1]
            current_function = {
                'name': func_name,
                'parameters': {}
            }
            result['functions'].append(current_function)

        # Парсим параметры функции
        elif current_function is not None and line:
            parts = line.split()
            if len(parts) >= 2:
                param_name = parts[0]

                # Парсим значение (может быть числом или строкой)
                try:
                    param_value = float(parts[1])
                except ValueError:
                    param_value = parts[1]  # оставляем строкой, если не число

                # Парсим ошибку
                error = None
                if '#' in line:
                    error_match = re.search(r'\+/-\s*([\d\.]+)', line)
                    if error_match:
                        try:
                            error = float(error_match.group(1))
                        except:
                            error = error_match.group(1)
                limits = None
                if len(line.split()) == 3:
                    limits = line.split()[-1]

                current_function['parameters'][param_name] = {
                    'value': param_value,
                    'error': error,
                    'limits': limits
                }

    return result


def get_ell(r, ell, x):
    foo = interp1d(r, ell)
    return foo(x)