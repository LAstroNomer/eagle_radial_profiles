from astropy.io import  fits
import subprocess as sp
import numpy as np
from scipy.interpolate import interp1d
from bin.common_functions import flux_to_sb
from scipy.ndimage import rotate

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


def get_ell(r, ell, x):
    foo = interp1d(r, ell)
    return foo(x)