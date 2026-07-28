import scipy.special as sc
from astropy.modeling.functional_models import Sersic1D
import  numpy as np
from matplotlib import pyplot as plt
import warnings
from scipy.optimize import  curve_fit
from scipy.interpolate import interp1d

def disk_edge(r: np.ndarray, I0: float, h: float, *args):
    x = r / h
    return I0 * x * sc.kn(1, x)


def sersic_e(r: np.ndarray, Ie: float, re: float, n: float):
    '''
    Модель Сёрсика в терминах "интенсивности". Из пакета astropy.
    Ie --- интенсивность на re
    '''
    s = Sersic1D(amplitude=Ie, r_eff=re, n=n)
    return s(r)


def bulge_fit_prep(r, J, Je, rin, I0, hs):
    #if rin == 0:


    ind = np.where(r < rin)
    x1 = r[ind]
    I_err1 = Je[ind]



    I1 = (J - disk_edge(r, I0, hs[0]))[ind]

    ind = np.where(I1 > 0)
    x1     = x1[ind]
    I_err1 = I_err1[ind]
    I1     = I1[ind]

    print('len', len(x1))
    if len(x1) < 3:
        return [0,0,0], 0, 0, 0, np.zeros((3,3))
    pb, covp = bulge_fit(x1, I1, I_err1, plot=False)


    Ie, re, n = pb
    return pb, Ie, re, n, covp


def bulge_fit(r, I, I_err, plot=False):
    ind = np.where(I > 0)
    r = r[ind]
    I = I[ind]
    I_err = I_err[ind]

    rmin = np.min(r)
    rmax = np.max(r)

    r_12 = (rmin + rmax) / 2.
    r_14 = (rmin + r_12) / 2.

    foo = interp1d(r, I, kind='cubic')
    t = np.linspace(rmin, rmax, 100)
    It = foo(t)

    I_12 = foo(r_12)
    I_14 = foo(r_14)

    SSE = np.inf
    p = None

    def foo(x, *p):
        Ie, re, n, C = p
        return sersic_e(x, Ie, re, n) + C

    with warnings.catch_warnings(action="ignore"):
        for n in np.arange(0.5, 6, 0.5):

            bn = sc.gammaincinv(2 * n, 0.5)

            re = bn * (r_14 ** (1 / n) - (r_12) ** (1 / n)) / (np.log(I_12) - np.log(I_14))
            if re > 50:
                re = np.max(r) / 2
            I0 = (I_12 * np.exp(bn * (r_12 / re) ** (1 / n)) + I_14 * np.exp(bn * (r_14 / re) ** (1 / n))) / 2.
            Ie = I0 * np.exp(-bn)
            C = np.min(It)

            try:
                p_, covp_, info_dict, mesg, ier = curve_fit(foo, xdata=t, ydata=It, p0=[Ie, re, n, C], full_output=True)
            except:
                continue
            SSE_ = np.sum(ier ** 2)

            if SSE_ < SSE:
                SSE = SSE_
                p = p_
                covp = covp_
    if p is None:
        print('Gauss')

        def foo(x, *p):
            return p[0] * np.exp(-0.5 * (x / p[1]) ** 2) + p[2]

        I0 = np.max(It)
        sigma = np.max(t) / 2
        C = np.min(It)

        p_, covp_, info_dict, mesg, ier = curve_fit(foo, xdata=t, ydata=It, p0=[I0, sigma, C], full_output=True)
        Ie = p_[0] * np.exp(-2 / 3)
        re = np.sqrt(3) * p_[1]
        n = 0.5
        return [Ie, re, n], covp_
    Ie, re, n, _ = p
    print('Ie: ', Ie)
    print('re: ', re)
    print('n: ', n)

    if plot:
        plt.figure()
        plt.plot(r, np.log10(I), '-')
        plt.plot(r, np.log10(foo(r, *p)), '-r')
        plt.show()
    return p[:-1], covp