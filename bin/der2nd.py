# %load bin/der2nd.py
from scipy import signal
from astropy.stats import sigma_clipped_stats
import numpy as np
from matplotlib import pyplot as plt


def get_rin_with_2nd_derivate(sma, I, zp=25, rmax=np.inf, plotting=False, ax=None):
    '''
    Определение области преобладания балджа по анализу второй изофоты поверхностной яркости (mu)
    Экспоненциальный диск ->(d/dr)-> константа ->(d/dr)-> 0

    '''

    # Сглаживание скользяшим спредним
    mu = zp - 2.5 * np.log10(I)
    window = signal.windows.boxcar(5) / 5
    mu = signal.convolve(mu, window)[4:-4]

    # Производные
    dmu = np.gradient(mu)
    d2mu = np.gradient(dmu)

    # Вычисление остаточных неоднородностей
    mean, med, std = sigma_clipped_stats(d2mu, sigma=3)
    r_in = 0.0  # float('nan')
    for i in range(len(d2mu) - 1):
        if (((d2mu)[i] <= med - 3*std) and ((d2mu)[i + 1] > med - 3*std)):
            r_in = (sma[2:-2])[i]
            break
    
    print('rin', r_in, rmax)
    if (r_in == 0) or (r_in > rmax):
        for i in range(1, len(d2mu) - 1):
            if (d2mu[i] < d2mu[i-1]) and (d2mu[i] < d2mu[i+1]):
                r_in = (sma[2:-2])[i]
                break
    if plotting:
        fig, axi = plt.subplots(1, 3, figsize=(20, 5))
        print(r_in)
        axi[0].plot(sma[2:-2], mu, '-')
        axi[0].invert_yaxis()
        axi[0].set_xlabel('r, pix')
        axi[0].set_ylabel(r'$\mu, mag \, arcsec^{-1}$')
        axi[1].plot(sma[2:-2], dmu, '-')
        axi[1].set_xlabel('r, pix')
        axi[1].set_ylabel(r"$\mu'_r$")
        axi[2].plot(sma[2:-2], d2mu, '-')
        axi[2].set_xlabel('r, pix')
        axi[2].set_ylabel(r"$\mu''_{rr}$")
        axi[2].axhline(med - std, color='red', ls='--', label='median-std')
        axi[0].axvline(r_in, ls='--', color='green', label=r'$r_{in} = %3.1f$' % r_in)
        axi[1].axvline(r_in, ls='--', color='green', label=r'$r_{in} = %3.1f$' % r_in)
        axi[2].axvline(r_in, ls='--', color='green', label=r'$r_{in} = %3.1f$' % r_in)
        axi[2].legend()
        axi[1].legend()
        axi[0].legend()
        plt.show()
    if not (ax is None):
        ax.plot(sma[2:-2], d2mu, '-', label=r'$d^2\, \mu/d\,r^2$')
        ax.axhline(med - std, ls='--', color='red', label='median-std')
        ax.axvline(r_in, ls='--', color='green', label=r'$r_{in}$')
        ax.set_xlabel('r, pix')
        ax.legend()

    return r_in