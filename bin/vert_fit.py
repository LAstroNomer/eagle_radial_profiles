import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import  curve_fit
from matplotlib import  pyplot as plt


def sech2_model(z, *pars):
    I0, z0, zc = pars
    return I0 * (1. / np.cosh((z - zc) / 2 / z0)) ** 2


def get_first_init(z, J):
    J0 = np.max(J)
    zc = np.nansum(z * J) / np.nansum(J)
    t = z - zc
    f = interp1d(t, J / J0)
    l = np.arange(np.min(t), np.max(t), 0.1)
    z0_1 = abs(l[np.argmin(abs(f(l) - 0.41997))])
    z0 = z0_1
    return np.array([J0, z0, zc])


def main_z0(data_, mask, smi=0, sma=500, step=1, plotting=False, ax=None):
    # centering by argmax of image
    data = data_.copy()
    data[mask > 0] = 0
    yc, xc = np.unravel_index(data.argmax(), data.shape)
    data[mask > 0] = float('nan')

    # print('Измерение z0 в диапазоне от smi до  sma ')
    # Измерение z0 в диапазоне от smi до  sma
    h, w = data.shape
    zcs = []
    z00 = []
    rs = []
    s = np.append(np.arange(-sma, -smi, step), np.arange(smi, sma, step))
    prof = []
    for i, k in enumerate(s):
        try:
            x = int(xc + k)
            J_ = np.nanmean(data[:, x - step // 2:x + 1 + step // 2], axis=1)
            z = np.arange(len(J_))

            ind = np.where(~np.isnan(J_))
            J = J_[ind]
            z = z[ind]

            z_cen = z - J.argmax()
            bounds = ([0.0, 0.0, -sma], [np.inf, np.inf, sma])
            p0 = get_first_init(z_cen, J)
            p, p_err = curve_fit(sech2_model, xdata=z_cen, ydata=J, p0=p0, bounds=bounds)
        except:
            print(k, 'fail')
            continue

        I0, z0, zc = p

        if (z0 > 0) and (I0 > 0):
            z00.append(z0)
            zcs.append(zc)
            rs.append(x)
            prof.append(J_ / np.nanmax(J_))

    z0 = np.median(z00)
    zerr = np.std(z00)

    if plotting:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(np.array(rs) - xc, z00, 'o')
        plt.xlabel('r, pix')
        plt.ylabel('z0, pix')
        plt.gca().axhline(z0, color='r')
        plt.ylim(0, 100)
        plt.subplot(122)
        plt.plot(np.array(rs) - xc, zcs, 'o')
        plt.xlabel('r, pix')
        plt.ylabel('z_c, pix')
        # plt.gca().axhline(z0, color='r')
        plt.ylim(-20, 20)
        plt.show()

    if not (ax is None):
        ax.plot(np.array(rs) - xc, z00, 'o')
        ax.axhline(z0, color='r', ls='--', label=r'$z_0 = %3.1f \pm %3.1f$' % (z0, zerr))
        ax.set_xlabel('r, pix')
        ax.set_ylabel(r'$z_0$')
        ax.set_ylim(0, 20)
        ax.legend()
    print('z0 =', z0, zerr)
    return z0, prof
