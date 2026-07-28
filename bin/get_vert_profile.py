import warnings
import numpy as np
from scipy.interpolate import  interp1d
from matplotlib import pyplot as plt
from astropy.stats import sigma_clipped_stats

def get_vertical_profile(image, mask, z0):
    masked_rotated = image.copy()
    masked_rotated[mask > 0] = float('nan')

    h, w = mask.shape

    J = np.nanmean(masked_rotated[int(h / 2 - z0):int(h / 2 + z0), :], axis=0)
    Jerr = np.nanstd(masked_rotated[int(h / 2 - z0):int(h / 2 + z0), :], axis=0)

    J[np.isnan(J)] = 0
    Jerr[np.isnan(J)] = 0
    z = np.arange(len(J))

    ind = np.where((~np.isnan(Jerr)) * (~np.isnan(J)))
    z = z[ind]
    J = J[ind]
    Jerr = Jerr[ind]

    '''
    ind = np.where(abs(np.arange(w)-w/2) <= r27)
    r_  = (np.arange(w)-w/2)[ind]
    I_  = J[ind]
    Ie_ = Jerr[ind]

    J_ = np.array(list(reversed(I_)))
    Je_ = np.array(list(reversed(Ie_)))

    J_ = (I_ + J_)/2
    Je_ = np.sqrt(Je_**2 + Ie_**2)/2
    inc = np.argmax(J_)

    r1 = r_[inc:]
    J1 = J_[inc:]
    J1e = Je_[inc:]

    '''

    xc = np.argmax(J)
    right_p = J[np.where(z >= xc)]
    right_pe = Jerr[np.where(z >= xc)]

    left_p = J[np.where(z <= xc)]
    left_pe = Jerr[np.where(z <= xc)]

    right_z = (z - xc)[np.where(z >= xc)]
    left_z = (z - xc)[np.where(z <= xc)]

    foo = interp1d(left_z, left_p, kind='cubic')
    ind = np.where(right_z < np.max(abs(left_z)))
    prof1 = (right_p[ind] + foo(-right_z[ind])) / 2

    J = prof1
    r = right_z[ind]

    foo = interp1d(left_z, left_pe, kind='cubic')
    ind = np.where(right_z < np.max(abs(left_z)))
    prof1 = np.sqrt(right_pe[ind] ** 2 + foo(-right_z[ind]) ** 2) / 2
    Je = prof1

    return r, J, Je


# Взвешанный вариант
def get_vertical_profile_w(image, mask, prof, z0, zp, test=False):
    masked_rotated = image.copy()
    masked_rotated[mask > 0] = float('nan')

    h, w = mask.shape
    prof = np.nanmedian(prof, axis=0)
    prof[np.isnan(prof)] = 0

    xc = np.argmax(prof)
    z = np.arange(len(prof))
    right_p = prof[np.where(z >= xc)]
    left_p = prof[np.where(z <= xc)]
    right_z = (z - xc)[np.where(z >= xc)]
    left_z = (z - xc)[np.where(z <= xc)]

    foo = interp1d(left_z, left_p, kind='cubic')
    ind = np.where(right_z < np.max(abs(left_z)))
    prof1 = (right_p[ind] + foo(-right_z[ind])) / 2
    foo1 = interp1d(right_z[ind], prof1, kind='cubic')

    '''
    TEST
    plt.figure()
    plt.plot(-left_z, (left_p))
    plt.plot(right_z, (right_p))
    plt.plot(right_z[ind], (prof1), '--r')
    plt.xlim(0, 3*z0)
    plt.ylim(0,1)
    plt.show()
    '''

    if test:
        with warnings.catch_warnings(action="ignore"):
            plt.figure(figsize=(15, 15))
            for k in [4, 2, 1, 0.5]:
                Js_ = []
                Jsw_ = []
                for p in np.arange(-z0 / k, z0 / k):
                    J = masked_rotated[int(h / 2 + p), :]
                    Jw = J / foo1(abs(p))
                    Js_.append(J)
                    Jsw_.append(Jw)

                Js_ = np.array(Js_)
                Jsw_ = np.array(Jsw_)
                h_, w_ = Jsw_.shape
                J = []
                Jw = []
                Jerr = []
                Jwerr = []
                for i in range(w_):
                    mean, med, std = sigma_clipped_stats(Js_[:, i])
                    J.append(mean)
                    Jerr.append(std)

                    mean, med, std = sigma_clipped_stats(Jsw_[:, i])
                    Jw.append(mean)
                    Jwerr.append(std)

                J = np.array(J)
                Jw = np.array(Jw)
                Jerr = np.array(Jerr)
                Jwerr = np.array(Jwerr)

                # J  = np.nanmean(Js_, axis=0)
                # Jerr = np.nanstd(Js_, axis=0)
                # Jw = np.nanmean(Jsw_, axis=0)
                # Jwerr = np.nanstd(Jsw_, axis=0)

                plt.subplot(221)
                plt.plot(zp - 2.5 * np.log10(J), label='k=%2.1f' % k)
                plt.legend()
                plt.ylim(27, 16)
                plt.subplot(222)
                plt.plot(zp - 2.5 * np.log10(Jw), label='k=%2.1f' % k)
                plt.ylim(27, 16)
                plt.subplot(223)
                plt.plot(Jerr / J, label='k=%2.1f' % k)
                plt.legend()
                plt.ylim(0, 0.5)

                plt.subplot(224)
                plt.plot(Jwerr / Jw, label='k=%2.1f' % k)
                plt.ylim(0, 0.5)

                plt.legend()

            plt.show()

    Jsw_ = []
    for p in np.arange(-z0, z0):
        J = masked_rotated[int(h / 2 + p), :]
        Jw = J / foo1(abs(p))

        Jsw_.append(Jw)

    Jsw_ = np.array(Jsw_)
    h, w = Jsw_.shape
    Jw = []
    Jerr = []
    with warnings.catch_warnings(action="ignore"):
        for i in range(w):
            mean, med, std = sigma_clipped_stats(Jsw_[:, i])
            Jw.append(mean)
            Jerr.append(std)

    Jw = np.array(Jw)
    Jerr = np.array(Jerr)
    z = np.arange(len(Jw))

    ind = np.where((~np.isnan(Jw)) * (~np.isnan(Jerr)))
    Jw = Jw[ind]
    Jerr = Jerr[ind]
    z    = z[ind]

    # Jw   = np.nanmean(Jsw_, axis=0)
    # Jerr = np.nanstd(Jsw_, axis=0)

    xc = z[np.argmax(Jw)]

    right_p = Jw[np.where(z >= xc)]
    right_pe = Jerr[np.where(z >= xc)]

    left_p = Jw[np.where(z <= xc)]
    left_pe = Jerr[np.where(z <= xc)]

    right_z = (z - xc)[np.where(z >= xc)]
    left_z = (z - xc)[np.where(z <= xc)]


    right_z, right_p, right_pe = resampling(right_z, right_p, right_pe)
    ind = np.where((~np.isnan(right_p)) * (~np.isnan(right_pe)))
    right_z = right_z[ind]
    right_p = right_p[ind]
    right_pe = right_pe[ind]

    left_z, left_p, left_pe = resampling(-left_z, left_p, left_pe)
    ind = np.where((~np.isnan(left_p)) * (~np.isnan(left_pe)))
    left_z = left_z[ind]
    left_p = left_p[ind]
    left_pe = left_pe[ind]


    #plt.figure()
    #plt.plot(right_z, np.log10(right_p))
    #plt.plot(left_z, np.log10(left_p))
    #plt.show()
    #print(left_p)
    #print(right_p)
    #print(left_z)
    #print(right_z)

    foo = interp1d(left_z, left_p, kind='cubic')
    ind = np.where(right_z < np.max(abs(left_z)))
    prof1 = (right_p[ind] + foo(right_z[ind])) / 2
    #print(prof1)
    J = prof1
    r = right_z[ind]

    #plt.figure()
    #plt.plot(r, np.log10(J))
    #plt.show()

    foo = interp1d(left_z, left_pe, kind='cubic')
    ind = np.where(right_z < np.max(abs(left_z)))
    prof1 = np.sqrt(right_pe[ind] ** 2 + foo(right_z[ind]) ** 2) / 2
    Je = prof1

    ind = np.where(J>0)
    r = r[ind]
    J = J[ind]
    Je = Je[ind]

    ind = np.where((~np.isnan(J)) * (~np.isnan(Je)))
    r = r[ind]
    J = J[ind]
    Je = Je[ind]

    return r, J, Je


def calc_r_crit(r, J, Jerr, rin):
    delta_mu = 2.5 / np.log(10) * Jerr / J
    for i in range(1, len(r) - 1):
        if (delta_mu[i] > delta_mu[i - 1]) and r[i] > rin:
            if (delta_mu[i] > 0.2) and (delta_mu[i + 1]) > 0.2:
                r_crit = r[i]
                return r_crit
    r_crit = r[-1]
    return r_crit


def resampling(r, J, Je, power=1.03, width=0):
    rmax = np.max(r)
    start = 0
    width = 1.0
    power = 1.03

    new_z = []
    new_J = []
    new_Je = []
    while True:
        end = start + width
        new_z.append((start + end) / 2.)
        ind = np.where((r >= start) * (r < end))
        new_J.append(np.mean(J[ind]))
        new_Je.append(np.sqrt(np.sum(Je[ind] ** 2)) / len(Je[ind]))
        width = width * power
        start = end
        if end > rmax:
            break
    new_z = np.array(new_z)
    new_J = np.array(new_J)
    new_Je = np.array(new_Je)
    return new_z, new_J, new_Je


def sigma_r_crit(r, J, sigma, zp):
    J1 = J + sigma
    J2 = J - sigma

    mu1 = - 2.5 * np.log10(J1)
    mu2 = - 2.5 * np.log10(J2)
    delta_mu = abs(mu1 - mu2)
    for i in range(1, len(r) - 1):
        if (delta_mu[i] > delta_mu[i - 1]):
            if (delta_mu[i] > 0.2) and (delta_mu[i + 1]) > 0.2:
                r_crit = r[i]
                foo = interp1d(r, zp - 2.5 * np.log10(J))
                mu_crit = foo(r_crit)
                return r_crit, mu_crit
    r_crit = r[-1]
    foo = interp1d(r, zp - 2.5 * np.log10(J))
    mu_crit = foo(r_crit)

    return r_crit, mu_crit

