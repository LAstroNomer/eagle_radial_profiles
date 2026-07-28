from photutils.isophote import EllipseGeometry
import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse
import numpy.ma as ma
from photutils.centroids import centroid_quadratic
from astropy.visualization import (MinMaxInterval, LogStretch, ImageNormalize)


def get_center(image):
    yc, xc = np.unravel_index(np.argmax(image), np.array(image).shape)
    return yc, xc


def moments(data):
    'Расчёт моментов первого и второго порядков галактики'

    h, w = data.shape
    x = np.arange(w)
    y = np.arange(h)

    xc = np.nansum(data @ x)/np.nansum(data)
    yc = np.nansum(data.transpose() @ y) / np.nansum(data)

    Ixx = np.nansum(data @ (x - xc) ** 2) / np.nansum(data)
    Iyy = np.nansum(data.transpose() @ (y - yc) ** 2) / np.nansum(data)

    xx, yy = np.meshgrid(x, y)

    Ixy = np.nansum((xx - xc) * (yy - yc) * data) / np.nansum(data)

    # Коррекция ориентации моментов
    if Ixx < 0:
        Ixx = -Ixx
        Iyy = -Iyy
        Ixy = -Ixy

    # Вычисление полуосей
    A_ = np.sqrt(2 * (Ixx + Iyy + np.sqrt(
        (Ixx - Iyy) ** 2 + 4 * Ixy ** 2)))  # ((Iyy - Ixx) + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))/2/Ixy
    B_ = np.sqrt(2 * abs(Ixx + Iyy - np.sqrt(
        (Ixx - Iyy) ** 2 + 4 * Ixy ** 2)))  # ((Iyy - Ixx) + np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))/2/Ixy
    A = np.max([A_, B_])
    B = np.min([A_, B_])
    eps = 1 - B / A

    # print(xc, yc)
    # print(Ixx, Ixy, Iyy ,A, B, eps)
    return xc, yc, A, eps


def main_iso(data, mask, pa=0.0, step=1, fix_center=False, fix_pa=False, maxsma=None, ax=None, linear=True):
    '''
    Построение азимутального профиля галактики с ребра
    '''

    data1 = data.copy()
    data1[mask > 0] = 0
    h, w = data.shape
    yc, xc = get_center(data1)

    xycen = centroid_quadratic(data1, xpeak=xc, ypeak=yc)
    print('xycen', xycen)

    xc_, yc_, A, eps = moments(data1)
    
    if maxsma is None:
        maxsma = w // 2

    # Если эллипс попадает на маску фит не проходит. На этот случай несколько подборов
    for k in [2, 4, 8, 1]:
        geometry = EllipseGeometry(x0=xycen[1], y0=xycen[0], sma=A//k, eps=eps, pa=pa)

        aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
                                  geometry.sma * (1 - geometry.eps),
                                  geometry.pa)

        data = ma.masked_array(data, mask)
        ellipse = Ellipse(data, geometry)
        isolist = ellipse.fit_image(linear=linear, step=step, maxsma=w // 2, fix_center=fix_center, fix_pa=fix_pa)
        print(isolist.to_table())
        if  len(isolist.to_table()['sma'].value) > 0:
            break

    if not (ax is None):
        norm = ImageNormalize(data, interval=MinMaxInterval(), stretch=LogStretch(10_000))
        ax.imshow(data, norm=norm, origin='lower')
        aper.plot(color='white', ax=ax)
    return isolist

def get_az_profile(image, mask=None, ax=None, linear=True, step=1, fix_pa=False, fix_center=False):
    if mask is None:
        mask = np.zeros(image.shape)
    isolist = main_iso(image, mask, ax=ax, linear=linear, step=step, fix_pa=fix_pa, fix_center=fix_center)

    tab = isolist.to_table()

    r = tab['sma'].value
    I = tab['intens'].value
    Ie = tab['intens_err'].value
    ell = tab['ellipticity'].value
    pa = tab['pa'].value
    xc = np.median(tab['x0'].value)
    yc = np.median(tab['y0'].value)

    ind = np.where(I > 0)
    r = r[ind]
    I = I[ind]
    Ie = Ie[ind]
    ell = ell[ind]
    pa = pa[ind]

    return r, I, Ie, ell, pa, xc, yc