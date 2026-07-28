import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm
import pandas as pd


def fast(theta, data, width=4):
    h, w = data.shape
    yc = h / 2
    r = rotate(data, theta, reshape=False, order=5)
    I1 = np.sum(r[int(yc - width):int(yc - 0), :])
    I2 = np.sum(r[int(yc + 0):int(yc + width), :])
    return I1 + I2


def find_theta(data, step=1, indent=40, plotting=False, ax=None):
    """
    Функция ищет позиционный угол галактики с ребра.
    Алгоритм взят из работы: https://ui.adsabs.harvard.edu/abs/2012MNRAS.427.1102M/abstract (Martín-Navarro et. al., 2012)


    input:
    - data     : исходное изображение
    - step     : шаг по углу                                     [deg]
    - indent   : Отступ для отображения пиков на крае            [deg]
    - plotting : Ключ построение графика
    - ax       : Вывод графика в общий стек (мой костыль)

    result:
    - theta    : P.A. галактики, относительно оси X изображения. [deg]

    P.S. Тут у меня было распараллеливание, но работало подозрительно, я убрал.
         Ещё был фит пика гауссианой, но я тоже убрал.
    """

    thetas = np.arange(0, 180, step)
    I = np.array([])
    for theta in tqdm(thetas):
        I = np.append(I, fast(theta, data))

    theta_fit = np.mean(thetas[np.where(I == np.max(I))])

    if (theta_fit - indent) < 0:
        I_new = np.append(I, I)
        theta_new = np.append(thetas - 180, thetas)
    elif (theta_fit + indent) > 180:
        I_new = np.append(I, I)
        theta_new = np.append(thetas, thetas + 180)
    else:
        I_new = np.array(I)
        theta_new = thetas

    ind = np.where((theta_new > theta_fit - indent) * (theta_new < theta_fit + indent))
    thetas = theta_new[ind]
    I = I_new[ind]

    theta_fit = thetas[np.where(I == np.max(I))][0]

    if plotting:
        plt.figure(figsize=(5, 5), dpi=300)
        plt.plot(thetas, I, '-', label='Image')
        plt.plot([theta_fit, theta_fit], [np.min(I), np.max(I)], '--g', label=r'$\theta = %3.1f ^{\circ} $' % float(theta_fit))
        plt.legend(fontsize=8)
        plt.xlabel(r'$\theta, \, deg$', fontsize=14)
        plt.ylabel(r'$I$', fontsize=14)
        plt.gca().set_aspect('auto')
        plt.tight_layout()
        plt.show()

    if not (ax is None):
        ax.plot(thetas, I, '-', label='Image')
        ax.plot([theta_fit, theta_fit], [np.min(I), np.max(I)], '--g',
                label=r'$\theta = %3.1f ^{\circ} \pm 0.2 $' % float(theta_fit))
        # plt.legend(fontsize=8)
        ax.set_xlabel(r'$\theta, \, deg$', fontsize=14)
        ax.set_ylabel(r'$I$', fontsize=14)
        ax.legend()

    return np.round(theta_fit, 1)