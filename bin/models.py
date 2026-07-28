import numpy as np
import scipy.special as sc
from astropy.modeling.functional_models import Sersic1D
'''
Эти модели расчитаны на работу с поверхностной плотностью массы. Поэтому необходим пересчёт коэфециентов.
Дело в том, что звездная величина mu = -2.5*log10(I0), для массы используется просто логарифм Sigma
'''

loge = np.log10(np.e)

def sersic(r :np.ndarray, I0 :float, re :float, n: float):
    '''
    Модель Сёрсика в терминах "интенсивности". Из пакета astropy.
    I0 --- пиковая интенсивность r = 0
    Производится пересчёт I0 в Ie
    '''
    bn = sc.gammaincinv(2 * n, 0.5)
    Ie = I0 * np.exp(-bn)
    s = Sersic1D(amplitude=Ie, r_eff=re, n=n)
    return s(r)

def sersic_e(r :np.ndarray, Ie :float, re :float, n: float):
    '''
    Модель Сёрсика в терминах "интенсивности". Из пакета astropy.
    Ie --- пиковая интенсивность на re
    '''
    s = Sersic1D(amplitude=Ie, r_eff=re, n=n)
    return s(r)


def exp(r :np.ndarray, I0 :float, h : float, *args):
    '''
    Диск без излома
    '''
    return I0*np.exp(-r/h)

def one_break(r :np.ndarray, I0 :float, h1 :float, h2 : float, rb :float, *args):
    '''
    Диск с одним резким изломом
    '''
    I01 = I0
    I02 = I01 * np.exp(-rb * (1./h1 + 1./h2))
    res = np.zeros(len(r))
                       
    ind = np.where(r < rb)
    res[ind] = I01 * np.exp(-r/h1)
    ind = np.where(r >= rb)
    res[ind] = I02 * np.exp(-r/h2)
    return res
def sersic_flux(I0, re, n, eps=0):
    bn = sc.gammaincinv(2 * n, 0.5)
    Ie = I0 * np.exp(-bn)
    return Ie * re**2 * 2 * np.pi * n * np.exp(bn) / bn**(2*n) *  sc.gamma(2*n)


####
'''

def disc_flux(I0, h1, h2, h3, rb1, rb2):
    if h2 is None:
        F =  2*np.pi * h1**2 * I0
    elif h3 is None:
        I1 = I0
        I2 = I1 * np.exp(rb1 * (1/h2 - 1/h1))
        #print(I2, I1)
        def foo(z):
            return - np.exp(-z) *(z+1)
        F  = 2*np.pi * h1**2 *I1 * (foo(rb1/h1) - foo(0.0))
        print('F', F, 2*np.pi * h1**2 *I1 * (foo(rb1/h1) - foo(0.0)))
        F += 2*np.pi * h2**2 *I2 * (-foo(rb1/h2))
        print('F', F, 2*np.pi * h2**2 *I2 * (-foo(rb1/h1)))
    else:
        I1 = I0
        I2 = I1 * np.exp(rb1 * (1/h2 - 1/h1))
        I3 = I2 * np.exp(rb2 * (1/h3 - 1/h2))
        def foo(z):
            return - np.exp(-z) *(z+1)
        F  = 2*np.pi * h1**2 * I1 * (foo(rb1/h1) - foo(0.0))
        F += 2*np.pi * h2**2 * I2 * (foo(rb2/h2) - foo(rb1/h2))
        F += 2*np.pi * h3**2 * I3 * (-foo(rb2/h3))
    return F

def brexp(r, I0, h1, h2, rb, alpha, *args):
    #print('alpha', alpha)
    S = (1.0 + np.exp(-alpha*rb))**(-1/alpha *(1/h1 - 1/h2))
    
    p = (1/alpha*(1/h1 - 1/h2))
    #print('S', S, 'p', p)
    return S*I0*np.exp(-r/h1)*(1 + np.exp(alpha*(r - rb)))**p
    # alpha*(r-R) < 100.
    """
    ind = np.where(alpha*(r-rb) < 100.)
    r1 = r[ind]
    res1 = S*I0*np.exp(-r1/h1)*(1 + np.exp(alpha*(r1 - rb)))**p
    
    ind = np.where(alpha*(r-rb) >= 100.)
    r2 = r[ind]
    res2 = S*I0*np.exp(rb/h2 - rb/h1 - r2/h2)
    return np.append(res1, res2) #-2.5*np.log10(I)
    """

def br2exp(r, I0, h1, h2, rb1, h3, rb2, *args):
    alpha = 0.5
    beta=0.5
    
    p1 = (1./h1 - 1./h2)/alpha
    p2 = (1./h2 - 1./h3)/beta
    S1 = (1.0 + np.exp(-alpha*rb1))**(-1/alpha *(1/h1 - 1/h2))
    S2 = (1.0 + np.exp(-beta*rb2))**(-1/beta *(1/h2 - 1/h3))
    
    res = []

    for ri in r:
        d1 = alpha*(ri - rb1)
        d2 = beta*(ri - rb2)

        if (d1 < 100.) and (d2 < 100):
            I = np.exp(-ri/h1)*(1 + np.exp(d1))**p1 * (1 + np.exp(d2))**p2
        elif (d1 < 100):
            I = (1 + np.exp(d1))**p1 * np.exp(d2*p2 - ri/h1)
        elif (d2 < 100):
            I = np.exp(rb1/h2 - rb1/h1 - ri/h2) * (1 + np.exp(d2))**p2
        else:
            I = np.exp(rb1/h2 - rb1/h1 + rb2/h3 - rb2/h2 - ri/h3)
        res.append(I)
    res = np.array(res)
    return res*I0*S2*S1

def two_break(r, I0, h1, h2, h3, rb1, rb2):

    mu01 = -2.5*np.log10(I0)

    mu02 = mu01 + 1.086*(1./h1 - 1./h2)*rb1
    mu03 = mu02 + 1.086*(1./h2 - 1./h3)*rb2

    res = np.array([])
    for ri in r:
        ri = abs(ri)
        if (ri < rb1):
            res1 = mu01 + 1.086*ri/h1
            res = np.append(res, res1)
        elif ((rb1 <= ri) * (ri < rb2)):
            res2 = mu02 + 1.086*ri/h2
            res = np.append(res, res2)
        else:
            res3 = mu03 + 1.086*ri/h3
            res = np.append(res, res3)
    return 10**(-0.4*res)

def sech_gen(z, *p):
    I0, z0, zc= p
    n=1
    return I0*2**(-2/n)*np.cosh(n*abs(z-zc)/2/z0)**(-2/n)
'''

