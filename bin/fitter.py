from scipy.optimize import curve_fit, minimize
import numpy as np
from bin.diff_evol import mpfit_fit
from bin.models import sersic

def curve_fitter(X, Y, model, params, bounds):
    def log_model(x, *p):
        return np.log10(model(x, *p))
    p, covp = curve_fit(log_model, xdata=X, ydata=np.log10(Y), p0=params, bounds=bounds)

    sigma = []
    for i in range(len(p)):
        sigma.append(np.sqrt(covp[i,i]))
    
    return p, sigma

def NM_fitter(X, Y, model, params, bounds):
    def foo(params):
        return np.sum((Y - model(X, *params))**2)
    res = minimize(foo, params, method='Nelder-Mead', bounds=bounds)
    return res.x

def itter_fitter(X, Y, d_model, s_model, p0d, p0s, s_bounds, d_bounds, maxiter=10):
    r = d_model.r0
    I = d_model.I00
    sl, sr = s_bounds
    rs_ = r[np.where((r >=sl)* (r <= sr))]
    Is_ = I[np.where((r >=sl)* (r <= sr))]
    
    dl, dr = d_bounds
    rd_ = r[np.where((r >= dl) * (r <= dr))]
    Id_ = I[np.where((r >= dl) * (r <= dr))]
    Id = Id_
    Is = Is_

    if d_model.h2 is None:
        db = ([None, None], [None, None])
    elif d_model.h3 is None:
        db = ([None, None, None, None], [None, None, None, None])
    else:
        db = ([None, None, None, None, None, None], [None, None, None, None, None, None])

    for _ in range(maxiter):
        pd, pd_err, chi2_d, stat = mpfit_fit(rd_, Id, np.sqrt(Id), d_model.model, p0=p0d, bounds=db) 
        Is = Is_ - d_model.model(rs_, *pd)
        ps, ps_err, chi2_s, stat = mpfit_fit(rs_, Is, np.sqrt(Is), sersic, p0=p0s, bounds=([None, None, None], [None, None, None])) 
        Id = Id_ - sersic(rd_, *ps)

    return pd, pd_err, ps, ps_err


if __name__ == '__main__':

    def mm(x, *p):
        a, b, c = p
        return a * x**2 + b*x + c
    
    a = -1
    b = 10
    c = 7

    x = np.linspace(-10, 10, 100)
    y = np.zeros(len(x))
    print(curve_fitter(x, y, mm, [a,b,c], None))
    print(NM_fitter(x, y, mm, [a, b, c]))