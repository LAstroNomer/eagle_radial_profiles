import bin.mpfit as mpfit
import numpy as np
from bin.models import *
import os
import pandas as pd




def borders(bounds, p0):
    parinfo = [{'value':0., 'fixed':0, 'limited':[0,0],
          'limits':[0.,0.]} for i in range(len(p0))]
    
    for i in range(len(p0)):
        parinfo[i]['value'] = p0[i]
        vmin, vmax = bounds[i]
        if not(vmin is None):
            parinfo[i]['limited'][0] = 1
            parinfo[i]['limits'][0] = vmin
        
        if not(vmax is None):
            parinfo[i]['limited'][1] = 1
            parinfo[i]['limits'][1] = vmax
    return parinfo

def mpfit_fit(r, I, err, model, p0, bounds, maxiter=1000):
    assert (len(p0) == len(bounds[0])) and (len(p0) == len(bounds[1])), 'Lenght p0 is not equal lenght bounds. Exit...'
    bounds = [(bounds[0][i], bounds[1][i]) for i in range(len(bounds[0]))]
    #print(bounds)
    def F(x, p):
        return  model(x, *p)

    def myfunct(p, fjac=None, x=None, y=None, err=None):
        # Parameter values are passed in "p"
        # If fjac==None then partial derivatives should not be
        # computed.  It will always be None if MPFIT is called with
        # default flag.
        model = F(x, p)
        # Non-negative status value means MPFIT should continue,
        # negative means stop the calculation.
        status = 0
        #print('err', err[0])
        return [status,((y - model)/err)]

    parinfo = borders(bounds=bounds, p0=p0) 
    #ye = np.sqrt(I)

    fa = {'x':r, 'y':I, 'err':err}
    m = mpfit.mpfit(myfunct, p0, parinfo=parinfo, functkw=fa, maxiter=maxiter, quiet=1)
    #print('status = ', m.status)
    #if (m.status <= 0):
    #    print('error message = ', m.errmsg)
        
    #print('parameters = ', m.params, m.perror, m.fnorm)
    import pylab as plt
    #plt.figure()
    #plt.plot(r, np.log10(I), '-k')
    #plt.plot(r, np.log10(F(r, m.params)), '--r')
    #plt.show()
    return m.params, m.perror, m.fnorm, m.status
