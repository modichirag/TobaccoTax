import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d

from yjtransform import *



def transform(alldata, idata=None, index=0, stdscale=False):
    if idata is None:
        idata = alldata[0]
        dd = alldata[1:]
    else: dd = alldata
        
    mmean, sstd = dd.mean(axis=0), dd.std(axis=0)    
    if not stdscale : sstd = 1
    dd = (dd - mmean)/sstd

    
    p0 = [dd.mean(), dd.std(), 0.1, 0.1, dd.std()]
    ppnl = minimize(lambda x: tominyjfit(dd.flatten(), x, fitbeta=False), p0,  options={'gtol': 1e-08, 'norm': np.inf, 'eps': 1.4901161193847656e-10,  'maxiter': 20000})
#     ppnl = minimize(lambda x: tominyjfit(dd.flatten(), x, fitbeta=False), p0, method='Nelder-Mead',
#                     options={'gtol': 1e-08, 'norm': np.inf, 'eps': 1.4901161193847656e-10,'maxiter': 20000})
    print(ppnl.x)

    mu, sig, eta, eps, beta = ppnl.x
    nl = np.array([eta, eps, beta])
    xp = ((dd - mmean)/sstd).flatten()
    xpg = to_gauss((xp.astype('float64') - mu), nl).reshape(dd.shape[0], -1)
    xc = (idata - mmean)/sstd
    xcg = to_gauss((xc.astype('float64') - mu), nl)

    return [ppnl.x, mmean, sstd], [xpg, xcg], [dd, idata]



def tominyjfit(ngdata, p, rety = False, nd=0, fitbeta=True):
    mu, sig, eta, eps, beta = p
    if not fitbeta: beta = ngdata.std()
    if abs(eps) > 1: return 1e10
    if sig < 0: return 1e10
    yy = (log_q(ngdata, np.array(mu), 1/sig**2, np.array([eta, eps, beta]), True))
    if rety : return np.exp(yy)
    else: return -sum(yy)
    


def fouriertox(p):
    u, v = p[:p.size//2], p[p.size//2:]
    s = u + 1j*v
    x = np.fft.irfft(s, norm='ortho')
    return x


def fourierposterior(p, data, priork, iyear, padl=0, sigma=1,verbose=False, pred=False):
    u, v = p[:p.size//2], p[p.size//2:]
    s = u + 1j*v
    x = np.fft.irfft(s, norm='ortho')
    ps = abs(s)**2
    if pred: return x
    res = (x[padl:padl+iyear] - data[:iyear])**2 / sigma**2
    prior = ps/priork 
    if verbose: print(sum(res), sum(prior))
    return sum(res) + sum(prior)
