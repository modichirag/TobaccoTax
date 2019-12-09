import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import stats


from casetup import *



def psfunc(x, x2=None, real=True, ortho=True):

    if ortho: norm='ortho'
    else: norm=None
    if x2 is None: x2 = x.copy()
    if real: 
        xc, xc2 = np.fft.rfft(x, norm=norm), np.fft.rfft(x2, norm=norm)
    else:
        xc, xc2 = np.fft.fft(x, norm=norm), np.fft.fft(x2, norm=norm)
    ps = (xc * xc2.conj()).real
    return ps

def DFT(x, real=False, matrix=False, ortho=True, inv=False):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    Minv = M.conj().T/N
    if ortho: 
        M /= np.sqrt(N)
        Minv *= np.sqrt(N)
    if matrix: 
        if inv: return Minv
        else: return M
    fft = np.dot(M, x)
    if real: fft = fft[:N//2+1]
    return fft



def getf(padl, padr, real=True, yy=years.size):
    return np.fft.rfftfreq(padl+padr+yy)

def setupdata(salearrayca, casales, padl, padr, al=1.):
    '''returns xp, means, pk, pkca'''
    d = np.pad(salearrayca.copy(), [(0, 0), (padl, padr)], mode='constant', constant_values=0)
    means = d.mean(axis=0)*al
#     std = d.mean(axis=0)*al
    xp = d - means
    std = xp.std(axis=0)
    std[std == 0] = 1
#     if norm is 'mean': xp /= means
#     elif norm is 'std': xp /= std
#     if trans is not None: xp = trans(xp)
    pk = []
    for i in range(xp.shape[0]): pk.append(psfunc(xp[i]))
    pk = np.array(pk)

    if padr: pkca = psfunc(np.pad(casales-means[padl:-padr], (padl, padr), mode='constant', constant_values=0))
    else: pkca = psfunc(np.pad(casales-means[padl:], (padl, padr), mode='constant', constant_values=0))
    return xp, means, pk, pkca
    
def pred(p, means, prior=False):
    u, v = p[:p.size//2], p[p.size//2:]
    s = u + 1j*v
    x = np.fft.irfft(s, norm='ortho')
    x += means
    if prior: 
        ps = abs(s)**2
        return x, ps
    else: return x


def chisq(p, means, casales, priork, padl, sigma=1, i1=18,verbose=False):
    x, ps = pred(p, means, prior=True)
    res = (x[padl:padl+i1] - casales[:i1])**2 / sigma**2
    prior = ps/priork 
    if verbose: print(sum(res), sum(prior))
    return sum(res) + sum(prior)


def getcov(ps, padl, padr, n0=0.01, ninf=1e10, real=True):
    if real: psf = np.concatenate([ps, ps[1:-1][::-1]])
    else: psf = ps.copy()
    invsnoisek = np.linalg.inv(np.diag(psf))

    ndiag = np.ones_like(psf)*n0
    ndiag[padl+i1:] = ninf
    ndiag[:padl] = ninf
#     ndiag[padl+i1:] = ninf
#     ndiag[padl+i1:-padr] = ninf

    noise = np.diag(ndiag)
    invnoise = np.linalg.inv(noise)
    ftmatrix = DFT(psf*0, matrix=True)
    ftmatrixdag = DFT(psf*0, matrix=True, inv=True)
    rtnr = np.dot(ftmatrixdag, np.dot(invnoise, ftmatrix))
    d = np.linalg.inv(invsnoisek + rtnr)
    cov = np.dot(ftmatrix, np.dot(d, ftmatrixdag)).real
    return cov       




def sampleps(ff, pk, n=100,  seed=100):
    ipk = interp1d(ff, pk)
    xxs, ps = [], []
    np.random.seed(seed)
    
    for i in range(n):
        uu = np.random.normal(scale=(ipk(ff)/2)**0.5)
        vv = np.random.normal(scale=(ipk(ff)/2)**0.5)
            
        xx = np.fft.irfft(uu+1j*vv, norm='ortho')
        xxs.append(xx)
        ps.append(psfunc(xxs[-1]))
    xxs, ps = np.array(xxs), np.array(ps)
    return xxs, ps


def getpadded(padl, padr, xxs):
    xxpad = np.zeros_like(xxs)
    pspad = []
    if padr: xxpad[:, padl:-padr] = xxs[:, padl:-padr]
    else: xxpad[:, padl:] = xxs[:, padl:]
    for i in range(xxpad.shape[0]): pspad.append(psfunc(xxpad[i]))
    pspad = np.array(pspad)
    return xxpad, pspad
    
def gettf(padl, padr, ff, pk, real=True, ny=yy, samples=False, seed=100):

    ffpad = np.fft.rfftfreq(ny + padl + padr)
    ppad = interp1d(ff, pk)(ffpad)
    
    xxs, ps = sampleps(ffpad, ppad, seed=seed)
    xxpad, pspad = getpadded(padl, padr, xxs)
    tf = pspad.T.mean(axis=1)/ps.T.mean(axis=1)
    
    if samples: return tf, [[xxs, ps], [xxpad, pspad]]
    else: return tf


def gettfprior(data1, padl, padr, n=2000, samples=False, seed=100, al=1.0, data2=None):

    yy = years.size
    yy = data1.shape[1] # -padl-padr
    print(data1.shape[1], padl, padr, yy)
    ff = np.fft.rfftfreq(yy)
    xp = data1 - data1.mean(axis=0)*al
    if data2 is not None: xp2 = data2 - data2.mean(axis=0)*al
    else: xp2 = xp
    p1 =  np.fft.rfft(xp, axis=1, norm='ortho')
    p2 =  np.fft.rfft(xp2, axis=1, norm='ortho')
    pkm = (p1*p2.conj()).mean(axis=0).real
#     pkm = (np.abs(np.fft.rfft(xp, axis=1, norm='ortho'))**2).mean(axis=0)
    return gettf (padl, padr, ff, pkm, real=True, ny=yy, samples=samples, seed=seed)
    



def gauss(x, a, mu, s):
# def gauss(x, p):
#     a, mu, s = p
    norm = 1/np.sqrt(2*np.pi*s**2)
#     norm = 1
    return a*norm*np.exp(-0.5*((x-mu)/s)**2)



def fitgauss(xx, bins, normed):
    h, x = np.histogram(xx, bins=bins, normed=normed)
    x = x[1:] + x[:-1]
    x /= 2 
    p0 = [h.max(), xx.mean(), xx.std()]
    ftomin = lambda p: ((gauss(x, *p) - h)**2).sum()
    pp = minimize(ftomin, p0, method='Nelder-Mead', options={'maxiter':5000})
    print('mu=%.2f, s=%.2f'%(pp.x[1],pp.x[2]))
#     pp = cf(gauss, x, h, p0, )[0]
    return (x, gauss(x, *pp.x))
