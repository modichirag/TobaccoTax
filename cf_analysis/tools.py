import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy import stats


#from casetup import *




def psfuncmean(x, x2=None, real=True, ortho=True):
    '''calculate mean power spectrum or cross-spectrum for a data-sample where samples along first axis'''
    if ortho: norm='ortho'
    else: norm=None
    if x2 is None: x2 = x.copy()
    if real: 
        xc, xc2 = np.fft.rfft(x, axis=1, norm=norm), np.fft.rfft(x2, axis=1, norm=norm)
    else:
        xc, xc2 = np.fft.fft(x, norm=norm), np.fft.fft(x2, norm=norm)
    ps = (xc * xc2.conj()).mean(axis=0).real
    return ps

def psfunc(x, x2=None, real=True, ortho=True):
    '''calculate power spectrum or cross-spectrum'''
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



def sampleps(ff, pk, n=100,  seed=100):
    """Draw 'n' samples from the power spectrum 'pk' at frequncies 'ff'"""
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




def pad_array(arr, padl, padr, ps=False):
    xxpad = np.zeros_like(arr)
    pspad = []
    if padr: xxpad[:, padl:-padr] = arr[:, padl:-padr]
    else: xxpad[:, padl:] = arr[:, padl:]
    if ps:
        for i in range(xxpad.shape[0]): pspad.append(psfunc(xxpad[i]))
        pspad = np.array(pspad)
        return xxpad, pspad
    else: return xxpad
    

def gettf(pk, ff, ny, padl, padr, real=True, samples=False, seed=100):
    ffpad = np.fft.rfftfreq(ny + padl + padr)
    ppad = interp1d(ff, pk, fill_value="extrapolate")(ffpad)
    xxs, ps = sampleps(ffpad, ppad, seed=seed)
    xxpad, pspad = pad_array(xxs, padl, padr, ps=True)
    tf = pspad.T.mean(axis=1)/ps.T.mean(axis=1)
    if samples: return tf, [[xxs, ps], [xxpad, pspad]]
    else: return tf



def gauss(x, a, mu, s):
# def gauss(x, p):
#     a, mu, s = p
    norm = 1/np.sqrt(2*np.pi*s**2)
#     norm = 1
    return a*norm*np.exp(-0.5*((x-mu)/s)**2)



def fitgauss(xx, bins, normed, verbose=0):
    h, x = np.histogram(xx, bins=bins, normed=normed)
    x = x[1:] + x[:-1]
    x /= 2 
    p0 = [h.max(), xx.mean(), xx.std()]
    ftomin = lambda p: ((gauss(x, *p) - h)**2).sum()
    pp = minimize(ftomin, p0, method='Nelder-Mead', options={'maxiter':5000})
    if verbose:
        if verbose > 1: print(pp)
        if verbose == 1: print('mu=%.2f, s=%.2f'%(pp.x[1],pp.x[2]))
#     pp = cf(gauss, x, h, p0, )[0]
    return (x, gauss(x, *pp.x))



def logpdf_gauss(x, mean, std, normalized=True, rety=False):
    result = -0.5 * (x - mean)**2/std**2
    if normalized:
        result += 0.5 * np.log(std**-2) - 0.5 * np.log(2 * np.pi)
    if rety: return np.exp(result)
    return result


def fitgausspdf(xx, normalized=True, verbose=0, method='Nelder-Mead'):
    p0 = [xx.mean(), xx.std()]
    ftomin = lambda p: (-logpdf_gauss(xx, *p, normalized=normalized)).sum()
    if verbose > 1: print(p0, ftomin(p0))
    #pp = minimize(ftomin, p0)
    pp = minimize(ftomin, p0, method=method, options={'maxiter':50000})
    if verbose:
        if verbose > 1: print(pp)
        if verbose == 1: print('mu=%.2f, s=%.2f'%(pp.x[0],pp.x[1]))
    return pp.x



def fouriertox(p):
    u, v = p[:p.size//2], p[p.size//2:]
    s = u + 1j*v
    x = np.fft.irfft(s, norm='ortho')
    return x





# compute approximate rank of matrix 
def approximate_rank(X, t=0.99):
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    total_energy = (s ** 2).cumsum() / (s ** 2).sum()
    rank = list((total_energy>t)).index(True) + 1
    return rank

def hsvt(X, rank=2):
    if rank is None:
        return X
    u, s, v = np.linalg.svd(X, full_matrices=False)
    s[rank:].fill(0)
    return np.dot(u * s, v) 

# ordinary least squares 
def linear_regression(X, y, rcond=1e-15):
    return np.linalg.pinv(X, rcond=rcond).dot(y)
