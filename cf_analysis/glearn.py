import numpy as np
from scipy.optimize import minimize
import yjtransform as yjt
import tools


class GLearn():

    def __init__(self, controls, padl=0, padr=0, ptransform=None, p0=None, n0=0.01, ninf=1e10, stdscale=False, seed=100, normalization=None):

        controls = controls.astype(np.float64)
        self.T = controls.shape[1]
        self.ff = np.fft.rfftfreq(self.T)
        self.seed = seed
        self.controls = controls
        self.means = controls.mean(axis=0) 
        self.stds = controls.mean(axis=0)
        self.normalization = normalization
        self.controlsnorm =   self.normalize(controls)
        self.padl, self.padr = padl, padr
        self.Tpad = self.T + self.padl + self.padr
        self.ffpad = np.fft.rfftfreq(self.Tpad)
        
        #
        if ptransform is None: self.ptransform = yjt.get_transform(self.controlsnorm, stdscale=stdscale, p0=p0) 
        else: self.ptransform = ptransform
        #  order in ptransform : mmean, sstd, mu, sig, eta, eps, beta
        self.controlsg = yjt.transform(self.controlsnorm.copy(), self.ptransform) 
        self.getprior()
        #

        
    def normalize(self, x):
        if self.normalization is None: return x
        if self.normalization == 'standard': return  (x - self.means)/self.stds
        if self.normalization == 'center': return  (x - self.means) 
        elif self.normalization == 'minmax' :
            off, norm = (self.cmax+self.cmin)/2., (self.cmax-self.cmin)/2.
            return (x-off)/norm

    def unnormalize(self, y):
        if self.normalization is None: return y
        if self.normalization == 'standard': return y * self.stds + self.means
        if self.normalization == 'center': return  (y + self.means) 
        elif self.normalization == 'minmax' :
            off, norm = (self.cmax+self.cmin)/2., (self.cmax-self.cmin)/2.
            return y*norm + off
       

    def getprior(self, demean=False, samples=False):
        #
        if demean: xx = self.controlsg - self.controlsg.mean(axis=0)
        else : xx = self.controlsg.copy()
        pkm = tools.psfuncmean(xx)
        #
        controlsgpad = np.pad(self.controlsg, [(0, 0), (self.padl, self.padr)], mode='constant', constant_values=0)
        if demean: xxpad = controlsgpad - controlsgpad.mean(axis=0)
        else : xxpad = controlsgpad.copy()
        self.pkraw = tools.psfuncmean(xxpad)
        #
        self.tfpad =  tools.gettf(pkm, self.ff, self.T, self.padl, self.padr, real=True, samples=samples, seed=self.seed)
        self.pkprior = self.pkraw/self.tfpad        

   


    def fit(self, treated, T0, sigma=None, ninf=1e10):

        treatednorm = self.normalize(treated)
        treatedg = yjt.transform(treatednorm, self.ptransform).flatten() #
        if sigma is None: sigma = abs(treatedg).mean()/100.
        p0 = np.ones(self.pkprior.size*2)

        def _neglogposterior(p):
            u, v = p[:p.size//2], p[p.size//2:]
            x = np.fft.irfft(u + 1j*v, norm='ortho', n=self.Tpad)
            ps = abs(u + 1j*v)**2
            res = (x[self.padl:self.padl+T0] - treatedg[:T0])**2 / sigma**2
            prior = ps/self.pkprior
            loss =  sum(res) + sum(prior)
            return loss
    
        #chisqmin = lambda p: fourierposterior(p, xcg, pkprior, iyear=T0, padl=padl, sigma=abs(xcg).mean()/100.)
        pp = minimize(_neglogposterior, p0).x
        predg = np.fft.irfft(pp[:pp.size//2] + 1j*pp[pp.size//2:], norm='ortho', n=self.Tpad)
        if self.padr: predg = predg[self.padl:-self.padr]
        else: predg = predg[self.padl:]
        pred = yjt.invtransform(predg, self.ptransform)

        cov = self.getcov(T0, n0=sigma, ninf=ninf)
        errg = cov.diagonal()**0.5

        nl = self.ptransform[-3:]
        #deriv = (yjt.from_gauss(predg*1.01, nl) - yjt.from_gauss(predg*0.99, nl))/(predg*0.02)
        pup, pdown = yjt.invtransform(predg*1.01, self.ptransform), yjt.invtransform(predg*0.99, self.ptransform)
        deriv = (pup - pdown)/(predg*0.02)
        if self.padr: err = (errg[self.padl:-self.padr]*deriv)*self.ptransform[1]
        else: err = (errg[self.padl:]*deriv)*self.ptransform[1]
        
        #
        #predup = (from_gauss(predg+errg, nl)[padl:-padr] + mu)*params[-1] + params[-2]
        #preddn = (from_gauss(predg-errg, nl)[padl:-padr]+ mu)*params[-1] + params[-2]
        pred = self.unnormalize(pred)
        if self.normalization == 'standard': err = err * self.stds
        if self.normalization == 'standard':
            if self.padr:
                cov[self.padl:-self.padr, self.padl:-self.padr]= cov[self.padl:-self.padr, self.padl:-self.padr] *self.stds.reshape(-1, 1)*self.stds.reshape(1, -1)
            else: 
                cov = cov *self.stds.reshape(-1, 1)*self.stds.reshape(1, -1)
        return pred, err, cov

    
    def getcov(self, T0, n0=0.01, ninf=1e10, real=True):
        psf = self.pkprior.copy()
        if real :
            if self.Tpad%2 == 0:psf = np.concatenate([psf, psf[1:-1][::-1]])
            else: psf = np.concatenate([psf, psf[1:][::-1]])
        invsnoisek = np.linalg.inv(np.diag(psf))
        ndiag = np.ones_like(psf)*n0
        ndiag[self.padl+T0:] = ninf
        ndiag[:self.padl] = ninf

        noise = np.diag(ndiag)
        invnoise = np.linalg.inv(noise)
        ftmatrix = tools.DFT(psf*0, matrix=True)
        ftmatrixdag = tools.DFT(psf*0, matrix=True, inv=True)
        rtnr = np.dot(ftmatrixdag, np.dot(invnoise, ftmatrix))
        d = np.linalg.inv(invsnoisek + rtnr)
        cov = np.dot(ftmatrix, np.dot(d, ftmatrixdag)).real
        return cov       







    

#
#def chisq(p, means, casales, priork, padl, i1, sigma=1, verbose=False):
#    x, ps = pred(p, means, prior=True)
#    res = (x[padl:padl+i1] - casales[:i1])**2 / sigma**2
#    prior = ps/priork 
#    if verbose: print(sum(res), sum(prior))
#    return sum(res) + sum(prior)
#
#
#def pred(p, means, prior=False):
#    u, v = p[:p.size//2], p[p.size//2:]
#    s = u + 1j*v
#    x = np.fft.irfft(s, norm='ortho')
#    x += means
#    if prior: 
#        ps = abs(s)**2
#        return x, ps
#    else: return x
#



##def getf(padl, padr, real=True, yy=years.size):
##    return np.fft.rfftfreq(padl+padr+yy)
##


##
##def gettfprior(data1, padl, padr, n=2000, samples=False, seed=100, al=1.0, data2=None):
##    yy = years.size
##    yy = data1.shape[1] # -padl-padr
##    print(data1.shape[1], padl, padr, yy)
##    ff = np.fft.rfftfreq(yy)
##    xp = data1 - data1.mean(axis=0)*al
##    if data2 is not None: xp2 = data2 - data2.mean(axis=0)*al
##    else: xp2 = xp
##    p1 =  np.fft.rfft(xp, axis=1, norm='ortho')
##    p2 =  np.fft.rfft(xp2, axis=1, norm='ortho')
##    pkm = (p1*p2.conj()).mean(axis=0).real
###     pkm = (np.abs(np.fft.rfft(xp, axis=1, norm='ortho'))**2).mean(axis=0)
##    return gettf (padl, padr, ff, pkm, real=True, ny=yy, samples=samples, seed=seed)
##
