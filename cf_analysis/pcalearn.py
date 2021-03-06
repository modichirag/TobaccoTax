import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import yjtransform as yjt

class PCAlearn():

    def __init__(self, controls, n_components, stdscale=False, seed=100, normalization='center', whiten=False):

        self.T = controls.shape[1]
        self.seed = seed
        self.controls = controls
        self.means = controls.mean(axis=0) 
        if stdscale: self.stds = controls.mean(axis=0)
        else : self.stds = 1
        self.cmin, self.cmax = self.controls.min(), self.controls.max()
        self.normalization = normalization
        #
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, whiten=whiten)
        self.pca.fit(self.normalize(self.controls))
        self.cov = self.pca.get_covariance()
        self.icov = self.pca.get_precision()
        
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
    
    def fit(self, treated, T0, ols=True, verbose=False, method='bfgs', regularization='l2', regwt=None, dT=3, retreg=False):
        
        yy = self.normalize(treated.copy())
        
        if regwt is None or regularization is None:
            yy = yy - self.pca.mean_
            yy = yy[:T0]
            pp = np.linalg.pinv(self.pca.components_[:, :T0].T).dot(yy)
            
        elif type(regwt) == float or type(regwt) == int or type(regwt) == np.float64 or type(regwt) == np.float32:
            
            def _chisq(p):
                yp = self.pca.inverse_transform(p)
                #yp = np.dot(self.pca.components_.T, p) + self.pca.mean_
                diff = yy-yp
                diff[T0:] = 0
                if regularization == 'l2': reg = sum(p**2) * regwt
                if regularization == 'l1': reg = sum(abs(p)) * regwt
                if ols : return np.sum(diff**2) + reg
                else: return np.dot(np.dot(diff, self.icov), diff) + reg
            #p0 = np.zeros(self.n_components)
            y0 = yy - self.pca.mean_
            y0 = y0[:T0].copy()
            p0 = np.linalg.pinv(self.pca.components_[:, :T0].T).dot(y0)
            
            if method == 'nelder-mead': pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
            elif method == 'lbfgs' : pp = minimize(_chisq, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
            elif method == 'bfgs' : pp = minimize(_chisq, p0, method='BFGS', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
            if verbose == 1: print(pp.fun, sum(pp.x**2)*regwt)
            if verbose > 1: print(pp)
            pp = pp.x

        else:
            rms = 1e10
            y0 = yy - self.pca.mean_
            y0 = y0[:T0].copy()
            p0 = np.linalg.pinv(self.pca.components_[:, :T0].T).dot(y0)
            
            for ir, rr in enumerate(regwt):
                rmst = 0
                for tt in np.arange(3, T0, dT).tolist() + [T0]:

                    def _chisq(p):
                        yp = self.pca.inverse_transform(p)
                        #yp = np.dot(self.pca.components_.T, p) + self.pca.mean_
                        diff = yy-yp
                        diff[tt:] = 0
                        if regularization == 'l2': reg = sum(p**2) * rr
                        if regularization == 'l1': reg = sum(abs(p)) * rr
                        if ols : return np.sum(diff**2) + reg
                        else: return np.dot(np.dot(diff, self.icov), diff) + reg
                    if method == 'nelder-mead': pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
                    elif method == 'lbfgs' : pp = minimize(_chisq, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
                    elif method == 'bfgs' : pp = minimize(_chisq, p0, method='BFGS', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
                    rmst += (yy - self.pca.inverse_transform(pp.x))[tt]**2.
                    #rmst += (yy - np.dot(self.pca.components_.T, pp.x) - self.pca.mean_)[tt]**2.
                if rmst**1. < rms:
                    rms = rmst**1.
                    #print(rr, pp.x)
                    pp0 = pp
                    ir0 = ir
                else: pass
            pp = pp0
            regwt = regwt[ir0]
            if verbose >= 1:
                print("min rms is at reg = %0.4f"%(regwt))
            if verbose > 1: print(pp.fun, sum(pp.x**2)*regwt)
            pp = pp.x

        yp = np.dot(self.pca.components_.T, pp) + self.pca.mean_
        #yp = self.pca.inverse_transform(pp)
        yp = self.unnormalize(yp)
        if retreg : return yp, pp, regwt
        return yp, pp

    def fitanalytic(self, treated, T0, verbose=False, regwt=None, dT=3, retreg=False, sigma=None, noise=1, unitwt=True):
        
        yy = self.normalize(treated.copy()) - self.pca.mean_

        def _fit(Sigmainv, t0):
            yy2 = yy.copy()
            yy2[t0:] = 0
            components = self.pca.components_.copy()
            if unitwt: components *= self.pca.explained_variance_.reshape(-1, 1)**0.5
            phi = components.copy()
            #if unitwt: phi *= self.pca.explained_variance_.reshape(-1, 1)**0.5
            phi[:, t0:] = 0
            noisescale = noise*t0/yy.size
            
            A = np.dot(phi, phi.T)/noisescale + Sigmainv
            Ainv =  np.linalg.pinv(A)
            w = np.dot(np.dot(Ainv, phi), yy2)/noisescale
            ypred = np.dot(components.T, w)
            cov = np.dot(np.dot(components.T, Ainv) , components)
            err = cov.diagonal()**0.5
            return ypred, err, w
            
        if sigma is not None:
            Sigmainv = np.diag(np.array(sigma)**-2)
            yp, err, w = _fit(Sigmainv, T0)
            yp += self.pca.mean_
            yp = self.unnormalize(yp)
            if self.normalization == 'standard': err *= self.stds
            return yp, err, w

        if regwt is None:
            Sigmainv = 0 #regwt*np.identity(self.pca.components_.shape[0])
            yp, err, w = _fit(Sigmainv, T0)
            yp += self.pca.mean_
            yp = self.unnormalize(yp)
            if self.normalization == 'standard': err *= self.stds
            return yp, err, w
#            yy = yy[:T0]
#            pp = np.linalg.pinv(self.pca.components_[:, :T0].T).dot(yy)
#            yp = np.dot(self.pca.components_.T, pp) + self.pca.mean_
#            yp = self.unnormalize(yp)
#            return yp, np.zeros_like(yp), pp
#            
        elif type(regwt) == float or type(regwt) == int or type(regwt) == np.float64 or type(regwt) == np.float32:
            Sigmainv = regwt*np.identity(self.pca.components_.shape[0])
            yp, err, w = _fit(Sigmainv, T0)
            yp += self.pca.mean_
            yp = self.unnormalize(yp)
            if self.normalization == 'standard': err *= self.stds
            return yp, err, w
        
        else:
            rms = 1e10

            for ir, rr in enumerate(regwt):
                rmst = 0
                Sigmainv = rr*np.identity(self.pca.components_.shape[0])
                for tt in np.arange(3, T0, dT).tolist() + [T0]:
                    yp, err, w = _fit(Sigmainv, tt)
                    rmst += (yy - yp)[tt+1]**2.
                if rmst**1. < rms:
                    rms = rmst**1.
                    #print(rr, w)
                    ypred, errpred = yp.copy(), err.copy()
                    pp0 = w
                    ir0 = ir
                else: pass
                
            yp = ypred + self.pca.mean_
            yp = self.unnormalize(yp)
            if self.normalization == 'standard': err = errpred * self.stds
            else: err = errpred
            w = pp0
            regwt = regwt[ir0]
            if verbose >= 1:
                print("min rms is at reg = %0.3e"%(regwt))
            #if verbose > 1: print(pp.fun, sum(pp.x**2)*regwt)
            if retreg : return yp, err, w, regwt #yp, pp, regwt
            else:  return yp, err, w
        


    
    def conditional_gaussian(self, vals, means=None, cov=None, given='left'):

        T0 = vals.size
        if means is None: means = self.means.copy()
        if cov is None: cov = self.cov.copy()
        if given == 'left':
            cov_yy = cov[T0:,T0:]
            cov_xx = cov[:T0,:T0]
            cov_xy = cov[:T0,T0:]
            mean_x = means[:T0]
            mean_y = means[T0:]
        elif given == 'right':
            cov_xx = cov[T0:,T0:]
            cov_yy = cov[:T0,:T0]
            cov_xy = cov[T0:,:T0]
            mean_x = means[T0:]
            mean_y = means[:T0]

        icov_xx = np.linalg.inv(cov_xx)
        cond_mean = mean_y + np.dot(cov_xy.T, np.dot(icov_xx, vals-mean_x))
        cond_cov  = cov_yy - np.dot(cov_xy.T, np.dot(icov_xx,cov_xy))
        return cond_mean, cond_cov 





    
class PCAYJ():
    
    def __init__(self, controls, n_components, stdscale=False, seed=100, normalization='center', whiten=False):

        self.T = controls.shape[1]
        self.seed = seed
        self.controls = controls
        self.means = controls.mean(axis=0)
        if stdscale: self.stds = controls.mean(axis=0)
        else : self.stds = 1
        self.cmin, self.cmax = self.controls.min(), self.controls.max()
        self.normalization = normalization
        self.controls_norm = self.normalize(self.controls)
        
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, whiten=whiten)
        self.pca.fit(self.normalize(self.controls))
        self.cov = self.pca.get_covariance()
        self.icov = self.pca.get_precision()
        
        self.control_wts = self.pca.transform(self.controls_norm)
        self.pyj = []
        for i in range(self.n_components): self.pyj.append(yjt.get_transform(self.control_wts[:, i]))
        
    def normalize(self, x):
        if self.normalization is None: return x
        if self.normalization == 'center': return  (x - self.means) 
        if self.normalization == 'standard': return  (x - self.means)/self.stds
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


    def fit(self, treated, T0, ols=True, verbose=False, method='bfgs', noise=1):

        yy = self.normalize(treated.copy())

        def _chisq(p):
            #yp = self.pca.inverse_transform(p)                                                                                                                                 
            yp = np.dot(self.pca.components_.T, p) + self.pca.mean_
            diff = yy-yp
            diff[T0:] = 0
            chisq = np.sum(diff**2) / noise
#             chisq = np.dot(np.dot(diff, self.icov), diff)
            return chisq
        
        def _logprior(p):
            reg = 0 
            for i in range(self.n_components): reg += np.log(1e-10 + yjt.getpdf(np.array(p[i]), self.pyj[i]))
            return reg
        
        def _loss(p):
            return _chisq(p) - _logprior(p)

        p0 = np.zeros(self.n_components)
        y0 = yy - self.pca.mean_
        y0 = y0[:T0].copy()
        p0 = np.linalg.pinv(self.pca.components_[:, :T0].T).dot(y0)
#         p0 = np.random.normal(size=self.n_components)
        
        if method == 'nelder-mead': pp = minimize(_loss, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
        elif method == 'lbfgs' : pp = minimize(_loss, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
        elif method == 'bfgs' : pp = minimize(_loss, p0, method='BFGS', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
        #elif method == 'bfgs' : pp = minimize(_loss, p0, method='BFGS', options={'maxiter':50000,  'maxfun': 50000})
        if verbose == 1: print(pp.fun)
        if verbose > 1:
            print(p0)
            print(pp)
        if verbose >=1 : print(_chisq(pp.x), _logprior(pp.x))
        pp = pp.x
        yp = np.dot(self.pca.components_.T, pp) + self.pca.mean_
        #yp = self.pca.inverse_transform(pp)                                                                                                                                        
        yp = self.unnormalize(yp)
        return yp, pp

    
    def fitanalytic(self, treated, T0, sigma, verbose=False, noise=1):
        
        yy = self.normalize(treated.copy()) - self.pca.mean_

        def _fit(Sigmainv, t0):
            yy2 = yy.copy()
            yy2[t0:] = 0 
            phi = self.pca.components_.copy()
            phi[:, t0:] = 0
            noisescale = noise * t0/yy.size
            
            A = np.dot(phi, phi.T)/noisescale + Sigmainv
            Ainv =  np.linalg.pinv(A)
            w = np.dot(np.dot(Ainv, phi), yy2)/noisescale
            ypred = np.dot(self.pca.components_.T, w)
            cov = np.dot(np.dot(self.pca.components_.T, Ainv) , self.pca.components_)
            err = cov.diagonal()**0.5
            return ypred, err, w
            
        Sigmainv = np.diag(np.array(sigma)**-2)
        yp, err, w = _fit(Sigmainv, T0)
        yp += self.pca.mean_
        yp = self.unnormalize(yp)
        if self.normalization == 'standard': err *= self.stds
        return yp, err, w
