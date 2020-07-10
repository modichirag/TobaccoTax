#Technically doing PCR but with SVD code so that we can do different kinds of normalization

import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA


class SVDlearn():

    def __init__(self, controls, stdscale=False, seed=100, normalization='center'):

        self.T = controls.shape[1]
        self.seed = seed
        self.controls = controls
        self.means = controls.mean(axis=0) 
        if stdscale: self.stds = controls.mean(axis=0)
        else : self.stds = 1
        self.cmin, self.cmax = self.controls.min(), self.controls.max()
        self.normalization = normalization

        self.z = self.normalize(self.controls).T
        self.u, self.s, self.vh = np.linalg.svd(self.z, full_matrices=False)
        #
        
    def normalize(self, x):
        if self.normalization is None: return x
        if self.normalization == 'center': return  (x - self.means)/self.stds
        elif self.normalization == 'minmax' :
            off, norm = (self.cmax+self.cmin)/2., (self.cmax-self.cmin)/2.
            return (x-off)/norm

    def unnormalize(self, y):
        if self.normalization is None: return y
        if self.normalization == 'center': return y * self.stds + self.means
        elif self.normalization == 'minmax' :
            off, norm = (self.cmax+self.cmin)/2., (self.cmax-self.cmin)/2.
            return y*norm + off
        
    def fit(self, treated, T0, n_components, verbose=False, method='lbfgs', regularization='l2', regwt=0.1, dT=3):
        
        yy = self.normalize(treated)
        zv = np.dot(self.z, self.vh[:n_components, :].T)
        
        _chisq = lambda p: (((yy[:T0]) - np.dot(zv, p)[:T0])**2).sum()
        p0 = np.zeros(n_components)
        if method == 'nelder-mead': pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
        elif method == 'lbfgs' : pp = minimize(_chisq, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
        elif method == 'exact':
            #rhs = np.dot(zv.T , yy[:T0])
            #pp = np.dot(np.linalg.inv(np.dot(zv.T, zv)), rhs)
            pp = np.dot(np.linalg.pinv(zv[:, :T0]), yy[:T0])
        
        if verbose == 1: print(pp)
        if method != 'exact': pp = pp.x
        ysvd = self.unnormalize(np.dot(zv, pp))
        return ysvd, pp
    
##    
#    def fit(self, treated, T0, ols=True, verbose=False, method='lbfgs', regularization='l2', regwt=0.1, dT=3):
#        
#        yy = self.normalize(treated.copy())
#        
#        if regwt is None or regularization is None:
#            yy = yy - self.pca.mean_
#            yy = yy[:T0]
#            pp = np.linalg.pinv(self.pca.components_[:, :T0].T).dot(yy)
#            
#        elif type(regwt) == float or type(regwt) == int:
#            
#            def _chisq(p):
#                #yp = self.pca.inverse_transform(p)
#                yp = np.dot(self.pca.components_.T, p) + self.pca.mean_
#                diff = yy-yp
#                diff[T0:] = 0
#                if regularization == 'l2': reg = sum(p**2) * regwt
#                if regularization == 'l1': reg = sum(abs(p)) * regwt
#                if ols : return np.sum(diff**2) + reg
#                else: return np.dot(np.dot(diff, self.icov), diff) + reg
#            p0 = np.zeros(self.n_components)
#            if method == 'nelder-mead': pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
#            elif method == 'lbfgs' : pp = minimize(_chisq, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
#            if verbose == 1: print(pp.fun, sum(pp.x**2)*regwt)
#            if verbose > 1: print(pp)
#            pp = pp.x
#
#        else:
#            rms = 1e10
#            for ir, rr in enumerate(regwt):
#                rmst = 0
#                for tt in range(3, T0, dT):
#                    p0 = np.zeros(self.n_components)
#                    def _chisq(p):
#                        #yp = self.pca.inverse_transform(p)
#                        yp = np.dot(self.pca.components_.T, p) + self.pca.mean_
#                        diff = yy-yp
#                        diff[tt:] = 0
#                        if regularization == 'l2': reg = sum(p**2) * rr
#                        if regularization == 'l1': reg = sum(abs(p)) * rr
#                        if ols : return np.sum(diff**2) + reg
#                        else: return np.dot(np.dot(diff, self.icov), diff) + reg
#                    if method == 'nelder-mead': pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
#                    elif method == 'lbfgs' : pp = minimize(_chisq, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
#                    rmst += (yy - self.pca.inverse_transform(pp.x))[tt]**2.
#                if rmst**1. < rms:
#                    rms = rmst**1.
#                    #print(pp.x)
#                    pp0 = pp
#                    ir0 = ir
#                else: pass
#            pp = pp0
#            regwt = regwt[ir0]
#            if verbose == 1:
#                print("min rms is at reg = %0.4f"%(regwt))
#            if verbose > 1: print(pp.fun, sum(pp.x**2)*regwt)
#            pp = pp.x
#
#        yp = np.dot(self.pca.components_.T, pp) + self.pca.mean_
#        #yp = self.pca.inverse_transform(pp)
#        yp = self.unnormalize(yp)
#        return yp, pp
#
