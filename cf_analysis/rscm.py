import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA


class RSCM():

    def __init__(self, controls, n_components, stdscale=False, seed=100, normalization='center'):

        self.N = controls.shape[0]
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
        self.svd_threshold()

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

    def svd_threshold(self):
        z = self.normalize(self.controls) 
        self.u, self.s, self.vh = np.linalg.svd(z, full_matrices=False)
        n = self.n_components
        self.controls_svd = np.dot(np.dot(self.u[:, :n], np.diag(self.s[:n])), self.vh[:n])
        
    
    def fit(self, treated, T0, verbose=False, method='lbfgs', regularization=None, regwt=None, dT=2):
        
        yy = self.normalize(treated)[:T0]
        #yy = (treated)[:T0]
        if regularization is None:
            print('No regularization')
            #z = self.controls_svd
            #rhs = np.dot(z , yy[:T0])
            #pp = np.dot(np.linalg.inv(np.dot(z, z.T)), rhs)
            pp = np.dot(np.linalg.pinv(self.controls_svd[:, :T0].T), yy[:T0])

            
        elif type(regwt) == float or type(regwt) == int:
            
            def _chisq(p, verbose=False):
                res = ((np.dot(p, self.controls_svd)[:T0] - yy)**2).sum()
                if regularization == 'l2': reg = sum(p**2) * regwt
                if regularization == 'l1': reg = sum(abs(p)) * regwt
                if verbose: print(res, reg)
                return res + reg

            p0 = np.zeros(self.N)*0
            if method == 'nelder-mead': pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
            elif method == 'lbfgs' : pp = minimize(_chisq, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
            if verbose:
                _chisq(pp.x, verbose=True)
                print(pp)
            pp = pp.x
        else:
            print('optimize')
            rms = 1e10
            for ir, rr in enumerate(regwt):
                rmst = 0
                for tt in range(3, T0+dT, dT):
                    if tt > T0: break
                    def _chisq(p, verbose=False):
                        res = ((np.dot(p, self.controls_svd)[:T0] - yy)[:tt]**2).sum()
                        if regularization == 'l2': reg = sum(p**2) * rr
                        if regularization == 'l1': reg = sum(abs(p)) * rr
                        if verbose: print(res, reg)
                        return res + reg

                    p0 = np.random.normal(size=self.N)*0
                    if method == 'nelder-mead': pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
                    elif method == 'lbfgs' : pp = minimize(_chisq, p0, method='L-BFGS-B', options={'maxiter':50000, 'ftol': 1e-10, 'gtol': 1e-8, 'eps': 1e-10, 'maxfun': 50000})
                    if tt <= T0-1: rmst += (yy - np.dot(pp.x, self.controls_svd)[:T0])[tt]**2.
                    else: rmst += (yy - np.dot(pp.x, self.controls_svd)[:T0])[T0-1]**2.
                if rmst < rms:
                    rms = rmst
                    #print(ir, rms, pp.fun)
                    pp0 = pp
                    ir0 = ir
                else: pass
            pp = pp0
            regwt = regwt[ir0]
            print("min rms is at reg = %0.4f with rms = %0.4e"%(regwt, rms))
            if verbose: print(pp)
            pp = pp.x
            
        yp = np.dot(pp, self.controls_svd)
        yp = self.unnormalize(yp)
        return yp, pp

        
