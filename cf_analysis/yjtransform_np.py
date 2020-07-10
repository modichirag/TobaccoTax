import numpy as np


x_sp = sp.Symbol('x_sp', real=True)
eta_sp = sp.Symbol('eta_sp', real=True)
eps_sp = sp.Symbol('eps_sp', real=True)
beta_sp = sp.Symbol('beta_sp', positive=True)
_yj_n = 4
_sp_eps = 1e-8


def yj_nested_tool(yj_n, n, yj_1 = None):
    
    if yj_1 is None:
        yj_1 = yj_n
    
    if n == 1:
        return yj_n
    
    elif n > 1:
        return yj_nested_tool( lambda x_sp, eps_sp: yj_1(yj_n(x_sp, eps_sp), eps_sp), n - 1, yj_1)
    
    else:
        raise ValueError
        
        
yj_sp_p = lambda x_sp, eps_sp: ((x_sp + 1)**(1 + eps_sp) - 1) / (1 + eps_sp)
yj_sp_n = lambda x_sp, eps_sp: -((-x_sp + 1)**(1 - eps_sp) - 1) / (1 - eps_sp)

sa_sp_p = lambda x_sp, eta_sp :  sp.sinh(eta_sp * x_sp) / eta_sp
sa_sp_o = lambda x_sp, eta_sp : x_s
sa_sp_n = lambda x_sp, eta_sp : sp.asinh(eta_sp * x_sp) / eta_sp 

def _to_gauss(x_sp, eta_sp, eps_sp, beta_sp):

    expr = np.Piecewise(x_sp,
                        [(eta_sp > -_sp_eps &  eta_sp < _sp_eps &  x_sp >= 0),
                         (eta_sp > -_sp_eps & eta_sp < _sp_eps & x_sp < 0),
                         (eta_sp > 0 & x_sp >= 0),
                         (eta_sp > 0 & x_sp < 0),
                         (eta_sp < 0 & x_sp >= 0),
                         eta_sp < 0 & x_sp < 0,
                         True
                        ],
                        [sa_sp_o(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, 
                        sa_sp_o(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, 
                        sa_sp_p(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, ,
                        sa_sp_p(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, 
                        sa_sp_n(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, 
                        sa_sp_n(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, 
                        sp.nan
                        ])
    return expr


def nl_vectorizer(fun):
    
    def fun_v(zz, nl):
        
        zz = np.atleast_1d(zz)
        nl = np.asarray(nl)
        
        if nl.shape[-1] == 2:
            
            nl = np.concatenate((nl, np.ones((*nl.shape[:-1], 1))), axis=-1)  # set default value of beta to 1
        
        if zz.ndim == 1 and nl.ndim == 1 and nl.shape[-1] == 3: # case I: zz is a bunch of the same variable, with the same NL parameter
                                                                # that is, zz is (# of samples,), nl is (3,)
            
            _x_sp = zz
            _eta_sp = np.tile(nl[0], zz.shape[0])
            _eps_sp = np.tile(nl[1], zz.shape[0])
            _beta_sp = np.tile(nl[2], zz.shape[0])
            _results = fun(_x_sp, _eta_sp, _eps_sp, _beta_sp)
            return _results
        
        elif zz.ndim == 2 and nl.ndim == 2 and nl.shape[-1] == 3: # case II: zz is (# of samples, # of dim), nl is (# of dim, 3)
            
            _x_sp = zz.flatten()
            _eta_sp = np.tile(nl[:, 0], zz.shape[0])
            _eps_sp = np.tile(nl[:, 1], zz.shape[0])
            _beta_sp = np.tile(nl[:, 2], zz.shape[0])
            _results = fun(_x_sp, _eta_sp, _eps_sp, _beta_sp).reshape(zz.shape)
            return _results
        
        elif zz.ndim == 1 and nl.ndim == 2 and zz.shape[0] == nl.shape[0] and nl.shape[-1] == 3:
            # case III: zz is one single multidimensional sample, with different NL parameters in each dim
            # that is, zz is (# of dim,), nl is (# of dim, 3)
            
            _x_sp = zz
            _eta_sp = np.copy(nl[:, 0])
            _eps_sp = np.copy(nl[:, 1])
            _beta_sp = np.copy(nl[:, 2])
            _results = fun(_x_sp, _eta_sp, _eps_sp, _beta_sp)
            
            return _results
        
        else:
            
            raise ValueError
            
    return fun_v


to_gauss = nl_vectorizer(_to_gauss)
to_gauss_g = nl_vectorizer(_to_gauss_g)
from_gauss = nl_vectorizer(_from_gauss)


def norm_logpdf(x, mean, hess, normalized=False):
    
    result = -0.5 * (x - mean) * hess * (x - mean)
    if normalized:
        result += 0.5 * np.log(hess) - 0.5 * np.log(2 * np.pi)
    return result

def log_q(xx, mean, hess, nl, normalized=False):
    
    yy = to_gauss(xx - mean, nl)
    return (norm_logpdf(yy, np.zeros_like(mean), hess, normalized) + np.log(np.abs(to_gauss_g(xx - mean, nl)))).reshape(xx.shape)



def _tomin_yjfit(ngdata, p, rety = False, nd=0, fitbeta=True):
    mu, sig, eta, eps, beta = p
    if not fitbeta: beta = ngdata.std()
    if abs(eps) > 1: return 1e10
    if sig < 0: return 1e10
    yy = (log_q(ngdata, np.array(mu), 1/sig**2, np.array([eta, eps, beta]), True))
    if rety : return np.exp(yy)
    else: return -sum(yy)
    

def get_transform(data, stdscale=False, verbose=False, p0=None):

    dd = data.copy()        
    mmean, sstd = dd.mean(axis=0), dd.std(axis=0)    
    if not stdscale : sstd = 1
    dd = (dd - mmean)/sstd
    
    if p0 is None: p0 = [dd.mean(), dd.std(), 0.1, 0.1, dd.std()]
    ppnl = minimize(lambda x: _tomin_yjfit(dd.flatten(), x, fitbeta=False), p0,
                    options={'gtol': 1e-08, 'norm': np.inf, 'eps': 1.4901161193847656e-10,  'maxiter': 20000})

    #     ppnl = minimize(lambda x: tominyjfit(dd.flatten(), x, fitbeta=False), p0, method='Nelder-Mead',
    #                     options={'gtol': 1e-08, 'norm': np.inf, 'eps': 1.4901161193847656e-10,'maxiter': 20000})
    if verbose: print(ppnl.x)
    return [mmean, sstd] + list(ppnl.x)



def transform(data, pp):
    mmean, sstd, mu, sig, eta, eps, beta = pp
    xp = (data - mmean)/sstd
    nl = np.array([eta, eps, beta])
    xpg = to_gauss((xp.flatten().astype('float64') - mu), nl).reshape(data.shape[0], -1)
    return xpg


def invtransform(datag, pp):
    xpg = datag.flatten()
    mmean, sstd, mu, sig, eta, eps, beta = pp
    nl = np.array([eta, eps, beta])
    data = ((from_gauss(xpg.astype('float64'), nl)+mu)*sstd + mmean).reshape(datag.shape)
    return data

def getpdf(xx, pp, normalized=True):
    mmean, sstd, mu, sig, eta, eps, beta = pp
    hess = sig**-2.
    nl = [eta, eps, beta]
    yy = to_gauss(xx - mmean, nl)    
    return np.exp(norm_logpdf(yy, np.zeros_like(mmean), hess, normalized) + np.log(np.abs(to_gauss_g(xx - mmean, nl)))).reshape(xx.shape)
