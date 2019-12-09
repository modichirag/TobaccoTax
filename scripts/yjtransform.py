import numpy as np
import sympy as sp
from sympy.utilities.autowrap import ufuncify


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
        return yj_nested_tool(sp.lambdify([x_sp, eps_sp], yj_1(yj_n(x_sp, eps_sp), eps_sp)), n - 1, yj_1)
    
    else:
        raise ValueError
        
        
yj_sp_p = sp.lambdify([x_sp, eps_sp], ((x_sp + 1)**(1 + eps_sp) - 1) / (1 + eps_sp), 'sympy')
yj_sp_n = sp.lambdify([x_sp, eps_sp], -((-x_sp + 1)**(1 - eps_sp) - 1) / (1 - eps_sp), 'sympy')

jy_sp_p = sp.lambdify([x_sp, eps_sp], ((1 + eps_sp) * x_sp + 1)**(1 / (1 + eps_sp)) - 1, 'sympy')
jy_sp_n = sp.lambdify([x_sp, eps_sp], -(-(1 - eps_sp) * x_sp + 1)**(1 / (1 - eps_sp)) + 1, 'sympy')

sa_sp_p = sp.lambdify([x_sp, eta_sp], sp.sinh(eta_sp * x_sp) / eta_sp, 'sympy')
sa_sp_o = sp.lambdify([x_sp, eta_sp], x_sp, 'sympy')
sa_sp_n = sp.lambdify([x_sp, eta_sp], sp.asinh(eta_sp * x_sp) / eta_sp, 'sympy')

_to_gauss = ufuncify(args = [x_sp, eta_sp, eps_sp, beta_sp],
                     expr = sp.Piecewise((sa_sp_o(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, sp.And(eta_sp > -_sp_eps, eta_sp < _sp_eps, x_sp >= 0)),
                                         (sa_sp_o(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, sp.And(eta_sp > -_sp_eps, eta_sp < _sp_eps, x_sp < 0)),
                                         (sa_sp_p(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, sp.And(eta_sp > 0, x_sp >= 0)),
                                         (sa_sp_p(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, sp.And(eta_sp > 0, x_sp < 0)),
                                         (sa_sp_n(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, sp.And(eta_sp < 0, x_sp >= 0)),
                                         (sa_sp_n(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, sp.And(eta_sp < 0, x_sp < 0)),
                                         (sp.nan, True)),
                     backend='cython')
#                      backend='numpy')

_to_gauss_g = ufuncify(args = [x_sp, eta_sp, eps_sp, beta_sp],
                     expr = sp.Piecewise((sp.diff(sa_sp_o(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, x_sp, 1), sp.And(eta_sp > -_sp_eps, eta_sp < _sp_eps, x_sp >= 0)),
                                         (sp.diff(sa_sp_o(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, x_sp, 1), sp.And(eta_sp > -_sp_eps, eta_sp < _sp_eps, x_sp < 0)),
                                         (sp.diff(sa_sp_p(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, x_sp, 1), sp.And(eta_sp > 0, x_sp >= 0)),
                                         (sp.diff(sa_sp_p(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, x_sp, 1), sp.And(eta_sp > 0, x_sp < 0)),
                                         (sp.diff(sa_sp_n(yj_nested_tool(yj_sp_p, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, x_sp, 1), sp.And(eta_sp < 0, x_sp >= 0)),
                                         (sp.diff(sa_sp_n(yj_nested_tool(yj_sp_n, _yj_n)(x_sp / beta_sp, eps_sp), eta_sp) * beta_sp, x_sp, 1), sp.And(eta_sp < 0, x_sp < 0)),
                                         (sp.nan, True)),
                     backend='cython')
#                      backend='numpy')

_from_gauss = ufuncify(args = [x_sp, eta_sp, eps_sp, beta_sp],
                     expr = sp.Piecewise((yj_nested_tool(jy_sp_p, _yj_n)(sa_sp_o(x_sp / beta_sp, eta_sp), eps_sp) * beta_sp, sp.And(eta_sp > -_sp_eps, eta_sp < _sp_eps, x_sp >= 0)),
                                         (yj_nested_tool(jy_sp_n, _yj_n)(sa_sp_o(x_sp / beta_sp, eta_sp), eps_sp) * beta_sp, sp.And(eta_sp > -_sp_eps, eta_sp < _sp_eps, x_sp < 0)),
                                         (yj_nested_tool(jy_sp_p, _yj_n)(sa_sp_n(x_sp / beta_sp, eta_sp), eps_sp) * beta_sp, sp.And(eta_sp > 0, x_sp >= 0)),
                                         (yj_nested_tool(jy_sp_n, _yj_n)(sa_sp_n(x_sp / beta_sp, eta_sp), eps_sp) * beta_sp, sp.And(eta_sp > 0, x_sp < 0)),
                                         (yj_nested_tool(jy_sp_p, _yj_n)(sa_sp_p(x_sp / beta_sp, eta_sp), eps_sp) * beta_sp, sp.And(eta_sp < 0, x_sp >= 0)),
                                         (yj_nested_tool(jy_sp_n, _yj_n)(sa_sp_p(x_sp / beta_sp, eta_sp), eps_sp) * beta_sp, sp.And(eta_sp < 0, x_sp < 0)),
                                         (sp.nan, True)),
                     backend='cython')
#                      backend='numpy')


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
