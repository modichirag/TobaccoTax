import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from tools import approximate_rank, hsvt, linear_regression


def hsvt_ols(X1, X2, y1, t=0.99, rcond=1e-15, include_pre=True):
    # find underlying ranks
    rank1 = approximate_rank(X1, t=t)
    rank2 = approximate_rank(X2, t=t)
    print(rank1, rank2)

    # de-noise donor matrices
    X1_hsvt = hsvt(X1, rank=rank1)
    X2_hsvt = hsvt(X2, rank=rank2)

    # learn synthetic control via linear regression
    beta = linear_regression(X1_hsvt, y1, rcond=rcond)
    # forecast counterfactuals
    y2h = X2_hsvt.dot(beta).T
    yh = np.concatenate([X1_hsvt.dot(beta).T, y2h]) if include_pre else y2h 

    # prediction intervals 
    std = np.sqrt(np.mean((X1 - X1_hsvt)**2))
    return yh


def hsvt_fit(controls, treated, T0, t=0.99, rcond=1e-15, include_pre=True, retbeta=True, verbose=False, combined=False):

    y1 = treated[:T0]

    if combined:
        X1, X2 = controls[:, :T0], controls[:, T0:]
        X1, X2 = X1.T, X2.T
        rank = approximate_rank(controls.T, t=t)
        X_hsvt = hsvt(controls.T, rank=rank)
        X1_hsvt = X_hsvt[:T0, :]
        X2_hsvt = X_hsvt[T0:, :]
        if verbose: print(rank)        
    else:
        X1, X2 = controls[:, :T0], controls[:, T0:]
        X1, X2 = X1.T, X2.T
        # find underlying ranks
        rank1 = approximate_rank(X1, t=t)
        rank2 = approximate_rank(X2, t=t)
        if verbose: print(rank1, rank2)        
        # de-noise donor matrices
        X1_hsvt = hsvt(X1, rank=rank1)
        X2_hsvt = hsvt(X2, rank=rank2)

    # learn synthetic control via linear regression
    beta = linear_regression(X1_hsvt, y1, rcond=rcond)
    # forecast counterfactuals
    y2h = X2_hsvt.dot(beta).T
    yh = np.concatenate([X1_hsvt.dot(beta).T, y2h]) if include_pre else y2h 

    # prediction intervals 
    std = np.sqrt(np.mean((X1 - X1_hsvt)**2))
    if retbeta: return yh, beta
    else: return yh

