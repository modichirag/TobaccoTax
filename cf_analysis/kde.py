""""The :mod:`sklearn.kernel_regressor` module implements the Kernel Regressor.
"""
# Author: Jan Hendrik Metzen <janmetzen@mailbox.de>
#
# License: BSD 3 clause

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, RegressorMixin


class KernelRegression(BaseEstimator, RegressorMixin):
    """Nadaraya-Watson kernel regression with automatic bandwidth selection.

    This implements Nadaraya-Watson kernel regression with (optional) automatic
    bandwith selection of the kernel via leave-one-out cross-validation. Kernel
    regression is a simple non-parametric kernelized technique for learning
    a non-linear relationship between input variable(s) and a target variable.

    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF ("bandwidth"), polynomial,
        exponential chi2 and sigmoid kernels. Interpretation of the default
        value is left to the kernel; see the documentation for
        sklearn.metrics.pairwise. Ignored by other kernels. If a sequence of
        values is given, one of these values is selected which minimizes
        the mean-squared-error of leave-one-out cross-validation.

    See also
    --------

    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
    """

    def __init__(self, kernel="rbf", gamma=None):
        self.kernel = kernel
        self.gamma = gamma

    def fit(self, X, y):
        """Fit the model

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values

        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X
        self.Y = y

        if hasattr(self.gamma, "__iter__"):
            if self.gamma is not None: 
                self.gamma = self._optimize_gamma(self.gamma)

        return self

    def predict(self, Xp, Yp=None, retK=False):
        """Predict target values for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted target value.
        """
        if Yp is None: Yp = self.Y #Usually we want this, but as diagnostics for pre-intervention
        if self.gamma is not None: K = pairwise_kernels(self.X, Xp, metric=self.kernel, gamma=self.gamma)
        else : K = pairwise_kernels(self.X, Xp, metric=self.kernel)
        y_pred = np.dot(K.T, Yp)/(K.sum(axis=0)[:, np.newaxis] + 1e-10)
        if retK: return y_pred, K
        else: return y_pred
        #return (K * self.y[:, None]).sum(axis=0) / K.sum(axis=0)

        
    def _optimize_gamma(self, gamma_values):
        # Select specific value of gamma from the range of given gamma_values
        # by minimizing mean-squared error in leave-one-out cross validation
        mse = np.empty_like(gamma_values, dtype=np.float)
        for i, gamma in enumerate(gamma_values):
            K = pairwise_kernels(self.X, self.X, metric=self.kernel,
                                 gamma=gamma)
            np.fill_diagonal(K, 0)  # leave-one-out
            #Ky = K * self.y[:, np.newaxis]            
            #y_pred = Ky.sum(axis=0) / K.sum(axis=0)
            y_pred = np.dot(K.T, self.Y)/(K.sum(axis=0)[:, np.newaxis] + 1e-10)
            mse[i] = ((y_pred - self.Y) ** 2).mean()

        return gamma_values[np.nanargmin(mse)]
