import numpy as np



##Synthetic controls
    
def v_rss(w, z0, z1):                                                                                                                    
    predictions = np.dot(w, z0)                                                                                                           
    errors = z1 - predictions                                                                                                            
    rss = sum(errors**2)                                                                                                                 
    return rss                                                                                                                           


def get_estimate2(z0, z1, z2):                                                                                                             
    j = z0.shape[0]                                                                                                                               
    w = np.array([1.0/j]*j)                                                                                                                       
    v = None                                                                                                                                  
#         controls = fmin_slsqp(v_rss, w, f_eqcons=w_constraintnopred, bounds=[(0.0, 1.0)]*len(w),                                                
#              args=(z0, z1), disp=False, full_output=True)[0]                                                                                    
    weights = minimize(v_rss, w, args=(z0, z1), constraints={'type':'eq', 'fun': lambda t: np.sum(t) - 1}, 
                        bounds=[(0.0, 1.0)]*len(w)).x    
    z_estimates = np.dot(weights, z2)                                                                                                         
    return z_estimates, weights                                                                                                        
                                                                 
def synth(controls, treated, T0):
    '''
    controls : shape(N, T) where N is number of units, T is time series data > T0
    treated : shape(1, T)
    T0 : int, index of intervention
    '''

    Z0, Z1 = controls[:, :T0], treated[:, :T0]
    Z2 = controls.copy()
                                                                                                                         
    return  get_estimate2(Z0, Z1, Z2)                                                   

#########################################################################################################

##PCA

from sklearn.decomposition import PCA


def pcafit(controls, treated, T0, n_components = 2):
    
    pca = PCA(n_components=n_components)
    means = controls.mean(axis=0) 

    pca.fit(controls-means)
    cov = pca.get_covariance()
    icov = pca.get_precision()
    
    def _chisq(p):
        yp = pca.inverse_transform(p)
        diff = treated-means-yp
        diff[T0:] = 0
        return np.dot(np.dot(diff, icov), diff)

    p0 = np.zeros(n_components)
#     pp = minimize(_chisq, p0, method='Nelder-Mead', options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
    pp = minimize(_chisq, p0, options={'maxiter':50000, 'maxfev':50000, 'tol':1e-10, 'rtol':1e-10})
    print(pp.fun)
    pp = pp.x
    yp = pca.inverse_transform(pp)
    yp += means
    return yp, pca, pp


def conditional_gaussian(mean, cov, vals, index, given='left'):
    if given == 'left':
        cov_yy = cov[index:,index:]
        cov_xx = cov[:index,:index]
        cov_xy = cov[:index,index:]
        mean_x = mean[:index]
        mean_y = mean[index:]
    elif given == 'right':
        cov_xx = cov[index:,index:]
        cov_yy = cov[:index,:index]
        cov_xy = cov[index:,:index]
        mean_x = mean[index:]
        mean_y = mean[:index]
        
    icov_xx = np.linalg.inv(cov_xx)
    cond_mean = mean_y + np.dot(cov_xy.T, np.dot(icov_xx, vals-mean_x))
    cond_cov  = cov_yy - np.dot(cov_xy.T, np.dot(icov_xx,cov_xy))
    return cond_mean, cond_cov 
