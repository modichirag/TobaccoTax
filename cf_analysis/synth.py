import numpy as np
from scipy.optimize import minimize


##Synthetic controls
    
def v_rss(w, z0, z1):                                                                                                                    
    predictions = np.dot(w, z0)                                                                                                           
    errors = z1 - predictions                                                                                                            
    rss = sum(errors**2)                                                                                                                 
    return rss                                                                                                                           


def get_estimate(z0, z1, z2, constraint=True, bound=True, reg=None):
    j = z0.shape[0]                                                                                                                               
    w = np.array([1.0/j]*j)*0                                                                                                                       
    #         controls = fmin_slsqp(v_rss, w, f_eqcons=w_constraintnopred, bounds=[(0.0, 1.0)]*len(w),                                                
    #              args=(z0, z1), disp=False, full_output=True)[0]

    if reg == 'l1' : chisq = lambda w, x, y: ((y - np.dot(w, x))**2).sum() + abs(w).sum()
    elif reg == 'l2' : chisq = lambda w, x, y: ((y - np.dot(w, x))**2).sum() + (w**2).sum()
    else : chisq = lambda w, x, y: ((y - np.dot(w, x))**2).sum()

    if bound : bounds=[(0.0, 1.0)]*len(w)
    else: bounds = None
    if constraint: constraints={'type':'eq', 'fun': lambda t: np.sum(t) - 1}
    else: constraints = None
    weights = minimize(chisq, w, args=(z0, z1), constraints=constraints, bounds=bounds).x    

    z_estimates = np.dot(weights, z2)                                                                                                         
    return z_estimates, weights                                                                                                        


def fit(controls, treated, T0, constraint=True, bound=True, reg=None):
    '''
    controls : shape(N, T) where N is number of units, T is time series data > T0
    treated : shape(T, )
    T0 : int, index of intervention
    '''

    Z0, Z1 = controls[:, :T0], treated[:T0]
    Z2 = controls.copy()
                                                                                                                         
    return  get_estimate(Z0, Z1, Z2, constraint, bound, reg=reg)  
