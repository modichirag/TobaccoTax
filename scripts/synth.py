import  numpy as np
from scipy.optimize import fmin_slsqp, minimize
import pandas as pd


def basic_dataprep(predictors_matrix, outcomes_matrix, 
             treated_unit, control_units, predictors_optimize, 
             outcomes_optimize, years_plot):
    print('\nDo data prep\n')

    # if the list of controls contains the treated unit, remove treated unit.
    while treated_unit in control_units:
        control_units.remove(treated_unit)

        
    
    try:
        X1 = predictors_matrix.loc[predictors_optimize][treated_unit]
        del predictors_matrix[treated_unit]
        X0 = predictors_matrix.loc[predictors_optimize][control_units]
        print('No exception')
    except Exception as e:
        print('Exception : ', e)
        X1, X0 = None, None
    
    Z3 = outcomes_matrix.loc[years_plot][treated_unit]
    Z2 = outcomes_matrix.loc[years_plot][control_units]
    Z1 = outcomes_matrix.loc[outcomes_optimize][treated_unit]
    Z0 = outcomes_matrix.loc[outcomes_optimize][control_units]
             
    return X0, X1, Z0, Z1, Z2, Z3


def w_rss(w, v, x0, x1):
    k = len(x1)
    importance = np.zeros((k,k))
    np.fill_diagonal(importance, v)
    predictions = np.dot(x0, w)
    errors = x1 - predictions
    weighted_errors = np.dot(errors.transpose(), importance)
    try: weighted_rss = np.dot(weighted_errors,errors).item(0)
    except: weighted_rss = np.dot(weighted_errors,errors)
    return weighted_rss


def v_rss(w, z0, z1):
    predictions = np.dot(z0,w)
    errors = z1 - predictions
    rss = sum(errors**2)
    return rss


def get_v_0(v, w, x0, x1, z0, z1):
    weights = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=0, full_output=True, iter=1000, acc=1e-7)[0]
    rss = v_rss(weights, z0, z1)
    return rss
    
def get_v_1(v, w, x0, x1, z0, z1):
    result = minimize(get_v_0, v, args=(w, x0, x1, z0, z1), bounds=[(0.0, 1.0)]*len(v), options={'maxiter':5000}) # 
    importance = result.x
    return importance
    

def w_constraint(w, v, x0, x1):
    return np.sum(w) - 1


def w_constraintnopred(w, x0, x1):
    return np.sum(w) - 1
    

def get_w(w, v, x0, x1):
    result = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=1, full_output=True, iter=1000, acc=1e-7)
    weights = result[0]
    return weights


def get_estimate(x0, x1, z0, z1, z2):
    #print('z0.shape, z1.shape, z2.shape: ',  z0.shape, z1.shape, z2.shape)
    #if x0 is not None: print('x0.shape, x1.shape : ', x0.shape, x1.shape)
    
    j = z0.shape[1]
    w = np.array([1.0/j]*j)
    use_predictors = True
    if x1 is not None: k = len(x1)
    else: use_predictors = False
    if use_predictors:
        print('Use predictors')
        v = [1.0/k]*k
        #print('len(v), w.shape : ',len(v), w.shape)
        predictors = get_v_1(v, w, x0, x1, z0, z1)
        controls = get_w(w, predictors, x0, x1)
        z_estimates = np.dot(z2,controls)
        return z_estimates, predictors, controls
    else:
        print('Do not use predictors')
        v = None
        #print('len(v), w.shape : ',v, w.shape)
#         controls = fmin_slsqp(v_rss, w, f_eqcons=w_constraintnopred, bounds=[(0.0, 1.0)]*len(w),
#              args=(z0, z1), disp=False, full_output=True)[0]
        controls = minimize(v_rss, w, args=(z0, z1), constraints={'type':'eq', 'fun': lambda t: np.sum(t) - 1},  bounds=[(0.0, 1.0)]*len(w)).x
        z_estimates = np.dot(z2,controls)
        return z_estimates, None, controls

    
def synth_tables(predictors_matrix, outcomes_matrix, treated_unit, control_units, 
    predictors_optimize, outcomes_optimize, years_plot):
    
    if predictors_matrix is not None:
        preds, outcomes = predictors_matrix.copy(), outcomes_matrix.copy()
    else:
        preds, outcomes = None, outcomes_matrix.copy()
        
    X0, X1, Z0, Z1, Z2, Z3 = basic_dataprep(preds, outcomes,
        treated_unit, control_units, predictors_optimize, outcomes_optimize, 
        years_plot)
        
    estimates, predictors, controls = get_estimate(X0, X1, Z0, Z1, Z2)
#     float:right
    if predictors is not None:
        estimated_predictors = np.dot(X0,controls)
        predictors_table = pd.DataFrame({'Synthetic':estimated_predictors, 'Actual': X1},index=X1.index)
    else: predictors_table = None
    
    estimated_outcomes = np.dot(Z2,controls)
    outcomes_table = pd.DataFrame({'Synthetic':estimated_outcomes, 'Actual':Z3},index=Z3.index)

    
    return estimates, Z3, predictors_table, outcomes_table, predictors, controls
    
