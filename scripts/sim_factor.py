import numpy as np
import matplotlib.pyplot as plt
import sys, os
from mpi4py import MPI

import pandas as pd
from scipy.integrate import simps
from time import time
import os, sys
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d
from scipy import stats


from casetup import *
from tools import *
from yjtransform import *
from synth import *
from glearn import *


comm = MPI.COMM_WORLD
rank, wsize = comm.rank, comm.size

odir = './output_sim/nopropensity/'
if rank == 0:
    try: os.makedirs(odir)
    except Exception as e: print(e)
    try: os.makedirs(odir + '/figs/')
    except Exception as e: print(e)
    



def getload(treat, nf, w0=2*0.8*3**0.5, lnt0=-3**0.5, lnt1=3**0.5, lt0=3**0.5, lt1=3**1.5, getfixed=False):
    if getfixed : nf += 1
#    if treat: return  np.random.uniform(lt0 - w0, lt1 - w0, size=nf)
    if treat: return  np.random.uniform(lnt0, lnt1, size=nf)
    else: return  np.random.uniform(lnt0, lnt1, size=nf)



def simdata(Nco, Ntr, T, T0, nfactors, ff, getfixed=True):
    N = Nco + Ntr
    Y0, Y1, loads = [], [], []

    for i in range(N):

        if i < Ntr: 
            load = getload(True, nfactors, getfixed=getfixed)            
            treatment = np.zeros(T)
            treatment[T0:] = np.arange(1, T-T0+1) + np.random.normal(size=T-T0)            
        else:
            load = getload(False, nfactors, getfixed=getfixed)
            treatment = np.zeros(T)
        
        if getfixed: load, alpha = load[:-1], load[-1]
        else: alpha = 0
        loads.append(load)
        
        facwt = np.dot(ff.T, load)
        eps, xi = np.random.normal(size=T), np.random.normal(size=T)
        
        cov = np.zeros((nfactors, T))
        for j in range(nfactors):
            cov[j] = (1 + facwt + load.sum() + ff.sum(axis=0) + np.random.normal(size=T))
            
#         covload = np.random.randint(1, 5, nfactors)
        covload = np.array((1, 3))
        
        Y0.append(np.dot(cov.T, covload) + facwt + alpha + xi + 5 + eps)
        Y1.append(treatment + np.dot(cov.T, covload) + facwt + alpha + xi + 5 + eps)
    return np.array(Y0), np.array(Y1), np.array(loads)    


def factormodel(seed, Nco, Ntr, T, T0, nfactors):
    np.random.seed(seed)
    ff = np.random.normal(size=T*nfactors).reshape(nfactors, T)       
    yy0, yy1, loads = simdata(Nco, Ntr, T, T0, nfactors, ff) 
    return yy0, yy1, loads



####################################

def getsynth(yy, T, T0, iss=0, ss=0):

    N = yy.shape[0]
    predictors = pd.DataFrame(index = np.arange(T0), columns=np.arange(N))
    outcomes = pd.DataFrame(index = np.arange(T), columns=np.arange(N))
    for iy in np.arange(T):
        outcomes.loc[int(iy)] = yy[:, iy]#subdata[subdata['Year'] == yy]['Data_Value'].values.astype('float32')
        if iy<=T0:
            predictors.loc[int(iy)] = yy[:, iy]#subdata[subdata['Year'] == yy]['Data_Value'].values.astype('float32')

    predictkeys = list(np.arange(T0).astype(int))

    output = synth_tables( predictors,
                   outcomes,
                   ss,
                   np.arange(1, Nco+Ntr),
                   predictkeys,
                   list(np.arange(T0).astype(int)),
                   list(np.arange(T).astype(int))
                 )

    return output


####################################


def lgfit(yy, T, T0, seed):
    params, gdata, ddata = transform(yy, stdscale=False)

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.plot(ddata[0].T)
    plt.ylim(ddata[0].min(), ddata[0].max())
    plt.title('Data')

    plt.subplot(132)
    plt.title('Gauss')
    plt.plot(gdata[0].T)
    plt.ylim(ddata[0].min(), ddata[0].max())

    plt.subplot(133)
    plt.hist(gdata[0].flatten(), normed=True, label='Gauss')
    plt.hist(ddata[0].flatten(), alpha=0.5, normed=True, label='raw');
    plt.legend()
    plt.savefig(odir + '/figs/transformcheck-%d.png'%(seed))
    plt.close()
    ####################################
    padl, padr = T, T
    timep = np.arange(T)
    xpg = gdata[0]
    xcg = gdata[1]
    mu, sig, eta, eps, beta = params[0]
    nl = np.array([eta, eps, beta])


    xpgpad = np.pad(xpg, [(0, 0), (padl, padr)], mode='constant', constant_values=0)
    ff = np.fft.rfftfreq(xpgpad[0].size)
    pk = []
    for i in range(xpgpad.shape[0]): pk.append(psfunc(xpgpad[i]))
    pk = np.array(pk)
    pkm = pk.mean(axis=0)
    tfpad = gettfprior(xpgpad[:, padl:-padr], padl, padr, al=0)
    pkprior = pkm/tfpad
    cov = getcov(pkprior, padl, padr)
    errg = cov.diagonal()**0.5

    pktreated = psfunc(np.pad(xcg, (padl, padr), mode='constant', constant_values=0))

    p0 = np.ones(pkm.size*2)
    chisqmin = lambda p: fourierposterior(p, xcg, pkprior, iyear=T0, padl=padl, sigma=abs(xcg).mean()/100.)
    pp = minimize(chisqmin, p0).x


    predg = fouriertox(pp)
    deriv = (from_gauss(predg*1.01, nl) - from_gauss(predg*0.99, nl))/(predg*0.02)
    err = (errg*deriv)[padl:-padr] *params[-1]

    predup = (from_gauss(predg+errg, nl)[padl:-padr] + mu)*params[-1] + params[-2]
    preddn = (from_gauss(predg-errg, nl)[padl:-padr]+ mu)*params[-1] + params[-2]
    pred = (from_gauss(predg, nl)[padl:-padr]+mu)*params[-1] + params[-2]

    return pred, err


# ####################################




if __name__=="__main__":


    for i in range(10):
        Nco, Ntr = 45, 5
        T, T0 = 30, 20
        nfactors = 2
        seed = rank*100 + i
        print(rank, seed)
        yy0, yy1, loads = factormodel(seed, Nco, Ntr, T, T0, nfactors)
        yy = yy0.copy()
        yy[0] = yy1[0]
        
        gpred, err = lgfit(yy, T, T0, seed)
        synth = getsynth(yy, T, T0)

        np.savetxt(odir + 'cf-%d.txt'%seed, np.stack((yy1[0], yy0[0], synth[0], gpred, err)).T, header='Treated, TrueCf, Synth (all lag), GL Pred, Error')
        np.savetxt(odir + 'unitwt-%d.txt'%seed, synth[-1])

                
