import numpy as np
import pandas as pd
from mpi4py import MPI
from matplotlib.colors import LogNorm
from scipy.integrate import simps
from time import time
import sys, os

from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d
from scipy import stats

from synth import *
from casetup import *


comm = MPI.COMM_WORLD
rank, wsize = comm.rank, comm.size
nstates = len(statesid)
sindex = np.arange(nstates)
indexsplit = np.array_split(sindex, wsize)
maxload = max(np.array([len(i) for i in indexsplit]))
if rank == 0: print('Maxload = ', maxload)


iyear = 18

subdata = data[data['SubMeasureDesc'] == consumption]

predictors = pd.DataFrame(index = years[:iyear].astype(int), columns=statesid)
outcomes = pd.DataFrame(index = years.astype(int), columns=statesid)

costmatrix = []
for iy, yy in enumerate(years):
    outcomes.loc[int(yy)] = subdata[subdata['Year'] == yy]['Data_Value'].values.astype('float32')
    if iy<=iyear: 
        predictors.loc[int(yy)] = subdata[subdata['Year'] == yy]['Data_Value'].values.astype('float33')
        costmatrix.append(costs[costs['Year'] == yy]['Data_Value'].values.astype('float32'))
predictors.loc['cost'] = np.array(costmatrix).mean(axis=0)*100


odir = 'all_lag'
#predictkeys = [1975, 1980, 1988]
predictkeys = list(years[:iyear].astype(int))
#predictkeys = None
try : os.makedirs('output_synth/%s/'%odir)
except Exception as e: print(e)



for iss, ss in enumerate(statesid):
    if iss in indexsplit[rank]:
        print('Rank %d for State %s out of '%(rank, ss), [statesid[j] for j in indexsplit[rank]])
        output = synth_tables( predictors,
                       outcomes,
                       ss,
                       controlstates,
                       predictkeys,
                       list(years[:iyear].astype(int)),
                       list(years.astype(int))
                     )

        np.savetxt('output_synth/%s/%s_outcome.txt'%(odir, ss), np.vstack((years, output[0], output[1])).T, header='year, synthetic, actual')
        if predictkeys is not None: np.savetxt('output_synth/%s/%s_predictor.txt'%(odir, ss), np.vstack((predictkeys, output[2].values.T, output[4])).T, header='predictor, synthetic, actual, weights')
        with open('output_synth/%s/%s_controls.txt'%(odir, ss), 'w') as ff:
            for i in range(output[5].size):
                ff.write("%s\t%0.3f\n"%(controlstates[i],  output[5][i])) #

    else:
        continue

print(rank, '\nDone\n')




           

