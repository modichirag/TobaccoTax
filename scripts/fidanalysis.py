import numpy as np
from lab import *
from transforms import *

ii = 0 
preds = []
# for padl, padr in [[1, 1],  [10, 10], [yy//2, yy//2], [yy, yy]]:
for padl, padr in [[yy, yy]]:
    ii +=1
    xp, means, pk, pkca = setupdata(salearrayskip, casales, padl, padr)
    pkm = pk.mean(axis=0)
    ffpad  = getf(padl, padr)
    tfpad = gettfprior(salearrayskip, padl, padr)
    pktrue = pkm/tfpad
    
    p0 = np.ones(pkm.size*2)
    tomin = lambda p: chisq(p, means, casales, pktrue, padl)
    pp = minimize(tomin, p0).x
    capred = pred(pp, means)
        
    cov = getcov(pktrue, padl, padr)
    err = cov.diagonal()**0.5
    preds.append([capred, err])

    if padr: plt.errorbar(years, capred[padl:-padr], err[padl:-padr], alpha=0.5, lw=2, elinewidth=1, label='Predict\n(padratio=%.2f)'%(2*padl/yy))
    else: plt.errorbar(years, capred[padl:], err[padl:], alpha=0.5, lw=2, elinewidth=1, label=ii)
    print('Expected difference : ', capred[padl:-padr][years==2000] - casales[years==2000])
    plt.title(' Expected difference : %.2f'%(capred[padl:-padr][years==2000] - casales[years==2000]))
    
plt.plot(years, casales, 'b--', lw=2, label='True sales')
plt.plot(years, meansales, 'r--', lw=2, label='National mean')
plt.grid()
plt.legend()

capredfid, errfid, covfid = capred[padl:-padr].copy(), err[padl:-padr].copy(), cov[padl:-padr, padl:-padr].copy()
# plt.xlim(years[i1], years[-1])
# plt.ylim(20, 100)
