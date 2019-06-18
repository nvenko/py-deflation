from samplers import sampler
#from solvers import solver
#from recyclers import recycler

import pylab as pl

nsmp = 1000
mcmc = sampler(nEl=1000, smp_type="mcmc", model="Exp", 
	        sig2=.357, L=0.1, delta2=1e-3)
mcmc.compute_KL()

kappa = []
ax = pl.subplot()
for i_smp in range(nsmp):
  mcmc.draw_realization()
  kappa += [mcmc.get_kappa()]
  ax.plot(mcmc.get_kappa())
pl.show()