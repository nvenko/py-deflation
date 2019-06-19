from samplers import sampler
from solvers import solver
import pylab as pl

nsmp = 500
mcmc = sampler(nEl=1000, smp_type="mcmc", model="Exp",
	        sig2=.357, L=0.1, delta2=1e-3)
mcmc.compute_KL()

pcg  = solver()

#pcgmo = recycler()

#for ismp in range(nsmp):
#  sampler.draw_realization()
#  sampler.do_assembly()
#  recycler.build_new_deflation_subspace()
#  recyler.solve_current_system()
  
"""
kappa = []
ax = pl.subplot()
for i_smp in range(nsmp):
  mcmc.draw_realization()
  kappa += [mcmc.get_kappa()]
  ax.plot(mcmc.get_kappa(), lw=.4)
pl.show()
"""


# Define sampler (smp_type, ...)
# Define solver (eps, precond, ...)
# Define recyler(sampler, solver, recycler_type, ...)
# For ismp in (1, nsmp):
#   sampler.draw_realization()
#   sampler.do_assembly()
#   recycler.solve()
# 



