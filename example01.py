from samplers import sampler
#from solvers import solver
#from recyclers import recycler

import pylab as pl

nsmp = 1000
mc = sampler(nEl=1000, smp_type="mc", model="SExp", sig2=.357, L=0.1, delta=1e-3)
mc.compute_KL()

kappa = []
ax = pl.subplot()
for i_smp in range(nsmp):
  mc.draw_realization()
  kappa += [mc.get_kappa()]
  ax.plot(mc.get_kappa())
pl.show()