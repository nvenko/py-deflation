from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl

nEl = 1000
nsmp = 50
sig2, L = .357, 0.05
model = "Exp"

mcmc = sampler(nEl=nEl, smp_type="mcmc", model=model, sig2=sig2, L=L)
mcmc.compute_KL()

pcg  = solver(solver_type="pcg", precond_id=1)
pcg.set_precond()

pcgmo = recycler(sampler=mcmc, solver=pcg, recylcer_type="pcgmo")

for i_smp in range(nsmp):
  recycler.draw_realization()
  recycler.prepare()
  recycler.solve()
