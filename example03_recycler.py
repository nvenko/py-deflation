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

pcg  = solver(n=mcmc.n, solver_type="pcg")
pcg.set_precond(Mat=mcmc.get_median_A(), precond_id=1)

pcgmo = recycler(sampler=mcmc, solver=pcg, recycler_type="pcgmo")

for i_smp in range(nsmp):
  pcgmo.draw_realization()
  pcgmo.prepare()
  pcgmo.solve()
