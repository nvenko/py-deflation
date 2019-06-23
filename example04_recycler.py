from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

nEl = 1000
nsmp = 100
sig2, L = .357, 0.05
model = "Exp"

mcmc = sampler(nEl=nEl, smp_type="mcmc", model=model, sig2=sig2, L=L, seed=123)
mcmc.compute_KL()

mcmc.draw_realization()
mcmc.do_assembly()

kl = 20

cg = solver(n=mcmc.n, solver_type="cg")
dcg = solver(n=mcmc.n, solver_type="dcg")

dcgmo = recycler(sampler=mcmc, solver=dcg, recycler_type="dcgmo", kl=kl)

cgmo_it = []
dcgmo_it, dcgmo_kdim, dcgmo_ell = [], [], []
while (mcmc.cnt_accepted_proposals < nsmp):
  mcmc.draw_realization()
  if (mcmc.proposal_accepted):
    dcgmo.do_assembly()
    dcgmo.prepare()
    dcgmo_kdim += [dcgmo.solver.kdim]
    dcgmo_ell += [dcgmo.solver.ell]
    dcgmo.solve()
    dcgmo_it += [dcg.it]

    cg.solve(A=mcmc.A, b=mcmc.b, x0=np.zeros(mcmc.n))
    cgmo_it += [cg.it]

    print("%d/%d" %(mcmc.cnt_accepted_proposals, nsmp))

fig, ax = pl.subplots(1, 2, figsize=(8.5,3.7))
ax[0].plot(dcgmo_it, label="dcgmo")
ax[0].plot(cgmo_it, label="cgmo")
ax[1].plot(dcgmo_it, label="dcgmo")
ax[1].plot(cgmo_it, label="cgmo")
ax[0].set_xlabel("Realization index, t"); ax[1].set_xlabel("Realization index, t")
ax[0].set_ylabel("Number of solver iterations, n_it")
fig.suptitle("DCGMO")
ax[0].legend(frameon=False)
#pl.show()
pl.savefig("example04_recycler.png", bbox_inches='tight')
