from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

nEl = 1000
nsmp = 1999
sig2, L = .357, 0.05
model = "Exp"

mcmc = sampler(nEl=nEl, smp_type="mcmc", model=model, sig2=sig2, L=L)
mcmc.compute_KL()

mcmc.draw_realization()
mcmc.do_assembly()

nb = 5
dt = [50, 100, 250, 500, 1000]
pcg_dtbJ = []
pcgmo_dtbJ = []

for i, dt_i in enumerate(dt):
  pcg_dtbJ += [solver(n=mcmc.n, solver_type="pcg")]
  pcg_dtbJ[i].set_precond(Mat=mcmc.A, precond_id=3, nb=nb)

pcg_medbJ  = solver(n=mcmc.n, solver_type="pcg")
pcg_medbJ.set_precond(Mat=mcmc.get_median_A(), precond_id=3, nb=nb)

for i, dt_i in enumerate(dt):
  pcgmo_dtbJ += [recycler(sampler=mcmc, solver=pcg_dtbJ[i], recycler_type="pcgmo", dt=dt_i)]
pcgmo_medbJ = recycler(sampler=mcmc, solver=pcg_medbJ, recycler_type="pcgmo")

pcgmo_dtbJ_it, pcgmo_medbJ_it = [[] for i in range(len(dt))], []
while (mcmc.cnt_accepted_proposals < nsmp):
  mcmc.draw_realization()
  if (mcmc.proposal_accepted):
    for i, dt_i in enumerate(dt):
      pcgmo_dtbJ[i].do_assembly()
      pcgmo_dtbJ[i].prepare()
      pcgmo_dtbJ[i].solve()
    
    pcgmo_medbJ.do_assembly()
    pcgmo_medbJ.prepare()
    pcgmo_medbJ.solve()

    for i, dt_i in enumerate(dt):
      pcgmo_dtbJ_it[i] += [pcg_dtbJ[i].it]
    pcgmo_medbJ_it += [pcg_medbJ.it]

fig, ax = pl.subplots(1, 2, figsize=(8.5,3.7))
ax[0].plot(pcgmo_medbJ_it, label="med-bJ#%d" %(nb))
for i, dt_i in enumerate(dt):
  ax[0].plot(pcgmo_dtbJ_it[i], label="%d-bJ#%d" %(dt_i,nb), lw=.4)
av_pcgmo_medbJ_it = np.mean(pcgmo_medbJ_it)
av_pcgmo_dtbJ_it = np.array([np.mean(pcgmo_it)/av_pcgmo_medbJ_it for pcgmo_it in pcgmo_dtbJ_it])
ax[1].plot(dt, av_pcgmo_dtbJ_it, "k")
ax[0].set_xlabel("Realization index, t"); ax[1].set_xlabel("Renewal period of preconditioner, dt")
ax[0].set_ylabel("Number of solver iterations, n_it")
ax[1].set_ylabel("Average relative number of solver iterations")
ax[0].legend(frameon=False, ncol=2)
fig.suptitle("MCMC / PCGMO / Realization dep. vs median bJ preconditioner")
#pl.show()
pl.savefig("example03_recycler.png", bbox_inches='tight')
