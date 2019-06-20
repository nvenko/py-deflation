from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl

nEl = 1000
nsmp = 1999
sig2, L = .357, 0.05
model = "Exp"

mcmc = sampler(nEl=nEl, smp_type="mcmc", model=model, sig2=sig2, L=L)
mcmc.compute_KL()

mcmc.draw_realization()
mcmc.do_assembly()

nb = 5

pcg_bJdt  = solver(n=mcmc.n, solver_type="pcg")
pcg_bJdt.set_precond(Mat=mcmc.A, precond_id=3, nb=nb)

pcg_med  = solver(n=mcmc.n, solver_type="pcg")
pcg_med.set_precond(Mat=mcmc.get_median_A(), precond_id=3, nb=nb)

dt = 400

pcgmo_bJdt = recycler(sampler=mcmc, solver=pcg_bJdt, recycler_type="pcgmo", dt=dt)
pcgmo_med = recycler(sampler=mcmc, solver=pcg_med, recycler_type="pcgmo")

pcgmo_bJdt_it, pcgmo_med_it = [], []
while (mcmc.cnt_accepted_proposals < nsmp):
  mcmc.draw_realization()
  if (mcmc.proposal_accepted):
    pcgmo_bJdt.do_assembly()
    pcgmo_bJdt.prepare()
    pcgmo_bJdt.solve()
    
    pcgmo_med.do_assembly()
    pcgmo_med.prepare()
    pcgmo_med.solve()

    pcgmo_bJdt_it += [pcg_bJdt.it]
    pcgmo_med_it += [pcg_med.it]

ax = pl.subplot()
ax.plot(pcgmo_med_it, label="med-bJ#%d" %(nb))
ax.plot(pcgmo_bJdt_it, label="%d-bJ#%d" %(dt,nb))
ax.set_xlabel("Realization index, t")
ax.set_ylabel("Number of solver iterations, n_it")
pl.legend(frameon=False)
ax.set_title("MCMC / PCGMO / Realization dep. vs median bJ preconditioner")
#pl.show()
pl.savefig("example03_recycler.png", bbox_inches='tight')
