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

pcg_dt1bJ  = solver(n=mcmc.n, solver_type="pcg")
pcg_dt1bJ.set_precond(Mat=mcmc.A, precond_id=3, nb=nb)

pcg_dt2bJ  = solver(n=mcmc.n, solver_type="pcg")
pcg_dt2bJ.set_precond(Mat=mcmc.A, precond_id=3, nb=nb)

pcg_medbJ  = solver(n=mcmc.n, solver_type="pcg")
pcg_medbJ.set_precond(Mat=mcmc.get_median_A(), precond_id=3, nb=nb)

dt1, dt2 = 200, 500
pcgmo_dt1bJ = recycler(sampler=mcmc, solver=pcg_dt1bJ, recycler_type="pcgmo", dt=dt1)
pcgmo_dt2bJ = recycler(sampler=mcmc, solver=pcg_dt2bJ, recycler_type="pcgmo", dt=dt2)
pcgmo_medbJ = recycler(sampler=mcmc, solver=pcg_medbJ, recycler_type="pcgmo")

pcgmo_dt1bJ_it, pcgmo_dt2bJ_it, pcgmo_medbJ_it = [], [], []
while (mcmc.cnt_accepted_proposals < nsmp):
  mcmc.draw_realization()
  if (mcmc.proposal_accepted):
    pcgmo_dt1bJ.do_assembly()
    pcgmo_dt1bJ.prepare()
    pcgmo_dt1bJ.solve()

    pcgmo_dt2bJ.do_assembly()
    pcgmo_dt2bJ.prepare()
    pcgmo_dt2bJ.solve()
    
    pcgmo_medbJ.do_assembly()
    pcgmo_medbJ.prepare()
    pcgmo_medbJ.solve()

    pcgmo_dt1bJ_it += [pcg_dt1bJ.it]
    pcgmo_dt2bJ_it += [pcg_dt2bJ.it]
    pcgmo_medbJ_it += [pcg_medbJ.it]

ax = pl.subplot()
ax.plot(pcgmo_medbJ_it, label="med-bJ#%d" %(nb))
ax.plot(pcgmo_dt1bJ_it, label="%d-bJ#%d" %(dt1,nb), lw=.4)
ax.plot(pcgmo_dt2bJ_it, label="%d-bJ#%d" %(dt2,nb), lw=.4)
ax.set_xlabel("Realization index, t")
ax.set_ylabel("Number of solver iterations, n_it")
pl.legend(frameon=False)
ax.set_title("MCMC / PCGMO / Realization dep. vs median bJ preconditioner")
#pl.show()
pl.savefig("example03_recycler.png", bbox_inches='tight')
