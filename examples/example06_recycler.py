import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

figures_path = '../figures/'

nEl = 1000
nsmp = 5000
sig2, L = .357, 0.05
model = "Exp"

kl = 20

smp, dpcg, dpcgmo = {}, {}, {}

for _smp in ("mc", "mcmc"):
  __smp = sampler(nEl=nEl, smp_type=_smp, model=model, sig2=sig2, L=L)
  __smp.compute_KL()
  __smp.draw_realization()
  __smp.do_assembly()
  smp[_smp] = __smp

pcg = solver(n=smp["mc"].n, solver_type="pcg")
#pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)

for __smp in ("mc", "mcmc"):
  for dp_seq in ("dp", "pd"):
      __dpcg = solver(n=smp["mc"].n, solver_type="dpcg")
      #__dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
      __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)
      dpcg[(__smp, dp_seq)] = __dpcg
      dpcgmo[(__smp, dp_seq)] = recycler(smp[__smp], __dpcg, "dpcgmo", kl=kl, dp_seq=dp_seq)

pcgmo_it = {"mc":[], "mcmc":[]}
dpcgmo_it = {}

for i_smp in range(nsmp):
  smp["mc"].draw_realization()
  pcg.presolve(smp["mc"].A, smp["mc"].b)
  pcg.solve(x0=np.zeros(smp["mc"].n))
  pcgmo_it["mc"] += [pcg.it]
  for dp_seq in ("dp", "pd"):
    _dpcgmo = ("mc", dp_seq)

    dpcgmo[_dpcgmo].do_assembly()
    dpcgmo[_dpcgmo].prepare()

    dpcgmo[_dpcgmo].solve()
    if dpcgmo_it.has_key(_dpcgmo):
      dpcgmo_it[_dpcgmo] += [dpcgmo[_dpcgmo].solver.it]
    else:
      dpcgmo_it[_dpcgmo] = [dpcgmo[_dpcgmo].solver.it]

  print("%d/%d" %(i_smp+1, nsmp))

while (smp["mcmc"].cnt_accepted_proposals < nsmp):
  smp["mcmc"].draw_realization()
  if (smp["mcmc"].proposal_accepted):
    pcg.presolve(smp["mcmc"].A, smp["mcmc"].b)
    pcg.solve(x0=np.zeros(smp["mcmc"].n))
    pcgmo_it["mcmc"] += [pcg.it]
    for dp_seq in ("dp", "pd"):
      _dpcgmo = ("mcmc", dp_seq)

      dpcgmo[_dpcgmo].do_assembly()
      dpcgmo[_dpcgmo].prepare()

      dpcgmo[_dpcgmo].solve()
      if dpcgmo_it.has_key(_dpcgmo):
        dpcgmo_it[_dpcgmo] += [dpcgmo[_dpcgmo].solver.it]
      else:
        dpcgmo_it[_dpcgmo] = [dpcgmo[_dpcgmo].solver.it]

    print("%d/%d" %(smp["mcmc"].cnt_accepted_proposals+1, nsmp))

fig, ax = pl.subplots(1, 2, figsize=(8.5,3.7), sharey=True)
ax[0].set_title("MC")
ax[0].plot(pcgmo_it["mc"], lw=0.5, label="pcgmo")
ax[0].plot(dpcgmo_it[("mc", "dp")], "-+", markersize=mrksize, label="dpcgmo-dp")
ax[0].plot(dpcgmo_it[("mc", "pd")], lw=0.5, label="dpcgmo-pd")
ax[1].set_title("MCMC")
ax[1].plot(pcgmo_it["mcmc"], lw=0.5, label="pcgmo")
ax[1].plot(dpcgmo_it[("mcmc", "dp")], "-+", markersize=mrksize, label="dpcgmo-dp")
ax[1].plot(dpcgmo_it[("mcmc", "pd")], lw=0.5, label="dpcgmo-pd")
ax[0].set_ylabel("Number of solver iterations, n_it")
ax[0].set_xlabel("Realization index, t"); ax[1].set_xlabel("Realization index, t")
#fig.suptitle("DPCGMO with median-bJ10")
fig.suptitle("DPCGMO with median")
ax[0].legend(frameon=False, ncol=2); ax[1].legend(frameon=False, ncol=2)
#pl.show()
#pl.savefig(figures_path+"example06_recycler_a.png", bbox_inches='tight')
pl.savefig(figures_path+"example06_recycler_b.png", bbox_inches='tight')


mrkr, mrksize = ".", 0.2
fig, ax = pl.subplots(1, 4, figsize=(16,3.7))
ax[0].set_title("MC")
ax[0].plot(np.array(dpcgmo_it[("mc", "dp")])/np.array(pcgmo_it["mc"], dtype=float), "r"+mrkr, markersize=mrksize, label="dpcgmo-dp")
ax[0].plot(np.array(dpcgmo_it[("mc", "pd")])/np.array(pcgmo_it["mc"], dtype=float), "g"+mrkr, markersize=mrksize, label="dpcgmo-pd")
ax[1].set_title("MC")
ax[1].plot(pcgmo_it["mc"], "k"+mrkr, markersize=mrksize, label="pcgmo")
ax[1].plot(dpcgmo_it[("mc", "dp")], "r"+mrkr, markersize=mrksize)
ax[1].plot(dpcgmo_it[("mc", "pd")], "g"+mrkr, markersize=mrksize)
ax[2].set_title("MCMC")
ax[2].plot(pcgmo_it["mcmc"], "k"+mrkr, markersize=mrksize, label="pcgmo")
ax[2].plot(dpcgmo_it[("mcmc", "dp")], "r"+mrkr, markersize=mrksize)
ax[2].plot(dpcgmo_it[("mcmc", "pd")], "g"+mrkr, markersize=mrksize)
ax[3].set_title("MCMC")
ax[3].plot(np.array(dpcgmo_it[("mcmc", "dp")])/np.array(pcgmo_it["mcmc"], dtype=float), "r"+mrkr, markersize=mrksize, label="dpcgmo-dp")
ax[3].plot(np.array(dpcgmo_it[("mcmc", "pd")])/np.array(pcgmo_it["mcmc"], dtype=float), "g"+mrkr, markersize=mrksize, label="dpcgmo-pd")
ax[0].set_ylim(0, 1); ax[3].set_ylim(0, 1)
ax[2].set_ylim(ax[1].get_ylim())
ax[0].set_ylabel("Relative number of solver iterations wrt PCG")
ax[1].set_ylabel("Number of solver iterations, n_it")
ax[2].set_ylabel("Number of solver iterations, n_it")
ax[3].set_ylabel("Relative number of solver iterations wrt PCG")
#fig.suptitle("DPCGMO with median-bJ10")
fig.suptitle("DPCGMO with median")
ax[0].legend(frameon=False, ncol=2); ax[1].legend(frameon=False)
ax[2].legend(frameon=False); ax[3].legend(frameon=False, ncol=2)
#pl.show()
#pl.savefig(figures_path+"example06_recycler_a.png", bbox_inches='tight')
pl.savefig(figures_path+"example06_recycler_b.png", bbox_inches='tight')