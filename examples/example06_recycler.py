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
case = "c" # {"a", "b", "c"}

smp, dpcg, dpcgmo = {}, {}, {}

for _smp in ("mc", "mcmc"):
  __smp = sampler(nEl=nEl, smp_type=_smp, model=model, sig2=sig2, L=L)
  __smp.compute_KL()
  __smp.draw_realization()
  __smp.do_assembly()
  smp[_smp] = __smp

pcg = solver(n=smp["mc"].n, solver_type="pcg")
if (case == "a"):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
elif (case == "b"):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)
elif (case == "c"):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=2)

for __smp in ("mc", "mcmc"):
  for dp_seq in ("dp", "pd"):
      __dpcg = solver(n=smp["mc"].n, solver_type="dpcg")
      if (case == "a"):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
      elif (case == "b"):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)
      elif (case == "c"):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=2)
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

lw = 0.3
fig, ax = pl.subplots(1, 4, figsize=(17.5,4.))
ax[0].set_title("MC")
ax[0].plot(np.array(dpcgmo_it[("mc", "dp")])/np.array(pcgmo_it["mc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
ax[0].plot(np.array(dpcgmo_it[("mc", "pd")])/np.array(pcgmo_it["mc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
ax[1].set_title("MC")
ax[1].plot(pcgmo_it["mc"], "k", lw=lw, label="pcgmo")
ax[1].plot(dpcgmo_it[("mc", "dp")], "r", lw=lw)
ax[1].plot(dpcgmo_it[("mc", "pd")], "g", lw=lw)
ax[2].set_title("MCMC")
ax[2].plot(pcgmo_it["mcmc"], "k", lw=lw, label="pcgmo")
ax[2].plot(dpcgmo_it[("mcmc", "dp")], "r", lw=lw)
ax[2].plot(dpcgmo_it[("mcmc", "pd")], "g", lw=lw)
ax[3].set_title("MCMC")
ax[3].plot(np.array(dpcgmo_it[("mcmc", "dp")])/np.array(pcgmo_it["mcmc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
ax[3].plot(np.array(dpcgmo_it[("mcmc", "pd")])/np.array(pcgmo_it["mcmc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
ax[0].set_ylim(0, 1); ax[3].set_ylim(0, 1)
ax[2].set_ylim(ax[1].get_ylim())
ax[0].set_ylabel("Relative number of solver iterations wrt PCG")
ax[1].set_ylabel("Number of solver iterations, n_it")
ax[2].set_ylabel("Number of solver iterations, n_it")
ax[3].set_ylabel("Relative number of solver iterations wrt PCG")
for j in range(4):
  ax[j].set_xlabel("Realization index, t")
if (case == "a"):
  fig.suptitle("DPCGMO with median-bJ10")
elif (case == "b"):
  fig.suptitle("DPCGMO with median")
elif (case == "c"):
  fig.suptitle("DPCGMO with median-AMG")
ax[0].legend(frameon=False, ncol=2); ax[1].legend(frameon=False)
ax[2].legend(frameon=False); ax[3].legend(frameon=False, ncol=2)
#pl.show()
if (case == "a"):
  pl.savefig(figures_path+"example06_recycler_a.png", bbox_inches='tight')
elif (case == "b"):
  pl.savefig(figures_path+"example06_recycler_b.png", bbox_inches='tight')
elif (case == "c"):
  pl.savefig(figures_path+"example06_recycler_c.png", bbox_inches='tight')