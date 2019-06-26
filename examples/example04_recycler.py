import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

figures_path = '../figures/'

nEl = 1000
nsmp = 1000
sig2, L = .357, 0.05
model = "Exp"

kl = 20

smp, dcg, dcgmo = {}, {}, {}

for _smp in ("mc", "mcmc"):
  __smp = sampler(nEl=nEl, smp_type=_smp, model=model, sig2=sig2, L=L)
  __smp.compute_KL()
  __smp.draw_realization()
  __smp.do_assembly()
  smp[_smp] = __smp

cg = solver(n=smp["mc"].n, solver_type="cg")

for __smp in ("mc", "mcmc"):
  for which_op in ("previous", "current"):
    for kl_strategy in range(2):
      __dcg = solver(n=smp["mc"].n, solver_type="dcg")
      dcg[(__smp, which_op, kl_strategy)] = __dcg
      dcgmo[(__smp, which_op, kl_strategy)] = recycler(smp[__smp], __dcg, "dcgmo", kl=kl, kl_strategy=kl_strategy, which_op=which_op)

cgmo_it = {"mc":[], "mcmc":[]}
dcgmo_it, dcgmo_kdim, dcgmo_ell = {}, {}, {}

for i_smp in range(nsmp):
  smp["mc"].draw_realization()
  cg.presolve(smp["mc"].A, smp["mc"].b)
  cg.solve(x0=np.zeros(smp["mc"].n))
  cgmo_it["mc"] += [cg.it]
  for which_op in ("previous", "current"):
    for kl_strategy in range(2):
      _dcgmo = ("mc", which_op, kl_strategy)

      dcgmo[_dcgmo].do_assembly()
      dcgmo[_dcgmo].prepare()

      if dcgmo_kdim.has_key(_dcgmo):
        dcgmo_kdim[_dcgmo] += [dcgmo[_dcgmo].solver.kdim]
        dcgmo_ell[_dcgmo] += [dcgmo[_dcgmo].solver.ell]
      else:
        dcgmo_kdim[_dcgmo] = [dcgmo[_dcgmo].solver.kdim]
        dcgmo_ell[_dcgmo] = [dcgmo[_dcgmo].solver.ell]

      dcgmo[_dcgmo].solve()
      if dcgmo_it.has_key(_dcgmo):
        dcgmo_it[_dcgmo] += [dcgmo[_dcgmo].solver.it]
      else:
        dcgmo_it[_dcgmo] = [dcgmo[_dcgmo].solver.it]

  print("%d/%d" %(i_smp+1, nsmp))

while (smp["mcmc"].cnt_accepted_proposals < nsmp):
  smp["mcmc"].draw_realization()
  if (smp["mcmc"].proposal_accepted):
    cg.presolve(smp["mcmc"].A, smp["mcmc"].b)
    cg.solve(x0=np.zeros(smp["mcmc"].n))
    cgmo_it["mcmc"] += [cg.it]
    for which_op in ("previous", "current"):
      for kl_strategy in range(2):
        _dcgmo = ("mcmc", which_op, kl_strategy)

        dcgmo[_dcgmo].do_assembly()
        dcgmo[_dcgmo].prepare()

        if dcgmo_kdim.has_key(_dcgmo):
          dcgmo_kdim[_dcgmo] += [dcgmo[_dcgmo].solver.kdim]
          dcgmo_ell[_dcgmo] += [dcgmo[_dcgmo].solver.ell]
        else:
          dcgmo_kdim[_dcgmo] = [dcgmo[_dcgmo].solver.kdim]
          dcgmo_ell[_dcgmo] = [dcgmo[_dcgmo].solver.ell]

        dcgmo[_dcgmo].solve()
        if dcgmo_it.has_key(_dcgmo):
          dcgmo_it[_dcgmo] += [dcgmo[_dcgmo].solver.it]
        else:
          dcgmo_it[_dcgmo] = [dcgmo[_dcgmo].solver.it]

    print("%d/%d" %(smp["mcmc"].cnt_accepted_proposals+1, nsmp))





lw = 0.3
fig, ax = pl.subplots(2, 3, figsize=(16,7.5), sharex="col")
fig.suptitle("DCGMO -- MC sampler")
# First row:
ax[0,0].set_title("kl_strategy #1")
ax[0,0].set_ylabel("kdim, ell")
ax[0,0].plot(dcgmo_kdim[("mc", "previous", 0)], label="kdim")
ax[0,0].plot(dcgmo_ell[("mc", "previous", 0)], label="ell")
ax[0,1].set_title("Number of solver iterations, n_it")
ax[0,1].plot(cgmo_it["mc"], label="cgmo", lw=lw)
ax[0,1].plot(dcgmo_it[("mc", "previous", 0)], "r", lw=lw)
ax[0,1].plot(dcgmo_it[("mc", "current", 0)], "g", lw=lw)
ax[0,2].set_title("Relative number of solver iterations wrt CG")
ax[0,2].plot(np.array(dcgmo_it[("mc", "previous", 0)])/np.array(cgmo_it["mc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
ax[0,2].plot(np.array(dcgmo_it[("mc", "current", 0)])/np.array(cgmo_it["mc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
ax[0,2].set_ylim(0.6,1)
ax[0,0].legend(frameon=False, ncol=2); ax[0,1].legend(frameon=False); ax[0,2].legend(frameon=False, ncol=2)
# Second row:
ax[1,0].set_title("kl_strategy #2")
ax[1,0].set_ylabel("kdim, ell")
ax[1,0].plot(dcgmo_kdim[("mc", "previous", 1)], label="kdim")
ax[1,0].plot(dcgmo_ell[("mc", "previous", 1)], label="ell")
ax[1,1].set_title("Number of solver iterations, n_it")
ax[1,1].plot(cgmo_it["mc"], label="cgmo", lw=lw)
ax[1,1].plot(dcgmo_it[("mc", "previous", 1)], "r", lw=lw)
ax[1,1].plot(dcgmo_it[("mc", "current", 1)], "g", lw=lw)
ax[1,2].set_title("Relative number of solver iterations wrt CG")
ax[1,2].plot(np.array(dcgmo_it[("mc", "previous", 1)])/np.array(cgmo_it["mc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
ax[1,2].plot(np.array(dcgmo_it[("mc", "current", 1)])/np.array(cgmo_it["mc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
ax[1,0].legend(frameon=False, ncol=2); ax[1,1].legend(frameon=False); ax[1,2].legend(frameon=False, ncol=2)
for j in range(3):
  ax[1,j].set_ylim(ax[0,j].get_ylim())
  ax[1,j].set_xlabel("Realization index, t")
#pl.show()
pl.savefig(figures_path+"example04_recycler_a.png", bbox_inches='tight')




fig, ax = pl.subplots(2, 4, figsize=(16,7.5), sharey="row")
# First row:
ax[0,0].set_title("MC / kl_strategy #1")
ax[0,0].set_ylabel("Number of solver iterations, n_it")
ax[0,0].plot(cgmo_it["mc"], label="cgmo")
ax[0,0].plot(dcgmo_it[("mc", "previous", 0)], "-+", label="dcgmo-prev")
ax[0,0].plot(dcgmo_it[("mc", "current", 0)], label="dcgmo-curr")
ax[0,1].set_title("MC / kl_strategy #2")
ax[0,1].plot(cgmo_it["mc"], label="cgmo")
ax[0,1].plot(dcgmo_it[("mc", "previous", 1)], "-+", label="dcgmo-prev")
ax[0,1].plot(dcgmo_it[("mc", "current", 1)], label="dcgmo-curr")
ax[0,2].set_title("MCMC / kl_strategy #1")
ax[0,2].plot(cgmo_it["mcmc"], label="cgmo")
ax[0,2].plot(dcgmo_it[("mcmc", "previous", 0)], "-+", label="dcgmo-prev")
ax[0,2].plot(dcgmo_it[("mcmc", "current", 0)], label="dcgmo-curr")
ax[0,3].set_title("MCMC / kl_strategy #2")
ax[0,3].plot(cgmo_it["mcmc"], label="cgmo")
ax[0,3].plot(dcgmo_it[("mcmc", "previous", 1)], "-+", label="dcgmo-prev")
ax[0,3].plot(dcgmo_it[("mcmc", "current", 1)], label="dcgmo-curr")
# Second row
ax[1,0].set_ylabel("kdim, ell")
ax[1,0].plot(dcgmo_kdim[("mc", "previous", 0)], label="kdim")
ax[1,0].plot(dcgmo_ell[("mc", "previous", 0)], label="ell")
ax[1,1].plot(dcgmo_kdim[("mc", "previous", 1)], label="kdim")
ax[1,1].plot(dcgmo_ell[("mc", "previous", 1)], label="ell")
ax[1,2].plot(dcgmo_kdim[("mcmc", "previous", 0)], label="kdim")
ax[1,2].plot(dcgmo_ell[("mcmc", "previous", 0)], label="ell")
ax[1,3].plot(dcgmo_kdim[("mcmc", "previous", 1)], label="kdim")
ax[1,3].plot(dcgmo_ell[("mcmc", "previous", 1)], label="ell")
ax[1,0].yaxis.set_ticks(np.linspace(0, kl, kl/2+1, dtype=int)); ax[1,0].set_ylim(-1, kl+1)
for i in range(2):
  for j in range(4):
    ax[i,j].legend(frameon=False, ncol=2)
for j in range(4):
  ax[0,j].grid()
  ax[1,j].set_xlabel("Realization index, t")
fig.suptitle("DCGMO")
#pl.show()
pl.savefig(figures_path+"example04_recycler.png", bbox_inches='tight')