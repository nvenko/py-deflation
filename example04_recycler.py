from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

nEl = 1000
nsmp = 50
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
for j in range(4):
  ax[0,j].grid()
# Second row
ax[1,0].set_xlabel("(kdim,ell)")
ax[1,0].plot(dcgmo_kdim[("mc", "previous", 0)], label="kdim")
ax[1,0].plot(dcgmo_ell[("mc", "previous", 0)], label="ell")
ax[1,1].plot(dcgmo_kdim[("mc", "previous", 1)], label="kdim")
ax[1,1].plot(dcgmo_ell[("mc", "previous", 1)], label="ell")
ax[1,2].plot(dcgmo_kdim[("mcmc", "previous", 0)], label="kdim")
ax[1,2].plot(dcgmo_ell[("mcmc", "previous", 0)], label="ell")
ax[1,3].plot(dcgmo_kdim[("mcmc", "previous", 1)], label="kdim")
ax[1,3].plot(dcgmo_ell[("mcmc", "previous", 1)], label="ell")
for i in range(2):
  for j in range(4):
    ax[i,j].legend(frameon=False, ncol=2)
fig.suptitle("DCGMO")
#pl.show()
pl.savefig("example04_recycler.png", bbox_inches='tight')