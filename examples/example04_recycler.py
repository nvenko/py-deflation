import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import numpy as np
from example04_recycler_plot import *

nEl = 1000
nsmp = 5000
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

kl_strategy = (0, 1, 1)
n_kl_strategies = len(kl_strategy)
t_end_kl = (0, 500, 1000)
ell_min = kl/2

for __smp in ("mc", "mcmc"):
  for which_op in ("previous", "current"):
    for _kl_strategy in range(n_kl_strategies):
      __dcg = solver(n=smp["mc"].n, solver_type="dcg")
      dcg[(__smp, which_op, _kl_strategy)] = __dcg
      dcgmo[(__smp, which_op, _kl_strategy)] = recycler(smp[__smp], __dcg, "dcgmo", kl=kl, 
                                               kl_strategy=kl_strategy[_kl_strategy], which_op=which_op, 
                                               t_end_kl=t_end_kl[_kl_strategy], ell_min=ell_min)

cgmo_it = {"mc":[], "mcmc":[]}
dcgmo_it, dcgmo_kdim, dcgmo_ell = {}, {}, {}

for i_smp in range(nsmp):
  smp["mc"].draw_realization()
  cg.presolve(smp["mc"].A, smp["mc"].b)
  cg.solve(x0=np.zeros(smp["mc"].n))
  cgmo_it["mc"] += [cg.it]
  for which_op in ("previous", "current"):
    for _kl_strategy in range(n_kl_strategies):
      _dcgmo = ("mc", which_op, _kl_strategy)

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

while (smp["mcmc"].cnt_accepted_proposals <= nsmp):
  smp["mcmc"].draw_realization()
  if (smp["mcmc"].proposal_accepted):
    cg.presolve(smp["mcmc"].A, smp["mcmc"].b)
    cg.solve(x0=np.zeros(smp["mcmc"].n))
    cgmo_it["mcmc"] += [cg.it]
    for which_op in ("previous", "current"):
      for _kl_strategy in range(n_kl_strategies):
        _dcgmo = ("mcmc", which_op, _kl_strategy)

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

#save_data(dcgmo_kdim, dcgmo_ell, dcgmo_it, cgmo_it)
plot()