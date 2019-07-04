import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import numpy as np
from example05_recycler_plot import *

nEl = 1000
nsmp = 5000
sig2, L = .357, 0.05
model = "Exp"

kl = 20
case = "c" # {"a", "b", "c"}


smp, dcg, dcgmo = {}, {}, {}

for _smp in ("mc", "mcmc"):
  __smp = sampler(nEl=nEl, smp_type=_smp, model=model, sig2=sig2, L=L)
  __smp.compute_KL()
  __smp.draw_realization()
  __smp.do_assembly()
  smp[_smp] = __smp

cg = solver(n=smp["mc"].n, solver_type="cg")

if (case == "a"):
  kl_strategy = (0,)
  t_end_kl = (0,)
elif (case == "b"):
  kl_strategy = (1,)
  t_end_kl = (500,)
elif (case == "c"):
  kl_strategy = (1,)
  t_end_kl = (1000,)

n_kl_strategies = len(kl_strategy)
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
dcgmo_it, dcgmo_kdim, dcgmo_ell, dcgmo_approx_eigvals = {}, {}, {}, {}
dcgmo_ritz_coef, dcgmo_eigen_error = {}, {}
smp_SpA, dcgmo_SpHtA = {"mc":[], "mcmc":[]}, {}

for i_smp in range(nsmp):
  smp["mc"].draw_realization()
  smp_SpA["mc"] += [np.linalg.eigvalsh(smp["mc"].A.A)]
  cg.presolve(smp["mc"].A, smp["mc"].b)
  cg.solve(x0=np.zeros(smp["mc"].n))
  cgmo_it["mc"] += [cg.it]
  for which_op in ("previous", "current"):
    for _kl_strategy in range(n_kl_strategies):
      _dcgmo = ("mc", which_op, _kl_strategy)

      if not (dcgmo_SpHtA.has_key(_dcgmo)):
        dcgmo_SpHtA[_dcgmo] = []
        dcgmo_kdim[_dcgmo], dcgmo_ell[_dcgmo], dcgmo_approx_eigvals[_dcgmo] = [], [], []
        dcgmo_ritz_coef[_dcgmo], dcgmo_eigen_error[_dcgmo] = [], []
        dcgmo_it[_dcgmo] = []

      dcgmo[_dcgmo].do_assembly()
      dcgmo[_dcgmo].prepare()

      dcgmo_kdim[_dcgmo] += [dcgmo[_dcgmo].solver.kdim]
      dcgmo_ell[_dcgmo] += [dcgmo[_dcgmo].solver.ell]

      if (dcgmo_kdim[_dcgmo][-1] > 0):
        HtA = dcgmo[_dcgmo].solver.get_deflated_op()
        dcgmo_SpHtA[_dcgmo] += [np.linalg.eigvalsh(HtA.A)]
        dcgmo_approx_eigvals[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigvals)]
        dcgmo_ritz_coef[_dcgmo] += [np.copy(dcgmo[_dcgmo].ritz_coef)]
        dcgmo_eigen_error[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigen_error)]

      else:
        dcgmo_SpHtA[_dcgmo] += [np.array(smp["mc"].n*[None])]
        dcgmo_approx_eigvals[_dcgmo] += [None]
        dcgmo_ritz_coef[_dcgmo] += [None]
        dcgmo_eigen_error[_dcgmo] += [None]

      dcgmo[_dcgmo].solve()
      dcgmo_it[_dcgmo] = [dcgmo[_dcgmo].solver.it]

  print("%d/%d" %(i_smp+1, nsmp))

while (smp["mcmc"].cnt_accepted_proposals <= nsmp):
  smp["mcmc"].draw_realization()
  if (smp["mcmc"].proposal_accepted):
    smp_SpA["mcmc"] += [np.linalg.eigvalsh(smp["mc"].A.A)]
    cg.presolve(smp["mcmc"].A, smp["mcmc"].b)
    cg.solve(x0=np.zeros(smp["mcmc"].n))
    cgmo_it["mcmc"] += [cg.it]
    for which_op in ("previous", "current"):
      for _kl_strategy in range(n_kl_strategies):
        _dcgmo = ("mcmc", which_op, _kl_strategy)

        if not (dcgmo_SpHtA.has_key(_dcgmo)):
          dcgmo_SpHtA[_dcgmo] = []
          dcgmo_kdim[_dcgmo], dcgmo_ell[_dcgmo], dcgmo_approx_eigvals[_dcgmo] = [], [], []
          dcgmo_ritz_coef[_dcgmo], dcgmo_eigen_error[_dcgmo] = [], []
          dcgmo_it[_dcgmo] = []

        dcgmo[_dcgmo].do_assembly()
        dcgmo[_dcgmo].prepare()

        dcgmo_kdim[_dcgmo] += [dcgmo[_dcgmo].solver.kdim]
        dcgmo_ell[_dcgmo] += [dcgmo[_dcgmo].solver.ell]

        dcgmo[_dcgmo].solve()
        dcgmo_it[_dcgmo] += [dcgmo[_dcgmo].solver.it]

        if (dcgmo_kdim[_dcgmo][-1] > 0):
          HtA = dcgmo[_dcgmo].solver.get_deflated_op()
          dcgmo_SpHtA[_dcgmo] += [np.linalg.eigvalsh(HtA.A)]
          dcgmo_approx_eigvals[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigvals)]
          dcgmo_ritz_coef[_dcgmo] += [np.copy(dcgmo[_dcgmo].ritz_coef)]
          dcgmo_eigen_error[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigen_error)]
        else:
          dcgmo_SpHtA[_dcgmo] += [np.array(smp["mcmc"].n*[None])]
          dcgmo_approx_eigvals[_dcgmo] += [None]
          dcgmo_ritz_coef[_dcgmo] += [None]
          dcgmo_eigen_error[_dcgmo] += [None]

    print("%d/%d" %(smp["mcmc"].cnt_accepted_proposals+1, nsmp))

save_data(smp, smp_SpA, dcgmo_SpHtA, dcgmo_kdim, dcgmo_approx_eigvals, dcgmo_ritz_coef, dcgmo_eigen_error, case)
plot(case=case)