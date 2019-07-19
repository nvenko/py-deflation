import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import numpy as np
from example04_recycler_dcgmo_plot import *
from example04_recycler_dcgmo_cases import get_params
import scipy.sparse as sparse
import scipy.sparse.linalg

nEl = 1000
# nsmp      in {200, 10000, 15000}
# (sig2, L) in {0.05, 0.50}x{0.02, 0.10, 0.50}
# model     in {"Exp", "SExp"}
# kl        in {20, 50}

case = "e7"
sig2, L, model, kl, kl_strategy, ell_min, nsmp, t_end_def, t_end_kl, t_switch_to_mc, ini_W, eigres_thresh = get_params(case)
case = "example04_"+case

smp, dcg, dcgmo = {}, {}, {}

for _smp in ("mc", "mcmc"):
  __smp = sampler(nEl=nEl, smp_type=_smp, model=model, sig2=sig2, L=L, t_switch_to_mc=t_switch_to_mc)
  __smp.compute_KL()
  __smp.draw_realization()
  __smp.do_assembly()
  smp[_smp] = __smp

cg = solver(n=smp["mc"].n, solver_type="cg")

for __smp in ("mc", "mcmc"):
  for which_op in ("previous", "current"):
    __dcg = solver(n=smp["mc"].n, solver_type="dcg")
    dcg[(__smp, which_op)] = __dcg
    dcgmo[(__smp, which_op)] = recycler(smp[__smp], __dcg, "dcgmo", kl=kl, kl_strategy=kl_strategy, ell_min=ell_min,
                                        t_end_def=t_end_def, t_end_kl=t_end_kl, ini_W=ini_W, which_op=which_op,
                                        eigres_thresh=eigres_thresh)

cgmo_it = {"mc":[], "mcmc":[]}
dcgmo_it, dcgmo_kdim, dcgmo_ell = {}, {}, {}
dcgmo_ritz_quotient, dcgmo_eigres = {}, {}
smp_SpA, dcgmo_SpHtA = {"mc":[], "mcmc":[]}, {}

# Possibly not needed:
dcgmo_sin_theta = {}
dcgmo_approx_eigvals = {}
#dcgmo_ave_gap_bound = {}


for i_smp in range(nsmp):
  smp["mc"].draw_realization()
  smp_SpA["mc"] += [np.linalg.eigvalsh(smp["mc"].A.A)]
  cg.presolve(smp["mc"].A, smp["mc"].b)
  cg.solve(x0=np.zeros(smp["mc"].n))
  cgmo_it["mc"] += [cg.it]

  _, U = sparse.linalg.eigsh(smp["mc"].A.tocsc(), k=kl, sigma=0, mode='normal')

  for which_op in ("previous", "current"):
    _dcgmo = ("mc", which_op)

    if not (dcgmo_SpHtA.has_key(_dcgmo)):
      dcgmo_SpHtA[_dcgmo] = []
      dcgmo_kdim[_dcgmo], dcgmo_ell[_dcgmo], dcgmo_approx_eigvals[_dcgmo] = [], [], []
      dcgmo_ritz_quotient[_dcgmo], dcgmo_eigres[_dcgmo] = [], []
      ##dcgmo_ave_gap_bound[_dcgmo] = []
      dcgmo_it[_dcgmo] = []
      dcgmo_sin_theta[_dcgmo] = []

    dcgmo[_dcgmo].do_assembly()
    dcgmo[_dcgmo].prepare()

    if (dcgmo[_dcgmo].solver.kdim > 0):
      QW,_ = np.linalg.qr(dcgmo[_dcgmo].solver.W, mode='reduced')
      C = QW.T.dot(U[:,:dcgmo[_dcgmo].solver.kdim])
      cos_theta = np.linalg.svd(C, compute_uv=False)
      sin_theta = np.sqrt(1.-cos_theta**2)
      dcgmo_sin_theta[_dcgmo] += [sin_theta]
    else:
      dcgmo_sin_theta[_dcgmo] += [None]


    print dcgmo[_dcgmo].solver.kdim
    dcgmo_kdim[_dcgmo] += [dcgmo[_dcgmo].solver.kdim]
    dcgmo_ell[_dcgmo] += [dcgmo[_dcgmo].solver.ell]

    if (dcgmo_kdim[_dcgmo][-1] > 0):
      HtA = dcgmo[_dcgmo].solver.get_deflated_op()
      dcgmo_SpHtA[_dcgmo] += [np.linalg.eigvalsh(HtA.A)]
      dcgmo_approx_eigvals[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigvals)]
      dcgmo_ritz_quotient[_dcgmo] += [np.copy(dcgmo[_dcgmo].ritz_quotient)]
      dcgmo_eigres[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigres)]
      #dcgmo_ave_gap_bound[_dcgmo] += [np.copy(dcgmo[_dcgmo].ave_gap_bound)]

    else:
      dcgmo_SpHtA[_dcgmo] += [np.array(smp["mc"].n*[None])]
      dcgmo_approx_eigvals[_dcgmo] += [None]
      dcgmo_ritz_quotient[_dcgmo] += [None]
      dcgmo_eigres[_dcgmo] += [None]
      #dcgmo_ave_gap_bound[_dcgmo] += [None]

    dcgmo[_dcgmo].solve()
    dcgmo_it[_dcgmo] += [dcgmo[_dcgmo].solver.it]

  print("%d/%d" %(i_smp+1, nsmp))


while (smp["mcmc"].cnt_accepted_proposals <= nsmp):
  smp["mcmc"].draw_realization()
  if (smp["mcmc"].proposal_accepted):
    smp_SpA["mcmc"] += [np.linalg.eigvalsh(smp["mc"].A.A)]
    cg.presolve(smp["mcmc"].A, smp["mcmc"].b)
    cg.solve(x0=np.zeros(smp["mcmc"].n))
    cgmo_it["mcmc"] += [cg.it]

    _, U = sparse.linalg.eigsh(smp["mc"].A.tocsc(), k=kl, sigma=0, mode='normal')

    for which_op in ("previous", "current"):
      _dcgmo = ("mcmc", which_op)

      if not (dcgmo_SpHtA.has_key(_dcgmo)):
        dcgmo_SpHtA[_dcgmo] = []
        dcgmo_kdim[_dcgmo], dcgmo_ell[_dcgmo], dcgmo_approx_eigvals[_dcgmo] = [], [], []
        dcgmo_ritz_quotient[_dcgmo], dcgmo_eigres[_dcgmo] = [], []
        #dcgmo_ave_gap_bound[_dcgmo] = []
        dcgmo_it[_dcgmo] = []
        dcgmo_sin_theta[_dcgmo] = []

      dcgmo[_dcgmo].do_assembly()
      dcgmo[_dcgmo].prepare()

      if (dcgmo[_dcgmo].solver.kdim > 0):
        QW,_ = np.linalg.qr(dcgmo[_dcgmo].solver.W, mode='reduced')
        C = QW.T.dot(U[:,:dcgmo[_dcgmo].solver.kdim])
        cos_theta = np.linalg.svd(C, compute_uv=False)
        sin_theta = np.sqrt(1.-cos_theta**2)
        dcgmo_sin_theta[_dcgmo] += [sin_theta]
      else:
        dcgmo_sin_theta[_dcgmo] += [None]

      print dcgmo[_dcgmo].solver.kdim
      dcgmo_kdim[_dcgmo] += [dcgmo[_dcgmo].solver.kdim]
      dcgmo_ell[_dcgmo] += [dcgmo[_dcgmo].solver.ell]

      dcgmo[_dcgmo].solve()
      dcgmo_it[_dcgmo] += [dcgmo[_dcgmo].solver.it]

      if (dcgmo_kdim[_dcgmo][-1] > 0):
        HtA = dcgmo[_dcgmo].solver.get_deflated_op()
        dcgmo_SpHtA[_dcgmo] += [np.linalg.eigvalsh(HtA.A)]
        dcgmo_approx_eigvals[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigvals)]
        dcgmo_ritz_quotient[_dcgmo] += [np.copy(dcgmo[_dcgmo].ritz_quotient)]
        dcgmo_eigres[_dcgmo] += [np.copy(dcgmo[_dcgmo].eigres)]
        #dcgmo_ave_gap_bound[_dcgmo] += [np.copy(dcgmo[_dcgmo].ave_gap_bound)]
      else:
        dcgmo_SpHtA[_dcgmo] += [np.array(smp["mcmc"].n*[None])]
        dcgmo_approx_eigvals[_dcgmo] += [None]
        dcgmo_ritz_quotient[_dcgmo] += [None]
        dcgmo_eigres[_dcgmo] += [None]
        #dcgmo_ave_gap_bound[_dcgmo] += [None]

    print("%d/%d" %(smp["mcmc"].cnt_accepted_proposals+1, nsmp))

save_data(smp, cgmo_it, dcgmo_it, smp_SpA, dcgmo_SpHtA, dcgmo_ell, dcgmo_kdim, dcgmo_approx_eigvals, dcgmo_ritz_quotient, dcgmo_eigres, dcgmo_sin_theta, case)
plot(_smp="mc", case_id=case)
plot(_smp="mcmc", case_id=case)