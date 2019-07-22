import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import numpy as np
import scipy
from example05_recycler_dpcgmo_plot import *
from example05_recycler_dpcgmo_cases import get_params
import scipy.sparse as sparse

nEl = 1000
# nsmp       in {200, 10000, 15000}
# (sig2, L)  in {0.05, 0.50}x{0.02, 0.10, 0.50}
# model      in {"Exp", "SExp"}
# kl         in {20, 50}
# nsmp       in {200}
# precond_id in {1, 2, 3}

case = "a4" # {"a", "b", "c"}
precond_id, sig2, L, model, kl, kl_strategy, ell_min, nsmp, t_end_def, t_end_kl, t_switch_to_mc, ini_W, eigres_thresh = get_params(case)
case = "example05_"+case

smp, dpcg, dpcgmo = {}, {}, {}

for _smp in ("mc", "mcmc"):
  __smp = sampler(nEl=nEl, smp_type=_smp, model=model, sig2=sig2, L=L)
  __smp.compute_KL()
  __smp.draw_realization()
  __smp.do_assembly()
  smp[_smp] = __smp

pcg = solver(n=smp["mc"].n, solver_type="pcg")
if (precond_id == 3):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
elif (precond_id == 1):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)
elif (precond_id == 2):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=2)

pcg.get_chol_M()
L = pcg.L_M
inv_L = pcg.invL_M

for __smp in ("mc", "mcmc"):
  for dp_seq in ("dp", "pd"):
    for which_op in ("previous", "current"):
      __dpcg = solver(n=smp["mc"].n, solver_type="dpcg")
      if (precond_id == 3):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
      elif (precond_id == 1):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)
      elif (precond_id ==2):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=2)
      __dpcg.get_chol_M()
      dpcg[(__smp, dp_seq, which_op)] = __dpcg
      dpcgmo[(__smp, dp_seq, which_op)] = recycler(smp[__smp], __dpcg, "dpcgmo", kl=kl, kl_strategy=kl_strategy, 
                                                   ell_min=ell_min, dp_seq=dp_seq, t_end_def=t_end_def, t_end_kl=t_end_kl, 
                                                   ini_W=ini_W, which_op=which_op, eigres_thresh=eigres_thresh)

pcgmo_it = {"mc":[], "mcmc":[]}
dpcgmo_it = {}

dpcgmo_it, dpcgmo_kdim, dpcgmo_ell = {}, {}, {}
dpcgmo_ritz_quotient, dpcgmo_eigres = {}, {}
smp_SpdA, dpcgmo_SpdHtdA = {"mc":[], "mcmc":[]}, {}
smp_SpA, dpcgmo_SpHtA = {"mc":[], "mcmc":[]}, {}
dpcgmo_SpdHtdA2 = {}

# Possibly not needed:
dpcgmo_sin_theta = {}
dpcgmo_approx_eigvals = {}

for i_smp in range(nsmp):
  smp["mc"].draw_realization()  

  dA = inv_L.dot(smp["mc"].A.dot(inv_L.T))
  if sparse.issparse(dA):
    smp_SpdA["mc"] += [np.linalg.eigvalsh(dA.A)]
  else:
    smp_SpdA["mc"] += [np.linalg.eigvalsh(dA)]
  smp_SpA["mc"] += [np.linalg.eigvalsh(smp["mc"].A.A)]

  pcg.presolve(smp["mc"].A, smp["mc"].b)
  pcg.solve(x0=np.zeros(smp["mc"].n))
  pcgmo_it["mc"] += [pcg.it]  

  _, U = sparse.linalg.eigsh(smp["mc"].A.tocsc(), k=kl, sigma=0, mode='normal')
  if sparse.issparse(dA):
    _, dU = sparse.linalg.eigsh(dA.tocsc(), k=kl, sigma=0, mode='normal')
  else:
    _, dU = scipy.linalg.eigh(dA, eigvals=(0,kl-1))

  for dp_seq in ("dp", "pd"):
    for which_op in ("previous", "current"):
      _dpcgmo = ("mc", dp_seq, which_op)

      if not (dpcgmo_SpdHtdA.has_key(_dpcgmo)):
        dpcgmo_SpdHtdA[_dpcgmo] = []
        dpcgmo_SpHtA[_dpcgmo] = []
        if (dp_seq == "pd"):
          dpcgmo_SpdHtdA2[_dpcgmo] = []
        dpcgmo_kdim[_dpcgmo], dpcgmo_ell[_dpcgmo], dpcgmo_approx_eigvals[_dpcgmo] = [], [], []
        dpcgmo_ritz_quotient[_dpcgmo], dpcgmo_eigres[_dpcgmo] = [], []
        dpcgmo_it[_dpcgmo] = []
        dpcgmo_sin_theta[_dpcgmo] = []  

      dpcgmo[_dpcgmo].do_assembly()
      dpcgmo[_dpcgmo].prepare()

      if (dpcgmo[_dpcgmo].solver.kdim > 0):
        if (dp_seq == "pd"):
          QW,_ = np.linalg.qr(dpcgmo[_dpcgmo].solver.W, mode='reduced')
          C = QW.T.dot(U[:,:dpcgmo[_dpcgmo].solver.kdim])
        elif (dp_seq == "dp"):
          QW,_ = np.linalg.qr(L.T.dot(dpcgmo[_dpcgmo].solver.W), mode='reduced')
          C = QW.T.dot(dU[:,:dpcgmo[_dpcgmo].solver.kdim])
        cos_theta = np.linalg.svd(C, compute_uv=False)
        sin_theta = np.sqrt(1.-cos_theta**2)
        dpcgmo_sin_theta[_dpcgmo] += [sin_theta]
      else:
        dpcgmo_sin_theta[_dpcgmo] += [None]

      print dpcgmo[_dpcgmo].solver.kdim
      dpcgmo_kdim[_dpcgmo] += [dpcgmo[_dpcgmo].solver.kdim]
      dpcgmo_ell[_dpcgmo] += [dpcgmo[_dpcgmo].solver.ell]  

      if (dpcgmo_kdim[_dpcgmo][-1] > 0):
        HtA = dpcgmo[_dpcgmo].solver.get_deflated_op()
        if sparse.issparse(inv_L):
          dHtdA = inv_L.A.dot(HtA.dot(inv_L.A.T))
        else:
          dHtdA = inv_L.dot(HtA.dot(inv_L.T))
        dpcgmo_SpdHtdA[_dpcgmo] += [np.linalg.eigvalsh(dHtdA.A)]
        dpcgmo_SpHtA[_dpcgmo] += [np.linalg.eigvalsh(HtA.A)]

        if (dp_seq == "pd"):
          AU = smp["mc"].A.dot(U[:,:dpcgmo[_dpcgmo].solver.kdim])
          UtAU = U[:,:dpcgmo[_dpcgmo].solver.kdim].T.dot(AU)
          HtA2 = smp["mc"].A-AU.dot(scipy.linalg.inv(UtAU).dot(AU.T))
          if sparse.issparse(inv_L):
            dHtdA2 = inv_L.A.dot(HtA2.dot(inv_L.A.T))
          else:
            dHtdA2 = inv_L.dot(HtA2.dot(inv_L.T))
          dpcgmo_SpdHtdA2[_dpcgmo] += [np.linalg.eigvalsh(dHtdA2)]  

        dpcgmo_approx_eigvals[_dpcgmo] += [np.copy(dpcgmo[_dpcgmo].eigvals)]
        dpcgmo_ritz_quotient[_dpcgmo] += [np.copy(dpcgmo[_dpcgmo].ritz_quotient)]
        dpcgmo_eigres[_dpcgmo] += [np.copy(dpcgmo[_dpcgmo].eigres)]
      else:
        dpcgmo_SpdHtdA[_dpcgmo] += [np.array(smp["mc"].n*[None])]
        dpcgmo_SpHtA[_dpcgmo] += [np.array(smp["mc"].n*[None])]
        if (dp_seq == "pd"):
          dpcgmo_SpdHtdA2[_dpcgmo] += [np.array(smp["mc"].n*[None])]
        dpcgmo_approx_eigvals[_dpcgmo] += [None]
        dpcgmo_ritz_quotient[_dpcgmo] += [None]
        dpcgmo_eigres[_dpcgmo] += [None]  

      dpcgmo[_dpcgmo].solve()
      dpcgmo_it[_dpcgmo] += [dpcgmo[_dpcgmo].solver.it]  

    print("%d/%d" %(i_smp+1, nsmp))



while (smp["mcmc"].cnt_accepted_proposals <= nsmp):
  smp["mcmc"].draw_realization()

  if (smp["mcmc"].proposal_accepted):
    dA = (inv_L.dot(smp["mc"].A.dot(inv_L.T)))
    if sparse.issparse(dA):
      smp_SpdA["mcmc"] += [np.linalg.eigvalsh(dA.A)]
    else:
      smp_SpdA["mcmc"] += [np.linalg.eigvalsh(dA)]
    smp_SpA["mcmc"] += [np.linalg.eigvalsh(smp["mc"].A.A)]
    
    pcg.presolve(smp["mcmc"].A, smp["mcmc"].b)
    pcg.solve(x0=np.zeros(smp["mcmc"].n))
    pcgmo_it["mcmc"] += [pcg.it]

    _, U = sparse.linalg.eigsh(smp["mc"].A.tocsc(), k=kl, sigma=0, mode='normal')
    if sparse.issparse(dA):
      _, dU = sparse.linalg.eigsh(dA.tocsc(), k=kl, sigma=0, mode='normal')
    else:
      _, dU = scipy.linalg.eigh(dA, eigvals=(0,kl-1))

    for dp_seq in ("dp", "pd"):
      for which_op in ("previous", "current"):
        _dpcgmo = ("mcmc", dp_seq, which_op)

        if not (dpcgmo_SpdHtdA.has_key(_dpcgmo)):
          dpcgmo_SpdHtdA[_dpcgmo] = []
          dpcgmo_SpHtA[_dpcgmo] = []
          if (dp_seq == "pd"):
            dpcgmo_SpdHtdA2[_dpcgmo] = []
          dpcgmo_kdim[_dpcgmo], dpcgmo_ell[_dpcgmo], dpcgmo_approx_eigvals[_dpcgmo] = [], [], []
          dpcgmo_ritz_quotient[_dpcgmo], dpcgmo_eigres[_dpcgmo] = [], []
          dpcgmo_it[_dpcgmo] = []
          dpcgmo_sin_theta[_dpcgmo] = []  

        dpcgmo[_dpcgmo].do_assembly()
        dpcgmo[_dpcgmo].prepare()

        if (dpcgmo[_dpcgmo].solver.kdim > 0):
          if (dp_seq == "pd"):
            QW,_ = np.linalg.qr(dpcgmo[_dpcgmo].solver.W, mode='reduced')
            C = QW.T.dot(U[:,:dpcgmo[_dpcgmo].solver.kdim])
          elif (dp_seq == "dp"):
            QW,_ = np.linalg.qr(L.T.dot(dpcgmo[_dpcgmo].solver.W), mode='reduced')
            C = QW.T.dot(dU[:,:dpcgmo[_dpcgmo].solver.kdim])
          cos_theta = np.linalg.svd(C, compute_uv=False)
          sin_theta = np.sqrt(1.-cos_theta**2)
          dpcgmo_sin_theta[_dpcgmo] += [sin_theta]
        else:
          dpcgmo_sin_theta[_dpcgmo] += [None]

        print dpcgmo[_dpcgmo].solver.kdim
        dpcgmo_kdim[_dpcgmo] += [dpcgmo[_dpcgmo].solver.kdim]
        dpcgmo_ell[_dpcgmo] += [dpcgmo[_dpcgmo].solver.ell]  

        if (dpcgmo_kdim[_dpcgmo][-1] > 0):
          HtA = dpcgmo[_dpcgmo].solver.get_deflated_op()
          if sparse.issparse(inv_L):
            dHtdA = inv_L.A.dot(HtA.dot(inv_L.A.T))
          else:
            dHtdA = inv_L.dot(HtA.dot(inv_L.T))
          dpcgmo_SpdHtdA[_dpcgmo] += [np.linalg.eigvalsh(dHtdA.A)]
          dpcgmo_SpHtA[_dpcgmo] += [np.linalg.eigvalsh(HtA.A)]

          if (dp_seq == "pd"):
            AU = smp["mc"].A.dot(U[:,:dpcgmo[_dpcgmo].solver.kdim])
            UtAU = U[:,:dpcgmo[_dpcgmo].solver.kdim].T.dot(AU)
            HtA2 = smp["mc"].A-AU.dot(scipy.linalg.inv(UtAU).dot(AU.T))
            if sparse.issparse(inv_L):
              dHtdA2 = inv_L.A.dot(HtA2.dot(inv_L.A.T))
            else:
              dHtdA2 = inv_L.dot(HtA2.dot(inv_L.T))
            dpcgmo_SpdHtdA2[_dpcgmo] += [np.linalg.eigvalsh(dHtdA2)]  

          dpcgmo_approx_eigvals[_dpcgmo] += [np.copy(dpcgmo[_dpcgmo].eigvals)]
          dpcgmo_ritz_quotient[_dpcgmo] += [np.copy(dpcgmo[_dpcgmo].ritz_quotient)]
          dpcgmo_eigres[_dpcgmo] += [np.copy(dpcgmo[_dpcgmo].eigres)]
        else:
          dpcgmo_SpdHtdA[_dpcgmo] += [np.array(smp["mc"].n*[None])]
          dpcgmo_SpHtA[_dpcgmo] += [np.array(smp["mc"].n*[None])]          
          if (dp_seq == "pd"):
            dpcgmo_SpdHtdA2[_dpcgmo] += [np.array(smp["mc"].n*[None])]
          dpcgmo_approx_eigvals[_dpcgmo] += [None]
          dpcgmo_ritz_quotient[_dpcgmo] += [None]
          dpcgmo_eigres[_dpcgmo] += [None]  

        dpcgmo[_dpcgmo].solve()
        dpcgmo_it[_dpcgmo] += [dpcgmo[_dpcgmo].solver.it]

    print("%d/%d" %(smp["mcmc"].cnt_accepted_proposals+1, nsmp))

save_data(smp, pcgmo_it, dpcgmo_it, smp_SpdA, smp_SpA, dpcgmo_SpdHtdA, dpcgmo_SpdHtdA2, dpcgmo_SpHtA, dpcgmo_ell, dpcgmo_kdim, dpcgmo_approx_eigvals, dpcgmo_ritz_quotient, dpcgmo_eigres, dpcgmo_sin_theta, case)

plot(_smp="mc", precond_id=precond_id, case_id=case)
plot(_smp="mcmc", precond_id=precond_id, case_id=case)