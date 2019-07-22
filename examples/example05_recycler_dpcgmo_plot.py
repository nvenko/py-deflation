import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
from matplotlib.ticker import MaxNLocator
import numpy as np

""" HOW TO:
    >>> from example05_dpcgmo_reclycler_plot import *
    >>> plot(case_id=case_id)
"""

figures_path = '../figures/'

lw = 0.5

"""
def save_data(dpcgmo_it, pcgmo_it, case_id="example05"):
  np.save(".recycler_dpcgmo_it_"+case_id, dpcgmo_it)
  np.save(".recycler_pcgmo_it_"+case_id, pcgmo_it)

def load_data(case_id):
  dpcgmo_it = np.load(".recycler_dpcgmo_it_"+case_id+".npy").item()
  pcgmo_it = np.load(".recycler_pcgmo_it_"+case_id+".npy").item()
  return dpcgmo_it, pcgmo_it
"""

def get_envelopes_dA(SpdA, dpcgmo_kdim):
  SpdA_0 = [Sp[0] for Sp in SpdA]
  SpdA_k = [SpdA[t][dpcgmo_kdim[t]] for t in range(len(SpdA))]
  SpdA_n = [Sp[-1] for Sp in SpdA]
  return SpdA_0, SpdA_k, SpdA_n

def get_envelopes_dHtdA(SpdHtdA, dpcgmo_kdim):
  SpdHtdA_k = [SpdHtdA[t][dpcgmo_kdim[t]] for t in range(len(SpdHtdA))]
  SpdHtdA_n = [Sp[-1] for Sp in SpdHtdA]
  return SpdHtdA_k, SpdHtdA_n

def get_eigres(eigres, kdim):
  kmax = max(kdim)
  errors = np.array(kmax*[len(kdim)*[None]])
  for t in range(len(kdim)):
    if (type(eigres[t]) != type(None)):
      errors[:kdim[t],t] = eigres[t][:kdim[t]]
  return errors

def save_data(smp, pcgmo_it, dpcgmo_it, smp_SpdA, smp_SpA, dpcgmo_SpdHtdA, dpcgmo_SpdHtdA2, dpcgmo_SpHtA, dpcgmo_ell, dpcgmo_kdim, dpcgmo_eigvals, dpcgmo_ritz_quotient, dpcgmo_eigres, dpcgmo_sin_theta, case_id="example05"):
  np.save(".recycler_smp_"+case_id, smp)
  np.save(".recycler_pcgmo_it_"+case_id, pcgmo_it)
  np.save(".recycler_dpcgmo_it_"+case_id, dpcgmo_it)
  np.save(".recycler_smp_SpdA_"+case_id, smp_SpdA)
  np.save(".recycler_smp_SpA_"+case_id, smp_SpA)
  np.save(".recycler_dpcgmo_SpdHtdA_"+case_id, dpcgmo_SpdHtdA)
  np.save(".recycler_dpcgmo_SpdHtdA2_"+case_id, dpcgmo_SpdHtdA2)
  np.save(".recycler_dpcgmo_SpHtA_"+case_id, dpcgmo_SpHtA)
  np.save(".recycler_dpcgmo_ell_"+case_id, dpcgmo_ell)
  np.save(".recycler_dpcgmo_kdim_"+case_id, dpcgmo_kdim)
  np.save(".recycler_dpcgmo_eigvals_"+case_id, dpcgmo_eigvals) # May not be needed
  np.save(".recycler_dpcgmo_ritz_quotient_"+case_id, dpcgmo_ritz_quotient)
  np.save(".recycler_dpcgmo_eigres_"+case_id, dpcgmo_eigres)
  np.save(".recycler_dpcgmo_sin_theta_"+case_id, dpcgmo_sin_theta)

def load_data(case_id):
    smp = np.load(".recycler_smp_"+case_id+".npy").item()
    pcgmo_it = np.load(".recycler_pcgmo_it_"+case_id+".npy").item()
    dpcgmo_it = np.load(".recycler_dpcgmo_it_"+case_id+".npy").item()
    smp_SpdA = np.load(".recycler_smp_SpdA_"+case_id+".npy").item()
    smp_SpA = np.load(".recycler_smp_SpA_"+case_id+".npy").item()
    dpcgmo_SpdHtdA = np.load(".recycler_dpcgmo_SpdHtdA_"+case_id+".npy").item()
    dpcgmo_SpdHtdA2 = np.load(".recycler_dpcgmo_SpdHtdA2_"+case_id+".npy").item()
    dpcgmo_SpHtA = np.load(".recycler_dpcgmo_SpHtA_"+case_id+".npy").item()
    dpcgmo_ell = np.load(".recycler_dpcgmo_ell_"+case_id+".npy").item()
    dpcgmo_kdim = np.load(".recycler_dpcgmo_kdim_"+case_id+".npy").item()
    dpcgmo_eigvals = np.load(".recycler_dpcgmo_eigvals_"+case_id+".npy").item() # May not be needed
    dpcgmo_ritz_quotient = np.load(".recycler_dpcgmo_ritz_quotient_"+case_id+".npy").item()
    dpcgmo_eigres = np.load(".recycler_dpcgmo_eigres_"+case_id+".npy").item()
    dpcgmo_sin_theta = np.load(".recycler_dpcgmo_sin_theta_"+case_id+".npy").item()
    return smp, pcgmo_it, dpcgmo_it, smp_SpdA, smp_SpA, dpcgmo_SpdHtdA, dpcgmo_SpdHtdA2, dpcgmo_SpHtA, dpcgmo_ell, dpcgmo_kdim, dpcgmo_eigvals, dpcgmo_ritz_quotient, dpcgmo_eigres, dpcgmo_sin_theta





#def plot(dpcgmo_it=None, pcgmo_it=None, precond_id=1; case_id="example05"):
def plot(_smp="mc", precond_id=1, smp=None, smp_SpA=None, dcgmo_SpHtA=None, dcgmo_kdim=None, dcgmo_eigvals=None, case_id=None):
 
  precond_label = ("",
                   "median",
                   "median-AMG",
                   "median-bJ10")

  #if (type(dpcgmo_it) == type(None)):
  #  dpcgmo_it, pcgmo_it  = load_data(case_id)

  smp, pcgmo_it, dpcgmo_it, smp_SpdA, smp_SpA, dpcgmo_SpdHtdA, dpcgmo_SpdHtdA2, dpcgmo_SpHtA, dpcgmo_ell, dpcgmo_kdim, dpcgmo_eigvals, dpcgmo_ritz_quotient, dpcgmo_eigres, dpcgmo_sin_theta = load_data(case_id)

  n = smp["mc"].n
  kl = 20

  solver_it_bnd = (0, 200)

  solver_rel_it_bnd = (0, 1)
  
  eigres_bnd = (1e-3, 5e3)
  
  Sp_bnd = (1e-4, 1e5)

  #snapshot = (3000, 9000)
  snapshot = (500,)
  #snapshot = (500, 3000)
  #snapshot = (5000, 8000)
  #snapshot = (5000, 10000)

  # Row1, _smp+"1_" : 
  fig, ax = pl.subplots(1, 5, figsize=(17.7,3.2))
  fig.suptitle("pdcg / %s / %s\n" %(_smp, precond_label[precond_id]), y=1.05, weight="bold")
  #
  ax[0].set_title(_smp+" sampler")
  ax[0].set_ylabel("kdim, ell")
  ax[0].plot(np.array(dpcgmo_kdim[(_smp, "pd", "previous")], dtype=int), label="kdim")
  ax[0].plot(np.array(dpcgmo_ell[(_smp, "pd", "previous")], dtype=int), label="ell")
  ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
  ax[0].legend(frameon=False, ncol=2)
  #
  ax[1].set_title("# solver iterations")
  ax[1].plot(pcgmo_it[_smp], "k", lw=lw, label="pcgmo")
  ax[1].plot(dpcgmo_it[(_smp, "pd", "previous")], "r", lw=lw)
  ax[1].plot(dpcgmo_it[(_smp, "pd", "current")], "g", lw=lw)
  ax[1].set_ylim(solver_it_bnd)
  #
  ax[2].set_title("Relative # solver iterations")
  ax[2].plot(np.array(dpcgmo_it[(_smp, "pd", "previous")])/np.array(pcgmo_it[_smp], dtype=float), "r", lw=lw, label="dpcgmo-prev")
  ax[2].plot(np.array(dpcgmo_it[(_smp, "pd", "current")])/np.array(pcgmo_it[_smp], dtype=float), "g", lw=lw, label="dpcgmo-curr")
  ax[2].set_ylim(solver_rel_it_bnd)
  #
  _dpcgmo = (_smp, "pd", "previous")
  ax[3].set_title('which_op="previous"')
  errors = get_eigres(dpcgmo_eigres[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  for k in range(errors.shape[0]):
    ax[3].semilogy(errors[k,:], lw=lw)
  ax[3].set_ylim(eigres_bnd)
  ax[3].plot((0), (0), lw=0, label=r"$||Aw_j-\rho(w_j,A)w_j||/||w_j||$")
  ax[3].legend(framealpha=1)
  #
  _dpcgmo = (_smp, "pd", "current")
  ax[4].set_title('which_op="current"')
  errors = get_eigres(dpcgmo_eigres[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  for k in range(errors.shape[0]):
    ax[4].semilogy(errors[k,:], lw=lw)
  ax[4].set_ylim(ax[3].get_ylim())
  ax[4].plot((0), (0), lw=0, label=r"$||Aw_j-\rho(w_j,A)w_j||/||w_j||$")
  ax[4].legend(framealpha=1)
  #
  for j in range(5):
    ax[j].grid()
    ax[j].set_xlabel("Realization index, t")
  pl.savefig(figures_path+"recycler_dpcgmo_%s_%s1.png" %(case_id, _smp), bbox_inches='tight')

  # Row2, _smp+"2_" : 
  fig, ax = pl.subplots(1, 5, figsize=(17.7,3.2))
  #
  _dpcgmo = (_smp, "pd", "previous")
  SpdA_0, SpdA_k, SpdA_n = get_envelopes_dA(smp_SpdA[_smp], dpcgmo_kdim[_dpcgmo])
  SpdHtdA_k, SpdHtdA_n = get_envelopes_dHtdA(dpcgmo_SpdHtdA[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  SpdHtdA2_k, SpdHtdA2_n = get_envelopes_dHtdA(dpcgmo_SpdHtdA2[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  #
  cond_A = np.array(SpdA_n)/np.array(SpdA_0)
  cond_HtA_theo = np.array(SpdA_n)/np.array(SpdA_k) 
  cond_HtA2_theo = np.concatenate(([cond_A[0]],np.array(SpdHtdA2_n[1:])/np.array(SpdHtdA2_k[1:])))
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpdHtdA_n[1:])/np.array(SpdHtdA_k[1:])))
  #
  ax[0].set_title(r"$\mathrm{cond}(\dot{H}^T\dot{A})/\mathrm{cond}(\dot{A})$")
  ax[0].semilogy(cond_HtA_theo/cond_A, "b", lw=lw, label="exact defl.")
  ax[0].legend(framealpha=1)
  ax[0].semilogy(cond_HtA/cond_A, "r", lw=lw)
  #
  ax[2].set_title("envelopes of "+r"$\mathrm{Sp}(\dot{H}^T\dot{A})$"+" and "+r"$\mathrm{Sp}(\dot{A})$") 
  ax[2].semilogy(SpdA_0, "k", lw=lw)
  ax[2].semilogy(SpdA_k, "b", lw=lw)
  ax[2].semilogy(SpdA_n, "k", lw=lw)
  ax[2].semilogy(SpdHtdA_k, "r", lw=lw, label="dpcgmo-previous")
  ax[2].semilogy(SpdHtdA_n, "r", lw=lw)
  #
  _dpcgmo = (_smp, "pd", "current")
  SpdHtdA_k, SpdHtdA_n = get_envelopes_dHtdA(dpcgmo_SpdHtdA[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpdHtdA_n[1:])/np.array(SpdHtdA_k[1:])))
  #
  ax[0].semilogy(cond_HtA/cond_A, "g", lw=lw)
  #
  ax[2].semilogy(SpdHtdA_k, "g", lw=lw, label="dpcgmo-current")
  ax[2].semilogy(SpdHtdA_n, "g", lw=lw)
  ax[2].legend(frameon=False)
  #

  _dpcgmo = (_smp, "pd", "previous")
  ax[1].set_title(r'$\{\sin(\theta_j)\}_{j=1}^k$'+', which_op="previous"')
  sin_theta = get_eigres(dpcgmo_sin_theta[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  for k in range(sin_theta.shape[0]):
    ax[1].semilogy(sin_theta[k,:], lw=lw)

  #
  # Snapshot #1
  ax[3].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA), \{\rho(w_j,A)\}_{j=1}^k$")
  for i in range(snapshot[0]-1, snapshot[0]+2):
    kdim = dpcgmo_kdim[(_smp, "pd", "previous")][i]
    ax[3].semilogy((n-kdim)*[i-.4], smp_SpA[_smp][i][kdim:], "k_", markersize=6)
    ax[3].semilogy(kdim*[i-.4], smp_SpA[_smp][i][:kdim], "k+", markersize=6)
    ax[3].semilogy((n-kdim)*[i-.2], dpcgmo_SpHtA[(_smp, "pd", "previous")][i][kdim:], "r_", markersize=6)
    ax[3].semilogy(kdim*[i], dpcgmo_ritz_quotient[(_smp, "pd", "previous")][i], "r+", markersize=6)
    kdim = dpcgmo_kdim[(_smp, "pd", "current")][i]
    ax[3].semilogy((n-kdim)*[i+.2], dpcgmo_SpHtA[(_smp, "pd", "current")][i][kdim:], "g_", markersize=6)
    ax[3].semilogy(kdim*[i+.4], dpcgmo_ritz_quotient[(_smp, "pd", "current")][i], "g+", markersize=6)
    if (i < snapshot[0]+1):
      ax[3].semilogy((i+.5,i+.5), (1e12, 1e-11),"k", lw=lw)
  #
  # Snapshot #2
  ax[4].set_title(r"$\mathrm{Sp}(\dot{A}), \mathrm{Sp}(\dot{H}^T\dot{A})$")
  for i in range(snapshot[0]-1, snapshot[0]+2):
    kdim = dpcgmo_kdim[(_smp, "pd", "previous")][i]
    ax[4].semilogy((n-kdim)*[i-.4], smp_SpdA[_smp][i][kdim:], "k_", markersize=6) # smp_SpdHtdA2
    ax[4].semilogy(kdim*[i-.4], smp_SpdA[_smp][i][:kdim], "k_", markersize=6)     # smp_SpdHtdA2
    ax[4].semilogy((n-kdim)*[i-.2], dpcgmo_SpdHtdA[(_smp, "pd", "previous")][i][kdim:], "r_", markersize=6)
    kdim = dpcgmo_kdim[(_smp, "pd", "current")][i]
    ax[4].semilogy((n-kdim)*[i+.2], dpcgmo_SpdHtdA[(_smp, "pd", "current")][i][kdim:], "g_", markersize=6)
    if (i < snapshot[0]+1):
      ax[4].semilogy((i+.5,i+.5), (1e12, 1e-11),"k", lw=lw)
  #
  ax[3].set_ylim(Sp_bnd)
  #ax[3].set_ylim(ax[2].get_ylim())
  ax[4].set_ylim(ax[2].get_ylim()); 
  for j in range(5):
    if (j < 3):
      ax[j].grid()
    ax[j].set_xlabel("Realization index, t")
  pl.savefig(figures_path+"recycler_dpcgmo_%s_%s2.png" %(case_id, _smp), bbox_inches='tight')







  # Row1, _smp+"1_" : 
  fig, ax = pl.subplots(1, 5, figsize=(17.7,3.2))
  fig.suptitle("dpcg / %s / %s\n" %(_smp, precond_label[precond_id]), y=1.05, weight="bold")
  #
  ax[0].set_title(_smp+" sampler")
  ax[0].set_ylabel("kdim, ell")
  ax[0].plot(np.array(dpcgmo_kdim[(_smp, "dp", "previous")], dtype=int), label="kdim")
  ax[0].plot(np.array(dpcgmo_ell[(_smp, "dp", "previous")], dtype=int), label="ell")
  ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
  ax[0].legend(frameon=False, ncol=2)
  #
  ax[1].set_title("# solver iterations")
  ax[1].plot(pcgmo_it[_smp], "k", lw=lw, label="pcgmo")
  ax[1].plot(dpcgmo_it[(_smp, "dp", "previous")], "r", lw=lw)
  ax[1].plot(dpcgmo_it[(_smp, "dp", "current")], "g", lw=lw)
  ax[1].set_ylim(solver_it_bnd)
  #
  ax[2].set_title("Relative # solver iterations")
  ax[2].plot(np.array(dpcgmo_it[(_smp, "dp", "previous")])/np.array(pcgmo_it[_smp], dtype=float), "r", lw=lw, label="dpcgmo-prev")
  ax[2].plot(np.array(dpcgmo_it[(_smp, "dp", "current")])/np.array(pcgmo_it[_smp], dtype=float), "g", lw=lw, label="dpcgmo-curr")
  ax[2].set_ylim(solver_rel_it_bnd)
  #
  _dpcgmo = (_smp, "dp", "previous")
  ax[3].set_title('which_op="previous"')
  errors = get_eigres(dpcgmo_eigres[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  for k in range(errors.shape[0]):
    ax[3].semilogy(errors[k,:], lw=lw)
  ax[3].set_ylim(eigres_bnd)
  ax[3].plot((0), (0), lw=0, label=r"$||\dot{A}\dot{w}_j-\rho(\dot{w}_j,\dot{A})\dot{w}_j||/||\dot{w}_j||$")
  ax[3].legend(framealpha=1)
  #
  _dpcgmo = (_smp, "dp", "current")
  ax[4].set_title('which_op="current"')
  errors = get_eigres(dpcgmo_eigres[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  for k in range(errors.shape[0]):
    ax[4].semilogy(errors[k,:], lw=lw)
  ax[4].set_ylim(ax[3].get_ylim())
  ax[4].plot((0), (0), lw=0, label=r"$||\dot{A}\dot{w}_j-\rho(\dot{w}_j,\dot{A})\dot{w}_j||/||\dot{w}_j||$")
  ax[4].legend(framealpha=1)
  #
  for j in range(5):
    ax[j].grid()
    ax[j].set_xlabel("Realization index, t")
  pl.savefig(figures_path+"recycler_dpcgmo_%s_%s3.png" %(case_id, _smp), bbox_inches='tight')

  # Row2, _smp+"2_" : 
  fig, ax = pl.subplots(1, 5, figsize=(17.7,3.2))
  #
  _dpcgmo = (_smp, "dp", "previous")
  SpdA_0, SpdA_k, SpdA_n = get_envelopes_dA(smp_SpdA[_smp], dpcgmo_kdim[_dpcgmo])
  SpdHtdA_k, SpdHtdA_n = get_envelopes_dHtdA(dpcgmo_SpdHtdA[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  #
  cond_A = np.array(SpdA_n)/np.array(SpdA_0)
  cond_HtA_theo = np.array(SpdA_n)/np.array(SpdA_k) 
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpdHtdA_n[1:])/np.array(SpdHtdA_k[1:])))
  #
  ax[0].set_title(r"$\mathrm{cond}(\dot{H}^T\dot{A})/\mathrm{cond}(\dot{A})$")
  ax[0].semilogy(cond_HtA_theo/cond_A, "b", lw=lw, label="exact defl.")
  ax[0].legend(framealpha=1)
  ax[0].semilogy(cond_HtA/cond_A, "r", lw=lw)
  #
  ax[2].set_title("envelopes of "+r"$\mathrm{Sp}(\dot{H}^T\dot{A})$"+" and "+r"$\mathrm{Sp}(\dot{A})$") 
  ax[2].semilogy(SpdA_0, "k", lw=lw)
  ax[2].semilogy(SpdA_k, "b", lw=lw)
  ax[2].semilogy(SpdA_n, "k", lw=lw)
  ax[2].semilogy(SpdHtdA_k, "r", lw=lw, label="dpcgmo-previous")
  ax[2].semilogy(SpdHtdA_n, "r", lw=lw)
  #
  _dpcgmo = (_smp, "dp", "current")
  SpdHtdA_k, SpdHtdA_n = get_envelopes_dHtdA(dpcgmo_SpdHtdA[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpdHtdA_n[1:])/np.array(SpdHtdA_k[1:])))
  #
  ax[0].semilogy(cond_HtA/cond_A, "g", lw=lw)
  #
  ax[2].semilogy(SpdHtdA_k, "g", lw=lw, label="dpcgmo-current")
  ax[2].semilogy(SpdHtdA_n, "g", lw=lw)
  ax[2].legend(frameon=False)
  #

  _dpcgmo = (_smp, "dp", "previous")
  ax[1].set_title(r'$\{\sin(\dot{\theta}_j)\}_{j=1}^k$'+', which_op="previous"')
  sin_theta = get_eigres(dpcgmo_sin_theta[_dpcgmo], dpcgmo_kdim[_dpcgmo])
  for k in range(sin_theta.shape[0]):
    ax[1].semilogy(sin_theta[k,:], lw=lw)

  #
  # Snapshot #1
  ax[3].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA)$")
  for i in range(snapshot[0]-1, snapshot[0]+2):
    kdim = dpcgmo_kdim[(_smp, "dp", "previous")][i]
    ax[3].semilogy((n-kdim)*[i-.4], smp_SpA[_smp][i][kdim:], "k_", markersize=6)
    ax[3].semilogy(kdim*[i-.4], smp_SpA[_smp][i][:kdim], "k_", markersize=6)
    ax[3].semilogy((n-kdim)*[i-.2], dpcgmo_SpHtA[(_smp, "dp", "previous")][i][kdim:], "r_", markersize=6)
    kdim = dpcgmo_kdim[(_smp, "dp", "current")][i]
    ax[3].semilogy((n-kdim)*[i+.2], dpcgmo_SpHtA[(_smp, "dp", "current")][i][kdim:], "g_", markersize=6)
    if (i < snapshot[0]+1):
      ax[3].semilogy((i+.5,i+.5), (1e12, 1e-11),"k", lw=lw)
  #
  # Snapshot #2
  ax[4].set_title(r"$\mathrm{Sp}(\dot{A}), \mathrm{Sp}(\dot{H}^T\dot{A}), \{\rho(\dot{w}_j,\dot{A})\}_{j=1}^k$")
  for i in range(snapshot[0]-1, snapshot[0]+2):
    kdim = dpcgmo_kdim[(_smp, "dp", "previous")][i]
    ax[4].semilogy((n-kdim)*[i-.4], smp_SpdA[_smp][i][kdim:], "k_", markersize=6)
    ax[4].semilogy(kdim*[i-.4], smp_SpdA[_smp][i][:kdim], "k+", markersize=6)
    ax[4].semilogy((n-kdim)*[i-.2], dpcgmo_SpdHtdA[(_smp, "dp", "previous")][i][kdim:], "r_", markersize=6)
    ax[4].semilogy(kdim*[i], dpcgmo_ritz_quotient[(_smp, "dp", "previous")][i], "r+", markersize=6)
    kdim = dpcgmo_kdim[(_smp, "dp", "current")][i]
    ax[4].semilogy((n-kdim)*[i+.2], dpcgmo_SpdHtdA[(_smp, "dp", "current")][i][kdim:], "g_", markersize=6)
    ax[4].semilogy(kdim*[i+.4], dpcgmo_ritz_quotient[(_smp, "dp", "current")][i], "g+", markersize=6)
    if (i < snapshot[0]+1):
      ax[4].semilogy((i+.5,i+.5), (1e12, 1e-11),"k", lw=lw)
  #
  ax[3].set_ylim(Sp_bnd); 
  ax[4].set_ylim(ax[2].get_ylim()); 
  for j in range(5):
    if (j < 3):
      ax[j].grid()
    ax[j].set_xlabel("Realization index, t")
  pl.savefig(figures_path+"recycler_dpcgmo_%s_%s4.png" %(case_id, _smp), bbox_inches='tight')








  """
  fig, ax = pl.subplots(2, 4, figsize=(17.5,7.5), sharex="col")
  ax[0,0].set_title("MC-previous")
  ax[0,0].plot(np.array(dpcgmo_it[("mc", "dp", "previous")])/np.array(pcgmo_it["mc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
  ax[0,0].plot(np.array(dpcgmo_it[("mc", "pd", "previous")])/np.array(pcgmo_it["mc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
  ax[0,1].set_title("MC-previous")
  ax[0,1].plot(pcgmo_it["mc"], "k", lw=lw, label="pcgmo")
  ax[0,1].plot(dpcgmo_it[("mc", "dp", "previous")], "r", lw=lw)
  ax[0,1].plot(dpcgmo_it[("mc", "pd", "previous")], "g", lw=lw)
  ax[0,2].set_title("MCMC-previous")
  ax[0,2].plot(pcgmo_it["mcmc"], "k", lw=lw, label="pcgmo")
  ax[0,2].plot(dpcgmo_it[("mcmc", "dp", "previous")], "r", lw=lw)
  ax[0,2].plot(dpcgmo_it[("mcmc", "pd", "previous")], "g", lw=lw)
  ax[0,3].set_title("MCMC-previous")
  ax[0,3].plot(np.array(dpcgmo_it[("mcmc", "dp", "previous")])/np.array(pcgmo_it["mcmc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
  ax[0,3].plot(np.array(dpcgmo_it[("mcmc", "pd", "previous")])/np.array(pcgmo_it["mcmc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
  ax[0,0].set_ylim(0, 1); ax[0,3].set_ylim(0, 1)  
  ax[0,2].set_ylim(ax[0,1].get_ylim())

  ax[1,0].set_title("MC-current")
  ax[1,0].plot(np.array(dpcgmo_it[("mc", "dp", "current")])/np.array(pcgmo_it["mc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
  ax[1,0].plot(np.array(dpcgmo_it[("mc", "pd", "current")])/np.array(pcgmo_it["mc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
  ax[1,1].set_title("MC-current")
  ax[1,1].plot(pcgmo_it["mc"], "k", lw=lw, label="pcgmo")
  ax[1,1].plot(dpcgmo_it[("mc", "dp", "current")], "r", lw=lw)
  ax[1,1].plot(dpcgmo_it[("mc", "pd", "current")], "g", lw=lw)
  ax[1,2].set_title("MCMC-current")
  ax[1,2].plot(pcgmo_it["mcmc"], "k", lw=lw, label="pcgmo")
  ax[1,2].plot(dpcgmo_it[("mcmc", "dp", "current")], "r", lw=lw)
  ax[1,2].plot(dpcgmo_it[("mcmc", "pd", "current")], "g", lw=lw)
  ax[1,3].set_title("MCMC-current")
  ax[1,3].plot(np.array(dpcgmo_it[("mcmc", "dp", "current")])/np.array(pcgmo_it["mcmc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
  ax[1,3].plot(np.array(dpcgmo_it[("mcmc", "pd", "current")])/np.array(pcgmo_it["mcmc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
  ax[1,0].set_ylim(0, 1); ax[1,3].set_ylim(0, 1)
  ax[1,1].set_ylim(ax[0,1].get_ylim()); ax[1,2].set_ylim(ax[0,1].get_ylim())
  ax[0,0].set_ylabel("Relative number of solver iterations wrt PCG"); ax[1,0].set_ylabel("Relative number of solver iterations wrt PCG")
  ax[0,1].set_ylabel("Number of solver iterations, n_it"); ax[1,1].set_ylabel("Number of solver iterations, n_it")
  ax[0,2].set_ylabel("Number of solver iterations, n_it"); ax[1,2].set_ylabel("Number of solver iterations, n_it")
  ax[0,3].set_ylabel("Relative number of solver iterations wrt PCG"); ax[1,3].set_ylabel("Relative number of solver iterations wrt PCG")
  for j in range(4):
    ax[1,j].set_xlabel("Realization index, t")

  if (precond_id == 1):
    fig.suptitle("DPCGMO with median")
  elif (precond_id == 2):
    fig.suptitle("DPCGMO with median-AMG")
  elif (precond_id == 3):
    fig.suptitle("DPCGMO with median-bJ10")


  for i in range(2):
    ax[i,0].legend(frameon=False, ncol=2); ax[i,1].legend(frameon=False)
    ax[i,2].legend(frameon=False); ax[i,3].legend(frameon=False, ncol=2)
  """


  pl.savefig(figures_path+"recycler_dpcgmo_%s.png" %case_id, bbox_inches='tight')
  """
  if (case == "a"):
    pl.savefig(figures_path+"example06_recycler_a.png", bbox_inches='tight')
  elif (case == "b"):
    pl.savefig(figures_path+"example06_recycler_b.png", bbox_inches='tight')
  elif (case == "c"):
    pl.savefig(figures_path+"example06_recycler_c.png", bbox_inches='tight')
  """