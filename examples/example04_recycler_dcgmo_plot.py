import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
from matplotlib.ticker import MaxNLocator
import numpy as np

""" HOW TO:
    >>> from example04_dcgmo_reclycler_plot import *
    >>> plot(case = case)
"""

figures_path = '../figures/'

lw = 0.5

def get_envelopes_A(SpA, dcgmo_kdim):
  SpA_0 = [Sp[0] for Sp in SpA]
  SpA_k = [SpA[t][dcgmo_kdim[t]] for t in range(len(SpA))]
  SpA_n = [Sp[-1] for Sp in SpA]
  return SpA_0, SpA_k, SpA_n

def get_envelopes_HtA(SpHtA, dcgmo_kdim):
  SpHtA_k = [SpHtA[t][dcgmo_kdim[t]] for t in range(len(SpHtA))]
  SpHtA_n = [Sp[-1] for Sp in SpHtA]
  return SpHtA_k, SpHtA_n

def get_eigres(eigres, kdim):
  kmax = max(kdim)
  errors = np.array(kmax*[len(kdim)*[None]])
  for t in range(len(kdim)):
    if (type(eigres[t]) != type(None)):
      errors[:kdim[t],t] = eigres[t][:kdim[t]]
  return errors

def save_data(smp, cgmo_it, dcgmo_it, smp_SpA, dcgmo_SpHtA, dcgmo_ell, dcgmo_kdim, dcgmo_eigvals, dcgmo_ritz_quotient, dcgmo_eigres, dcgmo_sin_theta, case_id="example04"):
  np.save(".recycler_smp_"+case_id, smp)
  np.save(".recycler_cgmo_it_"+case_id, cgmo_it)
  np.save(".recycler_dcgmo_it_"+case_id, dcgmo_it)
  np.save(".recycler_smp_SpA_"+case_id, smp_SpA)
  np.save(".recycler_dcgmo_SpHtA_"+case_id, dcgmo_SpHtA)
  np.save(".recycler_dcgmo_ell_"+case_id, dcgmo_ell)
  np.save(".recycler_dcgmo_kdim_"+case_id, dcgmo_kdim)
  np.save(".recycler_dcgmo_eigvals_"+case_id, dcgmo_eigvals) # May not be needed
  np.save(".recycler_dcgmo_ritz_quotient_"+case_id, dcgmo_ritz_quotient)
  np.save(".recycler_dcgmo_eigres_"+case_id, dcgmo_eigres)
  np.save(".recycler_dcgmo_sin_theta_"+case_id, dcgmo_sin_theta)

def load_data(case_id):
    smp = np.load(".recycler_smp_"+case_id+".npy").item()
    cgmo_it = np.load(".recycler_cgmo_it_"+case_id+".npy").item()
    dcgmo_it = np.load(".recycler_dcgmo_it_"+case_id+".npy").item()
    smp_SpA = np.load(".recycler_smp_SpA_"+case_id+".npy").item()
    dcgmo_SpHtA = np.load(".recycler_dcgmo_SpHtA_"+case_id+".npy").item()
    dcgmo_ell = np.load(".recycler_dcgmo_ell_"+case_id+".npy").item()
    dcgmo_kdim = np.load(".recycler_dcgmo_kdim_"+case_id+".npy").item()
    dcgmo_eigvals = np.load(".recycler_dcgmo_eigvals_"+case_id+".npy").item() # May not be needed
    dcgmo_ritz_quotient = np.load(".recycler_dcgmo_ritz_quotient_"+case_id+".npy").item()
    dcgmo_eigres = np.load(".recycler_dcgmo_eigres_"+case_id+".npy").item()
    dcgmo_sin_theta = np.load(".recycler_dcgmo_sin_theta_"+case_id+".npy").item()
    return smp, cgmo_it, dcgmo_it, smp_SpA, dcgmo_SpHtA, dcgmo_ell, dcgmo_kdim, dcgmo_eigvals, dcgmo_ritz_quotient, dcgmo_eigres, dcgmo_sin_theta

def plot(_smp="mc", smp=None, smp_SpA=None, dcgmo_SpHtA=None, dcgmo_kdim=None, dcgmo_eigvals=None, case_id=None):
  if (type(smp) == type(None)):
    smp, cgmo_it, dcgmo_it, smp_SpA, dcgmo_SpHtA, dcgmo_ell, dcgmo_kdim, dcgmo_eigvals, dcgmo_ritz_quotient, dcgmo_eigres, dcgmo_sin_theta = load_data(case_id)

  n = smp["mc"].n
  kl = 20
  
  solver_it_bnd = (600, 2700)

  solver_rel_it_bnd = (0, 1)
  
  eigres_bnd = (1e-3,5e3)
  
  #snapshot = (3000, 9000)
  snapshot = (30, 90)
  #snapshot = (500, 3000)
  #snapshot = (5000, 8000)
  #snapshot = (5000, 10000)

  # Row1, _smp+"1_" : 
  fig, ax = pl.subplots(1, 5, figsize=(17.7,3.2))
  #
  ax[0].set_title(_smp+" sampler")
  ax[0].set_ylabel("kdim, ell")
  ax[0].plot(np.array(dcgmo_kdim[(_smp, "previous")], dtype=int), label="kdim")
  ax[0].plot(np.array(dcgmo_ell[(_smp, "previous")], dtype=int), label="ell")
  ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
  ax[0].legend(frameon=False, ncol=2)
  #
  ax[1].set_title("# solver iterations")
  ax[1].plot(cgmo_it[_smp], "k", lw=lw, label="cgmo")
  ax[1].plot(dcgmo_it[(_smp, "previous")], "r", lw=lw)
  ax[1].plot(dcgmo_it[(_smp, "current")], "g", lw=lw)
  ax[1].set_ylim(solver_it_bnd)
  #
  ax[2].set_title("Relative # solver iterations")
  ax[2].plot(np.array(dcgmo_it[(_smp, "previous")])/np.array(cgmo_it[_smp], dtype=float), "r", lw=lw, label="dcgmo-prev")
  ax[2].plot(np.array(dcgmo_it[(_smp, "current")])/np.array(cgmo_it[_smp], dtype=float), "g", lw=lw, label="dcgmo-curr")
  ax[2].set_ylim(solver_rel_it_bnd)
  #
  _dcgmo = (_smp, "previous")
  ax[3].set_title('which_op="previous"')
  errors = get_eigres(dcgmo_eigres[_dcgmo], dcgmo_kdim[_dcgmo])
  for k in range(errors.shape[0]):
    ax[3].semilogy(errors[k,:], lw=lw)
  ax[3].set_ylim(eigres_bnd)
  ax[3].plot((0), (0), lw=0, label=r"$||Aw_j-\rho(w_j,A)w_j||/||w_j||$")
  ax[3].legend(framealpha=1)
  #
  _dcgmo = (_smp, "current")
  ax[4].set_title('which_op="current"')
  errors = get_eigres(dcgmo_eigres[_dcgmo], dcgmo_kdim[_dcgmo])
  for k in range(errors.shape[0]):
    ax[4].semilogy(errors[k,:], lw=lw)
  ax[4].set_ylim(ax[3].get_ylim())
  ax[4].plot((0), (0), lw=0, label=r"$||Aw_j-\rho(w_j,A)w_j||/||w_j||$")
  ax[4].legend(framealpha=1)
  #
  for j in range(5):
    ax[j].grid()
    ax[j].set_xlabel("Realization index, t")
  pl.savefig(figures_path+"recycler_dcgmo_%s_%s1.png" %(case_id, _smp), bbox_inches='tight')

  # Row2, _smp+"2_" : 
  fig, ax = pl.subplots(1, 5, figsize=(17.7,3.2))
  #
  _dcgmo = (_smp, "previous")
  SpA_0, SpA_k, SpA_n = get_envelopes_A(smp_SpA[_smp], dcgmo_kdim[_dcgmo])
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  #
  cond_A = np.array(SpA_n)/np.array(SpA_0)
  cond_HtA_theo = np.array(SpA_n)/np.array(SpA_k) 
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  #
  ax[0].set_title(r"$\mathrm{cond}(H^TA)/\mathrm{cond}(A)$")
  ax[0].semilogy(cond_HtA_theo/cond_A, "b", lw=lw)
  ax[0].semilogy(cond_HtA/cond_A, "r", lw=lw)
  #
  ax[2].set_title("envelopes") 
  ax[2].semilogy(SpA_0, "k", lw=lw)
  ax[2].semilogy(SpA_k, "b", lw=lw)
  ax[2].semilogy(SpA_n, "k", lw=lw)
  ax[2].semilogy(SpHtA_k, "r", lw=lw, label="dcgmo-previous")
  ax[2].semilogy(SpHtA_n, "r", lw=lw)
  #
  _dcgmo = (_smp, "current")
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  #
  ax[0].semilogy(cond_HtA/cond_A, "g", lw=lw)
  #
  ax[2].semilogy(SpHtA_k, "g", lw=lw, label="dcgmo-current")
  ax[2].semilogy(SpHtA_n, "g", lw=lw)
  ax[2].legend(frameon=False)
  #

  _dcgmo = (_smp, "previous")
  ax[1].set_title(r'$\{\sin(\theta_j)\}_{j=1}^k$'+', which_op="previous"')
  sin_theta = get_eigres(dcgmo_sin_theta[_dcgmo], dcgmo_kdim[_dcgmo])
  for k in range(sin_theta.shape[0]):
    ax[1].semilogy(sin_theta[k,:], lw=lw)

  #
  # Snapshot #1
  ax[3].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA), \{\rho(w_j,A)\}_{j=1}^k$")
  for i in range(snapshot[0]-1, snapshot[0]+2):
    kdim = dcgmo_kdim[(_smp, "previous")][i]
    ax[3].semilogy((n-kdim)*[i-.4], smp_SpA[_smp][i][kdim:], "k_", markersize=6)
    ax[3].semilogy(kdim*[i-.4], smp_SpA[_smp][i][:kdim], "k+", markersize=6)
    ax[3].semilogy((n-kdim)*[i-.2], dcgmo_SpHtA[(_smp, "previous")][i][kdim:], "r_", markersize=6)
    ax[3].semilogy(kdim*[i], dcgmo_ritz_quotient[(_smp, "previous")][i], "r+", markersize=6)
    kdim = dcgmo_kdim[(_smp, "current")][i]
    ax[3].semilogy((n-kdim)*[i+.2], dcgmo_SpHtA[(_smp, "current")][i][kdim:], "g_", markersize=6)
    ax[3].semilogy(kdim*[i+.4], dcgmo_ritz_quotient[(_smp, "current")][i], "g+", markersize=6)
    if (i < 501):
      ax[3].semilogy((i+.5,i+.5), (1e-3,5e4),"k", lw=lw)
  #
  # Snapshot #2
  ax[4].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA), \{\rho(w_j,A)\}_{j=1}^k$")
  for i in range(snapshot[1]-1, snapshot[1]+2):
    kdim = dcgmo_kdim[(_smp, "previous")][i]
    ax[4].semilogy((n-kdim)*[i-.4], smp_SpA[_smp][i][kdim:], "k_", markersize=6)
    ax[4].semilogy(kdim*[i-.4], smp_SpA[_smp][i][:kdim], "k+", markersize=6)
    ax[4].semilogy((n-kdim)*[i-.2], dcgmo_SpHtA[(_smp, "previous")][i][kdim:], "r_", markersize=6)
    ax[4].semilogy(kdim*[i], dcgmo_ritz_quotient[(_smp, "previous")][i], "r+", markersize=6)
    kdim = dcgmo_kdim[(_smp, "current")][i]
    ax[4].semilogy((n-kdim)*[i+.2], dcgmo_SpHtA[(_smp, "current")][i][kdim:], "g_", markersize=6)
    ax[4].semilogy(kdim*[i+.4], dcgmo_ritz_quotient[(_smp, "current")][i], "g+", markersize=6)
  #
  ax[3].set_ylim(ax[2].get_ylim()); ax[4].set_ylim(ax[2].get_ylim()); 
  for j in range(5):
    if (j < 3):
      ax[j].grid()
    ax[j].set_xlabel("Realization index, t")
  pl.savefig(figures_path+"recycler_dcgmo_%s_%s2.png" %(case_id, _smp), bbox_inches='tight')