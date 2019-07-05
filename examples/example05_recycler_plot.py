import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

""" HOW TO:
    >>> from example05_reclycler_plot import *
    >>> plot(case=case)
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

def get_eigen_errors(eigen_error, kdim):
  kmax = max(kdim)
  errors = np.array(kmax*[len(kdim)*[None]])
  for k in range(kmax):
    _k = k 
    while True:
      try:
        tk = kdim.index(_k+1)
        break
      except ValueError:
        _k += 1
    errors[k,tk:] = [err[k] for err in eigen_error[tk:]]
  return errors

def save_data(smp, smp_SpA, dcgmo_SpHtA, dcgmo_kdim, dcgmo_eigvals, dcgmo_ritz_coef, dcgmo_eigen_error, case):
  np.save(".example05_recycler_smp_"+case, smp)
  np.save(".example05_recycler_smp_SpA_"+case, smp_SpA)
  np.save(".example05_recycler_dcgmo_SpHtA_"+case, dcgmo_SpHtA)
  np.save(".example05_recycler_dcgmo_kdim_"+case, dcgmo_kdim)
  np.save(".example05_recycler_dcgmo_eigvals_"+case, dcgmo_eigvals)
  np.save(".example05_recycler_dcgmo_ritz_coef_"+case, dcgmo_ritz_coef)
  np.save(".example05_recycler_dcgmo_eigen_error_"+case, dcgmo_eigen_error)

def load_data(case):
    smp = np.load(".example05_recycler_smp_"+case+".npy").item()
    smp_SpA = np.load(".example05_recycler_smp_SpA_"+case+".npy").item()
    dcgmo_SpHtA = np.load(".example05_recycler_dcgmo_SpHtA_"+case+".npy").item()
    dcgmo_kdim = np.load(".example05_recycler_dcgmo_kdim_"+case+".npy").item()
    dcgmo_eigvals = np.load(".example05_recycler_dcgmo_eigvals_"+case+".npy").item()
    dcgmo_ritz_coef = np.load(".example05_recycler_dcgmo_ritz_coef_"+case+".npy").item()
    dcgmo_eigen_error = np.load(".example05_recycler_dcgmo_eigen_error_"+case+".npy").item()
    return smp, smp_SpA, dcgmo_SpHtA, dcgmo_kdim, dcgmo_eigvals, dcgmo_ritz_coef, dcgmo_eigen_error

def plot(smp=None, smp_SpA=None, dcgmo_SpHtA=None, dcgmo_kdim=None, dcgmo_eigvals=None, case=None):
  if (type(smp) == type(None)):
    smp, smp_SpA, dcgmo_SpHtA, dcgmo_kdim, dcgmo_eigvals, dcgmo_ritz_coef, dcgmo_eigen_error = load_data(case)

  n = smp["mc"].n
  fig, ax = pl.subplots(2, 4, figsize=(13.5,8), sharex="col")
  strategy_name = {"a":"strategy #1", "b":"strategy #2", "c":"strategy #3", "d":"strategy #4"}
  fig.suptitle("DCGMO -- %s -- Envelopes, conditioning numbers and spectra" %strategy_name[case])
  # Enveloppes
  ax[0,0].set_ylabel("MC sampler")
  ax[0,0].set_title("envelopes")
  _dcgmo = ("mc", "previous", 0)
  SpA_0, SpA_k, SpA_n = get_envelopes_A(smp_SpA["mc"], dcgmo_kdim[_dcgmo])
  cond_A = np.array(SpA_n)/np.array(SpA_0)
  cond_HtA = np.array(SpA_n)/np.array(SpA_k)  
  ax[0,1].set_title(r"$\mathrm{cond}(H^TA)/\mathrm{cond}(A)$")
  ax[0,1].semilogy(cond_HtA/cond_A, "b", lw=lw)
  ax[0,0].semilogy(SpA_0, "k", lw=lw)
  ax[0,0].semilogy(SpA_k, "b", lw=lw)
  ax[0,0].semilogy(SpA_n, "k", lw=lw)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[0,0].semilogy(SpHtA_k, "r", lw=lw, label="dcgmo-previous")
  ax[0,0].semilogy(SpHtA_n, "r", lw=lw)
  ax[0,1].semilogy(cond_HtA/cond_A, "r", lw=lw)
  _dcgmo = ("mc", "current", 0)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[0,0].semilogy(SpHtA_k, "g", lw=lw, label="dcgmo-current")
  ax[0,0].semilogy(SpHtA_n, "g", lw=lw)
  ax[0,1].semilogy(cond_HtA/cond_A, "g", lw=lw)
  ax[0,0].legend(frameon=False)
  # Snapshot #1
  ax[0,2].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA), \{\rho(w_j,A)\}_{j=1}^k$")
  for i in range(499,502):
    kdim = dcgmo_kdim[("mc", "previous", 0)][i]
    ax[0,2].semilogy((n-kdim)*[i-.4], smp_SpA["mc"][i][kdim:], "k_", markersize=6)
    ax[0,2].semilogy(kdim*[i-.4], smp_SpA["mc"][i][:kdim], "k+", markersize=6)
    ax[0,2].semilogy((n-kdim)*[i-.2], dcgmo_SpHtA[("mc", "previous", 0)][i][kdim:], "r_", markersize=6)
    #ax[0,2].semilogy(kdim*[i], dcgmo_eigvals[("mc", "previous", 0)][i], "r+", markersize=6)
    ax[0,2].semilogy(kdim*[i], dcgmo_ritz_coef[("mc", "previous", 0)][i], "r+", markersize=6)
    ax[0,2].semilogy((n-kdim)*[i+.2], dcgmo_SpHtA[("mc", "current", 0)][i][kdim:], "g_", markersize=6)
    #ax[0,2].semilogy(kdim*[i+.4], dcgmo_eigvals[("mc", "current", 0)][i], "g+", markersize=6)
    ax[0,2].semilogy(kdim*[i+.4], dcgmo_ritz_coef[("mc", "current", 0)][i], "g+", markersize=6)
    if (i < 501):
      ax[0,2].semilogy((i+.5,i+.5), (1e-3,5e4),"k", lw=lw)

  # Snapshot #2
  ax[0,3].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA), \{\rho(w_j,A)\}_{j=1}^k$")
  for i in range(2999,3002):
    kdim = dcgmo_kdim[("mc", "previous", 0)][i]
    ax[0,3].semilogy((n-kdim)*[i-.4], smp_SpA["mc"][i][kdim:], "k_", markersize=6)
    ax[0,3].semilogy(kdim*[i-.4], smp_SpA["mc"][i][:kdim], "k+", markersize=6)
    ax[0,3].semilogy((n-kdim)*[i-.2], dcgmo_SpHtA[("mc", "previous", 0)][i][kdim:], "r_", markersize=6)
    #ax[0,3].semilogy(kdim*[i], dcgmo_eigvals[("mc", "previous", 0)][i], "r+", markersize=6)
    ax[0,3].semilogy(kdim*[i], dcgmo_ritz_coef[("mc", "previous", 0)][i], "r+", markersize=6)
    ax[0,3].semilogy((n-kdim)*[i+.2], dcgmo_SpHtA[("mc", "current", 0)][i][kdim:], "g_", markersize=6)
    #ax[0,3].semilogy(kdim*[i+.4], dcgmo_eigvals[("mc", "current", 0)][i], "g+", markersize=6)
    ax[0,3].semilogy(kdim*[i+.4], dcgmo_ritz_coef[("mc", "current", 0)][i], "g+", markersize=6)
    if (i < 3001):
      ax[0,3].semilogy((i+.5,i+.5), (1e-3,5e4),"k", lw=lw)
  ax[0,2].set_ylim(ax[0,0].get_ylim()); ax[0,3].set_ylim(ax[0,0].get_ylim())  

  # Enveloppes
  ax[1,0].set_ylabel("MCMC sampler")
  ax[1,0].set_title("envelopes")
  SpA_0, SpA_k, SpA_n = get_envelopes_A(smp_SpA["mcmc"], dcgmo_kdim[_dcgmo])
  cond_A = np.array(SpA_n)/np.array(SpA_0)
  cond_HtA = np.array(SpA_n)/np.array(SpA_k)  
  ax[1,1].set_title(r"$\mathrm{cond}(H^TA)/\mathrm{cond}(A)$")
  ax[1,1].semilogy(cond_HtA/cond_A, "b", lw=lw)
  ax[1,0].semilogy(SpA_0, "k", lw=lw, label=r"$\lambda_{1}(A),\;\lambda_{n}(A)$")
  ax[1,0].semilogy(SpA_k, "b", lw=lw, label=r"$\lambda_{k+1}(A),\;\lambda_{n}(A)$")
  ax[1,0].semilogy(SpA_n, "k", lw=lw)
  _dcgmo = ("mcmc", "previous", 0)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[1,0].semilogy(SpHtA_k, "r", lw=lw, label=r"$\lambda_{k+1}(H^TA),\;\lambda_{n}(H^TA)$")
  ax[1,0].semilogy(SpHtA_n, "r", lw=lw)
  ax[1,1].semilogy(cond_HtA/cond_A, "r", lw=lw)
  _dcgmo = ("mcmc", "current", 0)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  ax[1,0].semilogy(SpHtA_k, "g", lw=lw, label=r"$\lambda_{k+1}(H^TA),\;\lambda_{n}(H^TA)$")
  ax[1,0].semilogy(SpHtA_n, "g", lw=lw)
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[1,1].semilogy(cond_HtA/cond_A, "g", lw=lw)
  ax[1,0].legend(loc="upper left", bbox_to_anchor=(0.05, 0.8), frameon=False)
  # Snapshot #1
  ax[1,2].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA), \{\rho(w_j,A)\}_{j=1}^k$")
  for i in range(499,502):
    kdim = dcgmo_kdim[("mcmc", "previous", 0)][i]
    ax[1,2].semilogy((n-kdim)*[i-.4], smp_SpA["mcmc"][i][kdim:], "k_", markersize=6)
    ax[1,2].semilogy(kdim*[i-.4], smp_SpA["mcmc"][i][:kdim], "k+", markersize=6)
    ax[1,2].semilogy((n-kdim)*[i-.2], dcgmo_SpHtA[("mcmc", "previous", 0)][i][kdim:], "r_", markersize=6)
    #ax[1,2].semilogy(kdim*[i], dcgmo_eigvals[("mcmc", "previous", 0)][i], "r+", markersize=6)
    ax[1,2].semilogy(kdim*[i], dcgmo_ritz_coef[("mcmc", "previous", 0)][i], "r+", markersize=6)
    ax[1,2].semilogy((n-kdim)*[i+.2], dcgmo_SpHtA[("mcmc", "current", 0)][i][kdim:], "g_", markersize=6)
    #ax[1,2].semilogy(kdim*[i+.4], dcgmo_eigvals[("mcmc", "current", 0)][i], "g+", markersize=6)
    ax[1,2].semilogy(kdim*[i+.4], dcgmo_ritz_coef[("mcmc", "current", 0)][i], "g+", markersize=6)
    if (i < 501):
      ax[1,2].semilogy((i+.5,i+.5), (1e-3,5e4),"k", lw=lw)
  # Snapshot #2
  ax[1,3].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA), \{\rho(w_j,A)\}_{j=1}^k$")
  for i in range(2999,3002):
    kdim = dcgmo_kdim[("mcmc", "previous", 0)][i]
    ax[1,3].semilogy((n-kdim)*[i-.4], smp_SpA["mcmc"][i][kdim:], "k_", markersize=6)
    ax[1,3].semilogy(kdim*[i-.4], smp_SpA["mcmc"][i][:kdim], "k+", markersize=6)
    ax[1,3].semilogy((n-kdim)*[i-.2], dcgmo_SpHtA[("mcmc", "previous", 0)][i][kdim:], "r_", markersize=6)
    #ax[1,3].semilogy(kdim*[i], dcgmo_eigvals[("mcmc", "previous", 0)][i], "r+", markersize=6)
    ax[1,3].semilogy(kdim*[i], dcgmo_ritz_coef[("mcmc", "previous", 0)][i], "r+", markersize=6)
    ax[1,3].semilogy((n-kdim)*[i+.2], dcgmo_SpHtA[("mcmc", "current", 0)][i][kdim:], "g_", markersize=6)
    #ax[1,3].semilogy(kdim*[i+.4], dcgmo_eigvals[("mcmc", "current", 0)][i], "g+", markersize=6)
    ax[1,3].semilogy(kdim*[i+.4], dcgmo_ritz_coef[("mcmc", "current", 0)][i], "g+", markersize=6)
    if (i < 3001):
      ax[1,3].semilogy((i+.5,i+.5), (1e-3,5e4),"k", lw=lw)
  ax[1,0].set_ylim(ax[0,0].get_ylim()); ax[1,2].set_ylim(ax[0,0].get_ylim()); ax[1,3].set_ylim(ax[0,0].get_ylim())
  ax[0,1].set_ylim(1e-3,1)
  ylim0 = ax[0,1].get_ylim(); ylim1 = ax[1,1].get_ylim()
  ylim = (min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1]))
  ax[0,1].set_ylim(ylim); ax[1,1].set_ylim(ylim)
  ax[0,1].grid(); ax[1,1].grid()
  for j in range(4):
    ax[1,j].set_xlabel("Realization index, t")
  #pl.show()
  pl.savefig(figures_path+"example05_recycler_"+case+".png", bbox_inches='tight')

  fig, ax = pl.subplots(1, 4, figsize=(13.5,3.5), sharey="row")
  _dcgmo = ("mc", "previous", 0)
  ax[0].set_title("MC--previous")
  errors = get_eigen_errors(dcgmo_eigen_error[_dcgmo], dcgmo_kdim[_dcgmo])
  for k in range(errors.shape[0]):
    ax[0].semilogy(errors[k,:], lw=lw)
  ax[0].set_ylim(1e-3,5e3)
  _dcgmo = ("mc", "current", 0)
  ax[1].set_title("MC--current")
  errors = get_eigen_errors(dcgmo_eigen_error[_dcgmo], dcgmo_kdim[_dcgmo])
  for k in range(errors.shape[0]):
    ax[1].semilogy(errors[k,:], lw=lw)
  _dcgmo = ("mcmc", "previous", 0)
  ax[2].set_title("MCMC--previous")
  errors = get_eigen_errors(dcgmo_eigen_error[_dcgmo], dcgmo_kdim[_dcgmo])
  for k in range(errors.shape[0]):
    ax[2].semilogy(errors[k,:], lw=lw)
  _dcgmo = ("mcmc", "current", 0)
  ax[3].set_title("MCMC--current")
  errors = get_eigen_errors(dcgmo_eigen_error[_dcgmo], dcgmo_kdim[_dcgmo])
  for k in range(errors.shape[0]):
    ax[3].semilogy(errors[k,:], lw=lw)
  for j in range(4):
    ax[j].grid()
    ax[j].set_xlabel("Realization index, t")
  ax[0].set_ylabel(r"$||Aw_j-\rho(w_j,A)w_j||/||w_j||$")
  #pl.show()
  pl.savefig(figures_path+"example05_recycler_err_"+case+".png", bbox_inches='tight')