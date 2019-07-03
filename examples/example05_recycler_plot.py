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

lw = 0.3

def get_envelopes_A(SpA, dcgmo_kdim):
  SpA_0 = [Sp[0] for Sp in SpA]
  SpA_k = [SpA[t][dcgmo_kdim[t]] for t in range(len(SpA))]
  SpA_n = [Sp[-1] for Sp in SpA]
  return SpA_0, SpA_k, SpA_n

def get_envelopes_HtA(SpHtA, dcgmo_kdim):
  SpHtA_k = [SpHtA[t][dcgmo_kdim[t]] for t in range(len(SpHtA))]
  SpHtA_n = [Sp[-1] for Sp in SpHtA]
  return SpHtA_k, SpHtA_n

def save_data(smp, smp_SpA, dcgmo_SpHtA, dcgmo_kdim, case):
  np.save(".example05_recycler_smp_"+case, smp)
  np.save(".example05_recycler_smp_SpA_"+case, smp_SpA)
  np.save(".example05_recycler_dcgmo_SpHtA_"+case, dcgmo_SpHtA)
  np.save(".example05_recycler_dcgmo_kdim_"+case, dcgmo_kdim)

def load_data(case):
    smp = np.load(".example05_recycler_smp_"+case+".npy").item()
    smp_SpA = np.load(".example05_recycler_smp_SpA_"+case+".npy").item()
    dcgmo_SpHtA = np.load(".example05_recycler_dcgmo_SpHtA_"+case+".npy").item()
    dcgmo_kdim = np.load(".example05_recycler_dcgmo_kdim_"+case+".npy").item()
    return smp, smp_SpA, dcgmo_SpHtA, dcgmo_kdim

def plot(smp=None, smp_SpA=None, dcgmo_SpHtA=None, dcgmo_kdim=None, case=None):
  if (type(smp) == type(None)):
    smp, smp_SpA, dcgmo_SpHtA, dcgmo_kdim = load_data(case)

  n = smp["mc"].n
  fig, ax = pl.subplots(2, 4, figsize=(13.5,9), sharex="col")
  strategy_name = {"a":"strategy #1", "b":"strategy #2", "c":"strategy #3"}
  fig.suptitle("DCGMO -- %s -- Envelopes and full spectra" %strategy_name[case])
  # Enveloppes
  ax[0,0].set_ylabel("MC sampler")
  ax[0,0].set_title("envelopes")
  _dcgmo = ("mc", "previous", 0)
  SpA_0, SpA_k, SpA_n = get_envelopes_A(smp_SpA["mc"], dcgmo_kdim[_dcgmo])
  cond_A = np.array(SpA_n)/np.array(SpA_0)
  ax[0,0].semilogy(SpA_0, "k", lw=lw)
  ax[0,0].semilogy(SpA_k, "k", lw=lw)
  ax[0,0].semilogy(SpA_n, "k", lw=lw)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[0,0].semilogy(SpHtA_k, "r", lw=lw, label="dcgmo-previous")
  ax[0,0].semilogy(SpHtA_n, "r", lw=lw)
  ax[0,1].set_title(r"$\mathrm{cond}(H^TA)/\mathrm{cond}(A)$")
  ax[0,1].semilogy(cond_HtA/cond_A, "r", lw=lw)
  _dcgmo = ("mc", "current", 0)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[0,0].semilogy(SpHtA_k, "g", lw=lw, label="dcgmo-current")
  ax[0,0].semilogy(SpHtA_n, "g", lw=lw)
  ax[0,1].semilogy(cond_HtA/cond_A, "g", lw=lw)
  ax[0,0].legend(frameon=False)
  # Snapshot #1
  ax[0,2].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA)$")
  for i in range(500,506):
    kdim = dcgmo_kdim[("mc", "previous", 0)][i]
    ax[0,2].semilogy((n-kdim)*[i], smp_SpA["mc"][i][kdim:], "k_", markersize=6)
    ax[0,2].semilogy(kdim*[i], smp_SpA["mc"][i][:kdim], "k+", markersize=6)
    ax[0,2].semilogy((n-kdim)*[i+.33], dcgmo_SpHtA[("mc", "previous", 0)][i][kdim:], "r_", markersize=6)
    ax[0,2].semilogy((n-kdim)*[i+.66], dcgmo_SpHtA[("mc", "current", 0)][i][kdim:], "g_", markersize=6)
  # Snapshot #2
  ax[0,3].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA)$")
  for i in range(3000,3006):
    kdim = dcgmo_kdim[("mc", "previous", 0)][i]
    ax[0,3].semilogy((n-kdim)*[i], smp_SpA["mc"][i][kdim:], "k_", markersize=6)
    ax[0,3].semilogy(kdim*[i], smp_SpA["mc"][i][:kdim], "k+", markersize=6)
    ax[0,3].semilogy((n-kdim)*[i+.33], dcgmo_SpHtA[("mc", "previous", 0)][i][kdim:], "r_", markersize=6)
    ax[0,3].semilogy((n-kdim)*[i+.66], dcgmo_SpHtA[("mc", "current", 0)][i][kdim:], "g_", markersize=6)
  ax[0,2].set_ylim(ax[0,0].get_ylim()); ax[0,3].set_ylim(ax[0,0].get_ylim())  

  # Enveloppes
  ax[1,0].set_ylabel("MCMC sampler")
  ax[1,0].set_title("envelopes")
  SpA_0, SpA_k, SpA_n = get_envelopes_A(smp_SpA["mcmc"], dcgmo_kdim[_dcgmo])
  cond_A = np.array(SpA_n)/np.array(SpA_0)
  ax[1,0].semilogy(SpA_0, "k", lw=lw, label=r"$\lambda_{1}(A),\;\lambda_{k}(A),\;\lambda_{n}(A)$")
  ax[1,0].semilogy(SpA_k, "k", lw=lw)
  ax[1,0].semilogy(SpA_n, "k", lw=lw)
  _dcgmo = ("mcmc", "previous", 0)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[1,0].semilogy(SpHtA_k, "r", lw=lw, label=r"$\lambda_{k}(H^TA),\;\lambda_{n}(H^TA)$")
  ax[1,0].semilogy(SpHtA_n, "r", lw=lw)
  ax[1,1].set_title(r"$\mathrm{cond}(H^TA)/\mathrm{cond}(A)$")
  ax[1,1].semilogy(cond_HtA/cond_A, "r", lw=lw)
  _dcgmo = ("mcmc", "current", 0)
  SpHtA_k, SpHtA_n = get_envelopes_HtA(dcgmo_SpHtA[_dcgmo], dcgmo_kdim[_dcgmo])
  ax[1,0].semilogy(SpHtA_k, "g", lw=lw, label=r"$\lambda_{k}(H^TA),\;\lambda_{n}(H^TA)$")
  ax[1,0].semilogy(SpHtA_n, "g", lw=lw)
  cond_HtA = np.concatenate(([cond_A[0]],np.array(SpHtA_n[1:])/np.array(SpHtA_k[1:])))
  ax[1,1].semilogy(cond_HtA/cond_A, "g", lw=lw)
  ax[1,0].legend(frameon=False)
  # Snapshot #1
  ax[1,2].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA)$")
  for i in range(500,506):
    kdim = dcgmo_kdim[("mcmc", "previous", 0)][i]
    ax[1,2].semilogy((n-kdim)*[i], smp_SpA["mcmc"][i][kdim:], "k_", markersize=6)
    ax[1,2].semilogy(kdim*[i], smp_SpA["mcmc"][i][:kdim], "k+", markersize=6)
    ax[1,2].semilogy((n-kdim)*[i+.33], dcgmo_SpHtA[("mcmc", "previous", 0)][i][kdim:], "r_", markersize=6)
    ax[1,2].semilogy((n-kdim)*[i+.66], dcgmo_SpHtA[("mcmc", "current", 0)][i][kdim:], "g_", markersize=6)
  # Snapshot #2
  ax[1,3].set_title(r"$\mathrm{Sp}(A), \mathrm{Sp}(H^TA)$")
  for i in range(3000,3006):
    kdim = dcgmo_kdim[("mcmc", "previous", 0)][i]
    ax[1,3].semilogy((n-kdim)*[i], smp_SpA["mcmc"][i][kdim:], "k_", markersize=6)
    ax[1,3].semilogy(kdim*[i], smp_SpA["mcmc"][i][:kdim], "k+", markersize=6)
    ax[1,3].semilogy((n-kdim)*[i+.33], dcgmo_SpHtA[("mcmc", "previous", 0)][i][kdim:], "r_", markersize=6)
    ax[1,3].semilogy((n-kdim)*[i+.66], dcgmo_SpHtA[("mcmc", "current", 0)][i][kdim:], "g_", markersize=6)
  ax[1,0].set_ylim(ax[0,0].get_ylim()); ax[1,2].set_ylim(ax[0,0].get_ylim()); ax[1,3].set_ylim(ax[0,0].get_ylim())
  ax[0,1].set_ylim(1e-3,1)
  ylim0 = ax[0,1].get_ylim(); ylim1 = ax[1,1].get_ylim()
  ylim = (min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1]))
  ax[0,1].set_ylim(ylim); ax[1,1].set_ylim(ylim)
  ax[0,1].grid(); ax[1,1].grid()
  for j in range(4):
    ax[1,j].set_xlabel("Realization index, t")
  pl.savefig(figures_path+"example05_recycler_"+case+".png", bbox_inches='tight')
  #pl.show()