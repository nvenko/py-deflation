import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

""" HOW TO:
    >>> from example06_reclycler_plot import *
    >>> plot(case=case)
"""

figures_path = '../figures/'

lw = 0.3

def save_data(dpcgmo_it, pcgmo_it, case):
  np.save(".example06_recycler_dpcgmo_it_"+case, dpcgmo_it)
  np.save(".example06_recycler_pcgmo_it_"+case, pcgmo_it)

def load_data(case):
  dpcgmo_it = np.load(".example06_recycler_dpcgmo_it_"+case+".npy").item()
  pcgmo_it = np.load(".example06_recycler_pcgmo_it_"+case+".npy").item()
  return dpcgmo_it, pcgmo_it

def plot(dpcgmo_it=None, pcgmo_it=None, case="a"):
  if (type(dpcgmo_it) == type(None)):
    dpcgmo_it, pcgmo_it  = load_data(case)

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
  if (case == "a"):
    fig.suptitle("DPCGMO with median-bJ10")
  elif (case == "b"):
    fig.suptitle("DPCGMO with median")
  elif (case == "c"):
    fig.suptitle("DPCGMO with median-AMG")
  for i in range(2):
    ax[i,0].legend(frameon=False, ncol=2); ax[i,1].legend(frameon=False)
    ax[i,2].legend(frameon=False); ax[i,3].legend(frameon=False, ncol=2)
  if (case == "a"):
    pl.savefig(figures_path+"example06_recycler_a.png", bbox_inches='tight')
  elif (case == "b"):
    pl.savefig(figures_path+"example06_recycler_b.png", bbox_inches='tight')
  elif (case == "c"):
    pl.savefig(figures_path+"example06_recycler_c.png", bbox_inches='tight')