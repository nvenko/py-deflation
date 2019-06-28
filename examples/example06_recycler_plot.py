import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

""" HOW TO:
    >>> from example05_reclycler_plot import *
    >>> plot()
"""

figures_path = '../figures/'

def plot():
  lw = 0.3
  fig, ax = pl.subplots(1, 4, figsize=(17.5,4.))
  ax[0].set_title("MC")
  ax[0].plot(np.array(dpcgmo_it[("mc", "dp")])/np.array(pcgmo_it["mc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
  ax[0].plot(np.array(dpcgmo_it[("mc", "pd")])/np.array(pcgmo_it["mc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
  ax[1].set_title("MC")
  ax[1].plot(pcgmo_it["mc"], "k", lw=lw, label="pcgmo")
  ax[1].plot(dpcgmo_it[("mc", "dp")], "r", lw=lw)
  ax[1].plot(dpcgmo_it[("mc", "pd")], "g", lw=lw)
  ax[2].set_title("MCMC")
  ax[2].plot(pcgmo_it["mcmc"], "k", lw=lw, label="pcgmo")
  ax[2].plot(dpcgmo_it[("mcmc", "dp")], "r", lw=lw)
  ax[2].plot(dpcgmo_it[("mcmc", "pd")], "g", lw=lw)
  ax[3].set_title("MCMC")
  ax[3].plot(np.array(dpcgmo_it[("mcmc", "dp")])/np.array(pcgmo_it["mcmc"], dtype=float), "r", lw=lw, label="dpcgmo-dp")
  ax[3].plot(np.array(dpcgmo_it[("mcmc", "pd")])/np.array(pcgmo_it["mcmc"], dtype=float), "g", lw=lw, label="dpcgmo-pd")
  ax[0].set_ylim(0, 1); ax[3].set_ylim(0, 1)
  ax[2].set_ylim(ax[1].get_ylim())
  ax[0].set_ylabel("Relative number of solver iterations wrt PCG")
  ax[1].set_ylabel("Number of solver iterations, n_it")
  ax[2].set_ylabel("Number of solver iterations, n_it")
  ax[3].set_ylabel("Relative number of solver iterations wrt PCG")
  for j in range(4):
    ax[j].set_xlabel("Realization index, t")
  if (case == "a"):
    fig.suptitle("DPCGMO with median-bJ10")
  elif (case == "b"):
    fig.suptitle("DPCGMO with median")
  elif (case == "c"):
    fig.suptitle("DPCGMO with median-AMG")
  ax[0].legend(frameon=False, ncol=2); ax[1].legend(frameon=False)
  ax[2].legend(frameon=False); ax[3].legend(frameon=False, ncol=2)
  #pl.show()
  if (case == "a"):
    pl.savefig(figures_path+"example06_recycler_a.png", bbox_inches='tight')
  elif (case == "b"):
    pl.savefig(figures_path+"example06_recycler_b.png", bbox_inches='tight')
  elif (case == "c"):
    pl.savefig(figures_path+"example06_recycler_c.png", bbox_inches='tight')