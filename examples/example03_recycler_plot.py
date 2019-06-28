import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

""" HOW TO:
    >>> from example03_reclycler_plot import *
    >>> plot()
"""

figures_path = '../figures/'

nb = 5
dt = [50, 100, 250, 500, 1000]

def save_data(pcgmo_medbJ_it, pcgmo_dtbJ_it):
  np.save(".example03_recycler_pcgmo_medbJ_it", pcgmo_medbJ_it)
  np.save(".example03_recycler_pcgmo_dtbJ_it", pcgmo_dtbJ_it)

def load_data():
    pcgmo_medbJ_it = np.load(".example03_recycler_pcgmo_medbJ_it.npy")
    pcgmo_dtbJ_it = np.load(".example03_recycler_pcgmo_dtbJ_it.npy")
    return pcgmo_medbJ_it, pcgmo_dtbJ_it

def plot(pcgmo_medbJ_it=None, pcgmo_dtbJ_it=None):
  if (type(pcgmo_medbJ_it) == type(None)):
    pcgmo_medbJ_it, pcgmo_dtbJ_it = load_data()

  fig, ax = pl.subplots(1, 2, figsize=(8.5,3.7))
  ax[0].plot(pcgmo_medbJ_it, label="med-bJ%d" %(nb))
  for i, dt_i in enumerate(dt):
    ax[0].plot(pcgmo_dtbJ_it[i], label="%d-bJ%d" %(dt_i,nb), lw=.4)
  av_pcgmo_medbJ_it = np.mean(pcgmo_medbJ_it)
  av_pcgmo_dtbJ_it = np.array([np.mean(pcgmo_it)/av_pcgmo_medbJ_it for pcgmo_it in pcgmo_dtbJ_it])
  ax[1].plot(dt, av_pcgmo_dtbJ_it, "k")
  ax[1].grid()
  ax[0].set_xlabel("Realization index, t"); ax[1].set_xlabel("Renewal period of preconditioner, dt")
  ax[0].set_ylabel("Number of solver iterations, n_it")
  ax[1].set_ylabel("Average relative number of solver iterations")
  ax[0].legend(frameon=False, ncol=2)
  fig.suptitle("MCMC sampled seq. solved by PCGMO w. realization dep. & constant bJ preconditioner")
  pl.savefig(figures_path+"example03_recycler.png", bbox_inches='tight')