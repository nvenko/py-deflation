import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import pylab as pl
import numpy as np

""" HOW TO:
    >>> from example04_reclycler_plot import *
    >>> plot()
"""

figures_path = '../figures/'

lw = 0.3

def save_data(dcgmo_kdim, dcgmo_ell, dcgmo_it, cgmo_it):
  np.save(".example04_recycler_dcgmo_kdim", dcgmo_kdim)
  np.save(".example04_recycler_dcgmo_ell", dcgmo_ell)
  np.save(".example04_recycler_dcgmo_it", dcgmo_it)
  np.save(".example04_recycler_cgmo_it", cgmo_it)

def load_data():
    dcgmo_kdim = np.load(".example04_recycler_dcgmo_kdim.npy").item()
    dcgmo_ell = np.load(".example04_recycler_dcgmo_ell.npy").item()
    dcgmo_it = np.load(".example04_recycler_dcgmo_it.npy").item()
    cgmo_it = np.load(".example04_recycler_cgmo_it.npy").item()
    return dcgmo_kdim, dcgmo_ell, dcgmo_it, cgmo_it

def plot(dcgmo_kdim=None, dcgmo_ell=None, dcgmo_it=None, cgmo_it=None):
  if (type(dcgmo_kdim) == type(None)):
    dcgmo_kdim, dcgmo_ell, dcgmo_it, cgmo_it = load_data()

  fig, ax = pl.subplots(3, 3, figsize=(13.5,10.5), sharex="col")
  fig.suptitle("DCGMO -- MC sampler")
  # First row:
  ax[0,0].set_title("kl_strategy #1")
  ax[0,0].set_ylabel("kdim, ell")
  ax[0,0].plot(dcgmo_kdim[("mc", "previous", 0)], label="kdim")
  ax[0,0].plot(dcgmo_ell[("mc", "previous", 0)], label="ell")
  ax[0,1].set_title("Number of solver iterations, n_it")
  ax[0,1].plot(cgmo_it["mc"], "k", lw=lw, label="cgmo")
  ax[0,1].plot(dcgmo_it[("mc", "previous", 0)], "r", lw=lw)
  ax[0,1].plot(dcgmo_it[("mc", "current", 0)], "g", lw=lw)
  ax[0,2].set_title("Relative number of solver iterations wrt CG")
  ax[0,2].plot(np.array(dcgmo_it[("mc", "previous", 0)])/np.array(cgmo_it["mc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
  ax[0,2].plot(np.array(dcgmo_it[("mc", "current", 0)])/np.array(cgmo_it["mc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
  ax[0,0].legend(frameon=False, ncol=2); ax[0,1].legend(frameon=False); ax[0,2].legend(frameon=False, ncol=2)
  # Second row:
  ax[1,0].set_title("kl_strategy #2")
  ax[1,0].set_ylabel("kdim, ell")
  ax[1,0].plot(dcgmo_kdim[("mc", "previous", 1)], label="kdim")
  ax[1,0].plot(dcgmo_ell[("mc", "previous", 1)], label="ell")
  ax[1,1].set_title("Number of solver iterations, n_it")
  ax[1,1].plot(cgmo_it["mc"], "k", lw=lw, label="cgmo")
  ax[1,1].plot(dcgmo_it[("mc", "previous", 1)], "r", lw=lw)
  ax[1,1].plot(dcgmo_it[("mc", "current", 1)], "g", lw=lw)
  ax[1,2].set_title("Relative number of solver iterations wrt CG")
  ax[1,2].plot(np.array(dcgmo_it[("mc", "previous", 1)])/np.array(cgmo_it["mc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
  ax[1,2].plot(np.array(dcgmo_it[("mc", "current", 1)])/np.array(cgmo_it["mc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
  ax[1,0].legend(frameon=False, ncol=2); ax[1,1].legend(frameon=False); ax[1,2].legend(frameon=False, ncol=2)
  # Third row:
  ax[2,0].set_title("kl_strategy #3")
  ax[2,0].set_ylabel("kdim, ell")
  ax[2,0].plot(dcgmo_kdim[("mc", "previous", 2)], label="kdim")
  ax[2,0].plot(dcgmo_ell[("mc", "previous", 2)], label="ell")
  ax[2,1].set_title("Number of solver iterations, n_it")
  ax[2,1].plot(cgmo_it["mc"], "k", lw=lw, label="cgmo")
  ax[2,1].plot(dcgmo_it[("mc", "previous", 2)], "r", lw=lw)
  ax[2,1].plot(dcgmo_it[("mc", "current", 2)], "g", lw=lw)
  ax[2,2].set_title("Relative number of solver iterations wrt CG")
  ax[2,2].plot(np.array(dcgmo_it[("mc", "previous", 2)])/np.array(cgmo_it["mc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
  ax[2,2].plot(np.array(dcgmo_it[("mc", "current", 2)])/np.array(cgmo_it["mc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
  ax[2,2].set_ylim(0.6,1)
  ax[2,0].legend(frameon=False, ncol=2); ax[2,1].legend(frameon=False); ax[2,2].legend(frameon=False, ncol=2)
  for j in range(3):
    ax[0,j].set_ylim(ax[2,j].get_ylim())
    ax[1,j].set_ylim(ax[2,j].get_ylim())
    ax[2,j].set_xlabel("Realization index, t")
  ax[0,2].grid(); ax[1,2].grid(); ax[2,2].grid()
  #pl.show()
  pl.savefig(figures_path+"example04_recycler_a.png", bbox_inches='tight')  

  fig, ax = pl.subplots(3, 3, figsize=(13.5,10.5), sharex="col")
  fig.suptitle("DCGMO -- MCMC sampler")
  # First row:
  ax[0,0].set_title("kl_strategy #1")
  ax[0,0].set_ylabel("kdim, ell")
  ax[0,0].plot(dcgmo_kdim[("mcmc", "previous", 0)], label="kdim")
  ax[0,0].plot(dcgmo_ell[("mcmc", "previous", 0)], label="ell")
  ax[0,1].set_title("Number of solver iterations, n_it")
  ax[0,1].plot(cgmo_it["mcmc"], "k", lw=lw, label="cgmo")
  ax[0,1].plot(dcgmo_it[("mcmc", "previous", 0)], "r", lw=lw)
  ax[0,1].plot(dcgmo_it[("mcmc", "current", 0)], "g", lw=lw)
  ax[0,2].set_title("Relative number of solver iterations wrt CG")
  ax[0,2].plot(np.array(dcgmo_it[("mcmc", "previous", 0)])/np.array(cgmo_it["mcmc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
  ax[0,2].plot(np.array(dcgmo_it[("mcmc", "current", 0)])/np.array(cgmo_it["mcmc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
  ax[0,0].legend(frameon=False, ncol=2); ax[0,1].legend(frameon=False); ax[0,2].legend(frameon=False, ncol=2)
  # Second row:
  ax[1,0].set_title("kl_strategy #2")
  ax[1,0].set_ylabel("kdim, ell")
  ax[1,0].plot(dcgmo_kdim[("mcmc", "previous", 1)], label="kdim")
  ax[1,0].plot(dcgmo_ell[("mcmc", "previous", 1)], label="ell")
  ax[1,1].set_title("Number of solver iterations, n_it")
  ax[1,1].plot(cgmo_it["mcmc"], "k", lw=lw, label="cgmo")
  ax[1,1].plot(dcgmo_it[("mcmc", "previous", 1)], "r", lw=lw)
  ax[1,1].plot(dcgmo_it[("mcmc", "current", 1)], "g", lw=lw)
  ax[1,2].set_title("Relative number of solver iterations wrt CG")
  ax[1,2].plot(np.array(dcgmo_it[("mcmc", "previous", 1)])/np.array(cgmo_it["mcmc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
  ax[1,2].plot(np.array(dcgmo_it[("mcmc", "current", 1)])/np.array(cgmo_it["mcmc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
  ax[1,0].legend(frameon=False, ncol=2); ax[1,1].legend(frameon=False); ax[1,2].legend(frameon=False, ncol=2)
  # Third row:
  ax[2,0].set_title("kl_strategy #3")
  ax[2,0].set_ylabel("kdim, ell")
  ax[2,0].plot(dcgmo_kdim[("mcmc", "previous", 2)], label="kdim")
  ax[2,0].plot(dcgmo_ell[("mcmc", "previous", 2)], label="ell")
  ax[2,1].set_title("Number of solver iterations, n_it")
  ax[2,1].plot(cgmo_it["mcmc"], "k", lw=lw, label="cgmo")
  ax[2,1].plot(dcgmo_it[("mcmc", "previous", 2)], "r", lw=lw)
  ax[2,1].plot(dcgmo_it[("mcmc", "current", 2)], "g", lw=lw)
  ax[2,2].set_title("Relative number of solver iterations wrt CG")
  ax[2,2].plot(np.array(dcgmo_it[("mcmc", "previous", 2)])/np.array(cgmo_it["mcmc"], dtype=float), "r", lw=lw, label="dcgmo-prev")
  ax[2,2].plot(np.array(dcgmo_it[("mcmc", "current", 2)])/np.array(cgmo_it["mcmc"], dtype=float), "g", lw=lw, label="dcgmo-curr")
  ax[2,2].set_ylim(0.6,1)
  ax[2,0].legend(frameon=False, ncol=2); ax[2,1].legend(frameon=False); ax[2,2].legend(frameon=False, ncol=2)
  for j in range(3):
    ax[0,j].set_ylim(ax[2,j].get_ylim())
    ax[1,j].set_ylim(ax[2,j].get_ylim())
    ax[2,j].set_xlabel("Realization index, t")
  ax[0,2].grid(); ax[1,2].grid(); ax[2,2].grid()
  pl.savefig(figures_path+"example04_recycler_b.png", bbox_inches='tight')