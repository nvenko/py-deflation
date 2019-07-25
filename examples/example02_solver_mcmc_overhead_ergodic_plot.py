import numpy as np
import pylab as pl
import glob

pl.rcParams['text.usetex'] = True
params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}
pl.rcParams.update(params)
pl.rcParams['axes.labelsize']=14#19.
pl.rcParams['legend.fontsize']=12.#16.
pl.rcParams['xtick.labelsize']=10.
pl.rcParams['ytick.labelsize']=10.
pl.rcParams['legend.numpoints']=1

figures_path = '../figures/'

def save_u_xb(u_smp, sig2, L, model, smp_type):
  if smp_type not in ("mc", "mcmc"):
    raise ValueError("Invalid value for smp_type.")
  if smp_type == "mc":
    np.save(".paper1D01_sig2%g_L%g_%s_u_xb_mc" %(sig2, L, model), u_smp["u"])
  elif smp_type == "mcmc":
    smp_dict = {"u":u_smp["u"], "ratio":u_smp["ratio"]}
    np.save(".paper1D01_sig2%g_L%g_%s_u_xb_mcmc" %(sig2, L, model), smp_dict)

def load_u_xb(sig2, L, model, smp_type):
  if smp_type not in ("mc", "mcmc"):
    raise ValueError("Invalid value for smp_type.")
  fname = ".paper1D01_sig2%g_L%g_%s_u_xb_%s.npy" %(sig2, L, model, smp_type)
  files = glob.glob(fname)
  if smp_type == "mc":
    if (len(files) > 0):
      u_xb_mc = np.load(files[0])
      return u_xb_mc
    else:
      return False
  elif smp_type == "mcmc":
    if (len(files) > 0):
      smp_dict = np.load(files[0]).item()
      u_xb_mcmc = smp_dict["u"]
      ratio = smp_dict["ratio"]
      return u_xb_mcmc, ratio
    else:
      return False, False


def plot(cov_u_xb_mcmc, cov_u_xb_mc, model):
  fig, ax = pl.subplots(1, 2, figsize=(7, 3.7), sharey=True)
  fig.suptitle("Correlation decay of MCMC sampled states and overhead / %s model" %model)
  ax[0].set_title("L = 0.20")
  gamma = .23*(cov_u_xb_mcmc[(0.5, 0.20)][0]+2*np.sum(cov_u_xb_mcmc[(0.5, 0.20)][1:200]))/cov_u_xb_mc[(0.5, 0.20)][0]
  ax[0].plot(cov_u_xb_mcmc[(0.5, 0.20)][:200]/cov_u_xb_mc[(0.5, 0.20)][0], label=r"$\sigma^2 = 0.50, \gamma\approx %d$" %gamma)
  gamma = .23*(cov_u_xb_mcmc[(0.05, 0.20)][0]+2*np.sum(cov_u_xb_mcmc[(0.05, 0.20)][1:200]))/cov_u_xb_mc[(0.05, 0.20)][0]
  ax[0].plot(cov_u_xb_mcmc[(0.05, 0.20)][:200]/cov_u_xb_mc[(0.05, 0.20)][0], label=r"$\sigma^2 = 0.05, \gamma\approx %d$" %gamma)

  ax[1].set_title("L = 0.02")
  gamma = .23*(cov_u_xb_mcmc[(0.5, 0.02)][0]+2*np.sum(cov_u_xb_mcmc[(0.5, 0.02)][1:2000]))/cov_u_xb_mc[(0.5, 0.02)][0]
  ax[1].plot(cov_u_xb_mcmc[(0.5, 0.02)][:2000]/cov_u_xb_mc[(0.5, 0.02)][0], label=r"$\sigma^2 = 0.50, \gamma\approx %d$" %gamma)
  gamma = .23*(cov_u_xb_mcmc[(0.05, 0.02)][0]+2*np.sum(cov_u_xb_mcmc[(0.05, 0.02)][1:2000]))/cov_u_xb_mc[(0.05, 0.02)][0]
  ax[1].plot(cov_u_xb_mcmc[(0.05, 0.02)][:2000]/cov_u_xb_mc[(0.05, 0.02)][0], label=r"$\sigma^2 = 0.05, \gamma\approx %d$" %gamma)
  ax[0].legend(), ax[1].legend()
  
  ax[0].set_ylabel(r"$Cov[u(x_b;\boldsymbol{\xi}_t), u(x_b;\boldsymbol{\xi}_{t+s})]/\mathbb{V}[u(x_b)]$")
  ax[0].set_xlabel("Lag of realization indexes, s"); ax[1].set_xlabel("Lag of realization indexes, s")
  pl.savefig(figures_path+"solver_mcmc_overhead_ergodic_%s.png" %model, bbox_inches='tight')

