import numpy as np
import pylab as pl
import glob

pl.rcParams['text.usetex'] = True
params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}
pl.rcParams.update(params)
pl.rcParams['axes.labelsize']=16#19.
pl.rcParams['axes.titlesize']=16#19.
pl.rcParams['legend.fontsize']=14.#16.
pl.rcParams['xtick.labelsize']=13.
pl.rcParams['ytick.labelsize']=13.
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


def plot(cov_u_xb_mcmc, cov_u_xb_mc, ratio, model, fig_ext=".png"):
  fig, ax = pl.subplots(1, 2, figsize=(7, 3.), sharey=True)
  fig.suptitle("%s model" %model, fontsize=16)

  ax[0].set_title("L = 0.20")
  s0 = 50000
  _case = (0.5, 0.20)
  gamma = ratio[_case]*(cov_u_xb_mcmc[_case][0]+2*np.sum(cov_u_xb_mcmc[_case][1:s0]))/cov_u_xb_mc[_case][0]
  ax[0].semilogx(cov_u_xb_mcmc[_case][:s0]/cov_u_xb_mc[_case][0], "r", label=r"$\sigma^2 = 0.50, \gamma\approx %d$" %gamma)
  _case = (0.05, 0.20)
  gamma = ratio[_case]*(cov_u_xb_mcmc[_case][0]+2*np.sum(cov_u_xb_mcmc[_case][1:s0]))/cov_u_xb_mc[_case][0]
  ax[0].semilogx(cov_u_xb_mcmc[_case][:s0]/cov_u_xb_mc[_case][0], "g", label=r"$\sigma^2 = 0.05, \gamma\approx %d$" %gamma)

  ax[1].set_title("L = 0.02")
  s0 = 50000
  _case = (0.5, 0.02)
  gamma = ratio[_case]*(cov_u_xb_mcmc[_case][0]+2*np.sum(cov_u_xb_mcmc[_case][1:s0]))/cov_u_xb_mc[_case][0]
  ax[1].semilogx(cov_u_xb_mcmc[_case][:s0]/cov_u_xb_mc[_case][0], "r", label=r"$\sigma^2 = 0.50, \gamma\approx %d$" %gamma)
  _case = (0.05, 0.02)
  gamma = ratio[_case]*(cov_u_xb_mcmc[_case][0]+2*np.sum(cov_u_xb_mcmc[_case][1:s0]))/cov_u_xb_mc[_case][0]
  ax[1].semilogx(cov_u_xb_mcmc[_case][:s0]/cov_u_xb_mc[_case][0], "g", label=r"$\sigma^2 = 0.05, \gamma\approx %d$" %gamma)

  ax[0].legend(frameon=False), ax[1].legend(frameon=False)
  ax[0].grid(), ax[1].grid()
  if (model == "SExp"):
    #ax[0].set_ylabel(r"$Cov[u(x_b;\boldsymbol{\xi}_t), u(x_b;\boldsymbol{\xi}_{t+s})]/\mathbb{V}[u(x_b)]$")
    ax[0].set_ylabel(r"$\mathrm{Cov}[u(1;\boldsymbol{\xi}_t), u(1;\boldsymbol{\xi}_{t+s})]/\mathbb{V}[u(1)]$")
  else:
    for ticklabel in ax[0].get_ymajorticklabels():
      ticklabel.set_visible(False)
      ticklabel.set_fontsize(0.0)
  ax[0].set_xlabel("States lag, s"); ax[1].set_xlabel("States lag, s")
  major_ticks = [10**j for j in range(5)]
  minor_ticks = []
  for maj_t in major_ticks:
    minor_ticks += [k*maj_t for k in range(2,10)]
  ax[0].set_xticks(major_ticks); ax[1].set_xticks(major_ticks)
  ax[0].set_xticks(minor_ticks, minor=True); ax[1].set_xticks(minor_ticks, minor=True)
  pl.savefig(figures_path+"solver_mcmc_overhead_ergodic_%s%s" %(model, fig_ext), bbox_inches='tight')