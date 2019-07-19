import numpy as np
import pylab as pl
import glob

figures_path = '../figures/'

def save_u_xb_mc(u_smp, case, model):
  np.save(".paper1D01_case%d_%s_u_xb_mc" %(case, model), u_smp)

def load_u_xb_mc(case, model):
  fname = ".paper1D01_case%d_%s_u_xb_mc.npy" %(case, model)
  files = glob.glob(fname)
  if (len(files) > 0):
    u_xb_mc = np.load(files[0])
    return u_xb_mc
  return False

def save_u_xb_mcmc(u_smp, ratio, case, model):
  smp_dict = {"u_xb_mcmc":u_smp, "ratio":ratio}
  np.save(".paper1D01_case%d_%s_u_xb_mcmc" %(case, model), smp_dict)

def load_u_xb_mcmc(case, model):
  fname = ".paper1D01_case%d_%s_u_xb_mcmc.npy" %(case, model)
  files = glob.glob(fname)
  if (len(files) > 0):
    smp_dict = np.load(files[0]).item()
    u_xb_mcmc = smp_dict["u_xb_mcmc"]
    ratio = smp_dict["ratio"]
    return u_xb_mcmc, ratio
  return False, False

def plot(sig2f_xb, case, model):
  fig, ax = pl.subplots(1, 1, figsize=(3.8, 3.8))
  ax.plot(sig2f_xb)
  ax.set_xlabel("Realization index, t")
  ax.set_ylabel(r"$\sigma_f^2(1)$")
  pl.savefig(figures_path+"solver_mcmc_overhead_%d_%s_sig2f.png" % (case, model), bbox_inches='tight')

