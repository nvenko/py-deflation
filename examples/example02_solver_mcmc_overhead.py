import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
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

nEl = 1000
nsmp_mcmc = 10000
nsmp_mc = 1000

nchains = 1000

case = 5
model = "Exp"

prms = [{"sig2":0.05, "L":0.02}, {"sig2":0.50, "L":0.02}, 
        {"sig2":0.05, "L":0.10}, {"sig2":0.50, "L":0.10}, 
        {"sig2":0.05, "L":0.50}, {"sig2":0.50, "L":0.50}]

sig2, L = prms[case]["sig2"], prms[case]["L"]

u_xb_mc = load_u_xb_mc(case, model)

if isinstance(u_xb_mc, type(False)):
  u_xb_mc = np.zeros(nsmp_mc)
  mc = sampler(nEl=nEl, smp_type="mc", model=model, sig2=sig2, L=L, u_xb=None, du_xb=0)
  mc.compute_KL()
  pcg = solver(n=mc.n, solver_type="pcg")
  pcg.set_precond(Mat=mc.get_median_A(), precond_id=1)
  for i_smp in range(nsmp_mc):
    mc.draw_realization()
    mc.do_assembly()
    pcg.presolve(A=mc.A, b=mc.b)
    pcg.solve(x0=np.zeros(mc.n))
    u_xb_mc[i_smp] = pcg.x[-1]
  save_u_xb_mc(u_xb_mc, case, model)
var_u_xb = np.var(u_xb_mc)



u_xb_mcmc, ratio = load_u_xb_mcmc(case, model)

if isinstance(u_xb_mcmc, type(False)):
  u_xb_mcmc = np.zeros((nchains, nsmp_mcmc))
  ratio = np.zeros(nchains)
  mcmc = sampler(nEl=nEl, smp_type="mcmc", model=model, sig2=sig2, L=L, u_xb=None, du_xb=0)
  mcmc.compute_KL()
  #mcmc.draw_realization()
  #mcmc.do_assembly()
  pcg = solver(n=mcmc.n, solver_type="pcg")
  pcg.set_precond(Mat=mcmc.get_median_A(), precond_id=1)  
  for ichain in range(nchains):
    print "chain %g/%g " % (ichain+1, nchains)
    mcmc.cnt_accepted_proposals = 0
    mcmc.reals = 0  
    for i_smp in range(nsmp_mcmc):
      mcmc.draw_realization()
      mcmc.do_assembly()
      if (mcmc.proposal_accepted) | (mcmc.reals == 1):
        pcg.presolve(A=mcmc.A, b=mcmc.b)
        pcg.solve(x0=np.zeros(mcmc.n))
        #print ichain, mcmc.cnt_accepted_proposals
      u_xb_mcmc[ichain][i_smp] = pcg.x[-1]
    ratio[ichain] = mcmc.cnt_accepted_proposals/float(mcmc.reals)
  save_u_xb_mcmc(u_xb_mcmc, ratio, case, model)

sig2f_xb = np.zeros(nsmp_mcmc)
sig2f_xb[0] = np.var(u_xb_mcmc[:,0])
for ismp in range(1, nsmp_mcmc):
  sig2f_xb[ismp] = sig2f_xb[ismp-1]+2.*np.cov(u_xb_mcmc[:,0], u_xb_mcmc[:,ismp])[0,1]

ax = pl.subplot()
ax.plot(sig2f_xb)
ax.set_xlabel("Realization index, t")
ax.set_ylabel(r"$\sigma_f^2(1)$")
pl.savefig(figures_path+"solver_mcmc_overhead_%d_%s_sig2f.png" % (case, model))
print np.mean(ratio)*sig2f_xb[-1]/var_u_xb

