import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
import numpy as np
from example02_solver_mcmc_overhead_ergodic_plot import *

nEl = 1000
nsmp_mcmc = 1e7
nsmp_mc = 1e5

model = "SExp"

prms = [{"sig2":0.05, "L":0.02}, {"sig2":0.50, "L":0.02}, 
        {"sig2":0.05, "L":0.20}, {"sig2":0.50, "L":0.20}]

u_xb_mc, u_xb_mcmc, ratio = {}, {}, {}
cov_u_xb_mc, cov_u_xb_mcmc = {}, {}

for case in range(4):
  sig2, L = prms[case]["sig2"], prms[case]["L"]
  _case = (sig2, L)
  print _case, "mc"
  u_xb_mc[_case] = load_u_xb(sig2, L, model, "mc")
  if isinstance(u_xb_mc[_case], bool):
    u_xb_mc[_case] = np.zeros(nsmp_mc)
    mc = sampler(nEl=nEl, smp_type="mc", model=model, sig2=sig2, L=L, u_xb=None, du_xb=0)
    mc.compute_KL()
    pcg = solver(n=mc.n, solver_type="pcg")
    pcg.set_precond(Mat=mc.get_median_A(), precond_id=1)
    for i_smp in range(nsmp_mc):
      mc.draw_realization()
      mc.do_assembly()
      pcg.presolve(A=mc.A, b=mc.b)
      pcg.solve(x0=np.zeros(mc.n))
      u_xb_mc[_case][i_smp] = pcg.x[-1]
    save_u_xb({"u":u_xb_mc[_case]}, sig2, L, model, "mc")
  fft_u_xb = np.fft.fft(u_xb_mc[_case]-u_xb_mc[_case].mean())
  cov_u_xb_mc[_case] = np.real(np.fft.ifft(fft_u_xb*np.conjugate(fft_u_xb))/nsmp_mc)

  print _case, "mcmc"
  u_xb_mcmc[_case], ratio[_case] = load_u_xb(sig2, L, model, "mcmc")
  if isinstance(u_xb_mcmc[_case], bool):
    u_xb_mcmc[_case] = np.zeros(nsmp_mcmc)
    mcmc = sampler(nEl=nEl, smp_type="mcmc", model=model, sig2=sig2, L=L, u_xb=None, du_xb=0)
    mcmc.compute_KL()
    pcg = solver(n=mcmc.n, solver_type="pcg")
    pcg.set_precond(Mat=mcmc.get_median_A(), precond_id=1)  
    for i_smp in range(nsmp_mcmc):
      mcmc.draw_realization()
      mcmc.do_assembly()
      if (mcmc.proposal_accepted):
        pcg.presolve(A=mcmc.A, b=mcmc.b)
        pcg.solve(x0=np.zeros(mcmc.n))
      u_xb_mcmc[_case][i_smp] = pcg.x[-1]
    ratio[_case] = mcmc.cnt_accepted_proposals/float(mcmc.reals)
    save_u_xb({"u":u_xb_mcmc[_case], "ratio":ratio[_case]}, sig2, L, model, "mcmc")
  fft_u_xb = np.fft.fft(u_xb_mcmc[_case]-u_xb_mcmc[_case].mean())
  cov_u_xb_mcmc[_case] = np.real(np.fft.ifft(fft_u_xb*np.conjugate(fft_u_xb))/nsmp_mcmc)

plot(cov_u_xb_mcmc, cov_u_xb_mc, ratio, model, fig_ext=".eps") 