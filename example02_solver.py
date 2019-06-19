from samplers import sampler
from solvers import solver
import pylab as pl
import numpy as np

nEl = 1000
nsmp = 50
sig2, L = .357, 0.05
model = "Exp"

mc = sampler(nEl=nEl, smp_type="mc", model=model, sig2=sig2, L=L)
mc.compute_KL()

A_median = mc.get_median_A()
pcg = solver(A_median.shape[0], solver_type="cg")
#pcg.set_precond(Mat=A_median, precond_id=3)

fig, ax = pl.subplots(1, 2, figsize=(8,3.5))
for i_smp in range(nsmp):
  mc.draw_realization()
  mc.do_assembly()
  pcg.solve(A=mc.A, b=mc.b, x0=np.zeros(pcg.n))
  ax[0].plot(mc.get_kappa(), lw=.1)
  ax[1].plot(pcg.x, lw=.1)
ax[0].set_title("kappa(x)")
ax[1].set_title("u(x)")
pl.show()