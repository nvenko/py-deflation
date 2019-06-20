from samplers import sampler
from solvers import solver
import numpy as np
import pylab as pl

nEl = 1000
nsmp = 50
sig2, L = .357, 0.05
model = "Exp"

mc = sampler(nEl=nEl, smp_type="mc", model=model, sig2=sig2, L=L, u_xb=0.005, du_xb=None)
mc.compute_KL()

pcg = solver(n=mc.n, solver_type="pcg")
pcg.set_precond(Mat=mc.get_median_A(), precond_id=3, nb=10)

fig, ax = pl.subplots(1, 2, figsize=(8,3.7))
for i_smp in range(nsmp):
  mc.draw_realization()
  mc.do_assembly()
  pcg.solve(A=mc.A, b=mc.b, x0=np.zeros(mc.n))
  ax[0].plot(mc.get_kappa(), lw=.1)
  ax[1].plot(pcg.x, lw=.2)
ax[0].set_title("kappa(x; theta_t)")
ax[1].set_title("u(x; theta_t)")
ax[0].set_xlabel("x"); ax[1].set_xlabel("x")
#pl.show()
pl.savefig("example02_solver.png")