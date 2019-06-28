import sys; sys.path += ["../"]
from samplers import sampler
from solvers import solver
from recyclers import recycler
import numpy as np
from example06_recycler_plot import *

nEl = 1000
nsmp = 5000
sig2, L = .357, 0.05
model = "Exp"

kl = 20
case = "b" # {"a", "b", "c"}

smp, dpcg, dpcgmo = {}, {}, {}

for _smp in ("mc", "mcmc"):
  __smp = sampler(nEl=nEl, smp_type=_smp, model=model, sig2=sig2, L=L)
  __smp.compute_KL()
  __smp.draw_realization()
  __smp.do_assembly()
  smp[_smp] = __smp

pcg = solver(n=smp["mc"].n, solver_type="pcg")
if (case == "a"):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
elif (case == "b"):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)
elif (case == "c"):
  pcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=2)

for __smp in ("mc", "mcmc"):
  for dp_seq in ("dp", "pd"):
    for which_op in ("previous", "current"):
      __dpcg = solver(n=smp["mc"].n, solver_type="dpcg")
      if (case == "a"):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=3, nb=10)
      elif (case == "b"):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=1)
      elif (case == "c"):
        __dpcg.set_precond(Mat=smp["mc"].get_median_A(), precond_id=2)
      dpcg[(__smp, dp_seq, which_op)] = __dpcg
      dpcgmo[(__smp, dp_seq, which_op)] = recycler(smp[__smp], __dpcg, "dpcgmo", kl=kl, dp_seq=dp_seq, which_op=which_op)

pcgmo_it = {"mc":[], "mcmc":[]}
dpcgmo_it = {}

for i_smp in range(nsmp):
  smp["mc"].draw_realization()
  pcg.presolve(smp["mc"].A, smp["mc"].b)
  pcg.solve(x0=np.zeros(smp["mc"].n))
  pcgmo_it["mc"] += [pcg.it]
  for dp_seq in ("dp", "pd"):
    for which_op in ("previous", "current"):
      _dpcgmo = ("mc", dp_seq, which_op)
  
      dpcgmo[_dpcgmo].do_assembly()
      dpcgmo[_dpcgmo].prepare()

      dpcgmo[_dpcgmo].solve()
      if dpcgmo_it.has_key(_dpcgmo):
        dpcgmo_it[_dpcgmo] += [dpcgmo[_dpcgmo].solver.it]
      else:
        dpcgmo_it[_dpcgmo] = [dpcgmo[_dpcgmo].solver.it]

  print("%d/%d" %(i_smp+1, nsmp))

while (smp["mcmc"].cnt_accepted_proposals <= nsmp):
  smp["mcmc"].draw_realization()
  if (smp["mcmc"].proposal_accepted):
    pcg.presolve(smp["mcmc"].A, smp["mcmc"].b)
    pcg.solve(x0=np.zeros(smp["mcmc"].n))
    pcgmo_it["mcmc"] += [pcg.it]
    for dp_seq in ("dp", "pd"):
      for which_op in ("previous", "current"):
        _dpcgmo = ("mcmc", dp_seq, which_op)

        dpcgmo[_dpcgmo].do_assembly()
        dpcgmo[_dpcgmo].prepare()

        dpcgmo[_dpcgmo].solve()
        if dpcgmo_it.has_key(_dpcgmo):
          dpcgmo_it[_dpcgmo] += [dpcgmo[_dpcgmo].solver.it]
        else:
          dpcgmo_it[_dpcgmo] = [dpcgmo[_dpcgmo].solver.it]

    print("%d/%d" %(smp["mcmc"].cnt_accepted_proposals+1, nsmp))

save_data(dpcgmo_it, pcgmo_it, case)
plot(case=case)