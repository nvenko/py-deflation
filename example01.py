import samplers

mc = sampler(nEL=1000, smp_type="mc", sig2=1., )
mc.compute_KL()
mc.draw_realization()
kappa = mc.get_kappa()