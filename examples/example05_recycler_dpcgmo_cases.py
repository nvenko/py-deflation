def get_params(case):

  eigres_thresh = 1e0

  if (case == "a"):
    precond_id = 3
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "b"):
    precond_id = 1
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "b2"):
    precond_id = 1
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/2; nsmp = 2000
    t_end_def = 0; t_end_kl = 500; t_switch_to_mc = 1000; ini_W = False

  return precond_id, sig2, L, model, kl, kl_strategy, ell_min, nsmp, t_end_def, t_end_kl, t_switch_to_mc, ini_W, eigres_thresh
