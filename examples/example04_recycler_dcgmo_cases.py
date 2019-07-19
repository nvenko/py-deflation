def get_params(case):
  eigres_thresh = 1e0
  if (case == "a"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 10000
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "b"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = 3*kl/4; nsmp = 10000
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "c"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 10000
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "d"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/2; nsmp = 10000
    t_end_def = 0; t_end_kl = 5000; t_switch_to_mc = 0; ini_W = False
  elif (case == "e"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/4; nsmp = 10000
    t_end_def = 0; t_end_kl = 5000; t_switch_to_mc = 0; ini_W = False
  elif (case == "f"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = 3*kl/4; nsmp = 10000
    t_end_def = 0; t_end_kl = 5000; t_switch_to_mc = 0; ini_W = False
  elif (case == "e5"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/4; nsmp = 15000
    t_end_def = 0; t_end_kl = 5000; t_switch_to_mc = 7000; ini_W = False
  elif (case == "e7"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/4; nsmp = 15000
    t_end_def = 0; t_end_kl = 1000; t_switch_to_mc = 7000; ini_W = False
  elif (case == "a3"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 10000
    t_end_def = nsmp; t_end_kl = 0; t_switch_to_mc = 7000; ini_W = True  

  elif (case == "a2"):
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 15000
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "e6"):
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/4; nsmp = 10000
    t_end_def = 0; t_end_kl = 5000; t_switch_to_mc = 7000; ini_W = False
  elif (case == "a4"):
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 10000
    t_end_def = nsmp; t_end_kl = 5000; t_switch_to_mc = 7000; ini_W = True
  elif (case == "a5"):
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 5000; t_switch_to_mc = 0; ini_W = True  

  elif (case == "g"):
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 2; ell_min = 0; nsmp = 200 # ell_min?
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
    eigres_thresh = 5e0

  return sig2, L, model, kl, kl_strategy, ell_min, nsmp, t_end_def, t_end_kl, t_switch_to_mc, ini_W, eigres_thresh
