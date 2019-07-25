def get_params(case):

  eigres_thresh = 1e0

  if (case == "a"):
    precond_id = 3
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "a2"):
    precond_id = 3
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = True
  elif (case == "a3"):
    precond_id = 3
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "a4"):
    precond_id = 3
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/4; nsmp = 2000
    t_end_def = 0; t_end_kl = 500; t_switch_to_mc = 500; ini_W = False
  elif (case == "a5"):
    precond_id = 3
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 2000
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 500; ini_W = False  

  elif (case == "a6"):
    precond_id = 3
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 200
    t_end_def = nsmp; t_end_kl = 0; t_switch_to_mc = 0; ini_W = True


  elif (case == "a7"):
    precond_id = 3
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 2000
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 500; ini_W = False  
  elif (case == "a8"):
    precond_id = 3
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 2000
    t_end_def = 500; t_end_kl = 0; t_switch_to_mc = 500; ini_W = False  


  elif (case == "a9"):
    precond_id = 3
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = True
  elif (case == "a10"):
    precond_id = 3
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = True



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


  elif (case == "c"):
    precond_id = 2
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "c2"):
    precond_id = 2
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/4; nsmp = 2000
    t_end_def = 0; t_end_kl = 1000; t_switch_to_mc = 1200; ini_W = False

  elif (case == "c3"):
    precond_id = 2
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "c4"):
    precond_id = 2
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = 3*kl/4; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False

  elif (case == "c5"):
    precond_id = 2
    sig2 = 0.05; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = 3*kl/4; nsmp = 2000
    t_end_def = 0; t_end_kl = 500; t_switch_to_mc = 500; ini_W = False

  elif (case == "c6"):
    precond_id = 2
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/2; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False
  elif (case == "c7"):
    precond_id = 2
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 0; ell_min = kl/4; nsmp = 200
    t_end_def = 0; t_end_kl = 0; t_switch_to_mc = 0; ini_W = False

  elif (case == "c8"):
    precond_id = 2
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/2; nsmp = 2000
    t_end_def = 0; t_end_kl = 500; t_switch_to_mc = 500; ini_W = False
  elif (case == "c9"):
    precond_id = 2
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/2; nsmp = 2000
    t_end_def = 500; t_end_kl = 500; t_switch_to_mc = 500; ini_W = False

  elif (case == "c10"):
    precond_id = 2
    sig2 = 0.50; L = 0.02; model = "Exp"
    kl = 20; kl_strategy = 1; ell_min = kl/4; nsmp = 2000
    t_end_def = 0; t_end_kl = 1000; t_switch_to_mc = 1000; ini_W = False


  return precond_id, sig2, L, model, kl, kl_strategy, ell_min, nsmp, t_end_def, t_end_kl, t_switch_to_mc, ini_W, eigres_thresh
