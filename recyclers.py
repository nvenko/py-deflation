import solvers
import samplers
import numpy as np
import scipy

import scipy.sparse as sparse
import scipy.sparse.linalg

class recycler:
  """ Recycles. 

      Public methods:
        do_assembly, prepare, solve.
  """
  def __init__(self, sampler, solver, recycler_type, dt=0, t_end_def=0, kl=5, 
               kl_strategy=0, t_end_kl=0, ell_min=0, dp_seq="pd", which_op="previous", 
               approx="HR", eigres_thresh=1., hr_ratio_thresh=2, ini_W=False):
    
    if not isinstance(sampler, samplers.sampler):
      raise ValueError("Invalid sampler.")
    else:
      self.sampler = sampler

    if not isinstance(solver, solvers.solver):
      raise ValueError("Invalid solver.")
    else:
      self.solver = solver

    if not isinstance(recycler_type, str):
      raise ValueError("Invalid recycler_type.")
    else:
      if recycler_type not in ("pcgmo", "dcgmo", "dpcgmo"):
        raise ValueError("Invalid recycler_type.")
      else:
        self.type = recycler_type

    if (sampler.type == "mcmc") & (solver.type == "pcg") & (self.type == "pcgmo"):
      pass
    elif (solver.type == "dcg") & (self.type == "dcgmo"):
      pass
    elif (solver.type == "dpcg") & (self.type == "dpcgmo"):
      pass
    else:
      raise ValueError("Recycler_type not compatible with solver and/or sampler.")

    if not isinstance(dt, int):
      raise ValueError("Invalid dt.")
    else:
      if (dt < 0):
        raise ValueError("Invalid dt.")
      else:
        self.dt = dt

    if not isinstance(t_end_def, int):
      raise ValueError("Invalid t_end_def.")
    else:
      if (t_end_def < 0):
        raise ValueError("Invalid t_end_def.")
      else:
        self.t_end_def = t_end_def

    if not isinstance(kl, int):
      raise ValueError("Invalid kl.")
    else:
      if (kl < 0) | (kl >= sampler.n):
        raise ValueError("Invalid kl.")
      else:
        self.kl = kl

    if not isinstance(kl_strategy, int):
      raise ValueError("Invalid kl_strategy.")
    else:
      if kl_strategy not in (0, 1, 2):
        raise ValueError("Invalid kl_strategy.")
      else:
        self.kl_strategy = kl_strategy

    if not isinstance(ell_min, int):
      raise ValueError("Invalid ell_min.")
    else:
      if (self.kl_strategy == 0):
        if (ell_min <= 0):
          raise ValueError("Invalid ell_min.")
        else:
          self.ell_min = ell_min
      if (self.kl_strategy == 1):
        if (ell_min < 0):
          raise ValueError("Invalid ell_min.")
        else:
          self.ell_min = ell_min

    if (self.kl_strategy == 1):
      if not isinstance(t_end_kl, int):
        raise ValueError("Invalid t_end_kl.")
      else:
        if (t_end_kl <= 0):
          raise ValueError("Invalid t_end_kl.")
        else:
          self.t_end_kl = t_end_kl
          self.dt_kl = self.t_end_kl/(self.kl-self.ell_min-1)

    elif (self.kl_strategy == 2):
      self.eigres_thresh = eigres_thresh
      self.hr_ratio_thresh = hr_ratio_thresh

    if (solver.type == "dpcg") & (self.type == "dpcgmo"):
      if not isinstance(dp_seq, str):
        raise ValueError("Invalid dp_seq.")
      else:
        if dp_seq not in ("pd", "dp"):
          raise ValueError("Invalid dp_seq.")
        else:
          self.dp_seq = dp_seq

    if not isinstance(which_op, str):
      raise ValueError("Invalid which_op.")
    else:
      if which_op not in ("previous", "current"):
        raise ValueError("Invalid which_op.")
      else:
        self.which_op = which_op

    if not isinstance(approx, str):
      raise ValueError("Invalid approx.")
    else:
      if approx not in ("HR", "RR"):
        raise ValueError("Invalid approx.")
      else:
        self.approx = approx

    if not isinstance(ini_W, bool):
      raise ValueError("Invalid ini_W.")
    else:
      self.ini_W = ini_W


  def do_assembly(self):
    if (self.sampler.type == "mc"):
      self.sampler.do_assembly()
      self.solver.A = self.sampler.A
    elif (self.sampler.type == "mcmc"):
      if (self.sampler.proposal_accepted):
        self.sampler.do_assembly()
        self.solver.A = self.sampler.A

  def __get_delta(self, rho_A, rho_B):
    delta = max(rho_A)
    for rho in rho_A:
      _delta = np.min(np.abs(rho-rho_B))
      if (_delta < delta):
        delta = _delta
    return delta
    
  def __pick_new_w(self, which_w, sk=0):
    J = np.where(which_w == False)[0]
    new_j = J[0]
    _which_w = np.copy(which_w); _which_w[new_j] = True
    #print J, which_w, new_j, type(J), J
    #print np.concatenate((self.ritz_quotient[which_w], [self.ritz_quotient[new_j]]))
    #print _which_w, self.ritz_quotient[~_which_w]
    delta = self.__get_delta(np.concatenate((self.ritz_quotient[which_w], [self.ritz_quotient[new_j]])), self.ritz_quotient[~_which_w])
    _sk = sk+self.eigres[new_j]**2
    _bound_gap_frobenius = _sk**.5/delta
    for j in J[1:-1]:
      _which_w = np.copy(which_w); _which_w[j] = True
      delta = self.__get_delta(np.concatenate((self.ritz_quotient[which_w], [self.ritz_quotient[j]])), self.ritz_quotient[~_which_w])
      __sk = sk+self.eigres[j]**2
      __bound_gap_frobenius = __sk**.5/delta
      if (__bound_gap_frobenius < _bound_gap_frobenius):
        _bound_gap_frobenius = __bound_gap_frobenius
        _sk = __sk
        new_j = j
    which_w[new_j] = True
    return _sk, _bound_gap_frobenius, which_w

  def __approx_eigvecs(self, G, F, new_kdim):
    if (self.approx == "HR"):
      #eigvals, eigvecs = scipy.linalg.eigh(G, F, eigvals=(0, new_kdim-1))
      #eigvals, eigvecs = scipy.linalg.eigh(G, F, eigvals=(0, new_kdim))
      eigvals, eigvecs = scipy.linalg.eigh(G, F)
      self.eigvals = eigvals
    elif (self.approx == "RR"):
      eigvals, eigvecs = scipy.linalg.eigh(G, F, eigvals=(self.solver.kdim+self.solver.ell-new_kdim, self.solver.kdim+self.solver.ell-1))
    #self.eigvals = eigvals
    
    if (self.kl_strategy < 2):
      if (self.solver.kdim > 0) & (self.solver.ell >0):
        self.solver.W = self.solver.W.dot(eigvecs[:self.solver.kdim,:new_kdim]) \
                        + self.solver.P.dot(eigvecs[self.solver.kdim:,:new_kdim])
      elif (self.solver.kdim > 0):
        self.solver.W = self.solver.W.dot(eigvecs[:self.solver.kdim,:new_kdim])
      else:
        self.solver.W = self.solver.P.dot(eigvecs[:,:new_kdim])
    elif (self.kl_strategy == 2):
      if (self.solver.kdim > 0) & (self.solver.ell >0):
        self.solver.W = self.solver.W.dot(eigvecs[:self.solver.kdim,:self.solver.kdim+1]) \
                        + self.solver.P.dot(eigvecs[self.solver.kdim:,:self.solver.kdim+1])
      elif (self.solver.kdim > 0):
        self.solver.W = self.solver.W.dot(eigvecs[:self.solver.kdim,:self.solver.kdim+1])
      else:
        self.solver.W = self.solver.P.dot(eigvecs[:,:self.solver.kdim+1])


    if (self.type == "pcgmo"): 
      Aeigvecs = self.sampler.A.dot(self.solver.W)
      wk2 = np.array([self.solver.W[:,k].T.dot(self.solver.W[:,k]) for k in range(new_kdim)])
      self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/wk2
      self.eigres = np.array([np.linalg.norm(Aeigvecs[:,k]-self.ritz_quotient[k]*self.solver.W[:,k])/wk2[k]**.5 for k in range(new_kdim)])
    elif (self.type == "dpcgmo"):
      if (self.dp_seq == "pd"):
        Aeigvecs = self.sampler.A.dot(self.solver.W)
        wk2 = np.array([self.solver.W[:,k].T.dot(self.solver.W[:,k]) for k in range(new_kdim)])
        self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/wk2
        self.eigres = np.array([np.linalg.norm(Aeigvecs[:,k]-self.ritz_quotient[k]*self.solver.W[:,k])/wk2[k]**.5 for k in range(new_kdim)])
      elif (self.dp_seq == "dp"):
        Aeigvecs = self.sampler.A.dot(self.solver.W)
        dwk2 = np.array([self.solver.W[:,k].T.dot(self.solver.M.dot(self.solver.W[:,k])) for k in range(new_kdim)])
        self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/dwk2
        self.eigres = np.array([np.linalg.norm(self.solver.invL_M.dot(Aeigvecs[:,k])-self.ritz_quotient[k]*self.solver.L_M.T.dot(self.solver.W[:,k]))/dwk2[k]**.5 for k in range(new_kdim)])


    #Aeigvecs = self.sampler.A.dot(self.solver.W)
    #wk2 = np.array([self.solver.W[:,k].T.dot(self.solver.W[:,k]) for k in range(new_kdim)])
    #self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/wk2
    #self.eigres = np.array([np.linalg.norm(Aeigvecs[:,k]-self.ritz_quotient[k]*self.solver.W[:,k])/wk2[k]**.5 for k in range(new_kdim)])
    
    if (self.kl_strategy == 2):
      if (self.solver.kdim > 0):
        if (self.eigvals[self.solver.kdim]/self.eigvals[self.solver.kdim-1] > self.hr_ratio_thresh) & (self.eigres[self.solver.kdim] < self.eigres_thresh):
          pass
        else:
          self.solver.W = self.solver.W[:,:self.solver.kdim]
          self.ritz_quotient = self.ritz_quotient[:self.solver.kdim]
          self.eigres = self.eigres[:self.solver.kdim]



    #self.ave_gap_bound = np.zeros(new_kdim)
    #which_w = np.array((new_kdim+1)*[False])
    #sk, bound_gap_frobenius, which_w = self.__pick_new_w(which_w)
    #self.ave_gap_bound[0] = bound_gap_frobenius    
    #for k in range(1, new_kdim):
    #  sk, bound_gap_frobenius, which_w = self.__pick_new_w(which_w, sk=sk)
    #  self.ave_gap_bound[k] = bound_gap_frobenius/(k+1)

    
    #eigres2 = self.eigres**2
    #delta = np.zeros(new_kdim)
    #for k in range(new_kdim):    
    #  diff_ritz_coeff = np.abs(self.ritz_quotient[k]-self.ritz_quotient)
    #  diff_ritz_coeff = diff_ritz_coeff[diff_ritz_coeff  > 0]
    #  delta[k] = min(diff_ritz_coeff)    
    #delta = [min(eigvals[1:k+2]-eigvals[:k+1]) for k in range(new_kdim)]
    #self.ave_gap_bound = np.array([sum(eigres2[:k+1])**.5/delta[k] for k in range(new_kdim)])


    #for k in range(new_kdim):    
    #  print self.eigres[k], delta[k]
    #  diff_ritz_coeff = np.abs(self.ritz_quotient[k]-self.ritz_quotient)
    #  diff_ritz_coeff = diff_ritz_coeff[diff_ritz_coeff  > 0]
    #  self.eigres[k] /= min(diff_ritz_coeff)

    """
    if (self.kl_strategy == 0) | (self.kl_strategy == 1):
      self.solver.W = self.solver.W[:,:new_kdim]
      self.ritz_quotient = self.ritz_quotient[:new_kdim]
      self.eigres = self.eigres[:new_kdim]

    elif (self.kl_strategy == 2):
      which_w = self.eigres < self.eigres_thres
      if (which_w.sum() < 1):
        which_w = self.eigres.argmin()
        self.solver.W = self.solver.W[:,which_w].reshape((self.solver.n,1))
        self.ritz_quotient = np.array([self.ritz_quotient[which_w]])
        self.eigres = np.array([self.eigres[which_w]])
      else:
        self.solver.W = self.solver.W[:,which_w]
        self.ritz_quotient = self.ritz_quotient[which_w]
        self.eigres = self.eigres[which_w]
    """

  def __update_W(self):

    G = np.zeros((self.solver.kdim+self.solver.ell,self.solver.kdim+self.solver.ell))
    F = np.copy(G)
    
    # Compute/get AW and AP for eigenvector approximation
    if (self.which_op == "previous"):
      if (self.solver.kdim > 0):
        AW = self.solver.AW                    # Already computed for last solver call
      if (self.solver.ell > 0):
        AP = self.solver.A.dot(self.solver.P)  # Recyclable from last solver call
    elif (self.which_op == "current"):
      if (self.solver.kdim > 0):
        AW = self.sampler.A.dot(self.solver.W)
      if (self.solver.ell > 0):
        AP = self.sampler.A.dot(self.solver.P)

    if (self.type == "dcgmo"):

      # Build (G, F) for G.hw = theta*F.hw
      if (self.approx == "HR"):
        # Harmonic-Ritz approximation
        # Build F:
        if (self.solver.kdim > 0):
          F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
          if (self.solver.ell > 0):
            if (self.which_op == "current"):
              F[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(AP)
              F[self.solver.kdim:,:self.solver.kdim] = F[:self.solver.kdim,self.solver.kdim:].T
        if (self.solver.ell > 0):
          F[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(AP)
        # Build G:
        if (self.solver.kdim > 0):
          G[:self.solver.kdim,:self.solver.kdim] = AW.T.dot(AW)
          if (self.solver.ell > 0):
            G[:self.solver.kdim,self.solver.kdim:] = AW.T.dot(AP)
            G[self.solver.kdim:,:self.solver.kdim] = G[:self.solver.kdim,self.solver.kdim:].T
        if (self.solver.ell > 0):
          G[self.solver.kdim:,self.solver.kdim:] = AP.T.dot(AP)

      elif (self.approx == "RR"):
        # Rayleigh-Ritz approximation
        # Buid F:
        if (self.solver.kdim > 0): 
          F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(self.solver.W)
          if (self.solver.ell > 0):
            F[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(self.solver.P)
            F[self.solver.kdim:,:self.solver.kdim] = F[:self.solver.kdim,self.solver.kdim:].T
        if (self.solver.ell > 0):
          F[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(self.solver.P)
        # Build G:
        if (self.solver.kdim > 0): 
          G[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
          if (self.solver.ell > 0): 
            if (self.which_op == "current"):
              G[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(AP)
              G[self.solver.kdim:,:self.solver.kdim] = G[:self.solver.kdim,self.solver.kdim:].T
        if (self.solver.ell > 0):
          G[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(AP)

    elif (self.type == "dpcgmo"):

      # Precondition after deflating  
      if (self.dp_seq == "pd"):
        # Build (G, F) for G.hw = theta*F.hw
        if (self.approx == "HR"):
          # Harmonic-Ritz approximation
          # Build F:
          if (self.solver.kdim > 0):
            F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
            if (self.solver.ell > 0):
              if (self.which_op == "current"):
                F[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(AP)
                F[self.solver.kdim:,:self.solver.kdim] = F[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            F[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(AP)
          # Build G:
          if (self.solver.kdim > 0):
            G[:self.solver.kdim,:self.solver.kdim] = AW.T.dot(AW)
            if (self.solver.ell > 0):
              G[:self.solver.kdim,self.solver.kdim:] = AW.T.dot(AP)
              G[self.solver.kdim:,:self.solver.kdim] = G[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            G[self.solver.kdim:,self.solver.kdim:] = AP.T.dot(AP)

        elif (self.approx == "RR"):
          # Rayleigh-Ritz approximation
          # Buid F:
          if (self.solver.kdim > 0): 
            F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(self.solver.W)
            if (self.solver.ell > 0):
              F[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(self.solver.P)
              F[self.solver.kdim:,:self.solver.kdim] = F[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            F[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(self.solver.P)
          # Build G:
          if (self.solver.kdim > 0): 
            G[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
            if (self.solver.ell > 0): 
              if (self.which_op == "current"):
                G[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(AP)
                G[self.solver.kdim:,:self.solver.kdim] = G[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            G[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(AP)

      # Deflate after preconditioning
      elif (self.dp_seq == "dp"):
        # Build (G, F) for G.hw = theta*F.hw
        if (self.approx == "HR"):
          # Harmonic-Ritz approximation (requires invM.AW and invM.AP)
          if (self.solver.kdim > 0):
            invMAW = np.zeros((self.sampler.n, self.solver.kdim))
            for k in range(self.solver.kdim):
              invMAW[:,k] = self.solver.apply_invM(AW[:,k])
          if (self.solver.ell > 0):
            invMAP = np.zeros((self.sampler.n, self.solver.ell))
            for l in range(self.solver.ell):
              invMAP[:,l] = self.solver.apply_invM(AP[:,l])          
          # Build F:
          if (self.solver.kdim > 0):
            F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
            if (self.solver.ell > 0):
              if (self.which_op == "current"):
                F[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(AP)
                F[self.solver.kdim:,:self.solver.kdim] = F[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            F[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(AP)
          # Build G:
          if (self.solver.kdim > 0):
            G[:self.solver.kdim,:self.solver.kdim] = AW.T.dot(invMAW)
            if (self.solver.ell > 0):
              G[:self.solver.kdim,self.solver.kdim:] = AW.T.dot(invMAP)
              G[self.solver.kdim:,:self.solver.kdim] = G[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            G[self.solver.kdim:,self.solver.kdim:] = AP.T.dot(invMAP)

        elif (self.approx == "RR"):
          # Rayleigh-Ritz approximation (requires M.W and M.P, i.e. not invM.W and invM.P)
          if (self.solver.precond_id == 2):
            print('Error: Rayleigh-Ritz approximation not available for dpcgmo-dp with AMG preconditioner')
          else:
            if (self.solver.kdim > 0):
              MW = self.solver.M.dot(self.solver.W) 
            if (self.solver.ell > 0):
              MP = self.solver.M.dot(self.solver.P) 
          # Buid F:
          if (self.solver.kdim > 0): 
            F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(MW)
            if (self.solver.ell > 0):
              F[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(MP)
              F[self.solver.kdim:,:self.solver.kdim] = F[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            F[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(MP)
          # Build G:
          if (self.solver.kdim > 0): 
            G[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
            if (self.solver.ell > 0): 
              if (self.which_op == "current"):
                G[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(AP)
                G[self.solver.kdim:,:self.solver.kdim] = G[:self.solver.kdim,self.solver.kdim:].T
          if (self.solver.ell > 0):
            G[self.solver.kdim:,self.solver.kdim:] = self.solver.P.T.dot(AP)            

    # Solve approximate eigenvalue problem
    new_kdim = self.__get_new_kdim()
    if (self.solver.kdim+self.solver.ell > 0):
      if (self.ini_W) & (self.t_end_def == 0): # W was previously set up with exact eigvecs. No update of W.
        Aeigvecs = self.sampler.A.dot(self.solver.W)
        wk2 = np.array([self.solver.W[:,k].T.dot(self.solver.W[:,k]) for k in range(new_kdim)])
        self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/wk2
        self.eigres = np.array([np.linalg.norm(Aeigvecs[:,k]-self.ritz_quotient[k]*self.solver.W[:,k])/wk2[k]**.5 for k in range(new_kdim)])
        if (self.kl_strategy == 2):
          if (self.solver.kdim > 0):
            if (self.eigvals[self.solver.kdim]/self.eigvals[self.solver.kdim-1] > self.hr_ratio_thresh) & (self.eigres[self.solver.kdim] < self.eigres_thresh):
              pass
            else:
              self.solver.W = self.solver.W[:,:self.solver.kdim]
              self.ritz_quotient = self.ritz_quotient[:self.solver.kdim]
              self.eigres = self.eigres[:self.solver.kdim]
      else:
        self.__approx_eigvecs(G, F, new_kdim)
        self.solver.kdim = self.solver.W.shape[1]
    else:
      if (self.ini_W): # Set up W with exact eigvecs.
        if (self.kl_strategy == 0):
          if (0 < self.ell_min < self.kl):
            new_kdim = self.kl-self.ell_min
          else:
            new_kdim = self.kl/2
        elif (self.kl_strategy == 1):
          new_kdim = 1

        if (self.type == "pcgmo"): 
          eigvals, eigvecs = sparse.linalg.eigsh(self.sampler.get_median_A(), k=new_kdim, which="SM")
          self.solver.W = np.copy(eigvecs)
          self.eigvals = eigvals
          Aeigvecs = self.sampler.A.dot(self.solver.W)
          wk2 = np.array([self.solver.W[:,k].T.dot(self.solver.W[:,k]) for k in range(new_kdim)])
          self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/wk2
          self.eigres = np.array([np.linalg.norm(Aeigvecs[:,k]-self.ritz_quotient[k]*self.solver.W[:,k])/wk2[k]**.5 for k in range(new_kdim)])
          self.solver.kdim = self.solver.W.shape[1]
        elif (self.type == "dpcgmo"):
          if (self.dp_seq == "pd"):
            eigvals, eigvecs = sparse.linalg.eigsh(self.sampler.get_median_A(), k=new_kdim, which="SM")
            self.solver.W = np.copy(eigvecs)
            self.eigvals = eigvals
            Aeigvecs = self.sampler.A.dot(self.solver.W)
            wk2 = np.array([self.solver.W[:,k].T.dot(self.solver.W[:,k]) for k in range(new_kdim)])
            self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/wk2
            self.eigres = np.array([np.linalg.norm(Aeigvecs[:,k]-self.ritz_quotient[k]*self.solver.W[:,k])/wk2[k]**.5 for k in range(new_kdim)])
            self.solver.kdim = self.solver.W.shape[1]
          elif (self.dp_seq == "dp"):
            dA = self.solver.invL_M.dot(self.sampler.get_median_A().dot(self.solver.invL_M.T))
            if (sparse.issparse(dA)):
              eigvals, eigvecs = sparse.linalg.eigsh(dA, k=new_kdim, which="SM")
            else:
              eigvals, eigvecs = scipy.linalg.eigh(dA, eigvals=(0, new_kdim-1))
            self.solver.W = self.solver.invL_M.T.dot(eigvecs)
            self.eigvals = eigvals    
            Aeigvecs = self.sampler.A.dot(self.solver.W)
            dwk2 = np.array([self.solver.W[:,k].T.dot(self.solver.M.dot(self.solver.W[:,k])) for k in range(new_kdim)])
            self.ritz_quotient = np.array([self.solver.W[:,k].T.dot(Aeigvecs[:,k]) for k in range(new_kdim)])/dwk2
            self.eigres = np.array([np.linalg.norm(self.solver.invL_M.dot(Aeigvecs[:,k])-self.ritz_quotient[k]*self.solver.L_M.T.dot(self.solver.W[:,k]))/dwk2[k]**.5 for k in range(new_kdim)])
            self.solver.kdim = self.solver.W.shape[1]

    self.__set_attempted_ell()

  def __get_new_kdim(self):
    if (self.solver.kdim == 0) & (self.solver.ell == 0):
      new_kdim = 0
    else:
      if (self.kl_strategy == 0):
        if (0 < self.ell_min < self.kl):
          new_kdim = min(self.kl-self.ell_min, self.solver.kdim+self.solver.ell)
        else:
          new_kdim = min(self.kl/2, self.solver.kdim+self.solver.ell)
      elif (self.kl_strategy == 1):
        if (self.solver.kdim == 0):
          new_kdim = 1
        else:
          if (self.sampler.type == "mc"):
            if (self.sampler.reals < self.t_end_kl):
              if (self.sampler.reals%self.dt_kl == 0):
                new_kdim = min(min(self.solver.kdim+1, self.solver.kdim+self.solver.ell), self.kl-self.ell_min)
              else:
                new_kdim = self.solver.kdim
            else:
              new_kdim = self.solver.kdim
          elif (self.sampler.type == "mcmc"):
            if (self.sampler.cnt_accepted_proposals < self.t_end_kl):
              if (self.sampler.cnt_accepted_proposals%self.dt_kl == 0):
                new_kdim = min(min(self.solver.kdim+1, self.solver.kdim+self.solver.ell), self.kl-self.ell_min)
              else:
                new_kdim = self.solver.kdim
            else:
              new_kdim = self.solver.kdim
      elif (self.kl_strategy == 2):
        new_kdim = min(self.solver.kdim+1, self.solver.kdim+self.solver.ell)
        # new_kdim may change to new_kdim-1 in __update_W
    return new_kdim

  def __set_attempted_ell(self):
    if (self.sampler.reals == 1):
      self.solver.ell = self.kl
    else:
      self.solver.ell = self.kl-self.solver.kdim

  def prepare(self):
    if (self.type == "pcgmo"):
      if (self.dt > 0) & (self.sampler.reals > 1):
        if (self.sampler.cnt_accepted_proposals%self.dt == 0):
          if (self.solver.precond_id == 1):
            self.solver.set_precond(Mat=self.sampler.A, precond_id=1, application_type=self.solver.application_type)
          elif (self.solver.precond_id == 2):
            self.solver.set_precond(Mat=self.sampler.A, precond_id=2)
          elif (self.solver.precond_id == 3):
            self.solver.set_precond(Mat=self.sampler.A, precond_id=3, nb=self.solver.nb, application_type=self.solver.application_type)

    elif (self.type == "dcgmo") | (self.type == "dpcgmo"):
      if (self.t_end_def == 0):
        self.__update_W()
      else:
        if (self.sampler.type == "mc") & (self.sampler.reals < self.t_end_def):
          self.__update_W()
        elif (self.sampler.type == "mcmc"):
          if (self.sampler.cnt_accepted_proposals < self.t_end_def):
            self.__update_W()

    self.solver.presolve(A=self.sampler.A, b=self.sampler.b, ell=self.solver.ell)

  def solve(self, x0=None):
    if not isinstance(x0, np.ndarray):
      x0 = np.zeros(self.sampler.n)
    self.solver.solve(x0=x0)
    # self.solver.ell was updated here