from solvers import solver
from samplers import sampler
import numpy as np
import scipy

class recycler:
  """ Recycles. 

      Public methods:
        do_assembly, prepare, solve.
  """
  def __init__(self, sampler, solver, recycler_type, dt=0, t_end_def=0,
  	       kl=5, kl_strategy=0, dp_seq="pd", which_op="previous", approx="HR"):
    self.sampler = sampler
    self.solver = solver
    self.type = recycler_type

    if (sampler.type == "mcmc") & (solver.type == "pcg") & (self.type == "pcgmo"):
      self.dt = int(dt)
      if (self.dt < 0):
  	    self.dt = 0
      return

    elif (solver.type == "dcg") & (self.type == "dcgmo"):
      self.t_end_def = int(t_end_def)
      if (self.t_end_def < 0):
  	    self.t_end_def = 0
      self.kl = int(kl)
      if (self.kl < 1) | (self.kl >= sampler.n):
        self.kl = min(5, sampler.n-1)
      self.kl_strategy = int(kl_strategy)
      if self.kl_strategy not in (0, 1):
        self.kl_strategy = 0
      self.which_op = str(which_op)
      if self.which_op not in ("previous", "current"):
        self.which_op = "previous"
      self.approx = str(approx)
      if self.approx not in ("HR", "RR"):
  	    self.approx = "HR"
      return

    elif (solver.type == "dpcg") & (self.type == "dpcgmo"):
      self.t_end_def = int(t_end_def)
      if (self.t_end_def < 0):
  	    self.t_end_def = 0
      self.kl = int(kl)
      if (self.kl < 1) | (self.kl >= sampler.n):
        self.kl = min(5, sampler.n-1)
      self.kl_strategy = int(kl_strategy)
      if self.kl_strategy not in (0, 1):
        self.kl_strategy = 0
      self.dp_seq = str(dp_seq)
      if self.dp_seq not in ("pd", "dp"):
  	    self.dp_seq = "pd"
      self.which_op = str(which_op)
      if self.which_op not in ("previous", "current"):
        self.which_op = "previous"
      self.approx = str(approx)
      if self.approx not in ("HR", "RR"):
  	    self.approx = "HR"
      return

    else:
      print 'Error: recycler_type not compatible with solver and/or sampler.'

  def do_assembly(self):
    if (self.sampler.type == "mc"):
      self.sampler.do_assembly()
      self.solver.A = self.sampler.A
    elif (self.sampler.type == "mcmc"):
      if (self.sampler.proposal_accepted):
        self.sampler.do_assembly()
        self.solver.A = self.sampler.A

  def __approx_eigvecs(self, G, F, new_kdim):
    if (self.approx == "HR"):
      _, eigvecs = scipy.linalg.eigh(G, F, eigvals=(0, new_kdim-1))
    elif (self.approx == "RR"):
      _, eigvecs = scipy.linalg.eigh(G, F, eigvals=(self.solver.kdim+self.solver.ell-new_kdim, self.solver.kdim+self.solver.ell-1))
    
    if (self.solver.kdim > 0):
      self.solver.W = self.solver.W.dot(eigvecs[:self.solver.kdim,:]) \
                      + self.solver.P.dot(eigvecs[self.solver.kdim:,:])
    else:
      self.solver.W = self.solver.P.dot(eigvecs)

  def __update_W(self):
    G = np.zeros((self.solver.kdim+self.solver.ell,self.solver.kdim+self.solver.ell))
    F = np.copy(G)

    if (self.type == "dcgmo"):
      # Compute/get AW and AP for eigenvector approximation
      if (self.which_op == "previous"):
        #AW = self.solver.A.dot(self.solver.W)
        if (self.solver.kdim > 0):
          AW = self.solver.AW                    # Computed for last solver call
        if (self.solver.ell > 0):
          AP = self.solver.A.dot(self.solver.P)  # Recyclable from last solver call
      elif (self.which_op == "current"):
        if (self.solver.kdim > 0):
          AW = self.sampler.A.dot(self.solver.W)
        if (self.solver.ell > 0):
          AP = self.sampler.A.dot(self.solver.P)

      # Build (G, F) for G.hw = theta*F.hw
      if (self.approx == "HR"):
        # Harmonic-Ritz approximation
        # Build F:
        if (self.solver.kdim > 0):
          F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
          if (self.solver.ell > 0):
            F[self.solver.kdim:,self.solver.kdim:] = self.solver.W.T.dot(AP)
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
        F[:self.solver.kdim,:self.solver.kdim] = self.solver.W.T.dot(AW)
        F[self.solver.kdim:,self.solver.kdim:] = self.solver.W.T.dot(AP)
        if (self.which_op == "current"):
          F[:self.solver.kdim,self.solver.kdim:] = self.solver.W.T.dot(AP)
          F[self.solver.kdim:,:self.solver.kdim] = F[:self.solver.kdim,self.solver.kdim:].T
        G[:self.solver.kdim,:self.solver.kdim] = AW.T.dot(AW)
        G[:self.solver.kdim,self.solver.kdim:] = AW.T.dot(AP)
        G[self.solver.kdim:,:self.solver.kdim] = G[:self.solver.kdim,self.solver.kdim:].T
        G[self.solver.kdim:,self.solver.kdim:] = AP.T.dot(AP)

      # Solve approximate eigenvalue problem
      new_kdim = self.__get_new_kdim()
      if (self.solver.kdim+self.solver.ell > 0):
        self.__approx_eigvecs(G, F, new_kdim)
        self.solver.kdim = self.solver.W.shape[1]
      self.__set_attempted_ell()

    elif (self.type == "dpcgmo"):
      if (self.which_op == "previous"):
        if (seq == "pd"):
          self.G = None
          self.F = None
        elif (seq == "pd"):
          self.G = None
          self.F = None
      elif (self.which_op == "current"):
        if (seq == "pd"):
          self.G = None
          self.F = None
        elif (seq == "pd"):
          self.G = None
          self.F = None
      self.set_kdim()
      self.__approx_eigvecs(G, F)
      self.__set_attempted_ell()

  def __get_new_kdim(self):
    if (self.solver.kdim == 0) & (self.solver.ell == 0):
      new_kdim = 0
    else:
      if (self.kl_strategy == 0):
        new_kdim = min(self.kl/2, self.solver.kdim+self.solver.ell)
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
      if not (self.t_end_def):
        self.__update_W()
      else:
        if (self.sampler.type == "mc") & (self.sampler.reals < self.t_end_def):
          self.__update_W()
        elif (self.sampler.type == "mcmc") & (self.sampler.cnt_accepted_proposals < self.t_end_def):
          self.__update_W()
    
    self.solver.presolve(A=self.sampler.A, b=self.sampler.b, ell=self.solver.ell)

  def solve(self, x0=None):
    if (type(x0) == type(None)):
      x0 = np.zeros(self.sampler.n)
    self.solver.solve(x0=x0)
    # self.solver.ell was updated here