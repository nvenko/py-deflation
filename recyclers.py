from solvers import solver
from samplers import sampler
import numpy as np

class recycler:
  """ Recycles. """

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

    elif (solver.type == "cg") & (self.type == "dcgmo"):
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

  def approx_eigvecs(self, G, F):
    if (self.approx == "HR"):
      eigvecs, _ = scipy.linalg.eigsh(G, F, self.solver.kdim)
    elif (self.approx == "RR"):
      eigvecs, _ = scipy.linalg.eigsh(G, F, self.solver.kdim)
    self.solver.W = self.solver.W.dot(eigvecs)

  def update_W(self):
    G = np.zeros((self.solver.kdim+self.solver.ell,self.solver.kdim+self.solver.ell))
    F = np.copy(G)

    if (self.type == "dcgmo"):
      # Compute/get AW and AP for eigenvector approximation
      if (self.which_op == "previous"):
        #AW = self.solver.A.dot(self.solver.W)
        if (self.solver.kdim):
          AW = self.solver.AW                    # Computed for last solver call
        if (self.solver.ell):
          AP = self.solver.A.dot(self.solver.P)  # Recyclable from last solver call
      elif (self.which_op == "current"):
        if (self.solver.kdim):
          AW = self.sampler.A.dot(self.solver.W)
        if (self.solver.ell):
          AP = self.sampler.A.dot(self.solver.P)

      # Build (G, F) for G.hw = theta*F.hw
      if (self.approx == "HR"):
        # Harmonic-Ritz approximation
        # Build F:
        if (self.solver.kdim):
          F[:self.solver.kdim,:self.solver.kdim] = self.W.T.dot(AW)
          if (self.solver.ell):
            F[self.solver.kdim:,self.solver.kdim:] = self.W.T.dot(AP)
            if (self.which_op == "current"):
              F[:self.kdim,self.kdim:] = self.W.T.dot(AP)
              F[self.kdim:,:self.kdim] = F[:self.kdim,self.kdim:].T
        if (self.solver.ell):
          F[self.solver.kdim:,self.solver.kdim:] = self.P.T.dot(AP)
        # Build G:
        if (self.solver.kdim):
          G[:self.kdim,:self.kdim] = AW.T.dot(AW)
          if (self.solver.ell):
            G[:self.kdim,self.kdim:] = AW.T.dot(AP)
            G[self.kdim:,:self.kdim] = G[:self.kdim,self.kdim:].T
        if (self.solver.ell):
          G[self.kdim:,self.kdim:] = AP.T.dot(AP)

      elif (self.approx == "RR"):
        # Rayleigh-Ritz approximation
        F[:self.solver.kdim,:self.solver.kdim] = self.W.T.dot(AW)
        F[self.solver.kdim:,self.solver.kdim:] = self.W.T.dot(AP)
        if (self.which_op == "current"):
          F[:self.kdim,self.kdim:] = self.W.T.dot(AP)
          F[self.kdim:,:self.kdim] = F[:self.kdim,self.kdim:].T
        G[:self.kdim,:self.kdim] = AW.T.dot(AW)
        G[:self.kdim,self.kdim:] = AW.T.dot(AP)
        G[self.kdim:,:self.kdim] = G[:self.kdim,self.kdim:].T
        G[self.kdim:,self.kdim:] = AP.T.dot(AP)

      # Solve approximate eigenvalue problem
      self.set_kdim()
      if (self.solver.kdim+self.solver.ell > 0):
        self.approx_eigvecs(G, F)
      self.set_attempted_ell()

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
      self.approx_eigvecs(G, F)
      self.set_attempted_ell()

  def set_kdim(self):
    if (self.solver.kdim == 0) & (self.solver.ell == 0):
      self.solver.kdim = 0
    else:
      if (self.kl_strategy == 0):
        self.solver.kdim = min(self.kl/2, self.solver.kdim+self.solver.ell)

  def set_attempted_ell(self):
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
        self.update_W()
      else:
        if (self.sampler.type == "mc") & (self.sampler.reals < self.t_end_def):
          self.update_W()
        elif (self.sampler.type == "mcmc") & (self.sampler.cnt_accepted_proposals < self.t_end_def):
          self.update_W()
    
    self.solver.A = self.sampler.A

  def solve(self):
    x0 = np.zeros(self.sampler.n)
    self.solver.solve(self.sampler.A, self.sampler.b, x0)
    # self.solver.ell was updated here