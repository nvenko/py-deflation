import numpy as np

class solver:
  """ Solves. """

  def __init__(self, solver_type="cg", eps=1e-7):
    self.type = solver_type
    self.eps = eps
    self.P = None
    self.W = None
    self.A = None
    self.b = None
    self.x0 = None
    self.x = None
    self.xstar = None
    self.bnorm = 0
    self.iterated_res_norm = None
    self.error_Anorm = None
    self.it = 0
    self.ell = 0

  def set_precond(self):
  	return

  def apply_invM(self):
  	return

  def cg(self, Error=False):
    self.it = 0
    self.x = np.copy(self.x0)
    r = self.b-self.A.dot(self.x)
    rTr = r.dot(r)
    p = np.copy(r)
    self.iterated_res_norm = [np.sqrt(rtr)]
    if (Error):
      err = self.x-self.xstar
      self.error_Anorm += [np.sqrt(err.dot(A.dot(err)))]
    while (self.it < self.itmax) & (self.iterated_res_norm[-1] > self.eps*self.bnorm):
      Ap = self.A.dot(p)
      d = Ap.dot(p)
      alpha = rTr/d
      beta = 1./rTr
      self.x += alpha*p
      r -= alpha*Ap
      if (Error):
  	    err = self.x-self.xstar
  	    self.error_Anorm += [np.sqrt(err.dot(A.dot(err)))]
      rTr = r.dot(r)
      beta *= rTr
      if (0 < self.it < self.ell): 
        self.P[:,self.it] = p
      p = r+beta*p
      self.iterated_res_norm += [np.sqrt(rTr)]
      self.it += 1
    if (0 < self.it < self.ell):
      self.P = self.P[:,:self.it]

  def pcg(self):
  	self.apply_invM()
  	return

  def dcg(self):
  	return

  def dpcg(self):
  	return