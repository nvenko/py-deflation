import numpy as np
import scipy
from scipy import sparse
import scipy.sparse.linalg
import samplers_etc
import time

class sampler:
  """ Assembles sampled operators in a sequence {A(theta_t)}_{t=1}^M for the 
      stochastic system A(theta).u(theta) = b of a P0-FE discretization of the SDE 
      d[kappa(x;theta)d(u)/dx(x;theta)]/dx = -f(x) in which the coefficient field 
      kappa(x;theta) is stationary lognormal. """

  xa, xb = 0., 1.

  def __init__(self, nEl=500, smp_type="mc", model="SExp", sig2=1, mu=0, L=0.1, vsig2=None, delta=1e-3, seed=123456789, verb=1):
  	self.nEl = int(abs(nEl))
  	if (self.nEl <= 0):
  	  self.nEl = 500
  	self.type = str(smp_type)
  	if (self.type not in ("mc", "mcmc")):
  	  self.type = "mc"
  	self.model = str(model)
  	if (self.model not in ("SExp", "Exp")):
  	  self.model = "SExp"
  	if (self.model == "Exp"):
  	  self.eval_cov = samplers_etc.CovExp
  	else:
  	  self.eval_cov = samplers_etc.CovSExp
  	self.sig2 = float(abs(sig2))
  	if (self.sig2 <= 0):
  	  self.sig2 = 1.
  	self.mu = float(mu)
  	self.L = float(abs(L))
  	if (self.L <= 0):
  	  self.L = 0.1
  	self.delta = float(delta)
  	if not (True):
  	  self.delta = 1e-3
  	self.seed = int(seed)
  	if (self.type == "mcmc"):
  	  if (vsig2 != None):  	
  	  	self.vsig2 = float(abs(vsig2))
  	  	if (self.vsig2 <= 0):
  	  	  self.vsig2 = None
  	self.verb = int(verb)
  	if (self.verb not in (0, 1, 2)):
  	  self.verb = 1

  	self.h = (self.xb-self.xa)/self.nEl
  	self.KL = None
  	self.nKL = nEl/3
  	self.kappa = None

  	if (self.verb):
  	  print("Sampler created.")
    
  def compute_KL(self):
  	if (self.KL):
  	  pass
  	else:
  	  KL_fname = samplers_etc.get_fname_of_KL()
  	  if (KL_fname):
  	  	self.KL = samplers_etc.load_KL()
  	  else:
  	  	K_mat = np.zeros((self.nEl, self.nEl))
        for k in range(self.nEl):
          K_mat[k,:] = range(-k, self.nEl-k)
        K_mat = self.eval_cov(self.h*K_mat, self.L, self.sig2)
        K_mat = self.h**2*sparse.csc_matrix(K_mat)
        M_mat = sparse.csc_matrix(self.h*sparse.eye(self.nEl))
        Eigvals, Eigvecs = sparse.linalg.eigsh(K_mat, M=M_mat, k=self.nKL, which='LM') 
        energy = np.array([Eigvals[self.nKL-i-1:].sum() for i in range(self.nKL)])
        N = sum((1.-self.delta**2/2.)*self.sig2 > energy) + 1
        if (N<self.nKL):
          self.nKL = N
        else:
          print('Need to compute more eigenpairs. Truncature error: %g fraction of variance' %(1-energy[-1]/self.sig2))
        self.KL = {"vals":Eigvals[-self.nKL:], "vecs":Eigvecs[:, -self.nKL:], "nKL":self.nKL}

        #fname=data_path+'kappa'+root_KL+'.KL'
        #np.save(fname,tmp)
        #print('Saved in %s.npy' %fname)
        

  def draw_realization(self):
  	if not self.KL:
  	  self.compute_KL()
  	xi = np.random.normal(size=self.KL["nKL"])
  	self.kappa = self.KL["vecs"].dot(self.KL["vals"]**.5*xi)

  def get_kappa(self):
  	return self.kappa
