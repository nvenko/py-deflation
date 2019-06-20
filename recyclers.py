from solvers import solver
from samplers import sampler

class recycler:
  """ Recycles. """

  def __init__(self, sampler, solver, recycler_type, dt=10, t_end_def=0,
  	           kl=5, kl_strategy=0, dp_seq="pd", which_op="previous", approx="HR"):
  	self.sampler = sampler
  	self.solver = solver
  	self.type = recycler_type
  	
    if (sampler.type == "mcmc") & (solver.type == "pcg") & (self.type == "pcgmo"):
      self.dt = int(dt)
      if (self.dt < 1):
        self.dt = 10
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


  def draw_realization(self):
  	self.sampler.draw_realization()

  def approx_eigvecs(self, y):
  	self.solver 

  def update_W(self):
  	if (self.type == "dcgmo"):
  	  self.solver.W = None

  	elif (self.type == "dpcgmo"):
      if (seq == "pd"):
  	    self.solver.W = None
  	  elif (seq == "dp"):
  	  	self.W = None

  def prepare(self):
  	if (self.type == "pcgmo"):
      if (self.sampler.cnt_accepted_proposals%self.dt == 0):
        self.solver.set_precond()

  	elif (self.type == "dcgmo"):
  	  if not (self.t_end_def):
  	    self.update_W()
  	  else:
  	    if (self.sampler.type == "mc") & (self.sampler.reals <= self.t_end_def):
          self.update_W()
  	    elif (self.sampler.type == "mcmc") & (self.sampler.cnt_accepted_proposals <= self.t_end_def):
          self.update_W()

  	elif (self.type == "dpcgmo"):
  	  if not (self.t_end_def):
  	    self.update_W()
  	  else:
  	    if (self.sampler.type == "mc") & (self.sampler.reals <= self.t_end_def):
          self.update_W()
  	    elif (self.sampler.type == "mcmc") & (self.sampler.cnt_accepted_proposals <= self.t_end_def):
          self.update_W()

  def solve(self):
  	if (self.type == "pcgmo"):
  	  self.solver.A = self.sampler.A
  	  self.solver.pcg()
  	elif (self.type == "dcgmo"):
  	  self.solver.dpcg()
  	elif (self.type == "dpcgmo"):
      self.solver.dpcg()



