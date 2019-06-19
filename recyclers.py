from solvers import solver
from samplers import sampler

class recycler:
  """ Recycles. """

  def __init__(self, sampler, solver, recycler_type="pcgmo"):
  	self.sampler = sampler
  	self.solver = solver
  	self.type = recycler_type
  	# check if recycler_type is compatible with solver.type

  def draw_realization(self):
  	self.sampler.draw_realization()

  def approx_eigvecs(self, y):
  	self.solver 

  def prepare(self):
  	if (self.type == "pcgmo"):
  	  # See if preconditioner need be updated
  	elif (self.type == "dcgmo"):
  	  # update deflation subspace
  	elif (self.type == "dpcgmo"):
  	  # update deflation subspace


  def solve(self):
  	if (self.type == "pcgmo"):
  	  self.solver.A = self.sampler.A
  	  self.solver.pcg()
  	elif (self.type == "dcgmo"):
  	  self.solver.dpcg()
  	elif (self.type == "dpcgmo"):
      self.solver.dpcg()



