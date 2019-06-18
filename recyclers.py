from solvers import solver
from samplers import sampler

class recycler:
  """ Recycles. """

  def __init__(self, sampler, solver, recycler_type="dcgmo"):
  	self.sampler = sampler
  	self.solver = solver
  	self.type = recycler_type
  	# check if recycler_type is compatible with solver.type
  	
  def approx_eigvecs(self, y):
  	self.solver 

