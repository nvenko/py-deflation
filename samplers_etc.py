import numpy as np
import scipy
import glob

def CovExp(dx, L, sig2):
  return sig2*np.exp(-np.abs(dx)/L)

def CovSExp(dx, L, sig2):
  return sig2*np.exp(-(dx/L)**2)

def get_root_KL(nEl, model, sig2, L, delta2):
  root = 'nEl%g_Cov%s_sig2%s_L%s_delta2%g' %(nEl, model, sig2, L, delta2)
  return root

def get_fname_of_KL(data_path, nEl, model, sig2, L, delta2):
  root = get_root_KL(nEl, model, sig2, L, delta2)
  fname = data_path+root+'.KL.npy'
  files = glob.glob(fname)
  if (len(files) > 0):
    return files[0], True
  return fname, False

def load_KL(fname):
  KL = np.load(fname)
  return KL.item()

def get_x0(nEl,h,BCtype,x0=None):
  if type(x0)==type(None):
    x0=np.zeros(nEl+1)
  if BCtype=='Neumann':
    return x0[1:]
  elif BCtype=='Dirichlet':
    return x0[1:-1]
  else:
    print '\nError: RHS BC not properly prescribed.\n'
    return 1