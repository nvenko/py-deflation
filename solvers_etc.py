from scipy import sparse

def get_invM_bJ(A, nb):
  n = A.shape[0]
  lb = n/nb 
  invM = A.tolil()
  if not (n%nb):
    invM = sparse.block_diag([invM[kb*lb:(kb+1)*lb,kb*lb:(kb+1)*lb] for kb in range(nb)])
  else:
    invM = sparse.block_diag([invM[kb*lb:(kb+1)*lb,kb*lb:(kb+1)*lb] for kb in range(nb)]+[invM[nb*lb:,nb*lb:]])
  invM = sparse.linalg.inv(invM.tocsc())
  return invM

def get_M_bJ(A, nb):
  n = A.shape[0]
  lb = n/nb 
  M = A.tolil()
  if not (n%nb):
    return sparse.block_diag([M[kb*lb:(kb+1)*lb,kb*lb:(kb+1)*lb] for kb in range(nb)])
  return sparse.block_diag([M[kb*lb:(kb+1)*lb,kb*lb:(kb+1)*lb] for kb in range(nb)]+[M[nb*lb:,nb*lb:]])

