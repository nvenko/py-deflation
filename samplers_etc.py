import numpy as np
import scipy

def CovExp(dx, L, sig2):
  return sig2*np.exp(-np.abs(dx)/L)

def CovSExp(dx, L, sig2):
  return sig2*np.exp(-(dx/L)**2)

def get_fname_of_KL():
  return None

def load_KL():
  return None



def old_load_KL(nEl,nKL,sig2,ell,xa=0.,xb=1.):
  h=(xb-xa)/nEl
  root_KL=get_root_KL(nEl,nKL,sig2,ell)
  fname=glob.glob(data_path+'kappa'+root_KL+'.KL.npy')
  if len(fname)>0:
    fname=fname[0]
    tmp=np.load(fname)
    Eigvals=tmp.item()['Eigvals']
    Eigvecs=tmp.item()['Eigvecs']
    C=tmp.item()['C']
    print('Loaded %s' %fname)
  else:
    print('\n##### Compute KL representation of kappa #####')
    print('nEL=%g, nKL=%g, sig2=%g, ell=%g' %(nEl,nKL,sig2,ell))
    # Initialize FE discretization 
    xDOF=np.linspace(xa,xb,nEl+1)
    x=np.linspace(xa+h/2.,xb-h/2.,nEl)  
    # Set covariance structure of coefficient field
    C=np.zeros((nEl,nEl))
    for k in range(nEl):
      C[k,:]=range(-k,nEl-k)
    C=CovSExp(h*C,ell,sig2)
    #C=CovExp(h*C,ell,sig2)
    C=sparse.coo_matrix(C)
    W=h*sparse.eye(nEl)
    # Resolve KL representation of coefficient field
    t0=time.time()
    #Eigvals,Eigvecs=np.linalg.eigh(C.dot(W),eigvals=(nEl-nKL,nEl-1)) # if C, W are np.ndarray
    Eigvals,Eigvecs=sparse.linalg.eigsh(C.dot(W),k=nKL,which='LM') # if C, W are Scipy sparse matrices
    t0=time.time()-t0
    EigvalsInvSqrt=Eigvals.real**-.5
    CdotEigvecs=h**.5*C.dot(Eigvecs.real)
    sumEigvals=(Eigvals.real).sum()
    print('Truncature error: %g fraction of variance' %(1-sumEigvals/sig2))
    # Save data into file
    tmp={}
    tmp['Eigvals']=Eigvals
    tmp['Eigvecs']=Eigvecs
    tmp['C']=C
    fname=data_path+'kappa'+root_KL+'.KL'
    np.save(fname,tmp)
    print('Saved in %s.npy' %fname)
    print('Done in %g sec' %(t0))
  return Eigvals.real, Eigvecs.real, C


def load_sim_kappa(nEl,nKL,sig2,ell,smp,nu,xa=0.,xb=1.):
  # Set (nr_MC_store << nr_MC) and (MC_stats=True) to plot only few realizations along with converged statistics
  #nr_MC=nu; nr_MC_store=nu; MC_stats=False; MC_plot=False; MC_gif=False
  #nr_MC=10; nr_MC_store=10; MC_stats=False; MC_gif=False
  _stats=False; _plot=False; _gif=False

  vsig2_opt=2.38**2/nKL # see Rosenthal (2010)
  #vsig2_opt=.6**2/nKL #
  #vsig2_opt=.4**2/nKL # 
  vsig2=1.*vsig2_opt
  comp_reals=True; s0=0
  root_sim_kappa=get_root_sim_kappa(nEl,nKL,sig2,ell,smp)
  fname=np.array(glob.glob(data_path+'kappa'+root_sim_kappa+'.sim*.npy'))
  if len(fname)>0:
    lnu=np.array([int(ifname[ifname.find('.sim')+4:].split('.')[0]) for ifname in fname],dtype=int)
    if np.any(nu<=lnu):
      fname=fname[nu<=lnu]
      lnu=lnu[nu<=lnu]
      fname=fname[np.argmin(lnu)]
      kappa=np.load(fname)
      print('Loaded %s' %fname)
      comp_reals=False
    else:
      s0=max(lnu)

  if comp_reals:
    #modes=[1,10,nKL]
    #plot_KL_modes(Eigvecs,Eigvals,sig2,modes,fname+'_modes')    
    # Sample KL representation by standard MC

    Eigvals,Eigvecs,C=load_KL(nEl,nKL,sig2,ell)

    np.random.seed(12345678)
    print('\n##### Sample realizations of kappa #####')
    print('nEL=%g, nKL=%g, sig2=%g, ell=%g' %(nEl,nKL,sig2,ell))
    print('smp=%s, nu=%g' %(smp,nu))

    kappa=np.zeros((nu,nEl))

    h=(xb-xa)/nEl
    EigvalsInvSqrt=Eigvals.real**-.5
    CdotEigvecs=h**.5*C.dot(Eigvecs.real)  

    t0=time.time(); task_bar(0,nu,t0)
    if (_stats) & (smp=='MC'):
        nCov=nEl/3
        Cov_MC_mean=np.zeros(nCov)
        Cov_lags=h*np.arange(nCov)
        Cov_x_MC=np.zeros((nu,nCov))
    if smp=='MCMC':
      xi0=np.random.normal(size=nKL)
      kappa[0,:]=np.exp(CdotEigvecs.dot(EigvalsInvSqrt*xi0))
      #kappa[0,:]=np.exp(Eigvecs.dot(Eigvals**.5*xi0))
      xi0txi0=xi0.dot(xi0)
      s=0;rej_cnt=0   
    if smp=='MC':
      for ir in range(nu):
        x_ir=CdotEigvecs.dot(EigvalsInvSqrt*np.random.normal(size=nKL))
        #x_ir=Eigvecs.real.dot(Eigvals**.5*np.random.normal(size=nKL))
        if (_stats) & (smp=='MC'):
          FFT_x=np.fft.fft(x_ir)
          Cov_x=np.real((np.fft.ifft(FFT_x*np.conjugate(FFT_x))/nEl)[:nCov])
          Cov_MC_mean+=Cov_x/nu
          Cov_x_MC[ir,:]=Cov_x
        kappa[ir,:]=np.exp(x_ir)
        task_bar(ir,nu,t0)
    elif smp=='MCMC':
      while (s<nu):
        xis=xi0+np.random.normal(scale=vsig2**.5,size=nKL) # scale is std
        xistxis=xis.dot(xis)
        alpha=min(np.exp((xi0txi0-xistxis)/2.),1)
        if (np.random.uniform()<alpha):
          s+=1
          x_MCMC_s=CdotEigvecs.dot(EigvalsInvSqrt*xis)
          #x_MCMC_s=Eigvecs.dot(Eigvals**.5*xis)
          if (s<nu):
            kappa[s,:]=np.exp(x_MCMC_s)
            task_bar(s,nu,t0)
          xi0=np.copy(xis)
          xi0txi0=xistxis
        else:
          rej_cnt+=1

    if smp=='MCMC':
      print('Acceptance rate: %g' %(nu/(nu+1.*rej_cnt)))

    if _plot:
      str_var='\kappa'; str_met='MC'; ylim=(0,3)
      fname_smp=plot_smp(x,kappa,nr_MC_store,nu,nKL,ell,sig2,vsig2,xa,xb,str_var,str_met,fname_MC+'_kappa',ylim)
      fname_smp=plot_smp(x,kappa_MCMC,nr_MCMC_store,nu,nKL,ell,sig2,vsig2,xa,xb,str_var,str_met,fname_MCMC+'_kappa',ylim)
      print('%s'%(fname_smp))
    if _gif:
      fname_gif=make_smp_gif(x,kappa,nr_MC_store,nu,nKL,ell,sig2,vsig2,xa,xb,str_var,str_met,fname_MC+'_kappa',ylim)
      fname_gif=make_smp_gif(x,kappa_MCMC,nr_MCMC_store,nu,nKL,ell,sig2,vsig2,xa,xb,str_var,str_met,fname_MCMC+'_kappa',ylim)
      print('%s'%(fname_gif))
    if (_stats) & (smp=='MC'):
      fname_cov=plot_x_MC_cov(Cov_lags,Cov_x_MC,Cov_MC_mean,nCov,nr_MC,nr_MC_store,nKL,h,ell,sig2,fname_MC+'_kappa')
      print('%s'%(fname_cov))

    fname=data_path+'kappa'+root_sim_kappa+'.sim%g'%nu 
    np.save(fname,kappa)
    print('Saved in %s.npy' %fname)
    if s0>0:
      fname=data_path+'kappa'+root_sim_kappa+'.sim%g.npy'%s0
      os.remove(fname)
      print('Removed %s'%fname)

  return kappa


def load_sim_A(nEl,nKL,sig2,ell,smp,nu,BCtype='Neumann',xa=0.,xb=1.):
  comp_A=True; s0=0
  root_sim_A=get_root_sim_A(nEl,nKL,sig2,ell,smp,BCtype)
  fname=np.array(glob.glob(data_path+'A'+root_sim_A+'.sim*.npy'))
  if len(fname)>0:
    lnu=np.array([int(ifname[ifname.find('.sim')+4:].split('.')[0]) for ifname in fname],dtype=int)
    if np.any(nu<=lnu):
      fname=fname[nu<=lnu]
      lnu=lnu[nu<=lnu]
      fname=fname[np.argmin(lnu)]
      A=scipy.load(fname)
      print('Loaded %s' %fname)
      comp_A=False
    else:
      s0=max(lnu)
  if comp_A:  
    #modes=[1,10,nKL]
    #plot_KL_modes(Eigvecs,Eigvals,sig2,modes,fname+'_modes')    
    # Sample KL representation by standard MC
    print('\n##### Assemble realizations of A #####')
    print('nEL=%g, nKL=%g, sig2=%g, ell=%g' %(nEl,nKL,sig2,ell))
    print('smp=%s, nu=%g' %(smp,nu))
    print('BCtype=%s' %(BCtype))
    kappa=load_sim_kappa(nEl,nKL,sig2,ell,smp,nu)
    h=(xb-xa)/nEl
    A=[]
    for ir in range(nu):
      A+=[get_A(nEl,h,BCtype,kappa[ir,:])]
    fname=data_path+'A'+root_sim_A+'.sim%g'%nu
    scipy.save(fname,A)
    print('Saved in %s.npy' %fname)
    if s0>0:
      fname=data_path+'A'+root_sim_A+'.sim%g.npy'%s0
      os.remove(fname)
      print('Removed %s'%fname) 
  return A[:nu]

def get_b(nEl,nKL,sig2,ell,smp,nu,du=0,u=None,xa=0.,xb=1.):
  # du==0    & u==None: Homogeneous Neumann RHS
  # du!=0    & u==None: Non-homogeneous Neumann RHS
  # du==None & u==0:    Homogeneous Dirichlet RHS
  # du==None & u!=0:    Non-homogeneous Dirichlet RHS
  #
  h=(xb-xa)/nEl
  M=h/6.*sparse.eye(nEl+1)
  f=np.ones(nEl+1)
  f[0];f[-1]=0.
  b=M.dot(f)
  #
  # By default, we have homogeneous Dirichlet LHS
  b=b[1:] # len(b)=nEl 
  #
  # Neumann RHS
  if (type(du)!=type(None)) & (type(u)==type(None)):
    # shape(A)=(nEl,nEl), len(b)=nEl
    if du==0:
      return b
    else:
      b=nu*[b]
      b=np.transpose(b)
      kappa=load_sim_kappa(nEl,nKL,sig2,ell,smp,nu)
      for r in range(nu):
        b[-1,r]=+du*kappa[r][-1]
      return b
  #
  # Dirichlet RHS
  elif (type(du)==type(None)) & (type(u)!=type(None)):
    # shape(A)=(nEl-1,nEl-1), len(b)=nEl-1
    if u==0:
      return b[:-1]
    else:
      b=nu*[b[:-1]]
      b=np.transpose(b)
      kappa=load_sim_kappa(nEl,nKL,sig2,ell,smp,nu)
      A_i_end=np.zeros(nEl-1)
      for r in range(nu):
        A_i_end[-1]=-kappa[r][-1]/h
        b[:,r]-=A_i_end*u
      return b
  else:
    print '\nError: RHS BC not properly prescribed.\n'

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

def get_A(nEl,h,BCtype,kappa):
  Ann=np.zeros(nEl+1)
  Ann[:nEl]+=kappa
  Ann[1:nEl+1]+=kappa
  if BCtype=='Neumann':
    return h**-1*sparse.diags([-kappa[1:],Ann[1:],-kappa[1:]],[-1,0,1])
  elif BCtype=='Dirichlet':
    return h**-1*sparse.diags([-kappa[1:-1],Ann[1:-1],-kappa[1:-1]],[-1,0,1])
  print '\nError: RHS BC not properly prescribed.\n'
  return 1