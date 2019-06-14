# deflation-precond-strategies-sde

#### Enables testing and applications of deflation and preconditioning strategies to solve sequences of sampled finite element (FE) discretization of stochastic differential equations (SDE).

Author: Nicolas Venkovic

email: [venkovic@cerfacs.fr](mailto:venkovic@cerfacs.fr)

_TeX expressions rendered by [TeXify](https://github.com/apps/texify)._

### Dependencies:

 - *Python* (2.x >= 2.6)
 - *SciPy* (>= 0.10)
 - *NumPy* (>= 1.6)

### File contents:

- _samplers.py_ : 

  Assembles sampled operator $\mathbf{A}(\theta)$  for the stochastic system $\mathbf{A}(\theta)\mathbf{u}(\theta)=\mathbf{b}$ coming from a P0-FE discretization of the SDE $\partial_x[\kappa(x;\theta)\partial_xu(x;\theta)]=-f(x)$. 

  Samplers of Karhunen-Loève (KL) representation of lognormal coefficient field $\kappa(x;\theta)$ :

  - Monte Carlo (MC)
  - Markov chain Monte Carlo (MCMC)

- _solvers.py_ : 

  Iterative solvers: 

  - Conjugate gradient (CG)
  - Preconditioned CG (PCG)
  - Deflated CG (DCG)
  - Preconditioned DCG (PDCG)

- _recycling.py_ : 

  Interfaces sampler and solver while handling W, P, A, exact and approximate eigenvectors after different strategies.

  Solvers for sequence of linear systems $\mathbf{A}(\theta_t)\mathbf{u}(\theta_t)=\mathbf{b}$  with $\{\mathbf{A}(\theta_t)\}_{t=1}^M$ and median operator $\hat{\mathbf{A}}$.

  - PCG for a sequence of systems with multiple operators (PCGMO) :

    - Preconditioner ID (none: 0, constant: 1-3, realization dependent: 4):

      (1) Median operator

      (2) Algebraic multi-grid (AMG) based on median operator

      (3) Block Jacobi (bJ) based on median operator with _nb_ (non-overlapping) blocks

      (4) Block Jacobi (bJ) based on periodically selected operator in sampled sequence with _nb_ (non-overlapping) blocks

  - DCG for a sequence with multiple operators (DCGMO) :

    _Rank of deflation subspace: _k_, Dimension of recycled Krylov subspace : _ell_

    - (_k_, _ell_)-strategy:

      (1) First strategy

      (2) Second strategy

    - Current/previous

    - Stop updating

    - HR vs RR

  - DPCG for a sequence with multiple operators (DPCGMO) :

    - Sequence (PD: 1, DP: 2)

      (1) PD is preconditioning after deflating

      (2) DP is deflating after preconditioning

    - Preconditioner ID

    - (_k_, _ell_)-strategy

    - Current/Previous

    - HR vs RR

    - Stop updating

- _post-recycling.py_ :

  - Plots results

### Usage:

```bash
./dist/GNU-Linux/npcf nx ny x0 y0 verb data.in
```

data.in: 2D csv data file.

nx, ny: Size of the 2D sub-slice of data to analyze.

x0, y0: upper left starting point of sub-slice in data file.

verb: Controls display settings.

 - 0: No output.
 - 1: Error messages only.
- 2: Error messages / 2D slice of data / Comparison of S2 and S3 results.

#### Example:

```bash
./dist/GNU-Linux/npcf 30 30 0 0 2 im00.csv

s2(0,0) = 0.668750, s2(1,0) = 0.626250, s2(2,0) = 0.590625
s2(0,0) = 0.668750, s2(0,1) = 0.631250, s2(0,2) = 0.602500
s3(0,0,0,0) = 0.668750, s3(1,0,0,0) = 0.626250, s3(2,0,0,0) = 0.590625
s3(0,0,0,0) = 0.668750, s3(0,1,0,0) = 0.631250, s3(0,2,0,0) = 0.602500
s3(0,0,0,0) = 0.668750, s3(0,0,1,0) = 0.626250, s3(0,0,2,0) = 0.590625
s3(0,0,0,0) = 0.668750, s3(0,0,0,1) = 0.631250, s3(0,0,0,2) = 0.602500


s2(0,0) = 0.668750, s2(-1,0) = 0.626250, s2(-2,0) = 0.590625
s2(0,0) = 0.668750, s2(0,-1) = 0.631250, s2(0,-2) = 0.602500
s3(0,0,0,0) = 0.668750, s3(-1,0,0,0) = 0.626250, s3(-2,0,0,0) = 0.590625
s3(0,0,0,0) = 0.668750, s3(0,-1,0,0) = 0.631250, s3(0,-2,0,0) = 0.602500
s3(0,0,0,0) = 0.668750, s3(0,0,-1,0) = 0.626250, s3(0,0,-2,0) = 0.590625
s3(0,0,0,0) = 0.668750, s3(0,0,0,-1) = 0.631250, s3(0,0,0,-2) = 0.602500

```

#### Public functions of NpcfTools:

- void **get_anisotropic_map_s3**(int **nx**, int **ny**, string **fname**="")

  Computes stuff.

  - nx, ny :

  - dx1, dy1 :

  - dx2, dy2 :

  - fname (optional) :

    ​

- void **get_anisotropic_map_s3**(int **nx**, int **ny**, int **dx1**, int **dy1**, int **dx2**, int **dy2**, string **fname**="")

  Computes stuff.

  - nx, ny :

  - dx1, dy1 :

  - dx2, dy2 :

  - fname (optional) 

    ​

Example:

```c++
#include "NpcfTools.h"

NpcfTools npcf("im00.csv");
npcf.get_anisotropic_map_s2(60,60,"im00.s2");
npcf.get_anisotropic_map_s3(60,60,0,1,1,0,"im00.s3");
```



#### Formats of output files

##### 	Anisotropic estimators:

- foo.s2 : Complete anisotropic 2-pcf estimator.

  ```
  nx,ny
  s2(-nx,0)  , s2(-nx,1)  , ...., s2(-nx,ny-1)  , s2(-nx,ny)
  s2(-nx+1,0), s2(-nx+1,1), ...., s2(-nx+1,ny-1), s2(-nx+1,ny)
  s2(-nx+2,0), s2(-nx+2,1), ...., s2(-nx+2,ny-1), s2(-nx+2,ny)
     :             :                  :               :
     :             :                  :               :
  s2(-1,0)   , s2(-1,1)   , ...., s2(-1,ny-1)   , s2(-1,ny)
  s2(0,0)    , s2(0,1)    , ...., s2(0,ny-1)    , s2(0,ny)
  s2(1,0)    , s2(1,1)    , ...., s2(1,ny-1)    , s2(1,ny)
     :             :                  :               :
     :             :                  :               :
  s2(nx-1,0) , s2(nx-1,1) , ...., s2(nx-1,ny-1) , s2(nx-1,ny)
  s2(nx,0)   , s2(nx,1)   , ...., s2(nx,ny-1)   , s2(nx,ny)
  ```


- foo.s3 : Anisotropic 3-pcf estimator of point configurations with fixed opening and rotation angles.

  ```
  nx,ny
  dx1,dy1,dx2,dyy
  s3(0,0,0,0)          , s3(0,0,dx2,dy2)          , ...., s3(0,0,ny*dx2,ny*dy2)
  s3(dx1,dy1,0,0)      , s3(dx1,dy1,dx2,dy2)      , ...., s3(dx,dy,ny*dx2,ny*dy2)
  s3(2*dx1,2*dy1,0,0)  , s3(2*dx1,2*dy1,dx2,dy2)  , ...., s3(2*dx,2*dy,ny*dx2,ny*dy2)
        :                          :                                   :
        :                          :                                   :
  s3(nx*dx1,nx*dy1,0,0), s3(nx*dx1,nx*dy1,dx2,dy2), ...., s3(nx*dx1,nx*dy1,ny*dx2,ny*dy2)
  ```


- foo.s4 : (?)-tropic 4-pcf estimator.

  ```
  nx,ny
  x1,y1,x2,y2
  s3(0,0,0,0)                  ,
  s3(dx1,dy1,0,0)              ,
  s3(2*dx1,2*dy1,0,0)          , s3(2*dx1,2*dy1,0,0),
        :
        :     
  s3((nx-1)*dx1,(nx-1)*dy1,0,0), 
  s3(nx*dx1,nx*dy1,0,0)        ,
  ```

  ##### Isotropic estimators:

- foo.iso-s2 : Isotropic 2-pcf estimator.

  ```
  -
  ```


- foo.iso-s3 : Isotropic 3-pcf estimator.

  ```
  -
  ```


- foo.iso-s4 : Isotropic 4-pcf estimator.

  ```
  -
  ```

#### Pending tasks:

 - ERROR to fix when almost all data entries are equal. For example, try

    ```bash
    ./dist/GNU-Linux/npcf 12 11 0 0 im00.csv
    ```

 - Compute hs3 on minimum domain and copy values to other components instead of repeating calculations.

 - Write subroutines to write output files.

 - Complete implementation of S4(dx1,dy1,dx2,dy2,dx3,dy3).

 - Verify implementation for odd nx and ny.

 - Add inference subroutines.

 - $a$