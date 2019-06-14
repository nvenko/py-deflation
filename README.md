## deflation-precond-strategies-sde

##### Enables testing and applications of deflation and preconditioning strategies to solve sequences of sampled finite element (FE) discretization of stochastic differential equations (SDE).

Author: Nicolas Venkovic

email: [venkovic@cerfacs.fr](mailto:venkovic@cerfacs.fr)

_TeX expressions rendered by [TeXify](https://github.com/apps/texify)._

#### Dependencies:

 - *Python* (2.x >= 2.6)
 - *SciPy* (>= 0.10)
 - *NumPy* (>= 1.6)

#### Files' content:

- samplers.py : 
  - Samples Karhunen-Loève (KL) representation of lognormal coefficient field <img src="/tex/4e923e372a3b20814e440dbab89a7369.svg?invert_in_darkmode&sanitize=true" align=middle width=47.13094649999999pt height=24.65753399999998pt/> proceeding either by (1) Monte Carlo (MC), or by (2) Markov chain Monte Carlo (MCMC). 
  - Assembles sampled operator <img src="/tex/5fa09f7b56ca22a36927cd31898c37bf.svg?invert_in_darkmode&sanitize=true" align=middle width=35.25112634999999pt height=24.65753399999998pt/>  for the stochastic system <img src="/tex/6e9d068d7e25f913f84dcfaacf3ff1ef.svg?invert_in_darkmode&sanitize=true" align=middle width=99.13217159999998pt height=24.65753399999998pt/> coming from a P0-FE discretization of the SDE <img src="/tex/4f6673d7b4b130cf0e1a5e39aa117b67.svg?invert_in_darkmode&sanitize=true" align=middle width=204.04130339999995pt height=24.65753399999998pt/>. 
- solvers.py : 

  - Iterative solvers: Conjugate gradient (CG), preconditioned CG (PCG), deflated CG (DCG) and preconditioned DCG (PDCG).
- recycling.py : 
  - Solves the sequence of linear systems <img src="/tex/392bdf51092e90ac7adfc4fe70a64bdd.svg?invert_in_darkmode&sanitize=true" align=middle width=109.79427854999999pt height=24.65753399999998pt/>  for a sample <img src="/tex/dfdd1736db6cf1ba2b6bb459252a9b95.svg?invert_in_darkmode&sanitize=true" align=middle width=78.63031109999999pt height=27.6567522pt/> with:
    - PCG for a sequence of systems with multiple operators (PCGMO) :
      - Preconditioners: 

  - Constant: Median operator <img src="/tex/01470c7d9dc1c9d0a9462973bbded215.svg?invert_in_darkmode&sanitize=true" align=middle width=14.29216634999999pt height=31.23293909999999pt/>, Algebraic multi-grid (AMG) based on <img src="/tex/01470c7d9dc1c9d0a9462973bbded215.svg?invert_in_darkmode&sanitize=true" align=middle width=14.29216634999999pt height=31.23293909999999pt/>, block Jacobi (bJ) based on <img src="/tex/01470c7d9dc1c9d0a9462973bbded215.svg?invert_in_darkmode&sanitize=true" align=middle width=14.29216634999999pt height=31.23293909999999pt/> with <img src="/tex/1b59df1ffb5656ef316bdd6b2bdc2dfb.svg?invert_in_darkmode&sanitize=true" align=middle width=15.647713949999991pt height=14.15524440000002pt/> (non-overlapping) blocks.

    - Realization dependent: Periodically selected operators in sampled sequence,

    - DCG for a sequence with multiple operators (DCGMO) :
      - b
    - DPCG for a sequence with multiple operators (DPCGMO) :
      - c 

  - Interfaces sampler and solver while handling W, P, A, exact and approximate eigenvectors after different strategies: 

    - Current/previous
- post_recycling :

  - Plots results

#### Installation: 

```bash
make -f Makefile CONF=Release
```

#### Usage:

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

 - <img src="/tex/44bc9d542a92714cac84e01cbbb7fd61.svg?invert_in_darkmode&sanitize=true" align=middle width=8.68915409999999pt height=14.15524440000002pt/>