import time, sys

def task_bar(i, n, t0=-1, bar=10):
  if n>1:
    if (i>=0) & (i<n):
      cnt=i*bar/(n-1)
      tbar='\r['+cnt*'='+(bar-cnt)*' '
      print(tbar+'] %g%% remaining' %((1.*bar-cnt)/bar*100)),
      sys.stdout.flush()
      if i==n-1:
        if t0>0:
          print('\rDone in %g sec\033[K' %(time.time()-t0))
        else:
          print('\rDone\033[K')
        sys.stdout.flush()