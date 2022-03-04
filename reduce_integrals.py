import scipy.special as sps
import numpy as np

def myrincbeta(x,a,b):
# compute the regularized incomplete beta function.
  if a < 0:
      cbf=(sps.gamma(a)*sps.gamma(b))/sps.gamma(a+b)
#      c1 = np.where(x <= 0.0)
#      c2 = np.where(x >= 1.0)
#      print x.shape
#      if np.sum(c1) > 0:
#        print "The problem is in X <=0"
#        import pdb; pdb.set_trace()
#      if np.sum(c2) > 0:
#        print "The problem is in X >=1"
#        import pdb; pdb.set_trace()
      res = (x**a * (1.0-x)**b) / (a * cbf)

#      print a,b,cbf,x.shape
      return myrincbeta(x,a+1.0,b) + res
  else:
#      cbf=(sps.gamma(a)*sps.gamma(b))/sps.gamma(a+b)
      cbf=1.0 # sps.betainc is the regularized inc. beta fun.
      res=(sps.betainc(a,b,x) / cbf)
      return res
    
def myredcosine(tmax,n):
# computes \int_0^tmax cos^n(x) dx
  if n < -2:
      res=np.cos(tmax)**(n+1)*np.sin(tmax)/(n+1) 
      return myredcosine(tmax,n+2)*(n+2)/(n+1) - res
  else:
      if n == -1.0:
          res=np.log(np.absolute(1.0/np.cos(tmax) + np.tan(tmax)) )
      if n == -2.0:
          res=np.tan(tmax) 

      return res
