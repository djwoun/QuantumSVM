import time
import numpy as np
from sklearn import datasets
from numba import jit


def go_fast(p,X,y):
    N = X.shape[0] 
    P = np.kron(np.eye(N), p)
    A = 0.5 * P.T @ (np.multiply(X@X.T, np.outer(y, y))) @ P
    b = - P.T @ np.ones(N)
    
    return A, b
    
    #return 0.5*(( P.T @np.multiply((X@X.T),
    #      np.outer(y, y)) @P ), -P.T@Ones
    #0.5*(( PrecisionM.T @np.multiply((X@X.T),
           #(y))) @PrecisionM ) - PrecisionM.T@Ones
     

standardDeviation = 0.001


# 256 data point 2**22 is the maximum for reasonable results - no virtual memory
# 512 data points 2**20 is the maximum for reasonable results
features = 2
Samples = 256
avgTime = 0


X, y = datasets.make_blobs(n_samples=Samples, centers=2, n_features=features,cluster_std=0.001,
                           random_state=1,center_box=(0,10))
p = np.array([0.5,1,2])



            
#print(time.process_time() )


a,b = go_fast(p,X,y) 
#print(time.process_time() )
start = time.time()


#print(y)
#0.5*(( PrecisionM.T @np.multiply((X@X.T),
#      (y[:,None]@y[None,:] ))) @PrecisionM ) - PrecisionM.T@Ones
#(timeit.timeit ( 'go_fast(PrecisionM,X,y,Ones)', 'from __main__ import go_fast, PrecisionM,X,y,Ones', number=10 )
   
#print(go_fast(p,X,y) )
#timeit.timeit
# 'add(a, b)', 'from __main__ import add, a, b'
end = time.time()
#print(time.process_time() )
#print(end-start)



import pyqubo as pq
from pyqubo import Spin
from pyqubo import Array
s = Array.create('s', shape=768, vartype='SPIN')
H =  (s.T@a)@s     - b@s.T

model = H.compile()
print("\n")
qubo, offset = model.to_qubo()
print(qubo)
                        
            