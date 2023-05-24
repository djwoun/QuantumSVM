import numpy as np 
import time
from sklearn import datasets

def svm2qubo(X, Y, p):
	N = X.shape[0]
	
	P = np.kron(np.eye(N), p)

	A = 0.5 * P.T @ (np.multiply(X@X.T, np.outer(Y, Y))) @ P

	b = - P.T @ np.ones(N)

	return A, b




N = 256
p = np.array([0.5, 1, 2])
#X = np.random.rand(N, d)
#Y = np.random.choice([-1, 1], N)
#X, Y = datasets.make_blobs(n_samples=N, centers=2, n_features=2**20,cluster_std=0.001,
#                           random_state=1,center_box=(0,10))

print(X.shape)
print(Y.shape)
print(p.shape)



start = time.time()

A, b = svm2qubo(X, Y, p)

end = time.time()


print(end - start)
