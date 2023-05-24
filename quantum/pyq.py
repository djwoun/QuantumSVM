import pyqubo
from pyqubo import Spin
from pyqubo import Array
s1 = Array.create('s', shape=2, vartype='BINARY')

print(s1)

f = [[2,3],[4,5]]
g = [4,5]
H = Array.dot (Array.dot(s1,f),s1 ) +Array.dot(Array(g),s1.T)
#H = Array(H)
#  ((np.outer(s1, s1))@f) #+ g@s2.T
#print(H)

model = H.compile()
qubo, offset = model.to_qubo()
print(qubo)
