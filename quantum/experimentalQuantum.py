import numpy as np
import pyqubo as pq
import neal

a = np.random.rand(2,2)
s1 =pq.Array.create('b', shape=2, vartype='BINARY')

H = pq.Array.dot(s1, pq.Array.dot(a,s1))

print(H)
model = H.compile()

bqm = model.to_bqm()

sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=10)
decoded_samples = model.decode_sampleset(sampleset)
best_sample = min(decoded_samples, key=lambda x: x.energy)
print(best_sample.sample)
