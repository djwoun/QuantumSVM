# -*- coding: utf-8 -*-
import csv
from sklearn.datasets import load_breast_cancer
import numpy as np
import random
Wbc = load_breast_cancer()

X =  Wbc.data[:, :]
y =  Wbc.target

#@print(X[0].size)
b = np.vstack(y)
print(b.size)

print(X.size)

a = np.concatenate([X, b],axis=1)

print(a)
print(a.size)
#virginica = X [100:150, :2]
#setosa = X [:50,:2]
#versicolor = X [50:100, :2]


for x in range (30):
    
   
    np.random.shuffle(a)
    
    ran = random.sample(range(569), int(170))
    
    
    np.savetxt('WbcCSVFiles/Wbc'+str(x)+'.csv', a, delimiter=',') 
 
    np.savetxt('WbcCSVFiles/random'+str(x)+'.csv', ran, delimiter=',')
   
    '''
    with open('WbcCSVFiles/test'+str(x)+'.csv', "w") as file:
        
        spamwriter = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_NONE)
        spamwriter.writerow([(str(1.1112233), str(1.3355)),(str(1.1112233), str(1.3355))])
    '''
        #fieldnames = ['setosa','versicolor']
        #theWriter = csv.DictWriter(f,fieldnames=fieldnames)
        #theWriter.writeheader()
        #for i in range(50):
         #   theWriter.writerow({'setosa':setosa[i],'versicolor':versicolor[i]})
